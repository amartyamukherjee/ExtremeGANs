from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn


def bce_discriminator_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor, label_smoothing: float = 0.05) -> torch.Tensor:
    real_targets = torch.full_like(real_logits, 1.0 - label_smoothing)
    fake_targets = torch.zeros_like(fake_logits)
    return F.binary_cross_entropy_with_logits(real_logits, real_targets) + F.binary_cross_entropy_with_logits(fake_logits, fake_targets)


def bce_generator_loss(fake_logits: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits))


def gradient_penalty(discriminator: nn.Module, real: torch.Tensor, fake: torch.Tensor, device: torch.device) -> torch.Tensor:
    b = real.shape[0]
    eps = torch.rand(b, 1, device=device)
    while eps.ndim < real.ndim:
        eps = eps.unsqueeze(-1)
    interp = (eps * real + (1 - eps) * fake).requires_grad_(True)
    pred = discriminator(interp)
    grad = torch.autograd.grad(pred.sum(), interp, create_graph=True, retain_graph=True, only_inputs=True)[0]
    grad = grad.reshape(b, -1)
    return ((grad.norm(2, dim=1) - 1.0) ** 2).mean()


def deterministic_neuralsort(s: torch.Tensor, tau: float) -> torch.Tensor:
    # s: [n_stats, batch, 1]
    n = s.size(1)
    one = torch.ones((n, 1), device=s.device, dtype=s.dtype)
    A_s = torch.abs(s - s.permute(0, 2, 1))
    B = torch.matmul(A_s, torch.matmul(one, one.T))
    scaling = n + 1 - 2 * (torch.arange(n, device=s.device, dtype=s.dtype) + 1)
    C = torch.matmul(s, scaling.unsqueeze(0))
    P_max = (C - B).permute(0, 2, 1)
    return torch.softmax(P_max / tau, dim=-1)


def G1(v): return v

def G2(e, scale: float = 1.0): return scale * torch.exp(e / scale)

def G2in(e, scale: float = 1.0): return scale ** 2 * torch.exp(e / scale)

def G1_quant(v, W: float): return -W * v ** 2 / 2

def G2_quant(e, alpha: float): return alpha * e

def G2in_quant(e, alpha: float): return alpha * e ** 2 / 2


def S_stats(v: torch.Tensor, e: torch.Tensor, X: torch.Tensor, alpha: float) -> torch.Tensor:
    if alpha < 0.5:
        return torch.mean(((X <= v).float() - alpha) * (G1(v) - G1(X)) + (1.0 / alpha) * G2(e) * (X <= v).float() * (v - X) + G2(e) * (e - v) - G2in(e))
    a = 1.0 - alpha
    return torch.mean(((X >= v).float() - a) * (G1(X) - G1(v)) + (1.0 / a) * G2(-e) * (X >= v).float() * (X - v) + G2(-e) * (v - e) - G2in(-e))


def S_quant(v: torch.Tensor, e: torch.Tensor, X: torch.Tensor, alpha: float, W: float) -> torch.Tensor:
    if alpha < 0.5:
        return torch.mean(((X <= v).float() - alpha) * (G1_quant(v, W) - G1_quant(X, W)) + (1.0 / alpha) * G2_quant(e, alpha) * (X <= v).float() * (v - X) + G2_quant(e, alpha) * (e - v) - G2in_quant(e, alpha))
    a = 1.0 - alpha
    return torch.mean(((X >= v).float() - a) * (G1_quant(v, W) - G1_quant(X, W)) + (1.0 / a) * G2_quant(-e, a) * (X >= v).float() * (X - v) + G2_quant(-e, a) * (v - e) - G2in_quant(-e, a))


def tail_score_loss(validity: torch.Tensor, stats: torch.Tensor, alphas: Sequence[float], score: str = "quant", W: float = 10.0) -> torch.Tensor:
    # validity: [n_stats, 2*len(alphas)], stats: [batch, n_stats]
    X = stats.T
    loss = torch.tensor(0.0, device=stats.device, dtype=stats.dtype)
    for i, alpha in enumerate(alphas):
        v = validity[:, [2 * i]]
        e = validity[:, [2 * i + 1]]
        loss = loss + (S_quant(v, e, X, alpha, W) if score == "quant" else S_stats(v, e, X, alpha))
    return loss

def root_euclidean_distance(x, y, gamma=2.0, eps=1e-8):
    diff = x[:, None, :] - y[None, :, :]
    dist = torch.sqrt(torch.sum(diff * diff, dim=-1) + eps)
    return dist.pow(1.0 / gamma)

def energy_distance_loss(real, fake, gamma=2.0):
    real = real.view(real.shape[0], -1)
    fake = fake.view(fake.shape[0], -1)

    d_rf = root_euclidean_distance(real, fake, gamma=gamma).mean()
    d_rr = root_euclidean_distance(real, real, gamma=gamma).mean()
    d_ff = root_euclidean_distance(fake, fake, gamma=gamma).mean()

    return 2.0 * d_rf - d_rr - d_ff

def gradient_penalty_conditional(discriminator, real, fake, y, device, lambda_gp=10.0):
    b = real.shape[0]
    eps_shape = [b] + [1] * (real.ndim - 1)
    eps = torch.rand(*eps_shape, device=device)

    interp = eps * real + (1.0 - eps) * fake
    interp.requires_grad_(True)

    d_interp = discriminator(interp.view(b, -1), y)

    grad = torch.autograd.grad(
        outputs=d_interp,
        inputs=interp,
        grad_outputs=torch.ones_like(d_interp),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    grad = grad.view(b, -1)
    return lambda_gp * ((grad.norm(2, dim=1) - 1.0) ** 2).mean()


def cdcgan_extreme_loss(fake, target_score, mode="mean", tail="high", eps=1e-6):
    from .stats import extreme_metric_torch

    fake_score = extreme_metric_torch(fake, mode=mode, tail=tail)
    target_score = target_score.view(-1)

    return torch.mean(torch.abs(fake_score - target_score) / (target_score.abs() + eps))
