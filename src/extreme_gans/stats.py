from __future__ import annotations

import numpy as np
import torch


def generic_time_series_stats(X: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Differentiable summary stats for arbitrary [B,H,F] time-series windows."""
    if X.ndim != 3:
        raise ValueError("X must have shape [batch,horizon,features]")
    flat = X.reshape(X.shape[0], -1)
    terminal_sum = X.sum(dim=1)                      # [B,F]
    mean = X.mean(dim=1)
    std = X.std(dim=1, unbiased=False)
    maxv = X.amax(dim=1)
    minv = X.amin(dim=1)
    absmax = X.abs().amax(dim=1)
    realized_vol = torch.sqrt((X ** 2).sum(dim=1).clamp_min(eps))
    path = torch.cumsum(X, dim=1)
    drawdown = torch.cummax(path, dim=1).values - path
    max_drawdown = drawdown.amax(dim=1)
    # Compress by averaging across features plus keep first few feature summaries.
    blocks = [
        terminal_sum.mean(dim=1, keepdim=True), mean.mean(dim=1, keepdim=True), std.mean(dim=1, keepdim=True),
        maxv.mean(dim=1, keepdim=True), minv.mean(dim=1, keepdim=True), absmax.mean(dim=1, keepdim=True),
        realized_vol.mean(dim=1, keepdim=True), max_drawdown.mean(dim=1, keepdim=True),
        terminal_sum[:, : min(4, X.shape[-1])], std[:, : min(4, X.shape[-1])],
    ]
    return torch.cat(blocks, dim=1)


def choose_stats(X: torch.Tensor, dataset: str, n_levels: int = 10) -> torch.Tensor:
    return generic_time_series_stats(X)


def extremeness_score_np(x, mode="mean", tail="high"):
    x = np.asarray(x)
    flat = x.reshape(x.shape[0], -1)

    if mode == "mean":
        score = flat.mean(axis=1)
    elif mode == "max":
        score = flat.max(axis=1)
    elif mode == "min":
        score = flat.min(axis=1)
    elif mode == "abs_mean":
        score = np.abs(flat).mean(axis=1)
    elif mode == "volatility":
        score = flat.std(axis=1)
    elif mode == "drawdown":
        path = np.cumsum(flat, axis=1)
        score = np.max(np.maximum.accumulate(path, axis=1) - path, axis=1)
    else:
        raise ValueError(f"Unknown extremeness mode: {mode}")

    return -score if tail == "low" else score


def make_extreme_labels_np(x, mode="mean", tail="high", n_bins=3):
    score = extremeness_score_np(x, mode=mode, tail=tail)
    qs = np.quantile(score, np.linspace(0, 1, n_bins + 1)[1:-1])
    labels = np.digitize(score, qs)
    return labels.astype(np.int64), score.astype(np.float32)


def one_hot_labels(labels, n_classes, device):
    labels = torch.as_tensor(labels, dtype=torch.long, device=device)
    return torch.nn.functional.one_hot(labels, num_classes=n_classes).float()


def extreme_metric_torch(x, mode="mean", tail="high"):
    flat = x.view(x.shape[0], -1)

    if mode == "mean":
        score = flat.mean(dim=1)
    elif mode == "max":
        score = flat.max(dim=1).values
    elif mode == "min":
        score = flat.min(dim=1).values
    elif mode == "abs_mean":
        score = flat.abs().mean(dim=1)
    elif mode == "volatility":
        score = flat.std(dim=1)
    elif mode == "drawdown":
        path = torch.cumsum(flat, dim=1)
        running_max = torch.cummax(path, dim=1).values
        score = torch.max(running_max - path, dim=1).values
    else:
        raise ValueError(f"Unknown extremeness mode: {mode}")

    return -score if tail == "low" else score
