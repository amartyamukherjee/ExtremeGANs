from __future__ import annotations

import json
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .models import MLPGenerator, MLPDiscriminator, EVGenerator, TailStatisticDiscriminator, parse_hidden
from .losses import bce_discriminator_loss, bce_generator_loss, gradient_penalty, deterministic_neuralsort, tail_score_loss
from .stats import choose_stats
from .transforms import EmpiricalFrechetTransform, PositiveTailTransform
from .metrics import evaluate_all


def seed_everything(seed: int):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def sample_noise(batch_size: int, latent_dim: int, device: torch.device, kind: str = "normal", gamma: float = 2/3):
    if kind == "uniform":
        return torch.rand(batch_size, latent_dim, device=device).clamp(1e-5, 1 - 1e-5)
    if kind == "pareto_log":
        u = torch.rand(batch_size, latent_dim, device=device).clamp(1e-6, 1 - 1e-6)
        return gamma * (-torch.log1p(-u))
    if kind == "student":
        return torch.from_numpy(np.random.standard_t(3, size=(batch_size, latent_dim)).astype("float32")).to(device)
    return torch.randn(batch_size, latent_dim, device=device)

def sample_pareto_noise(batch_size, latent_dim, xi, device, eps=1e-6, clip=1e6):
    u = torch.rand(batch_size, latent_dim, device=device).clamp(eps, 1.0 - eps)

    if abs(xi) < 1e-8:
        z = -torch.log(u)
    else:
        z = (u.pow(-xi) - 1.0) / xi

    return torch.clamp(z, max=clip)

def hill_tail_index_estimate(x, tail_fraction=0.05, eps=1e-8):
    x = np.asarray(x, dtype=np.float64)
    x = np.abs(x.reshape(-1))
    x = x[np.isfinite(x)]
    x = x[x > eps]

    if len(x) < 20:
        return 0.5

    x = np.sort(x)
    k = max(5, int(tail_fraction * len(x)))
    tail = x[-k:]
    threshold = x[-k]

    xi = np.mean(np.log(tail + eps) - np.log(threshold + eps))
    return float(max(xi, 1e-3))

@torch.no_grad()
def generate_flat(generator, n: int, latent_dim: int, device: torch.device, batch_size: int, noise_kind: str = "normal", gamma: float = 2/3) -> np.ndarray:
    generator.eval()
    outs = []
    while sum(len(x) for x in outs) < n:
        b = min(batch_size, n - sum(len(x) for x in outs))
        z = sample_noise(b, latent_dim, device, noise_kind, gamma)
        outs.append(generator(z).detach().cpu().numpy())
    return np.concatenate(outs, axis=0)


def train_bce_gan(data_flat: np.ndarray, args, device: torch.device, final_activation: str | None = "tanh", noise_kind: str = "normal"):
    dataset = TensorDataset(torch.tensor(data_flat, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    data_dim = data_flat.shape[1]
    latent_dim = args.latent_dim or min(128, data_dim)
    G = MLPGenerator(latent_dim, data_dim, parse_hidden(args.gen_hidden), final_activation=final_activation).to(device)
    D = MLPDiscriminator(data_dim, parse_hidden(args.disc_hidden)).to(device)
    optG = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    optD = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    history = []
    for epoch in range(args.epochs):
        g_loss_val = d_loss_val = 0.0
        for (real,) in loader:
            real = real.to(device)
            b = real.shape[0]
            # D
            z = sample_noise(b, latent_dim, device, noise_kind, gamma=args.ht_gamma)
            fake = G(z).detach()
            optD.zero_grad()
            d_loss = bce_discriminator_loss(D(real), D(fake), args.label_smoothing)
            d_loss.backward(); optD.step()
            # G
            z = sample_noise(b, latent_dim, device, noise_kind, gamma=args.ht_gamma)
            optG.zero_grad()
            g_loss = bce_generator_loss(D(G(z)))
            g_loss.backward(); optG.step()
            g_loss_val += float(g_loss.item()); d_loss_val += float(d_loss.item())
        if epoch % args.log_every == 0 or epoch == args.epochs - 1:
            history.append({"epoch": epoch, "g_loss": g_loss_val/max(len(loader),1), "d_loss": d_loss_val/max(len(loader),1)})
            print(history[-1])
    return G, {"history": history, "latent_dim": latent_dim, "noise_kind": noise_kind}


def train_wgan_gp(data_flat: np.ndarray, args, device: torch.device, final_activation: str | None = "tanh"):
    dataset = TensorDataset(torch.tensor(data_flat, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    data_dim = data_flat.shape[1]
    latent_dim = args.latent_dim or min(128, data_dim)
    G = MLPGenerator(latent_dim, data_dim, parse_hidden(args.gen_hidden), final_activation=final_activation).to(device)
    D = MLPDiscriminator(data_dim, parse_hidden(args.disc_hidden)).to(device)
    optG = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    optD = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    history = []
    for epoch in range(args.epochs):
        g_loss_val = d_loss_val = 0.0
        for i, (real,) in enumerate(loader):
            real = real.to(device); b = real.shape[0]
            for _ in range(args.n_critic):
                z = sample_noise(b, latent_dim, device)
                fake = G(z).detach()
                optD.zero_grad()
                d_loss = D(fake).mean() - D(real).mean() + args.gp_lambda * gradient_penalty(D, real, fake, device)
                d_loss.backward(); optD.step()
            z = sample_noise(b, latent_dim, device)
            optG.zero_grad()
            g_loss = -D(G(z)).mean()
            g_loss.backward(); optG.step()
            g_loss_val += float(g_loss.item()); d_loss_val += float(d_loss.item())
        if epoch % args.log_every == 0 or epoch == args.epochs - 1:
            history.append({"epoch": epoch, "g_loss": g_loss_val/max(len(loader),1), "d_loss": d_loss_val/max(len(loader),1)})
            print(history[-1])
    return G, {"history": history, "latent_dim": latent_dim, "noise_kind": "normal"}


def train_evgan(data_flat: np.ndarray, args, device: torch.device):
    pos = PositiveTailTransform(mode=args.ev_positive_mode, standardize_scale=True)
    y = pos.fit_transform(data_flat)
    dataset = TensorDataset(torch.tensor(y, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    data_dim = y.shape[1]
    latent_dim = max(args.latent_dim or data_dim, data_dim)
    G = EVGenerator(latent_dim, data_dim, hidden_units=args.ev_hidden_units, max_tif=args.ev_max_tif).to(device)
    D = MLPDiscriminator(data_dim, parse_hidden(args.disc_hidden)).to(device)
    optG = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    optD = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    history = []
    for epoch in range(args.epochs):
        gl = dl = 0.0
        for (real,) in loader:
            real = real.to(device); b = real.shape[0]
            z = sample_noise(b, latent_dim, device, "uniform")
            fake = G(z).detach()
            optD.zero_grad()
            d_loss = bce_discriminator_loss(D(torch.log1p(real)), D(torch.log1p(fake)), args.label_smoothing)
            d_loss.backward(); optD.step()
            z = sample_noise(b, latent_dim, device, "uniform")
            optG.zero_grad()
            g_loss = bce_generator_loss(D(torch.log1p(G(z))))
            g_loss.backward(); optG.step()
            gl += float(g_loss.item()); dl += float(d_loss.item())
        if epoch % args.log_every == 0 or epoch == args.epochs - 1:
            history.append({"epoch": epoch, "g_loss": gl/max(len(loader),1), "d_loss": dl/max(len(loader),1)})
            print(history[-1])
    return G, {"history": history, "latent_dim": latent_dim, "noise_kind": "uniform", "positive_transform": pos}

def tail_stats_gradient_penalty(TD, real_stats, fake_stats, args, device):
    b = real_stats.shape[0]
    eps = torch.rand(b, 1, device=device)

    interp_stats = eps * real_stats + (1.0 - eps) * fake_stats
    interp_stats.requires_grad_(True)

    stats_T = interp_stats.T
    s = stats_T.reshape(*stats_T.shape, 1)
    P = deterministic_neuralsort(s, args.neuralsort_temp)
    sorted_stats = torch.bmm(P, s).reshape(*stats_T.shape)

    validity = TD(sorted_stats)
    score = tail_score_loss(
        validity,
        interp_stats,
        args.alphas,
        args.tail_score,
        args.tail_W,
    )

    grad = torch.autograd.grad(
        outputs=score,
        inputs=interp_stats,
        grad_outputs=torch.ones_like(score),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    grad = grad.reshape(b, -1)
    gp = ((grad.norm(2, dim=1) - 1.0) ** 2).mean()
    return args.lambda_gp * gp

def train_tailgan(windows_norm: np.ndarray, args, device: torch.device):
    B,H,F = windows_norm.shape
    data_dim = H * F
    latent_dim = args.latent_dim or min(128, data_dim)
    dataset = TensorDataset(torch.tensor(windows_norm, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    G = MLPGenerator(latent_dim, data_dim, parse_hidden(args.gen_hidden), final_activation="tanh").to(device)
    # infer n_stats with one batch
    sample_stats = choose_stats(torch.tensor(windows_norm[:args.batch_size], dtype=torch.float32), args.dataset, args.n_levels)
    TD = TailStatisticDiscriminator(args.batch_size, len(args.alphas), parse_hidden(args.tail_disc_hidden)).to(device)
    optG = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    optD = torch.optim.Adam(TD.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    history = []
    for epoch in range(args.epochs):
        gl = dl = 0.0
        for (real_win,) in loader:
            real_win = real_win.to(device); b = real_win.shape[0]
            if b != args.batch_size:
                continue
            z = sample_noise(b, latent_dim, device, args.tail_noise)
            fake_win = G(z).view(b, H, F).detach()
            real_stats = choose_stats(real_win, args.dataset, args.n_levels)
            fake_stats = choose_stats(fake_win, args.dataset, args.n_levels)
            def validity_from_stats(stats):
                stats_T = stats.T
                s = stats_T.reshape(*stats_T.shape, 1)
                P = deterministic_neuralsort(s, args.neuralsort_temp)
                sorted_stats = torch.bmm(P, s).reshape(*stats_T.shape)
                return TD(sorted_stats)
            optD.zero_grad()
            real_valid = validity_from_stats(real_stats)
            fake_valid = validity_from_stats(fake_stats)
            real_score = tail_score_loss(
                real_valid,
                real_stats,
                args.alphas,
                args.tail_score,
                args.tail_W,
            )
            fake_score = tail_score_loss(
                fake_valid,
                fake_stats,
                args.alphas,
                args.tail_score,
                args.tail_W,
            )
            gp = tail_stats_gradient_penalty(
                TD,
                real_stats.detach(),
                fake_stats.detach(),
                args,
                device,
            )
            d_loss = -(real_score - fake_score) + gp
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(TD.parameters(), args.grad_clip)
            optD.step()
            z = sample_noise(b, latent_dim, device, args.tail_noise)
            fake_win = G(z).view(b, H, F)
            fake_stats = choose_stats(fake_win, args.dataset, args.n_levels)
            optG.zero_grad()
            fake_valid = validity_from_stats(fake_stats)
            g_loss = tail_score_loss(fake_valid, fake_stats, args.alphas, args.tail_score, args.tail_W)
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(G.parameters(), args.grad_clip)
            optG.step()
            gl += float(g_loss.item()); dl += float(d_loss.item())
        if epoch % args.log_every == 0 or epoch == args.epochs - 1:
            history.append({"epoch": epoch, "g_loss": gl/max(len(loader),1), "d_loss": dl/max(len(loader),1)})
            print(history[-1])
    return G, {"history": history, "latent_dim": latent_dim, "noise_kind": args.tail_noise, "horizon": H, "features": F}


def train_paretogan(
    windows_norm,
    args,
    device,
):
    from .models import MLPGenerator
    from .losses import energy_distance_loss

    n, horizon, n_features = windows_norm.shape
    data_dim = horizon * n_features

    x = torch.tensor(
        windows_norm.reshape(n, data_dim),
        dtype=torch.float32,
        device=device,
    )

    xi = args.pareto_xi
    if xi is None or xi <= 0:
        xi = hill_tail_index_estimate(
            windows_norm,
            tail_fraction=args.tail_fraction,
        )

    gamma = max(args.root_gamma, xi + 0.1)

    generator = MLPGenerator(
        latent_dim=args.latent_dim,
        output_dim=data_dim,
        hidden=parse_hidden(args.gen_hidden),
    ).to(device)

    opt_g = torch.optim.Adam(
        generator.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
    )

    history = []

    for epoch in range(args.epochs):
        perm = torch.randperm(n, device=device)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n, args.batch_size):
            idx = perm[start:start + args.batch_size]
            real = x[idx]
            b = real.shape[0]

            z = sample_pareto_noise(
                b,
                args.latent_dim,
                xi=xi,
                device=device,
                clip=args.pareto_clip,
            )

            fake = generator(z)
            loss = energy_distance_loss(real, fake, gamma=gamma)

            opt_g.zero_grad()
            loss.backward()

            opt_g.step()

            epoch_loss += float(loss.detach().cpu())
            n_batches += 1

        if epoch % args.log_every == 0 or epoch == args.epochs - 1:
            history.append(
                {
                    "epoch": epoch,
                    "loss": epoch_loss / max(n_batches, 1),
                    "xi": xi,
                    "root_gamma": gamma,
                }        
            )
            print(history[-1])

    return generator, {
        "history": history,
        "latent_dim": args.latent_dim,
        "pareto_xi": xi,
        "root_gamma": gamma,
    }


def train_cdcgan(windows_norm, args, device):
    from .models import ConditionalMLPGenerator, ConditionalMLPDiscriminator
    from .stats import make_extreme_labels_np, one_hot_labels
    from .losses import gradient_penalty_conditional, cdcgan_extreme_loss

    n, horizon, n_features = windows_norm.shape
    data_dim = horizon * n_features

    labels, scores = make_extreme_labels_np(
        windows_norm,
        mode=args.extreme_mode,
        tail=args.extreme_tail,
        n_bins=args.condition_dim,
    )

    x = torch.tensor(windows_norm.reshape(n, data_dim), dtype=torch.float32, device=device)
    labels_t = torch.tensor(labels, dtype=torch.long, device=device)
    scores_t = torch.tensor(scores, dtype=torch.float32, device=device)

    G = ConditionalMLPGenerator(args.latent_dim, args.condition_dim, data_dim, parse_hidden(args.gen_hidden)).to(device)
    D = ConditionalMLPDiscriminator(data_dim, args.condition_dim, parse_hidden(args.disc_hidden)).to(device)

    opt_g = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    opt_d = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

    history = []

    for epoch in range(args.epochs):
        perm = torch.randperm(n, device=device)

        for start in range(0, n, args.batch_size):
            idx = perm[start:start + args.batch_size]
            real = x[idx]
            b = real.shape[0]

            y = one_hot_labels(labels_t[idx], args.condition_dim, device)
            target_score = scores_t[idx]

            for _ in range(args.n_critic):
                z = torch.randn(b, args.latent_dim, device=device)
                fake = G(z, y).detach()

                d_real = D(real, y).mean()
                d_fake = D(fake, y).mean()
                gp = gradient_penalty_conditional(
                    D, real.view(b, horizon, n_features), fake.view(b, horizon, n_features),
                    y, device
                )

                loss_d = d_fake - d_real + gp

                opt_d.zero_grad()
                loss_d.backward()
                opt_d.step()

            z = torch.randn(b, args.latent_dim, device=device)
            fake = G(z, y)
            adv_loss = -D(fake, y).mean()
            ext_loss = cdcgan_extreme_loss(
                fake.view(b, horizon, n_features),
                target_score,
                mode=args.extreme_mode,
                tail=args.extreme_tail,
            )

            loss_g = adv_loss + args.extreme_loss_weight * ext_loss

            opt_g.zero_grad()
            loss_g.backward()
            opt_g.step()

        if epoch % args.log_every == 0 or epoch == args.epochs - 1:
            history.append({
                "epoch": epoch,
                "g_loss": float(loss_g.detach().cpu()),
                "d_loss": float(loss_d.detach().cpu()),
                "extreme_loss": float(ext_loss.detach().cpu()),
            })
            print(history[-1])

    return G, {
        "history": history,
        "latent_dim": args.latent_dim,
        "condition_dim": args.condition_dim,
        "extreme_mode": args.extreme_mode,
        "extreme_tail": args.extreme_tail,
        "horizon": horizon,
        "n_features": n_features,
    }


@dataclass
class TrainingResult:
    """Internal result of fitting an Extreme GAN algorithm."""

    algorithm: str
    generator: torch.nn.Module
    info: dict[str, Any]
    device: torch.device
    horizon: int
    n_features: int
    transform: Any = None


def fit_algorithm(windows_norm: np.ndarray, args) -> TrainingResult:
    """Fit an algorithm without generating samples, evaluating, or writing files.

    This function is intentionally side-effect-light so it can power both the
    public Python API and the legacy command-line interface.
    """

    seed_everything(args.seed)
    device = torch.device(
        args.device if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    windows_norm = np.asarray(windows_norm, dtype=np.float32)
    if windows_norm.ndim != 3:
        raise ValueError("windows_norm must have shape [n_samples, horizon, n_features]")

    H, F = windows_norm.shape[1], windows_norm.shape[2]
    flat_norm = windows_norm.reshape(len(windows_norm), -1)
    data_dim = flat_norm.shape[1]

    # A few legacy trainers expected latent_dim to be explicitly positive,
    # while others used 0 as an automatic sentinel. Normalize that behavior.
    if not getattr(args, "latent_dim", 0):
        args.latent_dim = min(128, data_dim)

    start = time.time()
    transform = None

    if args.algorithm == "htgan":
        gamma = 1.0 / args.ht_alpha
        # Keep the training and sampling tail parameter consistent.
        args.ht_gamma = gamma
        transform = EmpiricalFrechetTransform(
            gamma=gamma,
            train_space=args.ht_train_space,
        )
        y = transform.fit_transform(flat_norm)
        G, info = train_bce_gan(
            y,
            args,
            device,
            final_activation="softplus",
            noise_kind="pareto_log",
        )
        info["empirical_frechet"] = transform.state_dict()

    elif args.algorithm == "evgan":
        G, info = train_evgan(flat_norm, args, device)
        transform = info.pop("positive_transform")
        info["positive_transform"] = transform.state_dict()

    elif args.algorithm == "tailgan":
        G, info = train_tailgan(windows_norm, args, device)

    elif args.algorithm == "wgan":
        G, info = train_wgan_gp(flat_norm, args, device, final_activation="tanh")

    elif args.algorithm == "vanilla":
        G, info = train_bce_gan(
            flat_norm,
            args,
            device,
            final_activation="tanh",
            noise_kind="normal",
        )

    elif args.algorithm == "paretogan":
        G, info = train_paretogan(windows_norm, args, device)

    elif args.algorithm == "cdcgan":
        G, info = train_cdcgan(windows_norm, args, device)

    else:
        raise ValueError(f"Unknown algorithm {args.algorithm}")

    info["training_seconds"] = time.time() - start
    info["horizon"] = H
    info["n_features"] = F

    return TrainingResult(
        algorithm=args.algorithm,
        generator=G,
        info=info,
        device=device,
        horizon=H,
        n_features=F,
        transform=transform,
    )


def generate_algorithm(
    result: TrainingResult,
    n_samples: int,
    *,
    batch_size: int = 1024,
    cdcgan_label: int | None = None,
) -> np.ndarray:
    """Generate samples in the normalized dataset space."""

    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    G = result.generator
    info = result.info
    latent_dim = int(info["latent_dim"])

    if result.algorithm == "cdcgan":
        condition_dim = int(info["condition_dim"])
        target_label = condition_dim - 1 if cdcgan_label is None else int(cdcgan_label)
        if not 0 <= target_label < condition_dim:
            raise ValueError(
                f"cdcgan_label must be between 0 and {condition_dim - 1}, got {target_label}"
            )

        G.eval()
        outs = []
        remaining = n_samples
        with torch.no_grad():
            while remaining > 0:
                b = min(batch_size, remaining)
                z = torch.randn(b, latent_dim, device=result.device)
                labels = torch.full(
                    (b,), target_label, dtype=torch.long, device=result.device
                )
                y = torch.nn.functional.one_hot(
                    labels, num_classes=condition_dim
                ).float()
                outs.append(G(z, y).detach().cpu().numpy())
                remaining -= b
        gen_flat_space = np.concatenate(outs, axis=0)[:n_samples]

    else:
        noise_kind = info.get("noise_kind", "normal")
        gamma = 1.0
        if result.algorithm == "htgan":
            gamma = float(result.transform.gamma)
        gen_flat_space = generate_flat(
            G,
            n_samples,
            latent_dim,
            result.device,
            batch_size,
            noise_kind=noise_kind,
            gamma=gamma,
        )

    if result.transform is not None:
        norm_flat = result.transform.inverse_transform(gen_flat_space)
    else:
        norm_flat = gen_flat_space

    return norm_flat.reshape(n_samples, result.horizon, result.n_features).astype(np.float32)


def train_algorithm(windows_norm: np.ndarray, windows_raw: np.ndarray, scaler, args):
    """Legacy all-in-one training helper retained for CLI/backwards compatibility."""

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    result = fit_algorithm(windows_norm, args)
    G = result.generator
    info = result.info

    torch.save(G.state_dict(), out_dir / "generator.pt")
    with open(out_dir / "train_info.json", "w") as f:
        json.dump(info, f, indent=2)

    n_gen = args.n_generate
    gen_norm = generate_algorithm(
        result,
        n_gen,
        batch_size=args.gen_batch_size,
        cdcgan_label=getattr(args, "cdcgan_label", None),
    )
    H, F = result.horizon, result.n_features
    gen_raw = scaler.inverse_transform(gen_norm.reshape(-1, F)).reshape(n_gen, H, F)

    np.save(out_dir / f"generated_{n_gen}_raw.npy", gen_raw)
    np.save(out_dir / f"generated_{n_gen}_normalized.npy", gen_norm)

    pdict = evaluate_all(
        windows_raw,
        gen_raw,
        dataset=args.dataset,
        n_levels=args.n_levels,
        out_json=str(out_dir / "metrics.json"),
    )
    with open(out_dir / "metrics_pretty.txt", "w") as f:
        for k, v in sorted(pdict.items()):
            f.write(f"{k}: {v}\n")
    return G, pdict
