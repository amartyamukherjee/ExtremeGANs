from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Iterable, Sequence

import numpy as np


def _same_n(real: np.ndarray, fake: np.ndarray, seed: int = 0, max_n: int | None = None):
    rng = np.random.default_rng(seed)
    n = min(len(real), len(fake))
    if max_n is not None:
        n = min(n, max_n)
    ir = rng.choice(len(real), size=n, replace=False) if len(real) > n else np.arange(n)
    jf = rng.choice(len(fake), size=n, replace=False) if len(fake) > n else np.arange(n)
    return real[ir], fake[jf]


def flatten_windows(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).reshape(len(x), -1)


def ks_1d(x: np.ndarray, y: np.ndarray) -> float:
    x = np.sort(np.asarray(x)); y = np.sort(np.asarray(y))
    vals = np.sort(np.concatenate([x, y]))
    fx = np.searchsorted(x, vals, side="right") / max(len(x), 1)
    fy = np.searchsorted(y, vals, side="right") / max(len(y), 1)
    return float(np.max(np.abs(fx - fy)))


def dkss(real: np.ndarray, fake: np.ndarray, max_dims: int = 2048) -> float:
    r = flatten_windows(real); f = flatten_windows(fake)
    d = min(r.shape[1], max_dims)
    idx = np.linspace(0, r.shape[1] - 1, d).astype(int) if r.shape[1] > d else np.arange(d)
    return float(sum(ks_1d(r[:, j], f[:, j]) for j in idx) * (r.shape[1] / d))


def marginal_ks_mean(real: np.ndarray, fake: np.ndarray, max_dims: int = 2048) -> float:
    r = flatten_windows(real); f = flatten_windows(fake)
    d = min(r.shape[1], max_dims)
    idx = np.linspace(0, r.shape[1] - 1, d).astype(int) if r.shape[1] > d else np.arange(d)
    return float(np.mean([ks_1d(r[:, j], f[:, j]) for j in idx]))


def fid_gaussian(real: np.ndarray, fake: np.ndarray, eps: float = 1e-6, max_dims: int = 512) -> float:
    r = flatten_windows(real); f = flatten_windows(fake)
    if r.shape[1] > max_dims:
        idx = np.linspace(0, r.shape[1] - 1, max_dims).astype(int)
        r = r[:, idx]; f = f[:, idx]
    mu_r, mu_f = r.mean(axis=0), f.mean(axis=0)
    cr = np.cov(r, rowvar=False) + eps * np.eye(r.shape[1])
    cf = np.cov(f, rowvar=False) + eps * np.eye(f.shape[1])
    # symmetric eig sqrt of cr^{1/2} cf cr^{1/2}
    wr, Vr = np.linalg.eigh(cr)
    sr = (Vr * np.sqrt(np.maximum(wr, 0.0))) @ Vr.T
    prod = sr @ cf @ sr
    w, _ = np.linalg.eigh((prod + prod.T) / 2)
    tr_sqrt = np.sum(np.sqrt(np.maximum(w, 0.0)))
    return float(np.sum((mu_r - mu_f) ** 2) + np.trace(cr) + np.trace(cf) - 2 * tr_sqrt)


def tail_msle(real: np.ndarray, fake: np.ndarray, xi: float = 0.95, eps: float = 1e-12, max_dims: int = 2048) -> float:
    r = np.abs(flatten_windows(real)); f = np.abs(flatten_windows(fake))
    d = min(r.shape[1], max_dims)
    idx = np.linspace(0, r.shape[1] - 1, d).astype(int) if r.shape[1] > d else np.arange(d)
    n = min(len(r), len(f)); k = max(1, int(math.ceil((1 - xi) * n)))
    vals = []
    for j in idx:
        rr = np.sort(r[:n, j])[-k:]
        ff = np.sort(f[:n, j])[-k:]
        vals.append((np.log(rr + eps) - np.log(ff + eps)) ** 2)
    return float(np.mean(np.concatenate(vals)))


def quantile_mae(real: np.ndarray, fake: np.ndarray, quantiles=(0.01,0.05,0.5,0.95,0.99), max_dims: int = 2048) -> float:
    r = flatten_windows(real); f = flatten_windows(fake)
    d = min(r.shape[1], max_dims)
    idx = np.linspace(0, r.shape[1] - 1, d).astype(int) if r.shape[1] > d else np.arange(d)
    qr = np.quantile(r[:, idx], quantiles, axis=0)
    qf = np.quantile(f[:, idx], quantiles, axis=0)
    return float(np.mean(np.abs(qr - qf)))


def boxplot_error(real: np.ndarray, fake: np.ndarray) -> float:
    return quantile_mae(real, fake, quantiles=(0.0,0.25,0.5,0.75,1.0))


def autocorr(x: np.ndarray, max_lag: int = 10) -> np.ndarray:
    # returns [lag, feature], averaged over windows
    x = np.asarray(x, dtype=np.float64)
    B,H,F = x.shape
    outs = []
    for lag in range(1, min(max_lag, H-1)+1):
        a = x[:, :-lag, :] - x[:, :-lag, :].mean(axis=1, keepdims=True)
        b = x[:, lag:, :] - x[:, lag:, :].mean(axis=1, keepdims=True)
        num = (a*b).mean(axis=1)
        den = np.sqrt((a*a).mean(axis=1) * (b*b).mean(axis=1)) + 1e-12
        outs.append((num/den).mean(axis=0))
    return np.stack(outs, axis=0) if outs else np.zeros((0, x.shape[-1]))


def acf_error(real: np.ndarray, fake: np.ndarray, max_lag: int = 10) -> float:
    ar = autocorr(real, max_lag); af = autocorr(fake, max_lag)
    m = min(len(ar), len(af))
    return float(np.mean(np.abs(ar[:m] - af[:m]))) if m else float("nan")


def pearson_corr_error(real: np.ndarray, fake: np.ndarray) -> float:
    r = np.asarray(real).reshape(-1, real.shape[-1])
    f = np.asarray(fake).reshape(-1, fake.shape[-1])
    cr = np.corrcoef(r, rowvar=False)
    cf = np.corrcoef(f, rowvar=False)
    cr = np.nan_to_num(cr); cf = np.nan_to_num(cf)
    return float(np.linalg.norm(cr - cf, ord="fro") / max(cr.shape[0], 1))


def hist2d_l1(real: np.ndarray, fake: np.ndarray, bins: int = 30) -> float:
    r = np.asarray(real).reshape(-1, real.shape[-1])
    f = np.asarray(fake).reshape(-1, fake.shape[-1])
    if r.shape[1] < 2:
        return float("nan")
    mins = np.minimum(r[:, :2].min(axis=0), f[:, :2].min(axis=0))
    maxs = np.maximum(r[:, :2].max(axis=0), f[:, :2].max(axis=0))
    hr, _, _ = np.histogram2d(r[:,0], r[:,1], bins=bins, range=[[mins[0], maxs[0]], [mins[1], maxs[1]]], density=True)
    hf, _, _ = np.histogram2d(f[:,0], f[:,1], bins=bins, range=[[mins[0], maxs[0]], [mins[1], maxs[1]]], density=True)
    hr = hr / (hr.sum() + 1e-12); hf = hf / (hf.sum() + 1e-12)
    return float(np.abs(hr - hf).sum())


def extremeness_scores(x: np.ndarray, kind: str = "mean_abs") -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if kind == "mean":
        return x.mean(axis=(1,2))
    if kind == "mean_abs":
        return np.abs(x).mean(axis=(1,2))
    if kind == "max":
        return x.max(axis=(1,2))
    if kind == "max_abs":
        return np.abs(x).max(axis=(1,2))
    if kind == "norm":
        return np.linalg.norm(flatten_windows(x), axis=1)
    raise ValueError(f"Unknown extremeness kind {kind}")


def valid_rates(real: np.ndarray, fake: np.ndarray, kind: str = "mean_abs", quantiles=(0.9,0.95,0.99)) -> dict:
    sr = extremeness_scores(real, kind); sf = extremeness_scores(fake, kind)
    out = {}
    for q in quantiles:
        thr = float(np.quantile(sr, q))
        out[f"valid_rate_{kind}_q{int(q*100)}"] = float(np.mean(sf >= thr))
        out[f"real_rate_{kind}_q{int(q*100)}"] = float(np.mean(sr >= thr))
        out[f"threshold_{kind}_q{int(q*100)}"] = thr
    return out


def radial_angular_metrics(real: np.ndarray, fake: np.ndarray, q: float = 0.9) -> dict:
    r = np.abs(flatten_windows(real)); f = np.abs(flatten_windows(fake))
    rr = np.linalg.norm(r, axis=1); rf = np.linalg.norm(f, axis=1)
    radial_ks = ks_1d(rr, rf)
    tr = np.quantile(rr, q); tf = np.quantile(rf, q)
    ar = r[rr >= tr] / (r[rr >= tr].sum(axis=1, keepdims=True) + 1e-12)
    af = f[rf >= tf] / (f[rf >= tf].sum(axis=1, keepdims=True) + 1e-12)
    if len(ar) == 0 or len(af) == 0:
        ang = float("nan")
    else:
        mr = ar.mean(axis=0); mf = af.mean(axis=0)
        ang = float(np.mean(np.abs(mr - mf)))
    return {"radial_ks": float(radial_ks), "angular_mean_l1_tail": ang}


def loglog_ccdf_area(real, fake, eps=1e-12):
    """
    ParetoGAN tail metric: area between log-log empirical CCDF curves.
    Lower is better.
    """
    real = np.asarray(real, dtype=np.float64).reshape(-1)
    fake = np.asarray(fake, dtype=np.float64).reshape(-1)

    real = np.abs(real[np.isfinite(real)])
    fake = np.abs(fake[np.isfinite(fake)])

    real = real[real > eps]
    fake = fake[fake > eps]

    n = min(len(real), len(fake))
    if n < 2:
        return float("nan")

    real = np.sort(real)[-n:]
    fake = np.sort(fake)[-n:]

    # Inverse empirical CCDF at i/n, using largest-to-smallest tail order.
    real_tail = real[::-1]
    fake_tail = fake[::-1]

    weights = np.log((np.arange(1, n + 1) + 1) / np.arange(1, n + 1))

    area = np.sum(
        np.abs(np.log(real_tail + eps) - np.log(fake_tail + eps)) * weights
    )

    return float(area)


def evaluate_all(real: np.ndarray, fake: np.ndarray, dataset: str = "generic_csv", n_levels: int = 10, out_json: str | None = None) -> Dict[str, float]:
    real, fake = _same_n(np.asarray(real, dtype=np.float32), np.asarray(fake, dtype=np.float32), max_n=min(len(real), len(fake)))
    metrics = {
        "n_real_eval": int(len(real)),
        "n_fake_eval": int(len(fake)),
        "dkss": dkss(real, fake),
        "marginal_ks_mean": marginal_ks_mean(real, fake),
        "fid_gaussian_raw": fid_gaussian(real, fake),
        "tail_msle_90": tail_msle(real, fake, xi=0.90),
        "tail_msle_95": tail_msle(real, fake, xi=0.95),
        "tail_msle_99": tail_msle(real, fake, xi=0.99),
        "quantile_mae": quantile_mae(real, fake),
        "boxplot_error": boxplot_error(real, fake),
        "acf_error": acf_error(real, fake),
        "pearson_corr_error": pearson_corr_error(real, fake),
        "hist2d_l1": hist2d_l1(real, fake),
        "paretogan_loglog_ccdf_area": loglog_ccdf_area(real, fake),
    }
    metrics.update(valid_rates(real, fake, "mean_abs"))
    metrics.update(valid_rates(real, fake, "max_abs"))
    metrics.update(radial_angular_metrics(real, fake))
    if out_json:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w") as f:
            json.dump(metrics, f, indent=2)
    return metrics
