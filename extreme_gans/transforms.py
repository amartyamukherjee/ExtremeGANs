from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


class EmpiricalFrechetTransform:
    """Coordinatewise empirical-CDF transform used for HTGAN-style training."""
    def __init__(self, gamma: float = 2.0 / 3.0, train_space: str = "log_frechet", eps: float = 1e-6):
        if gamma <= 0:
            raise ValueError("gamma must be positive")
        if train_space not in {"frechet", "log_frechet"}:
            raise ValueError("train_space must be frechet or log_frechet")
        self.gamma = float(gamma)
        self.train_space = train_space
        self.eps = float(eps)
        self.sorted_x: Optional[np.ndarray] = None

    def fit(self, x: np.ndarray) -> "EmpiricalFrechetTransform":
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError("x must have shape [n, d]")
        self.sorted_x = np.sort(x, axis=0)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.sorted_x is None:
            raise RuntimeError("fit first")
        x = np.asarray(x, dtype=np.float64)
        n, d = self.sorted_x.shape
        ranks = np.empty_like(x, dtype=np.float64)
        for j in range(d):
            ranks[:, j] = np.searchsorted(self.sorted_x[:, j], x[:, j], side="right")
        u = np.clip(ranks / (n + 1.0), self.eps, 1.0 - self.eps)
        log_y = self.gamma * (-np.log1p(-u))
        return (log_y if self.train_space == "log_frechet" else np.exp(log_y)).astype(np.float32)

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)

    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        if self.sorted_x is None:
            raise RuntimeError("fit first")
        y = np.asarray(y, dtype=np.float64)
        log_y = y if self.train_space == "log_frechet" else np.log(np.maximum(y, 1.0 + self.eps))
        u = np.clip(1.0 - np.exp(-log_y / self.gamma), self.eps, 1.0 - self.eps)
        n, d = self.sorted_x.shape
        grid = np.arange(1, n + 1, dtype=np.float64) / (n + 1.0)
        out = np.empty_like(u, dtype=np.float64)
        for j in range(d):
            out[:, j] = np.interp(u[:, j], grid, self.sorted_x[:, j])
        return out.astype(np.float32)

    def state_dict(self) -> dict:
        return {"gamma": self.gamma, "train_space": self.train_space, "eps": self.eps, "sorted_x": self.sorted_x.astype(np.float32).tolist()}


@dataclass
class PositiveTailTransform:
    """Positive transform for EVGAN-style training.

    mode='shift' is safest for generic multivariate time-series because it is invertible
    up to coordinatewise min/scale. modes 'abs', 'losses', 'gains' are useful for
    finance tail magnitudes but are not sign-invertible.
    """
    mode: str = "shift"
    eps: float = 1e-8
    standardize_scale: bool = True
    mins: Optional[np.ndarray] = None
    scale: Optional[np.ndarray] = None

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if self.mode == "shift":
            self.mins = x.min(axis=0, keepdims=True).astype(np.float32)
            y = x - self.mins + self.eps
        elif self.mode == "abs":
            y = np.abs(x) + self.eps
        elif self.mode == "losses":
            y = np.maximum(-x, 0.0) + self.eps
        elif self.mode == "gains":
            y = np.maximum(x, 0.0) + self.eps
        else:
            raise ValueError("mode must be shift, abs, losses, or gains")
        self.scale = np.ones((1, y.shape[1]), dtype=np.float32)
        if self.standardize_scale:
            self.scale = np.maximum(np.median(y, axis=0, keepdims=True).astype(np.float32), self.eps)
            y = y / self.scale
        return y.astype(np.float32)

    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=np.float32)
        if self.scale is not None:
            y = y * self.scale
        if self.mode == "shift":
            return (y + self.mins - self.eps).astype(np.float32)
        if self.mode == "losses":
            return (-(y - self.eps)).astype(np.float32)
        if self.mode == "gains":
            return (y - self.eps).astype(np.float32)
        # abs mode cannot recover signs; return positive magnitudes.
        return (y - self.eps).astype(np.float32)

    def state_dict(self) -> dict:
        return {
            "mode": self.mode,
            "eps": self.eps,
            "standardize_scale": self.standardize_scale,
            "mins": None if self.mins is None else self.mins.tolist(),
            "scale": None if self.scale is None else self.scale.tolist(),
        }
