from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class FeatureScaler:
    """Feature-wise scaler fitted on the original time-series rows."""

    method: str = "zscore"  # none|minmax|zscore
    mean: Optional[np.ndarray] = None
    std: Optional[np.ndarray] = None
    min: Optional[np.ndarray] = None
    max: Optional[np.ndarray] = None
    eps: float = 1e-8

    def fit(self, x: np.ndarray) -> "FeatureScaler":
        x = np.asarray(x, dtype=np.float32)
        if self.method == "none":
            return self
        if self.method == "zscore":
            self.mean = np.nanmean(x, axis=0, keepdims=True).astype(np.float32)
            self.std = np.nanstd(x, axis=0, keepdims=True).astype(np.float32)
            self.std = np.maximum(self.std, self.eps)
        elif self.method == "minmax":
            self.min = np.nanmin(x, axis=0, keepdims=True).astype(np.float32)
            self.max = np.nanmax(x, axis=0, keepdims=True).astype(np.float32)
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if self.method == "none":
            return x
        if self.method == "zscore":
            return ((x - self.mean) / self.std).astype(np.float32)
        if self.method == "minmax":
            denom = np.maximum(self.max - self.min, self.eps)
            return (2.0 * (x - self.min) / denom - 1.0).astype(np.float32)
        raise ValueError(f"Unknown normalization method: {self.method}")

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if self.method == "none":
            return x
        if self.method == "zscore":
            return (x * self.std + self.mean).astype(np.float32)
        if self.method == "minmax":
            denom = np.maximum(self.max - self.min, self.eps)
            return (((x + 1.0) / 2.0) * denom + self.min).astype(np.float32)
        raise ValueError(f"Unknown normalization method: {self.method}")

    def state_dict(self) -> dict:
        d = asdict(self)
        for k in ("mean", "std", "min", "max"):
            if d[k] is not None:
                d[k] = np.asarray(d[k]).tolist()
        return d

    @classmethod
    def from_state_dict(cls, d: dict) -> "FeatureScaler":
        out = cls(method=d.get("method", "zscore"), eps=d.get("eps", 1e-8))
        for k in ("mean", "std", "min", "max"):
            v = d.get(k)
            setattr(out, k, None if v is None else np.asarray(v, dtype=np.float32))
        return out


@dataclass
class TimeSeriesDataset:
    """Dataset object used by the public Extreme GANs Python API.

    Users normally create this object through :func:`load_generic_csv` rather
    than constructing it directly.
    """

    windows: np.ndarray
    raw_windows: np.ndarray
    scaler: FeatureScaler
    feature_names: list[str]
    name: str = "generic_csv"
    source: Optional[str] = None

    def __post_init__(self) -> None:
        self.windows = np.asarray(self.windows, dtype=np.float32)
        self.raw_windows = np.asarray(self.raw_windows, dtype=np.float32)
        if self.windows.ndim != 3:
            raise ValueError("windows must have shape [n_samples, horizon, n_features]")
        if self.raw_windows.shape != self.windows.shape:
            raise ValueError("raw_windows and windows must have the same shape")
        if self.windows.shape[2] != len(self.feature_names):
            raise ValueError("feature_names length must match the number of features")

    def __len__(self) -> int:
        return int(self.windows.shape[0])

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(self.windows.shape)

    @property
    def horizon(self) -> int:
        return int(self.windows.shape[1])

    @property
    def n_features(self) -> int:
        return int(self.windows.shape[2])

    @property
    def n_samples(self) -> int:
        return len(self)

    def __repr__(self) -> str:
        return (
            f"TimeSeriesDataset(n_samples={self.n_samples}, horizon={self.horizon}, "
            f"n_features={self.n_features}, normalize={self.scaler.method!r})"
        )


class WindowTensorDataset(Dataset):
    def __init__(self, windows: np.ndarray):
        self.windows = torch.tensor(windows, dtype=torch.float32)

    def __len__(self) -> int:
        return self.windows.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.windows[idx]


def parse_csv_list(s: Optional[str]) -> list[str]:
    if s is None or str(s).strip() == "":
        return []
    return [x.strip() for x in str(s).split(",") if x.strip()]


def make_windows(values: np.ndarray, horizon: int, stride: int = 1) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("values must have shape [time, features]")
    if horizon <= 1:
        raise ValueError("horizon must be > 1")
    if len(values) < horizon:
        raise ValueError(f"Need at least horizon={horizon} rows, got {len(values)}")
    starts = range(0, len(values) - horizon + 1, stride)
    return np.stack([values[i : i + horizon] for i in starts], axis=0).astype(np.float32)


def _load_generic_csv_arrays(
    csv_path: str | Path,
    horizon: int,
    stride: int = 1,
    drop_cols: Sequence[str] = ("date", "OT"),
    feature_cols: Optional[Sequence[str]] = None,
    normalize: str = "zscore",
    fill_method: str = "ffill_bfill",
) -> tuple[np.ndarray, np.ndarray, FeatureScaler, list[str]]:
    """Internal array-level CSV loader used by the public dataset loader."""

    df = pd.read_csv(csv_path)
    if feature_cols:
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing requested feature columns: {missing}")
        df = df.loc[:, list(feature_cols)]
    else:
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])
        df = df.select_dtypes(include=[np.number])

    if df.shape[1] == 0:
        raise ValueError("No numeric feature columns remain after dropping columns.")

    if fill_method == "ffill_bfill":
        df = df.ffill().bfill()
    elif fill_method == "interpolate":
        df = df.interpolate(limit_direction="both").ffill().bfill()
    elif fill_method == "drop":
        df = df.dropna(axis=0, how="any")
    else:
        raise ValueError("fill_method must be ffill_bfill, interpolate, or drop")

    values_raw = df.to_numpy(dtype=np.float32)
    values_raw = np.nan_to_num(values_raw, nan=0.0, posinf=0.0, neginf=0.0)
    scaler = FeatureScaler(method=normalize).fit(values_raw)
    values_norm = scaler.transform(values_raw)
    return (
        make_windows(values_norm, horizon, stride),
        make_windows(values_raw, horizon, stride),
        scaler,
        list(df.columns),
    )


def load_generic_csv(
    csv_path: str | Path,
    horizon: int,
    stride: int = 1,
    drop_cols: Sequence[str] = ("date", "OT"),
    feature_cols: Optional[Sequence[str]] = None,
    normalize: str = "zscore",
    fill_method: str = "ffill_bfill",
) -> TimeSeriesDataset:
    """Load a generic multivariate time-series CSV.

    The returned :class:`TimeSeriesDataset` contains normalized training windows,
    raw windows, the fitted scaler, and feature metadata. Those implementation
    details are carried with the dataset so users do not need to pass them around
    manually.
    """

    windows, raw_windows, scaler, features = _load_generic_csv_arrays(
        csv_path=csv_path,
        horizon=horizon,
        stride=stride,
        drop_cols=drop_cols,
        feature_cols=feature_cols,
        normalize=normalize,
        fill_method=fill_method,
    )
    return TimeSeriesDataset(
        windows=windows,
        raw_windows=raw_windows,
        scaler=scaler,
        feature_names=features,
        name="generic_csv",
        source=str(csv_path),
    )


def load_dataset_from_args(args) -> TimeSeriesDataset:
    """Compatibility helper for the command-line interface."""

    feature_cols = parse_csv_list(getattr(args, "feature_cols", None)) or None
    drop_cols = parse_csv_list(getattr(args, "drop_cols", None)) or ["date", "OT"]
    if args.dataset == "generic_csv":
        return load_generic_csv(
            args.csv_path,
            args.horizon,
            args.stride,
            drop_cols,
            feature_cols,
            args.normalize,
            args.fill_method,
        )
    raise ValueError(f"Unknown dataset: {args.dataset}")


def save_dataset_metadata(
    out_dir: str | Path,
    scaler: FeatureScaler,
    feature_cols: Sequence[str],
    args=None,
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "scaler.json", "w") as f:
        json.dump(scaler.state_dict(), f, indent=2)
    with open(out / "features.json", "w") as f:
        json.dump(list(feature_cols), f, indent=2)
