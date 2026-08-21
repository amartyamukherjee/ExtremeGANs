"""Extreme GANs: GAN models for extreme multivariate time-series generation."""

from . import algos, data, metrics
from .data import TimeSeriesDataset

__version__ = "0.2.0"

__all__ = ["algos", "data", "metrics", "TimeSeriesDataset", "__version__"]
