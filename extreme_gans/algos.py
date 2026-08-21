from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np

from .data import FeatureScaler, TimeSeriesDataset
from .metrics import evaluate_all
from .trainers import TrainingResult, fit_algorithm, generate_algorithm


def _hidden(value: str | Sequence[int]) -> tuple[int, ...]:
    if isinstance(value, str):
        return tuple(int(x.strip()) for x in value.split(",") if x.strip())
    return tuple(int(x) for x in value)


def _hidden_string(value: str | Sequence[int]) -> str:
    return ",".join(str(x) for x in _hidden(value))


class GAN:
    """Public estimator-style interface shared by Extreme GANs algorithms.

    Instances are created with functions such as :func:`htgan` and then follow
    the familiar ``fit -> evaluate/generate`` lifecycle.
    """

    def __init__(self, algorithm: str, **algorithm_config: Any):
        self.algorithm = algorithm
        self.algorithm_config = dict(algorithm_config)
        self._result: TrainingResult | None = None
        self._scaler: FeatureScaler | None = None
        self._feature_names: list[str] | None = None
        self._dataset_name: str | None = None
        self._fit_config: dict[str, Any] | None = None
        self.metrics_: dict[str, float] | None = None

    def __repr__(self) -> str:
        state = "fitted" if self.is_fitted else "unfitted"
        return f"GAN(algorithm={self.algorithm!r}, state={state}, config={self.algorithm_config})"

    @property
    def is_fitted(self) -> bool:
        return self._result is not None

    @property
    def history_(self) -> list[dict[str, Any]]:
        self._require_fitted()
        return list(self._result.info.get("history", []))

    @property
    def training_info_(self) -> dict[str, Any]:
        self._require_fitted()
        return deepcopy(self._result.info)

    @property
    def generator_(self):
        """Underlying trained PyTorch generator for advanced users."""
        self._require_fitted()
        return self._result.generator

    def fit(
        self,
        dataset: TimeSeriesDataset,
        *,
        epochs: int = 200,
        batch_size: int = 256,
        lr: float = 1e-4,
        beta1: float = 0.5,
        beta2: float = 0.999,
        label_smoothing: float = 0.05,
        seed: int = 0,
        device: str = "auto",
        log_every: int = 10,
        n_levels: int = 10,
    ) -> "GAN":
        """Train the model on a :class:`TimeSeriesDataset`.

        Algorithm/model choices belong in the constructor. Training-process
        choices such as epochs, batch size, learning rate, and device belong
        here.
        """

        if not isinstance(dataset, TimeSeriesDataset):
            raise TypeError(
                "dataset must be a TimeSeriesDataset. "
                "Use extreme_gans.data.load_generic_csv(...) to create one."
            )
        if epochs <= 0:
            raise ValueError("epochs must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if lr <= 0:
            raise ValueError("lr must be positive")

        fit_config = {
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "lr": float(lr),
            "beta1": float(beta1),
            "beta2": float(beta2),
            "label_smoothing": float(label_smoothing),
            "seed": int(seed),
            "device": device,
            "log_every": int(log_every),
            "n_levels": int(n_levels),
        }

        args = SimpleNamespace(
            algorithm=self.algorithm,
            dataset=dataset.name,
            **fit_config,
            **self.algorithm_config,
        )

        self._result = fit_algorithm(dataset.windows, args)
        self._scaler = deepcopy(dataset.scaler)
        self._feature_names = list(dataset.feature_names)
        self._dataset_name = dataset.name
        self._fit_config = fit_config
        self.metrics_ = None
        return self

    def generate(
        self,
        n_samples: int = 10_000,
        *,
        batch_size: int = 1024,
        normalized: bool = False,
        condition_label: int | None = None,
    ) -> np.ndarray:
        """Generate synthetic windows.

        By default samples are returned in the original data scale. Set
        ``normalized=True`` to receive the normalized training representation.
        """

        self._require_fitted()
        gen_norm = generate_algorithm(
            self._result,
            int(n_samples),
            batch_size=int(batch_size),
            cdcgan_label=condition_label,
        )
        if normalized:
            return gen_norm

        H, F = self._result.horizon, self._result.n_features
        return self._scaler.inverse_transform(gen_norm.reshape(-1, F)).reshape(
            n_samples, H, F
        )

    def evaluate(
        self,
        dataset: TimeSeriesDataset,
        *,
        n_samples: int = 10_000,
        batch_size: int = 1024,
        n_levels: int | None = None,
        condition_label: int | None = None,
    ) -> dict[str, float]:
        """Generate samples and evaluate them against a dataset."""

        self._require_fitted()
        if not isinstance(dataset, TimeSeriesDataset):
            raise TypeError("dataset must be a TimeSeriesDataset")
        if dataset.horizon != self._result.horizon or dataset.n_features != self._result.n_features:
            raise ValueError(
                "Evaluation dataset shape does not match the fitted model: "
                f"expected horizon={self._result.horizon}, n_features={self._result.n_features}; "
                f"got horizon={dataset.horizon}, n_features={dataset.n_features}."
            )

        fake = self.generate(
            n_samples=n_samples,
            batch_size=batch_size,
            normalized=False,
            condition_label=condition_label,
        )
        if n_levels is None:
            n_levels = int(self._fit_config.get("n_levels", 10))
        self.metrics_ = evaluate_all(
            dataset.raw_windows,
            fake,
            dataset=dataset.name,
            n_levels=int(n_levels),
            out_json=None,
        )
        return dict(self.metrics_)

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("This model is not fitted yet. Call model.fit(dataset) first.")


# ---------------------------------------------------------------------------
# Public algorithm factories
# ---------------------------------------------------------------------------


def htgan(
    *,
    alpha: float = 1.5,
    train_space: str = "log_frechet",
    latent_dim: int | None = None,
    gen_hidden: Sequence[int] = (256, 512, 512),
    disc_hidden: Sequence[int] = (512, 256),
) -> GAN:
    """Create an HTGAN model."""
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    if train_space not in {"log_frechet", "frechet"}:
        raise ValueError("train_space must be 'log_frechet' or 'frechet'")
    return GAN(
        "htgan",
        ht_alpha=float(alpha),
        ht_gamma=1.0 / float(alpha),
        ht_train_space=train_space,
        latent_dim=0 if latent_dim is None else int(latent_dim),
        gen_hidden=_hidden_string(gen_hidden),
        disc_hidden=_hidden_string(disc_hidden),
        # unused defaults kept because the shared legacy trainer expects them
        tail_disc_hidden="256,128",
        tail_noise="student",
        alphas=[0.05, 0.95],
        tail_score="quant",
        tail_W=10.0,
        neuralsort_temp=1.0,
        lambda_gp=10.0,
        grad_clip=1.0,
        n_critic=1,
        gp_lambda=10.0,
        ev_positive_mode="shift",
        ev_hidden_units=128,
        ev_max_tif=5.0,
        pareto_xi=None,
        tail_fraction=0.05,
        root_gamma=2.0,
        pareto_clip=1e6,
        condition_dim=3,
        extreme_mode="mean",
        extreme_tail="high",
        extreme_loss_weight=1.0,
    )


def evgan(
    *,
    positive_mode: str = "shift",
    hidden_units: int = 128,
    max_tif: float = 5.0,
    latent_dim: int | None = None,
    disc_hidden: Sequence[int] = (512, 256),
) -> GAN:
    """Create an EVGAN model."""
    cfg = _base_config(latent_dim=latent_dim, disc_hidden=disc_hidden)
    cfg.update(
        ev_positive_mode=positive_mode,
        ev_hidden_units=int(hidden_units),
        ev_max_tif=float(max_tif),
    )
    return GAN("evgan", **cfg)


def tailgan(
    *,
    alphas: Sequence[float] = (0.05, 0.95),
    score: str = "quant",
    W: float = 10.0,
    noise: str = "student",
    neuralsort_temp: float = 1.0,
    lambda_gp: float = 10.0,
    grad_clip: float = 1.0,
    latent_dim: int | None = None,
    gen_hidden: Sequence[int] = (256, 512, 512),
    tail_disc_hidden: Sequence[int] = (256, 128),
) -> GAN:
    """Create a TailGAN model."""
    cfg = _base_config(latent_dim=latent_dim, gen_hidden=gen_hidden)
    cfg.update(
        alphas=list(float(a) for a in alphas),
        tail_score=score,
        tail_W=float(W),
        tail_noise=noise,
        neuralsort_temp=float(neuralsort_temp),
        lambda_gp=float(lambda_gp),
        grad_clip=float(grad_clip),
        tail_disc_hidden=_hidden_string(tail_disc_hidden),
    )
    return GAN("tailgan", **cfg)


def wgan(
    *,
    n_critic: int = 1,
    gp_lambda: float = 10.0,
    latent_dim: int | None = None,
    gen_hidden: Sequence[int] = (256, 512, 512),
    disc_hidden: Sequence[int] = (512, 256),
) -> GAN:
    """Create a WGAN-GP baseline."""
    cfg = _base_config(latent_dim=latent_dim, gen_hidden=gen_hidden, disc_hidden=disc_hidden)
    cfg.update(n_critic=int(n_critic), gp_lambda=float(gp_lambda))
    return GAN("wgan", **cfg)


def vanilla(
    *,
    latent_dim: int | None = None,
    gen_hidden: Sequence[int] = (256, 512, 512),
    disc_hidden: Sequence[int] = (512, 256),
) -> GAN:
    """Create a vanilla BCE GAN baseline."""
    cfg = _base_config(latent_dim=latent_dim, gen_hidden=gen_hidden, disc_hidden=disc_hidden)
    return GAN("vanilla", **cfg)


def paretogan(
    *,
    xi: float | None = None,
    tail_fraction: float = 0.05,
    root_gamma: float = 2.0,
    clip: float = 1e6,
    latent_dim: int | None = None,
    gen_hidden: Sequence[int] = (256, 512, 512),
) -> GAN:
    """Create a ParetoGAN model."""
    cfg = _base_config(latent_dim=latent_dim, gen_hidden=gen_hidden)
    cfg.update(
        pareto_xi=xi,
        tail_fraction=float(tail_fraction),
        root_gamma=float(root_gamma),
        pareto_clip=float(clip),
    )
    return GAN("paretogan", **cfg)


def cdcgan(
    *,
    condition_dim: int = 3,
    extreme_mode: str = "mean",
    extreme_tail: str = "high",
    extreme_loss_weight: float = 1.0,
    n_critic: int = 1,
    latent_dim: int | None = None,
    gen_hidden: Sequence[int] = (256, 512, 512),
    disc_hidden: Sequence[int] = (512, 256),
) -> GAN:
    """Create a conditional DCGAN model."""
    cfg = _base_config(latent_dim=latent_dim, gen_hidden=gen_hidden, disc_hidden=disc_hidden)
    cfg.update(
        condition_dim=int(condition_dim),
        extreme_mode=extreme_mode,
        extreme_tail=extreme_tail,
        extreme_loss_weight=float(extreme_loss_weight),
        n_critic=int(n_critic),
    )
    return GAN("cdcgan", **cfg)


def _base_config(
    *,
    latent_dim: int | None = None,
    gen_hidden: Sequence[int] = (256, 512, 512),
    disc_hidden: Sequence[int] = (512, 256),
) -> dict[str, Any]:
    """Defaults required by the current internal trainers.

    The public API intentionally exposes only parameters relevant to each
    algorithm; these compatibility defaults stay private.
    """
    return {
        "latent_dim": 0 if latent_dim is None else int(latent_dim),
        "gen_hidden": _hidden_string(gen_hidden),
        "disc_hidden": _hidden_string(disc_hidden),
        "tail_disc_hidden": "256,128",
        # HTGAN-specific defaults
        "ht_alpha": 1.5,
        "ht_gamma": 2.0 / 3.0,
        "ht_train_space": "log_frechet",
        # EVGAN-specific defaults
        "ev_positive_mode": "shift",
        "ev_hidden_units": 128,
        "ev_max_tif": 5.0,
        # TailGAN-specific defaults
        "alphas": [0.05, 0.95],
        "tail_score": "quant",
        "tail_W": 10.0,
        "tail_noise": "student",
        "neuralsort_temp": 1.0,
        "lambda_gp": 10.0,
        "grad_clip": 1.0,
        "n_critic": 1,
        "gp_lambda": 10.0,
        # ParetoGAN-specific defaults
        "pareto_xi": None,
        "tail_fraction": 0.05,
        "root_gamma": 2.0,
        "pareto_clip": 1e6,
        # C-DCGAN-specific defaults
        "condition_dim": 3,
        "extreme_mode": "mean",
        "extreme_tail": "high",
        "extreme_loss_weight": 1.0,
    }


__all__ = [
    "GAN",
    "htgan",
    "evgan",
    "tailgan",
    "wgan",
    "vanilla",
    "paretogan",
    "cdcgan",
]
