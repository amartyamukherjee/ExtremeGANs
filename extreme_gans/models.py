from __future__ import annotations

from typing import Sequence, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def parse_hidden(s: str | Sequence[int]) -> tuple[int, ...]:
    if isinstance(s, str):
        return tuple(int(x.strip()) for x in s.split(",") if x.strip())
    return tuple(int(x) for x in s)


class MLPGenerator(nn.Module):
    def __init__(self, latent_dim: int, output_dim: int, hidden: Sequence[int] = (256, 512, 512), final_activation: Optional[str] = "none"):
        super().__init__()
        layers: list[nn.Module] = []
        prev = latent_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.LeakyReLU(0.2, inplace=True)]
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        if final_activation == "tanh":
            layers.append(nn.Tanh())
        elif final_activation == "softplus":
            layers.append(nn.Softplus())
        elif final_activation is None or final_activation == "none":
            pass
        else:
            raise ValueError(f"Unknown final activation {final_activation}")
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class MLPDiscriminator(nn.Module):
    def __init__(self, input_dim: int, hidden: Sequence[int] = (512, 256), use_sigmoid: bool = False):
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.LeakyReLU(0.2, inplace=True)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        if use_sigmoid:
            layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(-1)


class EVGenerator(nn.Module):
    """EVGAN quantile-style generator for positive heavy-tail variables."""
    def __init__(self, latent_dim: int, data_dim: int, hidden_units: int = 128, positive_tif: bool = True, max_tif: float = 5.0, eps: float = 1e-5):
        super().__init__()
        if latent_dim < data_dim:
            raise ValueError("EVGenerator requires latent_dim >= data_dim")
        self.latent_dim = latent_dim
        self.data_dim = data_dim
        self.positive_tif = positive_tif
        self.max_tif = max_tif
        self.eps = eps
        self.fc1 = nn.Linear(latent_dim, hidden_units)
        self.fc2 = nn.Linear(hidden_units, data_dim, bias=False)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = torch.clamp(z, self.eps, 1.0 - self.eps)
        h = F.relu(self.fc1(z))
        tif = self.fc2(h)
        if self.positive_tif:
            tif = F.softplus(tif) + 1e-4
        tif = torch.clamp(tif, min=1e-4, max=self.max_tif)
        z_m = z[:, : self.data_dim]
        log_base = torch.log(torch.clamp((1.0 - z_m.pow(2)) / 2.0, min=self.eps))
        log_x = torch.clamp(-tif * log_base, min=-30.0, max=30.0)
        return torch.exp(log_x)


class TailStatisticDiscriminator(nn.Module):
    """TailGAN discriminator over sorted batchwise statistics.

    It maps each statistic's NeuralSort-relaxed order statistics to VaR/ES-like
    parameters for each alpha.
    """
    def __init__(self, batch_size: int, n_alphas: int, hidden: Sequence[int] = (256, 128)):
        super().__init__()
        layers: list[nn.Module] = []
        prev = batch_size
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.LeakyReLU(0.2, inplace=True)]
            prev = h
        layers.append(nn.Linear(prev, 2 * n_alphas))
        self.net = nn.Sequential(*layers)

    def forward(self, sorted_stats: torch.Tensor) -> torch.Tensor:
        # sorted_stats: [n_stats, batch]
        return self.net(sorted_stats)


class ConditionalMLPGenerator(MLPGenerator):
    def __init__(self, latent_dim, condition_dim, output_dim, hidden_dims=(256, 256, 256)):
        super().__init__(latent_dim + condition_dim, output_dim, hidden_dims)

    def forward(self, z, y):
        return self.net(torch.cat([z, y], dim=1))


class ConditionalMLPDiscriminator(MLPDiscriminator):
    def __init__(self, data_dim, condition_dim, hidden_dims=(256, 256, 256)):
        super().__init__(data_dim + condition_dim, hidden_dims)

    def forward(self, x, y):
        return self.net(torch.cat([x, y], dim=1)).view(-1)
