"""Frozen output preprocessing and deterministic Task 02A benchmark data."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.quasirandom import SobolEngine


@dataclass(frozen=True)
class FrozenOutputTransform:
    """Affine output transform estimated once from the initial design."""

    mean: float
    scale: float

    def __post_init__(self) -> None:
        if not torch.isfinite(torch.tensor([self.mean, self.scale], dtype=torch.double)).all():
            raise ValueError("transform parameters must be finite")
        if self.scale <= 0:
            raise ValueError("transform scale must be positive")

    @classmethod
    def fit(cls, initial_y: torch.Tensor) -> "FrozenOutputTransform":
        values = initial_y.detach().to(dtype=torch.double).reshape(-1)
        if values.numel() < 2 or not torch.isfinite(values).all():
            raise ValueError("at least two finite initial outcomes are required")
        scale = values.std(unbiased=False)
        if not scale > 0:
            raise ValueError("initial outcomes must have positive population standard deviation")
        return cls(mean=float(values.mean()), scale=float(scale))

    def transform(self, y: torch.Tensor) -> torch.Tensor:
        return (y.to(dtype=torch.double) - self.mean) / self.scale

    def untransform(self, y: torch.Tensor) -> torch.Tensor:
        return y.to(dtype=torch.double) * self.scale + self.mean

    def transform_variance(self, variance: float | torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(variance, dtype=torch.double) / self.scale**2


def negative_branin(x: torch.Tensor) -> torch.Tensor:
    """Negative Branin embedded through the first two coordinates of [0, 1]^D."""

    x = x.to(dtype=torch.double)
    if x.ndim != 2 or x.shape[1] < 2:
        raise ValueError("negative_branin expects [n, D] with D >= 2")
    x1 = 15.0 * x[:, 0] - 5.0
    x2 = 15.0 * x[:, 1]
    a = 1.0
    b = 5.1 / (4.0 * torch.pi**2)
    c = 5.0 / torch.pi
    r = 6.0
    s = 10.0
    t = 1.0 / (8.0 * torch.pi)
    value = a * (x2 - b * x1.square() + c * x1 - r).square() + s * (1.0 - t) * torch.cos(x1) + s
    return -value


def deterministic_benchmark_inputs(count: int, dimension: int, seed: int) -> torch.Tensor:
    if count < 1 or dimension < 2:
        raise ValueError("count must be positive and dimension must be at least two")
    return SobolEngine(dimension=dimension, scramble=True, seed=seed).draw(count).double()
