"""Known scalar distributions used by the Task 01 oracle checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from scipy.integrate import quad
from scipy.special import ndtr

SQRT_2PI = float(np.sqrt(2.0 * np.pi))


def normal_pdf(value: np.ndarray | float) -> np.ndarray | float:
    value = np.asarray(value, dtype=float)
    return np.exp(-0.5 * np.square(value)) / SQRT_2PI


def normal_ei(
    mean: np.ndarray | float, scale: np.ndarray | float, best_f: float
) -> np.ndarray:
    """Analytic maximization expected improvement for a Gaussian predictive law."""
    mean_array = np.asarray(mean, dtype=float)
    scale_array = np.asarray(scale, dtype=float)
    if np.any(scale_array <= 0.0):
        raise ValueError("scale must be strictly positive")
    standardized = (mean_array - best_f) / scale_array
    return scale_array * (
        normal_pdf(standardized) + standardized * ndtr(standardized)
    )


@dataclass(frozen=True)
class GaussianMixture:
    """A univariate Gaussian mixture with analytic moments and EI."""

    weights: tuple[float, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]

    def __post_init__(self) -> None:
        if not (
            len(self.weights) == len(self.means) == len(self.scales) and self.weights
        ):
            raise ValueError("mixture parameters must have one or more matching entries")
        if not np.isclose(sum(self.weights), 1.0):
            raise ValueError("mixture weights must sum to one")
        if min(self.weights) <= 0.0 or min(self.scales) <= 0.0:
            raise ValueError("mixture weights and scales must be positive")

    @property
    def mean(self) -> float:
        return float(np.dot(self.weights, self.means))

    @property
    def variance(self) -> float:
        means = np.asarray(self.means)
        scales = np.asarray(self.scales)
        return float(np.dot(self.weights, np.square(scales) + np.square(means)) - self.mean**2)

    def pdf(self, value: np.ndarray | float) -> np.ndarray:
        value_array = np.asarray(value, dtype=float)
        total = np.zeros_like(value_array, dtype=float)
        for weight, mean, scale in zip(self.weights, self.means, self.scales, strict=True):
            total += weight * normal_pdf((value_array - mean) / scale) / scale
        return total

    def logpdf(self, value: np.ndarray | float) -> np.ndarray:
        return np.log(self.pdf(value))

    def cdf(self, value: np.ndarray | float) -> np.ndarray:
        value_array = np.asarray(value, dtype=float)
        total = np.zeros_like(value_array, dtype=float)
        for weight, mean, scale in zip(self.weights, self.means, self.scales, strict=True):
            total += weight * ndtr((value_array - mean) / scale)
        return total

    def sample(self, count: int, generator: torch.Generator) -> torch.Tensor:
        """Draw double-precision iid residual samples with a supplied seed generator."""
        weights = torch.tensor(self.weights, dtype=torch.double)
        component = torch.multinomial(weights, count, replacement=True, generator=generator)
        means = torch.tensor(self.means, dtype=torch.double)[component]
        scales = torch.tensor(self.scales, dtype=torch.double)[component]
        return means + scales * torch.randn(count, dtype=torch.double, generator=generator)

    def expected_improvement(
        self,
        mean: np.ndarray | float,
        scale: np.ndarray | float,
        best_f: float,
    ) -> np.ndarray:
        """Exact EI after the affine transformation ``Y = mean + scale * Z``."""
        mean_array = np.asarray(mean, dtype=float)
        scale_array = np.asarray(scale, dtype=float)
        if np.any(scale_array <= 0.0):
            raise ValueError("scale must be strictly positive")
        total = np.zeros(np.broadcast(mean_array, scale_array).shape, dtype=float)
        for weight, component_mean, component_scale in zip(
            self.weights, self.means, self.scales, strict=True
        ):
            total += weight * normal_ei(
                mean_array + scale_array * component_mean,
                scale_array * component_scale,
                best_f,
            )
        return total

    def expected_improvement_quadrature(
        self, mean: float, scale: float, best_f: float
    ) -> tuple[float, float]:
        """Independent adaptive integration of EI over the standardized residual."""
        threshold = (best_f - mean) / scale
        value, error = quad(
            lambda z: (mean + scale * z - best_f) * float(self.pdf(z)),
            threshold,
            np.inf,
            epsabs=1e-11,
            epsrel=1e-11,
            limit=200,
        )
        return float(value), float(error)

    def moments_quadrature(self) -> tuple[float, float]:
        """Independent adaptive integration of the first two raw moments."""
        mean, _ = quad(
            lambda z: z * float(self.pdf(z)), -np.inf, np.inf, epsabs=1e-12, epsrel=1e-12
        )
        second, _ = quad(
            lambda z: z * z * float(self.pdf(z)),
            -np.inf,
            np.inf,
            epsabs=1e-12,
            epsrel=1e-12,
        )
        return float(mean), float(second - mean * mean)

    def tail_probability(self, threshold: float) -> float:
        return float(1.0 - self.cdf(threshold))


STANDARD_NORMAL = GaussianMixture(weights=(1.0,), means=(0.0,), scales=(1.0,))
ORACLE_MIXTURE = GaussianMixture(
    weights=(0.8, 0.2), means=(-0.3, 1.2), scales=(0.8, 0.8)
)


def kl_true_to_model(
    true_distribution: GaussianMixture,
    model_pdf: Callable[[float], float],
    lower: float = -10.0,
    upper: float = 10.0,
) -> tuple[float, float]:
    """High-accuracy finite-interval KL; omitted mixture mass is negligible here."""
    value, error = quad(
        lambda z: float(true_distribution.pdf(z))
        * (float(true_distribution.logpdf(z)) - np.log(max(model_pdf(z), 1e-300))),
        lower,
        upper,
        epsabs=1e-10,
        epsrel=1e-10,
        limit=300,
    )
    return float(value), float(error)
