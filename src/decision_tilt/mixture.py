"""Exact scalar Gaussian-mixture utilities for expected-improvement studies."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.special import ndtr, ndtri, roots_hermitenorm

_SQRT_2PI = np.sqrt(2.0 * np.pi)


def _normal_pdf(value: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * np.square(value)) / _SQRT_2PI


def normal_improvement_moments(
    mean: np.ndarray | float,
    std: np.ndarray | float,
    incumbent: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return first and second moments of ``(Y - incumbent)_+`` for a normal Y."""

    mean_array, std_array = np.broadcast_arrays(
        np.asarray(mean, dtype=np.float64), np.asarray(std, dtype=np.float64)
    )
    if np.any(~np.isfinite(mean_array)) or np.any(~np.isfinite(std_array)):
        raise ValueError("normal parameters must be finite")
    if np.any(std_array <= 0.0):
        raise ValueError("standard deviations must be positive")
    delta = mean_array - float(incumbent)
    standardized = delta / std_array
    probability = ndtr(standardized)
    density = _normal_pdf(standardized)
    first = delta * probability + std_array * density
    second = (
        (np.square(delta) + np.square(std_array)) * probability
        + delta * std_array * density
    )
    return np.maximum(first, 0.0), np.maximum(second, 0.0)


def softplus_utility(
    value: np.ndarray | float, incumbent: float, temperature: float
) -> np.ndarray:
    """Strictly positive smooth improvement, evaluated without overflow."""

    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    value_array = np.asarray(value, dtype=np.float64)
    return temperature * np.logaddexp(
        0.0, (value_array - float(incumbent)) / temperature
    )


@lru_cache(maxsize=None)
def _gauss_hermite_rule(order: int) -> tuple[np.ndarray, np.ndarray]:
    if order < 8:
        raise ValueError("Gauss-Hermite order must be at least 8")
    # Probabilists' Hermite nodes integrate against exp(-z^2 / 2) directly and
    # remain stable at the high order needed by the sharp positive control.
    nodes, weights = roots_hermitenorm(order)
    nodes = nodes.astype(np.float64)
    weights = weights.astype(np.float64) / np.sqrt(2.0 * np.pi)
    return nodes, weights


def normal_softplus_moments(
    mean: np.ndarray | float,
    std: np.ndarray | float,
    incumbent: float,
    temperature: float,
    *,
    order: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic one-dimensional quadrature for softplus utility moments."""

    mean_array, std_array = np.broadcast_arrays(
        np.asarray(mean, dtype=np.float64), np.asarray(std, dtype=np.float64)
    )
    if np.any(~np.isfinite(mean_array)) or np.any(~np.isfinite(std_array)):
        raise ValueError("normal parameters must be finite")
    if np.any(std_array <= 0.0):
        raise ValueError("standard deviations must be positive")
    nodes, weights = _gauss_hermite_rule(order)
    values = mean_array[..., None] + std_array[..., None] * nodes
    utility = softplus_utility(values, incumbent, temperature)
    return utility @ weights, np.square(utility) @ weights


def chi_square_decision_shift(first: float, second: float) -> float:
    """Return chi-square divergence from utility moments."""

    if not np.isfinite(first) or first <= 0.0:
        raise ValueError("the first utility moment must be finite and positive")
    if not np.isfinite(second) or second < 0.0:
        raise ValueError("the second utility moment must be finite and nonnegative")
    return float(second / np.square(first) - 1.0)


def population_ess_fraction(first: float, second: float) -> float:
    """Return the population limit of self-normalized utility-weight ESS/N."""

    shift = chi_square_decision_shift(first, second)
    return float(1.0 / (1.0 + shift))


def mc_relative_variance(first: float, second: float, sample_count: int) -> float:
    """Exact relative variance of an iid sample-mean utility estimator."""

    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    return chi_square_decision_shift(first, second) / sample_count


@dataclass(frozen=True)
class GaussianMixture1D:
    """A finite one-dimensional Gaussian mixture with exact utility moments."""

    weights: np.ndarray
    means: np.ndarray
    stds: np.ndarray

    def __post_init__(self) -> None:
        weights = np.asarray(self.weights, dtype=np.float64)
        means = np.asarray(self.means, dtype=np.float64)
        stds = np.asarray(self.stds, dtype=np.float64)
        if weights.ndim != 1 or means.shape != weights.shape or stds.shape != weights.shape:
            raise ValueError("weights, means, and stds must be one-dimensional and aligned")
        if np.any(~np.isfinite(weights)) or np.any(weights <= 0.0):
            raise ValueError("weights must be finite and positive")
        if not np.isclose(weights.sum(), 1.0, atol=1e-14, rtol=0.0):
            raise ValueError("weights must sum to one")
        if np.any(~np.isfinite(means)) or np.any(~np.isfinite(stds)) or np.any(stds <= 0.0):
            raise ValueError("component parameters must be finite with positive stds")
        object.__setattr__(self, "weights", weights.copy())
        object.__setattr__(self, "means", means.copy())
        object.__setattr__(self, "stds", stds.copy())

    def density(self, value: np.ndarray | float) -> np.ndarray:
        value_array = np.asarray(value, dtype=np.float64)
        standardized = (value_array[..., None] - self.means) / self.stds
        return np.sum(self.weights * _normal_pdf(standardized) / self.stds, axis=-1)

    def improvement_component_moments(
        self, incumbent: float
    ) -> tuple[np.ndarray, np.ndarray]:
        return normal_improvement_moments(self.means, self.stds, incumbent)

    def improvement_moments(self, incumbent: float) -> tuple[float, float]:
        first, second = self.improvement_component_moments(incumbent)
        return float(self.weights @ first), float(self.weights @ second)

    def softplus_component_moments(
        self, incumbent: float, temperature: float, *, order: int = 4096
    ) -> tuple[np.ndarray, np.ndarray]:
        return normal_softplus_moments(
            self.means,
            self.stds,
            incumbent,
            temperature,
            order=order,
        )

    def softplus_moments(
        self, incumbent: float, temperature: float, *, order: int = 4096
    ) -> tuple[float, float]:
        first, second = self.softplus_component_moments(
            incumbent, temperature, order=order
        )
        return float(self.weights @ first), float(self.weights @ second)

    def tilted_component_weights(
        self,
        incumbent: float,
        *,
        utility: str = "improvement",
        temperature: float | None = None,
        order: int = 4096,
    ) -> np.ndarray:
        if utility == "improvement":
            first, _ = self.improvement_component_moments(incumbent)
        elif utility == "softplus":
            if temperature is None:
                raise ValueError("temperature is required for softplus utility")
            first, _ = self.softplus_component_moments(
                incumbent, temperature, order=order
            )
        else:
            raise ValueError(f"unknown utility: {utility}")
        contributions = self.weights * first
        return contributions / contributions.sum()

    def tilted_density(
        self,
        value: np.ndarray | float,
        incumbent: float,
        *,
        utility: str = "improvement",
        temperature: float | None = None,
        order: int = 4096,
    ) -> np.ndarray:
        value_array = np.asarray(value, dtype=np.float64)
        if utility == "improvement":
            utility_value = np.maximum(value_array - incumbent, 0.0)
            normalizer, _ = self.improvement_moments(incumbent)
        elif utility == "softplus":
            if temperature is None:
                raise ValueError("temperature is required for softplus utility")
            utility_value = softplus_utility(value_array, incumbent, temperature)
            normalizer, _ = self.softplus_moments(
                incumbent, temperature, order=order
            )
        else:
            raise ValueError(f"unknown utility: {utility}")
        return self.density(value_array) * utility_value / normalizer

    def sample_from_unit(
        self, component_uniform: np.ndarray, normal_uniform: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Transform two unit-uniform arrays into exact mixture samples."""

        component_uniform, normal_uniform = np.broadcast_arrays(
            np.asarray(component_uniform, dtype=np.float64),
            np.asarray(normal_uniform, dtype=np.float64),
        )
        if np.any((component_uniform < 0.0) | (component_uniform >= 1.0)):
            raise ValueError("component uniforms must lie in [0, 1)")
        if np.any((normal_uniform < 0.0) | (normal_uniform >= 1.0)):
            raise ValueError("normal uniforms must lie in [0, 1)")
        indices = np.searchsorted(np.cumsum(self.weights), component_uniform, side="right")
        tiny = np.nextafter(0.0, 1.0)
        clipped = np.clip(normal_uniform, tiny, np.nextafter(1.0, 0.0))
        normal_scores = ndtri(clipped)
        samples = self.means[indices] + self.stds[indices] * normal_scores
        return samples, indices, normal_scores

    def standardized_shape(self) -> dict[str, float]:
        """Return variance, skewness, and excess kurtosis of the mixture."""

        mean = float(self.weights @ self.means)
        offsets = self.means - mean
        variance = float(self.weights @ (np.square(self.stds) + np.square(offsets)))
        third = float(
            self.weights
            @ (np.power(offsets, 3) + 3.0 * offsets * np.square(self.stds))
        )
        fourth = float(
            self.weights
            @ (
                np.power(offsets, 4)
                + 6.0 * np.square(offsets) * np.square(self.stds)
                + 3.0 * np.power(self.stds, 4)
            )
        )
        return {
            "mean": mean,
            "variance": variance,
            "skewness": third / variance**1.5,
            "excess_kurtosis": fourth / variance**2 - 3.0,
        }
