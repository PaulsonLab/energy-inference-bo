"""Direct augmented-probability identities and raw-weight importance sampling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from energy_bo.metrics import effective_sample_size, histogram_target_masses, normalized_grid

from .acquisition import oracle_ei_curve
from .distributions import GaussianMixture
from .scenarios import OracleScenario


def augmented_grid_marginal(
    expected_utility: np.ndarray, x: np.ndarray, replicas: int = 1
) -> np.ndarray:
    """Uniform-prior x marginal proportional to expected_utility ** replicas."""
    if replicas < 1:
        raise ValueError("replicas must be a positive integer")
    return normalized_grid(np.power(expected_utility, replicas), x)


def oracle_augmented_marginal_by_quadrature(
    distribution: GaussianMixture,
    scenario: OracleScenario,
    x: np.ndarray,
    replicas: int = 1,
) -> tuple[np.ndarray, float]:
    """Compute the augmented marginal using adaptive integration of raw utility."""
    expected_utility = np.empty_like(x, dtype=float)
    largest_error = 0.0
    for index, x_value in enumerate(x):
        value, error = distribution.expected_improvement_quadrature(
            float(scenario.mean(np.array([x_value]))[0]),
            float(scenario.scale(np.array([x_value]))[0]),
            scenario.best_f,
        )
        expected_utility[index] = value
        largest_error = max(largest_error, error)
    return augmented_grid_marginal(expected_utility, x, replicas), largest_error


@dataclass(frozen=True)
class ImportanceSamplingResult:
    replicas: int
    particle_count: int
    l1_marginal_error: float
    max_bin_error: float
    inferred_mode: float
    exact_mode: float
    effective_sample_size: float
    effective_sample_fraction: float
    nonzero_weight_fraction: float


def importance_sample_augmented_oracle(
    distribution: GaussianMixture,
    scenario: OracleScenario,
    x_grid: np.ndarray,
    replicas: int,
    particle_count: int,
    seed: int,
    bins: int = 100,
) -> ImportanceSamplingResult:
    """Sample x uniformly and reweight independent latent outcomes by raw utility."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x_particles = torch.rand(particle_count, dtype=torch.double, generator=generator)
    residuals = distribution.sample(particle_count * replicas, generator).reshape(
        particle_count, replicas
    )
    x_numpy = x_particles.numpy()
    means = torch.as_tensor(scenario.mean(x_numpy), dtype=torch.double).unsqueeze(1)
    scales = torch.as_tensor(scenario.scale(x_numpy), dtype=torch.double).unsqueeze(1)
    utility = torch.clamp(means + scales * residuals - scenario.best_f, min=0.0)
    weights = torch.prod(utility, dim=1).numpy()

    expected_utility = oracle_ei_curve(distribution, scenario, x_grid)
    target_density = augmented_grid_marginal(expected_utility, x_grid, replicas)
    edges = np.linspace(0.0, 1.0, bins + 1)
    exact_masses = histogram_target_masses(target_density, x_grid, edges)
    weighted_counts, _ = np.histogram(x_numpy, bins=edges, weights=weights)
    if np.sum(weighted_counts) <= 0.0:
        raise RuntimeError("all raw importance weights were zero")
    empirical_masses = weighted_counts / np.sum(weighted_counts)
    mode_index = int(np.argmax(empirical_masses))
    exact_mode = float(x_grid[int(np.argmax(target_density))])
    ess = effective_sample_size(weights)
    return ImportanceSamplingResult(
        replicas=replicas,
        particle_count=particle_count,
        l1_marginal_error=float(np.sum(np.abs(empirical_masses - exact_masses))),
        max_bin_error=float(np.max(np.abs(empirical_masses - exact_masses))),
        inferred_mode=float((edges[mode_index] + edges[mode_index + 1]) / 2.0),
        exact_mode=exact_mode,
        effective_sample_size=ess,
        effective_sample_fraction=ess / particle_count,
        nonzero_weight_fraction=float(np.mean(weights > 0.0)),
    )
