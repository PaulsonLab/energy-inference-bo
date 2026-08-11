"""Particle-averaged analytic q=1 expected improvement."""

from __future__ import annotations

import math

import torch

from .exact_gp import ExactGPBatchState
from .particles import ParticleWeights


def gaussian_expected_improvement(
    mean: torch.Tensor, variance: torch.Tensor, best_f: float | torch.Tensor
) -> torch.Tensor:
    """Analytic maximization EI for independent Gaussian marginals."""

    standard_deviation = variance.clamp_min(torch.finfo(torch.double).tiny).sqrt()
    improvement = mean - torch.as_tensor(best_f, dtype=torch.double)
    standardized = improvement / standard_deviation
    normal_pdf = torch.exp(-0.5 * standardized.square()) / math.sqrt(2.0 * math.pi)
    normal_cdf = 0.5 * (1.0 + torch.erf(standardized / math.sqrt(2.0)))
    # The analytic expression is nonnegative, but cancellation can produce tiny
    # negative values far into the no-improvement tail in double precision.
    return (improvement * normal_cdf + standard_deviation * normal_pdf).clamp_min(0.0)


def weighted_expected_improvement(
    state: ExactGPBatchState,
    weights: ParticleWeights,
    candidate_x: torch.Tensor,
    best_f: float | torch.Tensor,
    *,
    chunk_size: int | None = None,
) -> torch.Tensor:
    if weights.log_weights.numel() != state.particles.num_particles:
        raise ValueError("one weight is required per GP particle")
    mean, variance = state.predict(candidate_x, chunk_size=chunk_size)
    particle_ei = gaussian_expected_improvement(mean, variance, best_f)
    return torch.einsum("p,pc->c", weights.probabilities, particle_ei)
