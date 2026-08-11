"""Acquisition signatures induced by fixed structural particles."""

from __future__ import annotations

import torch

from energy_bo.structural.acquisition import gaussian_expected_improvement
from energy_bo.structural.exact_gp import ExactGPBatchState
from energy_bo.structural.particles import SaasParticles


def particle_ei_signatures(
    state: ExactGPBatchState,
    candidate_x: torch.Tensor,
    best_f: float | torch.Tensor,
    *,
    chunk_size: int | None = None,
) -> torch.Tensor:
    """Return analytic q=1 EI for every particle and candidate, shape ``[P, J]``."""

    mean, variance = state.predict(candidate_x, chunk_size=chunk_size)
    signatures = gaussian_expected_improvement(mean, variance, best_f)
    if signatures.shape != (state.particles.num_particles, candidate_x.shape[0]):
        raise RuntimeError("unexpected acquisition-signature shape")
    if not torch.isfinite(signatures).all() or not torch.all(signatures >= 0):
        raise RuntimeError("EI signatures must be finite and nonnegative")
    return signatures


def transformed_particle_features(particles: SaasParticles) -> torch.Tensor:
    """Return ``[log lengthscales, mean, log outputscale]`` structural coordinates."""

    return torch.cat(
        (
            particles.lengthscales.log(),
            particles.means[:, None],
            particles.outputscales.log()[:, None],
        ),
        dim=1,
    )
