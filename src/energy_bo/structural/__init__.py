"""Exact structural-posterior diagnostics for Task 02A."""

from .exact_gp import ExactGPBatchState, matern52_covariance
from .particles import ParticleWeights, SaasParticles
from .preprocessing import FrozenOutputTransform
from .map_saas import MapSaasReference, fit_map_saas_reference, seeded_half_cauchy_taus

__all__ = [
    "ExactGPBatchState",
    "FrozenOutputTransform",
    "MapSaasReference",
    "ParticleWeights",
    "SaasParticles",
    "fit_map_saas_reference",
    "matern52_covariance",
    "seeded_half_cauchy_taus",
]
