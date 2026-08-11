"""Exact structural-posterior diagnostics for Task 02A."""

from .exact_gp import ExactGPBatchState, matern52_covariance
from .particles import ParticleWeights, SaasParticles
from .preprocessing import FrozenOutputTransform

__all__ = [
    "ExactGPBatchState",
    "FrozenOutputTransform",
    "ParticleWeights",
    "SaasParticles",
    "matern52_covariance",
]
