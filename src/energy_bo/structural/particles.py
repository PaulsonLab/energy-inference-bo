"""Immutable SAAS operational particles and separately updated weights."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch


def _frozen_double(value: torch.Tensor) -> torch.Tensor:
    result = value.detach().to(dtype=torch.double, device="cpu").clone().contiguous()
    result.requires_grad_(False)
    return result


@dataclass(frozen=True)
class SaasParticles:
    """Fixed operational hyperparameters for exact Matérn-5/2 GPs.

    Observation noise is intentionally not part of the particle location. Task 02A
    fixes it once and passes it to the exact GP cache separately.
    """

    lengthscales: torch.Tensor
    means: torch.Tensor
    outputscales: torch.Tensor

    def __post_init__(self) -> None:
        lengthscales = _frozen_double(self.lengthscales)
        means = _frozen_double(self.means).reshape(-1)
        outputscales = _frozen_double(self.outputscales).reshape(-1)
        if lengthscales.ndim != 2:
            raise ValueError("lengthscales must have shape [particles, dimensions]")
        if means.shape != (lengthscales.shape[0],):
            raise ValueError("means must have shape [particles]")
        if outputscales.shape != (lengthscales.shape[0],):
            raise ValueError("outputscales must have shape [particles]")
        if lengthscales.shape[0] < 1 or lengthscales.shape[1] < 1:
            raise ValueError("at least one particle and one dimension are required")
        if not all(torch.isfinite(tensor).all() for tensor in (lengthscales, means, outputscales)):
            raise ValueError("particle values must be finite")
        if not torch.all(lengthscales > 0) or not torch.all(outputscales > 0):
            raise ValueError("lengthscales and outputscales must be positive")
        object.__setattr__(self, "lengthscales", lengthscales)
        object.__setattr__(self, "means", means)
        object.__setattr__(self, "outputscales", outputscales)

    @property
    def num_particles(self) -> int:
        return self.lengthscales.shape[0]

    @property
    def dimension(self) -> int:
        return self.lengthscales.shape[1]

    @property
    def feature_matrix(self) -> torch.Tensor:
        """Coordinates used for multivariate posterior-drift diagnostics."""

        return torch.cat(
            (
                self.lengthscales.log(),
                self.means[:, None],
                self.outputscales.log()[:, None],
            ),
            dim=1,
        )

    @classmethod
    def from_botorch(cls, model: Any) -> "SaasParticles":
        """Extract every retained operational sample from a fitted SAAS model."""

        lengthscales = model.covar_module.base_kernel.lengthscale.detach()
        while lengthscales.ndim > 2 and lengthscales.shape[-2] == 1:
            lengthscales = lengthscales.squeeze(-2)
        means = model.mean_module.constant.detach().reshape(-1)
        outputscales = model.covar_module.outputscale.detach().reshape(-1)
        if lengthscales.ndim == 1:
            lengthscales = lengthscales.unsqueeze(0)
        return cls(lengthscales=lengthscales, means=means, outputscales=outputscales)


@dataclass(frozen=True)
class ParticleWeights:
    """Normalized log weights, kept distinct from immutable particles."""

    log_weights: torch.Tensor

    def __post_init__(self) -> None:
        log_weights = _frozen_double(self.log_weights).reshape(-1)
        if log_weights.numel() < 1 or not torch.isfinite(log_weights).all():
            raise ValueError("log weights must be a finite, nonempty vector")
        log_weights = log_weights - torch.logsumexp(log_weights, dim=0)
        object.__setattr__(self, "log_weights", log_weights)

    @classmethod
    def uniform(cls, num_particles: int) -> "ParticleWeights":
        if num_particles < 1:
            raise ValueError("num_particles must be positive")
        return cls(torch.full((num_particles,), -math.log(num_particles), dtype=torch.double))

    @property
    def probabilities(self) -> torch.Tensor:
        return self.log_weights.exp()

    @property
    def ess(self) -> float:
        return float(1.0 / self.probabilities.square().sum())

    @property
    def ess_fraction(self) -> float:
        return self.ess / self.log_weights.numel()

    def update(self, log_likelihood: torch.Tensor) -> "ParticleWeights":
        increment = _frozen_double(log_likelihood).reshape(-1)
        if increment.shape != self.log_weights.shape or not torch.isfinite(increment).all():
            raise ValueError("log likelihoods must be finite with one value per particle")
        return ParticleWeights(self.log_weights + increment)

    def conditional_ess_fraction(self, log_likelihood: torch.Tensor) -> float:
        """Relative conditional ESS of an incremental likelihood update."""

        increment = _frozen_double(log_likelihood).reshape(-1)
        if increment.shape != self.log_weights.shape or not torch.isfinite(increment).all():
            raise ValueError("log likelihoods must be finite with one value per particle")
        log_first_moment = torch.logsumexp(self.log_weights + increment, dim=0)
        log_second_moment = torch.logsumexp(self.log_weights + 2.0 * increment, dim=0)
        return float(torch.exp(2.0 * log_first_moment - log_second_moment))

    def weighted_variance(self, values: torch.Tensor) -> float:
        values = _frozen_double(values).reshape(-1)
        if values.shape != self.log_weights.shape:
            raise ValueError("values must have one entry per particle")
        mean = torch.dot(self.probabilities, values)
        return float(torch.dot(self.probabilities, (values - mean).square()))
