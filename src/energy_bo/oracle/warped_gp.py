"""Exact sparse latent-GP oracle with an invertible exponential warp."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch.quasirandom import SobolEngine


def _matern52(x1: torch.Tensor, x2: torch.Tensor, lengthscale: float = 0.2) -> torch.Tensor:
    delta = (x1[:, None, :2] - x2[None, :, :2]) / lengthscale
    distance = delta.square().sum(dim=-1).sqrt()
    scaled = math.sqrt(5.0) * distance
    return (1.0 + scaled + 5.0 * distance.square() / 3.0) * torch.exp(-scaled)


@dataclass(frozen=True)
class LatentPosterior:
    mean: torch.Tensor
    variance: torch.Tensor
    alpha: float

    def __post_init__(self) -> None:
        if self.mean.shape != self.variance.shape or not torch.all(self.variance > 0):
            raise ValueError("posterior mean and variance must match with positive variance")

    def transform(self, latent: torch.Tensor) -> torch.Tensor:
        return latent if self.alpha == 0 else torch.expm1(self.alpha * latent) / self.alpha

    def inverse(self, value: torch.Tensor) -> torch.Tensor:
        return value if self.alpha == 0 else torch.log1p(self.alpha * value) / self.alpha

    @property
    def predictive_mean(self) -> torch.Tensor:
        if self.alpha == 0:
            return self.mean
        return (torch.exp(self.alpha * self.mean + 0.5 * self.alpha**2 * self.variance) - 1) / self.alpha

    @property
    def predictive_variance(self) -> torch.Tensor:
        if self.alpha == 0:
            return self.variance
        first = torch.exp(self.alpha * self.mean + 0.5 * self.alpha**2 * self.variance)
        second = torch.exp(2 * self.alpha * self.mean + 2 * self.alpha**2 * self.variance)
        return (second - first.square()) / self.alpha**2

    def cdf(self, value: torch.Tensor) -> torch.Tensor:
        latent = self.inverse(torch.as_tensor(value, dtype=torch.double))
        z = (latent - self.mean) / self.variance.sqrt()
        return 0.5 * (1 + torch.erf(z / math.sqrt(2)))

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        latent = self.inverse(torch.as_tensor(value, dtype=torch.double))
        z = (latent - self.mean) / self.variance.sqrt()
        log_jacobian = torch.zeros_like(z) if self.alpha == 0 else -self.alpha * latent
        return -0.5 * z.square() - 0.5 * self.variance.log() - 0.5 * math.log(2 * math.pi) + log_jacobian

    def probability_improvement(self, best_f: float) -> torch.Tensor:
        threshold = self.inverse(torch.as_tensor(best_f, dtype=torch.double))
        z = (self.mean - threshold) / self.variance.sqrt()
        return 0.5 * (1 + torch.erf(z / math.sqrt(2)))

    def expected_improvement(self, best_f: float) -> torch.Tensor:
        scale = self.variance.sqrt()
        if self.alpha == 0:
            improvement = self.mean - best_f
            z = improvement / scale
            return improvement * 0.5 * (1 + torch.erf(z / math.sqrt(2))) + scale * torch.exp(-0.5*z.square()) / math.sqrt(2*math.pi)
        threshold = self.inverse(torch.as_tensor(best_f, dtype=torch.double))
        first_z = (self.mean + self.alpha * self.variance - threshold) / scale
        second_z = (self.mean - threshold) / scale
        first = torch.exp(self.alpha*self.mean + 0.5*self.alpha**2*self.variance) * 0.5 * (1 + torch.erf(first_z/math.sqrt(2)))
        second = torch.exp(self.alpha*threshold) * 0.5 * (1 + torch.erf(second_z/math.sqrt(2)))
        return ((first - second) / self.alpha).clamp_min(0)


@dataclass(frozen=True)
class SparseWarpedGPOracle:
    train_x_all: torch.Tensor
    latent_all: torch.Tensor
    alpha: float
    jitter: float = 1e-8

    @classmethod
    def generate(cls, dimension: int, seed: int, count: int = 64, alpha: float = 0.0) -> "SparseWarpedGPOracle":
        if dimension < 2 or count < 2 or alpha < 0:
            raise ValueError("invalid oracle configuration")
        train_x = SobolEngine(dimension, scramble=True, seed=seed).draw(count).double()
        covariance = _matern52(train_x, train_x) + 1e-8 * torch.eye(count, dtype=torch.double)
        generator = torch.Generator().manual_seed(100_000 + seed)
        latent = torch.linalg.cholesky(covariance) @ torch.randn(count, dtype=torch.double, generator=generator)
        return cls(train_x, latent, float(alpha))

    def outcomes(self, count: int) -> torch.Tensor:
        latent = self.latent_all[:count]
        return latent if self.alpha == 0 else torch.expm1(self.alpha * latent) / self.alpha

    def posterior(self, test_x: torch.Tensor, count: int) -> LatentPosterior:
        device=test_x.device
        train_x = self.train_x_all[:count].to(device)
        train_y = self.latent_all[:count].to(device)
        kernel = _matern52(train_x, train_x) + self.jitter * torch.eye(count, dtype=torch.double,device=device)
        chol = torch.linalg.cholesky(kernel)
        cross = _matern52(train_x, test_x.double())
        alpha = torch.cholesky_solve(train_y[:, None], chol).squeeze(-1)
        mean = cross.T @ alpha
        solved = torch.linalg.solve_triangular(chol, cross, upper=False)
        variance = (1.0 - solved.square().sum(dim=0)).clamp_min(1e-12)
        return LatentPosterior(mean, variance, self.alpha)
