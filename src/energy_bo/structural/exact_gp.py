"""Batched exact GP calculations and rank-one cache updates."""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass

import torch

from .particles import SaasParticles


def matern52_covariance(
    x1: torch.Tensor,
    x2: torch.Tensor,
    lengthscales: torch.Tensor,
    outputscales: torch.Tensor,
) -> torch.Tensor:
    """Batched ARD Matérn-5/2 covariance using GPyTorch's distance convention."""

    x1 = x1.to(dtype=torch.double)
    x2 = x2.to(dtype=torch.double)
    lengthscales = lengthscales.to(dtype=torch.double)
    outputscales = outputscales.to(dtype=torch.double).reshape(-1)
    if x1.ndim != 2 or x2.ndim != 2 or lengthscales.ndim != 2:
        raise ValueError("x1, x2, and lengthscales must be matrices")
    if x1.shape[1] != x2.shape[1] or x1.shape[1] != lengthscales.shape[1]:
        raise ValueError("input and lengthscale dimensions must agree")
    if outputscales.shape != (lengthscales.shape[0],):
        raise ValueError("one outputscale is required per particle")
    scaled = (x1[None, :, None, :] - x2[None, None, :, :]) / lengthscales[:, None, None, :]
    distance = scaled.square().sum(dim=-1).sqrt()
    sqrt5_distance = math.sqrt(5.0) * distance
    base = (1.0 + sqrt5_distance + 5.0 * distance.square() / 3.0) * torch.exp(-sqrt5_distance)
    return outputscales[:, None, None] * base


@dataclass
class CacheCounters:
    full_factorizations: int = 0
    rank_one_appends: int = 0
    likelihood_updates: int = 0
    validation_factorizations: int = 0
    full_factorization_seconds: float = 0.0
    rank_one_seconds: float = 0.0
    prediction_seconds: float = 0.0
    likelihood_seconds: float = 0.0

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def full_log_marginal_likelihood(
    particles: SaasParticles,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    noise_variance: float,
) -> torch.Tensor:
    """Independent full-factorization exact log marginal likelihood per particle."""

    train_x = train_x.to(dtype=torch.double)
    train_y = train_y.to(dtype=torch.double).reshape(-1)
    covariance = matern52_covariance(
        train_x, train_x, particles.lengthscales, particles.outputscales
    )
    identity = torch.eye(train_x.shape[0], dtype=torch.double)
    covariance = covariance + float(noise_variance) * identity
    chol = torch.linalg.cholesky(covariance)
    centered = train_y[None, :] - particles.means[:, None]
    alpha = torch.cholesky_solve(centered[..., None], chol).squeeze(-1)
    quadratic = (centered * alpha).sum(dim=1)
    logdet = 2.0 * chol.diagonal(dim1=-2, dim2=-1).log().sum(dim=1)
    return -0.5 * (quadratic + logdet + train_x.shape[0] * math.log(2.0 * math.pi))


@dataclass
class ExactGPBatchState:
    """Exact state for fixed particles, updated without changing locations."""

    particles: SaasParticles
    noise_variance: float
    train_x: torch.Tensor
    train_y: torch.Tensor
    chol: torch.Tensor
    alpha: torch.Tensor
    counters: CacheCounters

    @classmethod
    def build(
        cls,
        particles: SaasParticles,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        noise_variance: float,
    ) -> "ExactGPBatchState":
        train_x = train_x.detach().to(dtype=torch.double, device="cpu").clone()
        train_y = train_y.detach().to(dtype=torch.double, device="cpu").reshape(-1).clone()
        if train_x.shape != (train_y.numel(), particles.dimension):
            raise ValueError("training data shapes do not match particle dimension")
        if not noise_variance > 0:
            raise ValueError("noise variance must be positive")
        counters = CacheCounters()
        start = time.perf_counter()
        covariance = matern52_covariance(
            train_x, train_x, particles.lengthscales, particles.outputscales
        )
        covariance = covariance + float(noise_variance) * torch.eye(train_x.shape[0], dtype=torch.double)
        chol = torch.linalg.cholesky(covariance)
        centered = train_y[None, :] - particles.means[:, None]
        alpha = torch.cholesky_solve(centered[..., None], chol).squeeze(-1)
        counters.full_factorizations += particles.num_particles
        counters.full_factorization_seconds += time.perf_counter() - start
        return cls(particles, float(noise_variance), train_x, train_y, chol, alpha, counters)

    def predict(
        self,
        test_x: torch.Tensor,
        *,
        observation_noise: bool = False,
        chunk_size: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return particle-wise posterior mean and marginal variance."""

        test_x = test_x.to(dtype=torch.double).reshape(-1, self.particles.dimension)
        chunk_size = test_x.shape[0] if chunk_size is None else int(chunk_size)
        means: list[torch.Tensor] = []
        variances: list[torch.Tensor] = []
        start = time.perf_counter()
        for chunk in test_x.split(chunk_size):
            cross = matern52_covariance(
                self.train_x, chunk, self.particles.lengthscales, self.particles.outputscales
            )
            mean = self.particles.means[:, None] + torch.einsum("pnm,pn->pm", cross, self.alpha)
            solved = torch.linalg.solve_triangular(self.chol, cross, upper=False)
            variance = self.particles.outputscales[:, None] - solved.square().sum(dim=1)
            variance = variance.clamp_min(torch.finfo(torch.double).eps)
            if observation_noise:
                variance = variance + self.noise_variance
            means.append(mean)
            variances.append(variance)
        self.counters.prediction_seconds += time.perf_counter() - start
        return torch.cat(means, dim=1), torch.cat(variances, dim=1)

    def predictive_log_likelihood(self, new_x: torch.Tensor, new_y: float | torch.Tensor) -> torch.Tensor:
        start = time.perf_counter()
        mean, variance = self.predict(new_x.reshape(1, -1), observation_noise=True)
        residual = torch.as_tensor(new_y, dtype=torch.double) - mean[:, 0]
        result = -0.5 * (residual.square() / variance[:, 0] + variance[:, 0].log() + math.log(2.0 * math.pi))
        self.counters.likelihood_updates += self.particles.num_particles
        self.counters.likelihood_seconds += time.perf_counter() - start
        return result

    def append(self, new_x: torch.Tensor, new_y: float | torch.Tensor) -> None:
        """Append one observation using the block-Cholesky identity."""

        start = time.perf_counter()
        x = new_x.detach().to(dtype=torch.double, device="cpu").reshape(1, self.particles.dimension)
        y = torch.as_tensor(new_y, dtype=torch.double).reshape(1)
        cross = matern52_covariance(
            self.train_x, x, self.particles.lengthscales, self.particles.outputscales
        )
        solved = torch.linalg.solve_triangular(self.chol, cross, upper=False)
        schur = self.particles.outputscales + self.noise_variance - solved[:, :, 0].square().sum(dim=1)
        if not torch.all(schur > 0):
            raise RuntimeError("nonpositive Schur complement in rank-one GP update")
        old_n = self.train_x.shape[0]
        new_chol = torch.zeros(
            (self.particles.num_particles, old_n + 1, old_n + 1), dtype=torch.double
        )
        new_chol[:, :old_n, :old_n] = self.chol
        new_chol[:, old_n, :old_n] = solved[:, :, 0]
        new_chol[:, old_n, old_n] = schur.sqrt()
        self.train_x = torch.cat((self.train_x, x), dim=0)
        self.train_y = torch.cat((self.train_y, y), dim=0)
        self.chol = new_chol
        centered = self.train_y[None, :] - self.particles.means[:, None]
        self.alpha = torch.cholesky_solve(centered[..., None], self.chol).squeeze(-1)
        self.counters.rank_one_appends += self.particles.num_particles
        self.counters.rank_one_seconds += time.perf_counter() - start

    def validate_against_full(self, test_x: torch.Tensor) -> dict[str, float]:
        """Rebuild exactly and report cache discrepancies (validation cost is separate)."""

        start = time.perf_counter()
        rebuilt = ExactGPBatchState.build(
            self.particles, self.train_x, self.train_y, self.noise_variance
        )
        cached_mean, cached_variance = self.predict(test_x)
        full_mean, full_variance = rebuilt.predict(test_x)
        self.counters.validation_factorizations += self.particles.num_particles
        elapsed = time.perf_counter() - start
        return {
            "chol_max_abs": float((self.chol - rebuilt.chol).abs().max()),
            "alpha_max_abs": float((self.alpha - rebuilt.alpha).abs().max()),
            "mean_max_abs": float((cached_mean - full_mean).abs().max()),
            "variance_max_abs": float((cached_variance - full_variance).abs().max()),
            "validation_seconds": elapsed,
        }
