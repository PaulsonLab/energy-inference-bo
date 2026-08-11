"""Posterior, prediction, and decision diagnostics for Task 02A."""

from __future__ import annotations

import math

import numpy as np
import torch
from scipy.stats import spearmanr, wasserstein_distance

from energy_bo.metrics import top_fraction_overlap

from .exact_gp import ExactGPBatchState
from .particles import ParticleWeights, SaasParticles


def weighted_quantile(values: torch.Tensor, weights: torch.Tensor, quantile: float) -> float:
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must lie in [0, 1]")
    values = values.detach().double().reshape(-1)
    weights = weights.detach().double().reshape(-1)
    order = torch.argsort(values)
    sorted_weights = weights[order]
    cumulative = torch.cumsum(sorted_weights, dim=0)
    index = int(torch.searchsorted(cumulative, torch.tensor(quantile, dtype=torch.double)))
    return float(values[order[min(index, values.numel() - 1)]])


def lengthscale_summary(
    particles: SaasParticles, weights: ParticleWeights
) -> list[dict[str, float | int]]:
    probabilities = weights.probabilities
    inverse = particles.lengthscales.reciprocal()
    top_two = torch.topk(inverse, k=min(2, particles.dimension), dim=1).indices
    rows: list[dict[str, float | int]] = []
    for dimension in range(particles.dimension):
        values = particles.lengthscales[:, dimension]
        rows.append(
            {
                "dimension": dimension,
                "median": weighted_quantile(values, probabilities, 0.5),
                "q25": weighted_quantile(values, probabilities, 0.25),
                "q75": weighted_quantile(values, probabilities, 0.75),
                "top2_relevance_probability": float(
                    probabilities[(top_two == dimension).any(dim=1)].sum()
                ),
            }
        )
    return rows


def log_lengthscale_wasserstein(
    reused: SaasParticles,
    reused_weights: ParticleWeights,
    fresh: SaasParticles,
) -> list[float]:
    return [
        float(
            wasserstein_distance(
                reused.lengthscales[:, d].log().numpy(),
                fresh.lengthscales[:, d].log().numpy(),
                u_weights=reused_weights.probabilities.numpy(),
            )
        )
        for d in range(reused.dimension)
    ]


def standardized_rbf_mmd(
    reused: SaasParticles,
    reused_weights: ParticleWeights,
    fresh: SaasParticles,
) -> float:
    """Biased weighted RBF MMD using pooled scaling and median bandwidth."""

    x = reused.feature_matrix
    y = fresh.feature_matrix
    pooled = torch.cat((x, y), dim=0)
    scale = pooled.std(dim=0, unbiased=False).clamp_min(1e-12)
    x = (x - pooled.mean(dim=0)) / scale
    y = (y - pooled.mean(dim=0)) / scale
    distances = torch.pdist(pooled / scale)
    positive = distances[distances > 0]
    bandwidth = torch.median(positive) if positive.numel() else torch.tensor(1.0, dtype=torch.double)

    def kernel(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return torch.exp(-torch.cdist(left, right).square() / (2.0 * bandwidth.square()))

    wx = reused_weights.probabilities
    wy = torch.full((fresh.num_particles,), 1.0 / fresh.num_particles, dtype=torch.double)
    value = wx @ kernel(x, x) @ wx + wy @ kernel(y, y) @ wy - 2.0 * wx @ kernel(x, y) @ wy
    return float(value.clamp_min(0.0).sqrt())


def mixture_prediction(
    state: ExactGPBatchState,
    weights: ParticleWeights,
    test_x: torch.Tensor,
    *,
    observation_noise: bool,
    chunk_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    mean, variance = state.predict(
        test_x, observation_noise=observation_noise, chunk_size=chunk_size
    )
    probabilities = weights.probabilities[:, None]
    mixture_mean = (probabilities * mean).sum(dim=0)
    total_variance = (probabilities * (variance + mean.square())).sum(dim=0) - mixture_mean.square()
    return mixture_mean, total_variance.clamp_min(torch.finfo(torch.double).eps)


def mixture_log_score(
    state: ExactGPBatchState,
    weights: ParticleWeights,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    *,
    chunk_size: int | None = None,
) -> float:
    mean, variance = state.predict(test_x, observation_noise=True, chunk_size=chunk_size)
    outcomes = test_y.double().reshape(1, -1)
    log_density = -0.5 * (
        (outcomes - mean).square() / variance + variance.log() + math.log(2.0 * math.pi)
    )
    mixture = torch.logsumexp(weights.log_weights[:, None] + log_density, dim=0)
    return float(mixture.mean())


def prediction_comparison(
    reused_state: ExactGPBatchState,
    reused_weights: ParticleWeights,
    fresh_state: ExactGPBatchState,
    fresh_weights: ParticleWeights,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    *,
    chunk_size: int | None = None,
) -> dict[str, float]:
    reused_mean, reused_variance = mixture_prediction(
        reused_state, reused_weights, test_x, observation_noise=False, chunk_size=chunk_size
    )
    fresh_mean, fresh_variance = mixture_prediction(
        fresh_state, fresh_weights, test_x, observation_noise=False, chunk_size=chunk_size
    )
    return {
        "predictive_mean_rmse": float(torch.mean((reused_mean - fresh_mean).square()).sqrt()),
        "predictive_variance_rmse": float(
            torch.mean((reused_variance - fresh_variance).square()).sqrt()
        ),
        "reused_log_score": mixture_log_score(
            reused_state, reused_weights, test_x, test_y, chunk_size=chunk_size
        ),
        "fresh_log_score": mixture_log_score(
            fresh_state, fresh_weights, test_x, test_y, chunk_size=chunk_size
        ),
    }


def ei_comparison(
    reused_ei: torch.Tensor,
    fresh_ei: torch.Tensor,
    candidate_x: torch.Tensor,
) -> dict[str, float | int]:
    reused = reused_ei.detach().numpy()
    fresh = fresh_ei.detach().numpy()
    reused_index = int(np.argmax(reused))
    fresh_index = int(np.argmax(fresh))
    maximum = max(float(fresh.max()), 1e-15)
    return {
        "ei_spearman": float(spearmanr(fresh, reused).statistic),
        "ei_top5_overlap": float(top_fraction_overlap(fresh, reused, fraction=0.05)),
        "ei_max_relative_error": float(abs(reused.max() - fresh.max()) / maximum),
        "reused_ei_index": reused_index,
        "fresh_ei_index": fresh_index,
        "ei_candidate_distance": float(
            torch.linalg.vector_norm(candidate_x[reused_index] - candidate_x[fresh_index])
        ),
    }
