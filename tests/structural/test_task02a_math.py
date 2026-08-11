from __future__ import annotations

import math
from types import SimpleNamespace

import gpytorch
import torch

from energy_bo.structural.acquisition import (
    gaussian_expected_improvement,
    weighted_expected_improvement,
)
from energy_bo.structural.exact_gp import (
    ExactGPBatchState,
    full_log_marginal_likelihood,
    matern52_covariance,
)
from energy_bo.structural.particles import ParticleWeights, SaasParticles
from energy_bo.structural.preprocessing import FrozenOutputTransform


def _case() -> tuple[SaasParticles, torch.Tensor, torch.Tensor]:
    particles = SaasParticles(
        lengthscales=torch.tensor(
            [[0.3, 0.7], [0.8, 0.4], [0.5, 1.1]], dtype=torch.double
        ),
        means=torch.tensor([-0.2, 0.1, 0.4], dtype=torch.double),
        outputscales=torch.tensor([0.8, 1.2, 0.6], dtype=torch.double),
    )
    x = torch.tensor([[0.05, 0.1], [0.35, 0.8], [0.7, 0.25], [0.9, 0.65]], dtype=torch.double)
    y = torch.tensor([-0.4, 0.9, 0.2, -0.1], dtype=torch.double)
    return particles, x, y


def test_marginal_likelihood_increment_equals_predictive_likelihood() -> None:
    particles, x, y = _case()
    noise = 1e-4
    state = ExactGPBatchState.build(particles, x[:3], y[:3], noise)
    predictive = state.predictive_log_likelihood(x[3], y[3])
    increment = full_log_marginal_likelihood(particles, x, y, noise) - full_log_marginal_likelihood(
        particles, x[:3], y[:3], noise
    )
    torch.testing.assert_close(predictive, increment, atol=2e-12, rtol=2e-12)


def test_rank_one_cholesky_append_equals_full_factorization() -> None:
    particles, x, y = _case()
    state = ExactGPBatchState.build(particles, x[:3], y[:3], 1e-4)
    state.append(x[3], y[3])
    full = ExactGPBatchState.build(particles, x, y, 1e-4)
    torch.testing.assert_close(state.chol, full.chol, atol=2e-12, rtol=2e-12)
    torch.testing.assert_close(state.alpha, full.alpha, atol=5e-12, rtol=5e-12)


def test_cached_prediction_equals_full_rebuild() -> None:
    particles, x, y = _case()
    state = ExactGPBatchState.build(particles, x[:2], y[:2], 1e-4)
    state.append(x[2], y[2])
    state.append(x[3], y[3])
    check = state.validate_against_full(torch.tensor([[0.2, 0.3], [0.6, 0.6]], dtype=torch.double))
    assert check["chol_max_abs"] < 2e-12
    assert check["alpha_max_abs"] < 1e-11
    assert check["mean_max_abs"] < 1e-11
    assert check["variance_max_abs"] < 1e-11


def test_log_weight_normalization_is_stable() -> None:
    weights = ParticleWeights.uniform(3).update(
        torch.tensor([-1e4, -1e4 - 1.0, -1e4 - 2.0], dtype=torch.double)
    )
    assert torch.isfinite(weights.log_weights).all()
    torch.testing.assert_close(weights.probabilities.sum(), torch.tensor(1.0, dtype=torch.double))
    assert int(torch.argmax(weights.probabilities)) == 0


def test_equal_weight_ess() -> None:
    weights = ParticleWeights.uniform(17)
    assert math.isclose(weights.ess, 17.0, rel_tol=1e-14)
    assert math.isclose(weights.ess_fraction, 1.0, rel_tol=1e-14)


def test_constant_likelihood_leaves_weights_unchanged() -> None:
    weights = ParticleWeights(torch.tensor([-3.0, -1.0, -2.0], dtype=torch.double))
    updated = weights.update(torch.full((3,), 1234.5, dtype=torch.double))
    torch.testing.assert_close(updated.probabilities, weights.probabilities, atol=2e-13, rtol=2e-13)


def test_weighted_ei_is_particle_average() -> None:
    particles, x, y = _case()
    state = ExactGPBatchState.build(particles, x, y, 1e-4)
    weights = ParticleWeights(torch.log(torch.tensor([0.2, 0.3, 0.5], dtype=torch.double)))
    candidates = torch.tensor([[0.1, 0.7], [0.8, 0.2]], dtype=torch.double)
    mean, variance = state.predict(candidates)
    expected = (weights.probabilities[:, None] * gaussian_expected_improvement(mean, variance, 0.9)).sum(0)
    actual = weighted_expected_improvement(state, weights, candidates, 0.9)
    torch.testing.assert_close(actual, expected, atol=1e-14, rtol=1e-14)


def test_output_preprocessing_is_frozen_after_initial_design() -> None:
    initial = torch.tensor([-3.0, 1.0, 5.0, 9.0], dtype=torch.double)
    transform = FrozenOutputTransform.fit(initial)
    assert transform.mean == 3.0
    assert transform.scale == math.sqrt(20.0)
    before = transform.transform(initial)
    later = torch.cat((initial, torch.tensor([1000.0], dtype=torch.double)))
    after = transform.transform(later)[:4]
    torch.testing.assert_close(before, after)
    assert transform == FrozenOutputTransform(mean=3.0, scale=math.sqrt(20.0))


def test_custom_matern52_matches_gpytorch() -> None:
    particles, x, _ = _case()
    kernel = gpytorch.kernels.ScaleKernel(
        gpytorch.kernels.MaternKernel(
            nu=2.5,
            ard_num_dims=particles.dimension,
            batch_shape=torch.Size([particles.num_particles]),
        ),
        batch_shape=torch.Size([particles.num_particles]),
    ).double()
    kernel.base_kernel.lengthscale = particles.lengthscales[:, None, :]
    kernel.outputscale = particles.outputscales
    expected = kernel(x[:3], x[1:]).to_dense()
    actual = matern52_covariance(
        x[:3], x[1:], particles.lengthscales, particles.outputscales
    )
    torch.testing.assert_close(actual, expected, atol=2e-14, rtol=2e-14)


def test_particle_extraction_from_batched_botorch_shape() -> None:
    lengthscales = torch.tensor(
        [[[0.2, 0.4]], [[0.5, 0.7]], [[0.9, 1.1]]], dtype=torch.double
    )
    model = SimpleNamespace(
        covar_module=SimpleNamespace(
            base_kernel=SimpleNamespace(lengthscale=lengthscales),
            outputscale=torch.tensor([0.6, 0.8, 1.2], dtype=torch.double),
        ),
        mean_module=SimpleNamespace(constant=torch.tensor([0.1, -0.2, 0.3], dtype=torch.double)),
    )
    particles = SaasParticles.from_botorch(model)
    assert particles.lengthscales.shape == (3, 2)
    torch.testing.assert_close(particles.lengthscales, lengthscales[:, 0, :])
    assert particles.num_particles == 3
