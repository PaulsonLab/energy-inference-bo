from __future__ import annotations

import torch

from energy_bo.decision.coresets import (
    acquisition_frank_wolfe,
    posterior_k_medoids,
    random_equal,
)
from energy_bo.decision.joint_target import joint_target_marginals
from energy_bo.decision.metrics import acquisition_metrics, signature_spectrum
from energy_bo.decision.signatures import particle_ei_signatures
from energy_bo.structural.acquisition import weighted_expected_improvement
from energy_bo.structural.acquisition import gaussian_expected_improvement
from energy_bo.structural.exact_gp import ExactGPBatchState
from energy_bo.structural.particles import ParticleWeights, SaasParticles


def _state() -> ExactGPBatchState:
    particles = SaasParticles(
        lengthscales=torch.tensor([[0.3, 0.7], [0.8, 0.4], [1.1, 0.9]]),
        means=torch.tensor([-0.1, 0.0, 0.2]),
        outputscales=torch.tensor([0.8, 1.0, 1.3]),
    )
    train_x = torch.tensor([[0.1, 0.2], [0.7, 0.8], [0.4, 0.6]], dtype=torch.double)
    train_y = torch.tensor([-0.2, 0.4, 0.1], dtype=torch.double)
    return ExactGPBatchState.build(particles, train_x, train_y, 1e-4)


def test_particle_signature_average_matches_weighted_ei() -> None:
    state = _state()
    candidates = torch.tensor(
        [[0.2, 0.3], [0.5, 0.4], [0.9, 0.7], [0.3, 0.9]], dtype=torch.double
    )
    signatures = particle_ei_signatures(state, candidates, best_f=0.4)
    direct = weighted_expected_improvement(
        state, ParticleWeights.uniform(3), candidates, best_f=0.4
    )
    assert signatures.shape == (3, 4)
    assert torch.allclose(signatures.mean(dim=0), direct, atol=1e-14, rtol=1e-13)


def test_analytic_ei_remains_nonnegative_in_extreme_tail() -> None:
    value = gaussian_expected_improvement(
        torch.tensor([-100.0], dtype=torch.double),
        torch.tensor([1e-6], dtype=torch.double),
        best_f=1.0,
    )
    assert torch.isfinite(value).all()
    assert torch.all(value >= 0)


def test_signature_spectrum_detects_known_centered_rank_one() -> None:
    direction = torch.tensor([1.0, -2.0, 0.5, 3.0], dtype=torch.double)
    coefficients = torch.tensor([-2.0, -1.0, 1.0, 2.0], dtype=torch.double)
    matrix = 5.0 + coefficients[:, None] * direction[None, :]
    spectrum = signature_spectrum(matrix)
    assert spectrum["rank90"] == 1
    assert spectrum["rank99"] == 1
    assert abs(spectrum["entropy_effective_rank"] - 1.0) < 1e-12


def test_random_and_medoid_coresets_are_deterministic_simplexes() -> None:
    signatures = torch.arange(60, dtype=torch.double).reshape(10, 6).sin().abs()
    features = torch.arange(40, dtype=torch.double).reshape(10, 4).cos()
    first = random_equal(signatures, 4, seed=7)
    second = random_equal(signatures, 4, seed=7)
    medoids = posterior_k_medoids(features, 4)
    assert torch.equal(first.indices, second.indices)
    for coreset in (first, medoids):
        assert coreset.indices.numel() == 4
        assert torch.all(coreset.weights >= 0)
        assert torch.isclose(coreset.weights.sum(), torch.tensor(1.0, dtype=torch.double))


def test_medoid_coreset_handles_duplicate_short_chain_particles() -> None:
    features = torch.tensor(
        [[0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [1.0, 1.0]], dtype=torch.double
    )
    coreset = posterior_k_medoids(features, 4)
    assert coreset.indices.numel() == 4
    assert torch.all(coreset.weights >= 0)
    assert torch.isclose(coreset.weights.sum(), torch.tensor(1.0, dtype=torch.double))


def test_frank_wolfe_is_deterministic_and_monotone() -> None:
    generator = torch.Generator().manual_seed(11)
    signatures = torch.rand((20, 12), generator=generator, dtype=torch.double)
    first = acquisition_frank_wolfe(signatures, 8)
    second = acquisition_frank_wolfe(signatures, 8)
    assert torch.equal(first.indices, second.indices)
    assert torch.allclose(first.weights, second.weights)
    assert torch.all(first.weights >= 0)
    assert torch.isclose(first.weights.sum(), torch.tensor(1.0, dtype=torch.double))
    assert all(
        later <= earlier + 1e-15
        for earlier, later in zip(first.error_history[:-1], first.error_history[1:], strict=True)
    )


def test_finite_candidate_decision_regret_bound() -> None:
    teacher = torch.tensor([0.2, 1.0, 0.8, 0.1], dtype=torch.double)
    approximate = torch.tensor([0.2, 0.7, 0.9, 0.1], dtype=torch.double)
    metrics = acquisition_metrics(teacher, approximate)
    assert metrics["teacher_index"] == 1
    assert metrics["approximate_index"] == 2
    assert abs(metrics["absolute_decision_regret"] - 0.2) < 1e-15
    assert metrics["regret_bound_pass"]
    assert metrics["absolute_decision_regret"] <= metrics["twice_delta"] + 1e-15


def test_joint_m1_recovers_teacher_acquisition() -> None:
    signatures = torch.tensor(
        [[0.2, 0.8, 0.1], [0.5, 0.4, 0.3], [0.1, 0.9, 0.2]], dtype=torch.double
    )
    weights = torch.tensor([0.2, 0.3, 0.5], dtype=torch.double)
    result = joint_target_marginals(signatures, weights)
    assert torch.allclose(result["m1"], result["normalized_teacher"], atol=1e-15)


def test_independent_m2_recovers_squared_teacher_not_common_particle() -> None:
    signatures = torch.tensor(
        [[0.1, 1.0, 0.2], [1.2, 0.2, 0.4], [0.3, 0.8, 1.1]], dtype=torch.double
    )
    result = joint_target_marginals(signatures)
    assert torch.allclose(
        result["independent_m2"], result["normalized_teacher_squared"], atol=1e-15
    )
    assert not torch.allclose(result["common_particle_m2"], result["independent_m2"])
