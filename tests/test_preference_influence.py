from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from conditioned_bo.preference_bo import gp_reference_posterior
from conditioned_bo.preference_influence import (
    comparison_diagnostics,
    comparison_matrix_from_precision,
    ei_decision_footprint,
    factor_block_metadata,
    factor_sensitivity_matrix,
    logistic_preference_energy,
    logistic_preference_gradient,
    logistic_preference_hessian,
    omitted_factor_load,
    preference_block_sensitivity,
    preference_blocks,
    ranked_omitted_contributions,
    structural_bound,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads(
    (REPOSITORY_ROOT / "experiments/preference_bo/configs/minimal_pilot.json").read_text()
)
GRID = np.linspace(
    CONFIG["action_grid"]["minimum"],
    CONFIG["action_grid"]["maximum"],
    CONFIG["action_grid"]["count"],
)
PAIRS = np.asarray(CONFIG["preference_bank"]["endpoint_index_pairs"], dtype=int)
TAU = CONFIG["preference_bank"]["temperature"]


def _finite_gradient(function, value: np.ndarray, step: float = 1e-6) -> np.ndarray:
    result = np.empty_like(value)
    for coordinate in range(value.size):
        perturbation = np.zeros_like(value)
        perturbation[coordinate] = step
        result[coordinate] = (
            function(value + perturbation) - function(value - perturbation)
        ) / (2.0 * step)
    return result


def _finite_jacobian(function, value: np.ndarray, step: float = 2e-5) -> np.ndarray:
    baseline = np.asarray(function(value), dtype=float)
    result = np.empty((baseline.size, value.size), dtype=float)
    for coordinate in range(value.size):
        perturbation = np.zeros_like(value)
        perturbation[coordinate] = step
        result[:, coordinate] = (
            function(value + perturbation) - function(value - perturbation)
        ) / (2.0 * step)
    return result


def _prior():
    gp = CONFIG["gp_reference"]
    return gp_reference_posterior(
        GRID,
        observed_indices=[],
        observed_values=[],
        kernel_amplitude=gp["kernel_amplitude"],
        kernel_lengthscale=gp["kernel_lengthscale"],
        observation_noise_standard_deviation=gp[
            "observation_noise_standard_deviation"
        ],
    )


def test_logistic_energy_is_stable_at_extreme_margins() -> None:
    latent = np.zeros(GRID.size)
    latent[0] = 1.0e300
    latent[16] = -1.0e300
    favorable = logistic_preference_energy(latent, (0, 16), 1, TAU)
    unfavorable = logistic_preference_energy(latent, (0, 16), -1, TAU)
    assert favorable == 0.0
    assert np.isfinite(unfavorable)
    np.testing.assert_allclose(unfavorable, 2.0e300 / TAU)


def test_logistic_gradient_and_hessian_match_finite_differences() -> None:
    latent = np.linspace(-0.7, 0.9, GRID.size)
    pair = (3, 13)
    sign = -1
    analytic_gradient = logistic_preference_gradient(latent, pair, sign, TAU)
    finite_gradient = _finite_gradient(
        lambda value: logistic_preference_energy(value, pair, sign, TAU), latent
    )
    np.testing.assert_allclose(
        analytic_gradient, finite_gradient, rtol=2e-9, atol=2e-10
    )

    analytic_hessian = logistic_preference_hessian(latent, pair, sign, TAU)
    finite_hessian = _finite_jacobian(
        lambda value: logistic_preference_gradient(value, pair, sign, TAU), latent
    )
    np.testing.assert_allclose(
        analytic_hessian, finite_hessian, rtol=3e-9, atol=3e-10
    )


def test_logistic_convexity_and_exact_sensitivity_bound() -> None:
    bound = preference_block_sensitivity(TAU)
    np.testing.assert_allclose(bound, np.sqrt(2.0) / TAU)
    for margin in np.linspace(-30.0, 30.0, 121):
        latent = np.zeros(GRID.size)
        latent[2] = margin
        gradient = logistic_preference_gradient(latent, (2, 14), 1, TAU)
        hessian = logistic_preference_hessian(latent, (2, 14), 1, TAU)
        assert np.linalg.norm(gradient) <= bound + 2e-15
        assert np.linalg.eigvalsh(hessian).min() >= -2e-15


def test_factor_support_metadata_and_zero_cross_block_hessian() -> None:
    blocks = preference_blocks(GRID.size)
    metadata = factor_block_metadata(PAIRS, blocks)
    assert len(metadata) == PAIRS.shape[0]
    assert [item.block_index for item in metadata] == [8, 7, 6, 5, 4, 3, 2, 1]
    for item in metadata:
        assert set(item.support) == set(PAIRS[item.factor_index])
        assert set(item.support).issubset(set(blocks[item.block_index]))

        hessian = logistic_preference_hessian(
            np.linspace(-0.2, 0.3, GRID.size), item.support, 1, TAU
        )
        for other_index, other_block in enumerate(blocks):
            if other_index == item.block_index:
                continue
            cross = hessian[np.ix_(blocks[item.block_index], other_block)]
            np.testing.assert_array_equal(cross, np.zeros_like(cross))


def test_frozen_prior_comparison_matrix_regression_is_derived() -> None:
    posterior = _prior()
    blocks = preference_blocks(GRID.size)
    matrix, rho, kappa = comparison_matrix_from_precision(
        posterior.precision, blocks
    )
    diagnostics = comparison_diagnostics(matrix)
    assert diagnostics.is_spd
    np.testing.assert_allclose(
        diagnostics.minimum_eigenvalue, 5.63301154, rtol=0.0, atol=5e-8
    )
    np.testing.assert_allclose(diagnostics.condition_number, 3.18, atol=0.01)
    assert np.all(rho > 0.0)
    assert np.all(kappa >= 0.0)
    assert not np.allclose(matrix, np.diag(np.diag(matrix)))


def test_scalar_observations_cannot_reduce_diagonal_curvature_or_destroy_spd() -> None:
    prior = _prior()
    gp = CONFIG["gp_reference"]
    posterior = gp_reference_posterior(
        GRID,
        observed_indices=[2, 8, 14, 11],
        observed_values=[0.1, -0.2, 0.4, 0.3],
        kernel_amplitude=gp["kernel_amplitude"],
        kernel_lengthscale=gp["kernel_lengthscale"],
        observation_noise_standard_deviation=gp[
            "observation_noise_standard_deviation"
        ],
    )
    blocks = preference_blocks(GRID.size)
    prior_matrix, prior_rho, prior_kappa = comparison_matrix_from_precision(
        prior.precision, blocks
    )
    posterior_matrix, posterior_rho, posterior_kappa = (
        comparison_matrix_from_precision(posterior.precision, blocks)
    )
    assert np.all(posterior_rho >= prior_rho - 1e-12)
    np.testing.assert_allclose(posterior_kappa, prior_kappa, atol=2e-10)
    assert comparison_diagnostics(prior_matrix).is_spd
    assert comparison_diagnostics(posterior_matrix).is_spd


def test_structural_accounting_zero_active_and_monotonicity() -> None:
    posterior = _prior()
    blocks = preference_blocks(GRID.size)
    matrix, _, _ = comparison_matrix_from_precision(posterior.precision, blocks)
    metadata = factor_block_metadata(PAIRS, blocks)
    sensitivities = factor_sensitivity_matrix(metadata, len(blocks), TAU)
    footprint = ei_decision_footprint(3, 13, blocks)

    all_active = np.ones(PAIRS.shape[0], dtype=bool)
    all_active_load = omitted_factor_load(sensitivities, ~all_active)
    assert structural_bound(matrix, footprint, all_active_load) == 0.0

    omitted = np.ones(PAIRS.shape[0], dtype=bool)
    previous = structural_bound(
        matrix, footprint, omitted_factor_load(sensitivities, omitted)
    )
    for removed in (0, 3, 7):
        contributions = ranked_omitted_contributions(
            matrix, footprint, sensitivities, omitted
        )
        bound = structural_bound(
            matrix, footprint, omitted_factor_load(sensitivities, omitted)
        )
        np.testing.assert_allclose(
            bound, contributions[omitted].sum(), rtol=3e-15, atol=3e-15
        )
        assert bound <= previous + 2e-15
        omitted[removed] = False
        previous = bound
    reduced = structural_bound(
        matrix, footprint, omitted_factor_load(sensitivities, omitted)
    )
    assert reduced <= previous + 2e-15


def test_structural_runtime_path_never_calls_explicit_inverse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    posterior = _prior()
    blocks = preference_blocks(GRID.size)
    matrix, _, _ = comparison_matrix_from_precision(posterior.precision, blocks)
    metadata = factor_block_metadata(PAIRS, blocks)
    sensitivities = factor_sensitivity_matrix(metadata, len(blocks), TAU)
    load = omitted_factor_load(sensitivities, np.ones(PAIRS.shape[0], dtype=bool))
    footprint = ei_decision_footprint(0, 16, blocks)

    def forbidden_inverse(*_args, **_kwargs):
        raise AssertionError("explicit inversion is forbidden")

    monkeypatch.setattr(np.linalg, "inv", forbidden_inverse)
    assert structural_bound(matrix, footprint, load) > 0.0
    scores = ranked_omitted_contributions(
        matrix, footprint, sensitivities, np.ones(PAIRS.shape[0], dtype=bool)
    )
    assert np.all(scores > 0.0)
