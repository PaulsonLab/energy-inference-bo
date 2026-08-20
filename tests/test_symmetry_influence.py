from __future__ import annotations

import csv
import hashlib
import json
import runpy
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.special import expit

from conditioned_bo.symmetry_influence import (
    active_set_mask,
    conditional_ei_gradient,
    conditional_expected_improvement,
    ei_action_coefficients,
    ei_block_decision_footprint,
    factor_sensitivity,
    log_exponential_block_footprint,
    omitted_factor_load,
    ou_ar1_precision,
    ou_symmetry_comparison,
    ranked_omitted_contributions,
    reflection_blocks,
    row_dominance_margins,
    solve_comparison,
    structural_bound,
    symmetry_logcosh_energy,
    symmetry_logcosh_gradient,
    symmetry_logcosh_hessian,
)


N_FACTORS = 40
DX = 0.05
ELL = 0.125
GAMMA = 0.05
TAU = 0.5
BETA = 0.7


def _latent_points() -> np.ndarray:
    radii = (np.arange(N_FACTORS) + 0.5) * DX
    return np.concatenate((-radii[::-1], radii))


def _archived_setup():
    precision, blocks, rho, kappa, matrix = ou_symmetry_comparison(
        N_FACTORS, DX, ELL
    )
    return precision, blocks, rho, kappa, matrix


def test_analytic_ou_precision_matches_dense_covariance_inverse() -> None:
    points = _latent_points()
    covariance = np.exp(-np.abs(points[:, None] - points[None, :]) / ELL)
    analytic = ou_ar1_precision(points.size, DX, ELL).toarray()
    numerical = np.linalg.solve(covariance, np.eye(points.size))

    np.testing.assert_allclose(analytic, numerical, rtol=2e-13, atol=2e-13)


def test_archived_comparison_constants_and_matrix_properties() -> None:
    _, _, rho, kappa, matrix = _archived_setup()
    dense = matrix.toarray()
    q = np.exp(-DX / ELL)

    np.testing.assert_allclose(q, 0.6703200460356393, rtol=2e-15)
    np.testing.assert_allclose(rho[0], 1.4146538810285458, rtol=2e-15)
    np.testing.assert_allclose(rho[1:-1], 2.631932441832188, rtol=2e-15)
    np.testing.assert_allclose(rho[-1], 1.815966220916094, rtol=2e-15)
    np.testing.assert_allclose(
        np.diag(kappa, 1), 1.2172785608036423, rtol=2e-15
    )
    assert np.count_nonzero(kappa) == 2 * (N_FACTORS - 1)

    assert sparse.isspmatrix_csr(matrix)
    np.testing.assert_allclose(dense, dense.T, atol=0.0)
    margins = row_dominance_margins(matrix)
    expected_margin = (1.0 - q) / (1.0 + q)
    np.testing.assert_allclose(margins.min(), expected_margin, rtol=2e-14)
    np.testing.assert_allclose(margins.min(), 0.197375320224904, rtol=2e-14)
    assert np.all(margins > 0.0)

    np.linalg.cholesky(dense)
    np.testing.assert_allclose(
        np.linalg.eigvalsh(dense).min(), 0.199035930388094, rtol=2e-13
    )


def test_comparison_solve_residual_and_inverse_nonnegativity() -> None:
    _, _, _, _, matrix = _archived_setup()
    dense = matrix.toarray()
    right_hand_side = np.linspace(0.1, 1.0, N_FACTORS)
    solution = solve_comparison(matrix, right_hand_side)

    np.testing.assert_allclose(dense @ solution, right_hand_side, rtol=2e-14)
    inverse_from_solve = solve_comparison(matrix, np.eye(N_FACTORS))
    assert inverse_from_solve.min() >= -2e-14


def test_logcosh_factor_gradient_hessian_and_exact_bound() -> None:
    value = np.array([-0.31, 0.44])
    step_gradient = 1e-6
    finite_gradient = np.empty(2)
    for coordinate in range(2):
        perturbation = np.zeros(2)
        perturbation[coordinate] = step_gradient
        finite_gradient[coordinate] = (
            symmetry_logcosh_energy(value + perturbation, GAMMA, TAU)
            - symmetry_logcosh_energy(value - perturbation, GAMMA, TAU)
        ) / (2.0 * step_gradient)

    analytic_gradient = symmetry_logcosh_gradient(value, GAMMA, TAU)
    np.testing.assert_allclose(analytic_gradient, finite_gradient, rtol=2e-9, atol=2e-11)

    step_hessian = 2e-5
    finite_hessian = np.empty((2, 2))
    for coordinate in range(2):
        perturbation = np.zeros(2)
        perturbation[coordinate] = step_hessian
        finite_hessian[:, coordinate] = (
            symmetry_logcosh_gradient(value + perturbation, GAMMA, TAU)
            - symmetry_logcosh_gradient(value - perturbation, GAMMA, TAU)
        ) / (2.0 * step_hessian)

    analytic_hessian = symmetry_logcosh_hessian(value, GAMMA, TAU)
    np.testing.assert_allclose(analytic_hessian, finite_hessian, rtol=2e-9, atol=2e-10)
    assert np.linalg.eigvalsh(analytic_hessian).min() >= -1e-15

    exact_bound = float(factor_sensitivity(GAMMA, TAU))
    np.testing.assert_allclose(exact_bound, np.sqrt(2.0) * GAMMA / TAU)
    sampled_differences = np.linspace(-20.0, 20.0, 1001)
    sampled_norms = [
        np.linalg.norm(
            symmetry_logcosh_gradient(np.array([0.0, difference]), GAMMA, TAU)
        )
        for difference in sampled_differences
    ]
    assert max(sampled_norms) <= exact_bound + 1e-15
    np.testing.assert_allclose(max(sampled_norms), exact_bound, rtol=2e-14)


def test_conditional_ei_gradient_matches_finite_difference() -> None:
    precision, _, _, _, _ = _archived_setup()
    points = _latent_points()
    coefficients, variance = ei_action_coefficients(-0.217, points, ELL, precision)
    rng = np.random.default_rng(20260820)
    latent_value = rng.normal(size=points.size)
    incumbent = 0.55
    mean = float(coefficients @ latent_value)
    analytic = conditional_ei_gradient(mean, variance, incumbent, coefficients)

    step = 2e-6
    finite = np.empty(points.size)
    for coordinate in range(points.size):
        shift = step * coefficients[coordinate]
        finite[coordinate] = (
            conditional_expected_improvement(mean + shift, variance, incumbent)
            - conditional_expected_improvement(mean - shift, variance, incumbent)
        ) / (2.0 * step)

    np.testing.assert_allclose(analytic, finite, rtol=2e-7, atol=2e-10)


def test_zero_variance_ei_uses_weak_derivative_convention() -> None:
    coefficients = np.array([0.4, -0.7])
    means = np.array([0.49, 0.50, 0.51])
    gradient = conditional_ei_gradient(means, 0.0, 0.50, coefficients)
    expected = np.array([0.0, 0.5, 1.0])[:, None] * coefficients
    np.testing.assert_allclose(gradient, expected)
    np.testing.assert_allclose(
        conditional_expected_improvement(means, 0.0, 0.50), [0.0, 0.0, 0.01]
    )


def test_smooth_positive_part_converges_at_zero_variance_kink() -> None:
    centered_means = np.array([-0.3, 0.0, 0.3])
    target_values = np.maximum(centered_means, 0.0)
    target_slopes = np.array([0.0, 0.5, 1.0])

    errors = []
    for smoothing_scale in (0.2, 0.1, 0.05, 0.01, 1e-12):
        smooth_values = smoothing_scale * np.logaddexp(
            0.0, centered_means / smoothing_scale
        )
        smooth_slopes = expit(centered_means / smoothing_scale)
        errors.append(float(np.max(np.abs(smooth_values - target_values))))
        assert np.all((smooth_slopes >= 0.0) & (smooth_slopes <= 1.0))

    assert all(later < earlier for earlier, later in zip(errors, errors[1:]))
    np.testing.assert_allclose(smooth_values, target_values, atol=1e-12)
    np.testing.assert_allclose(smooth_slopes, target_slopes, atol=1e-12)


def test_ei_footprint_dominates_sampled_gap_gradients() -> None:
    precision, blocks, _, _, _ = _archived_setup()
    points = _latent_points()
    action_coefficients, action_variance = ei_action_coefficients(
        -0.359, points, ELL, precision
    )
    leader_coefficients, leader_variance = ei_action_coefficients(
        -0.2082, points, ELL, precision
    )
    footprint = ei_block_decision_footprint(
        action_coefficients, leader_coefficients, blocks
    )

    rng = np.random.default_rng(17)
    for latent_value in rng.normal(size=(128, points.size)):
        action_mean = float(action_coefficients @ latent_value)
        leader_mean = float(leader_coefficients @ latent_value)
        gap_gradient = conditional_ei_gradient(
            action_mean, action_variance, 0.55, action_coefficients
        ) - conditional_ei_gradient(
            leader_mean, leader_variance, 0.55, leader_coefficients
        )
        actual_norms = np.linalg.norm(gap_gradient[blocks], axis=1)
        assert np.all(actual_norms <= footprint + 2e-14)


def test_archived_log_acquisition_structural_bound_progression() -> None:
    precision, blocks, _, _, matrix = _archived_setup()
    points = _latent_points()
    rounds = [
        ((), -0.2134, -0.2758, 0.9954513349887654, (5, 4, 3)),
        ((5, 4, 3), -0.2082, -0.3746, 0.6665729957378025, (7, 6, 8)),
        (
            (5, 4, 3, 7, 6, 8),
            -0.2069,
            -0.1250,
            0.4720071420523526,
            (2, 1, 0),
        ),
        (
            (5, 4, 3, 7, 6, 8, 2, 1, 0),
            -0.2082,
            -0.5761,
            0.44791757896613626,
            (11, 10, 12),
        ),
        (
            (5, 4, 3, 7, 6, 8, 2, 1, 0, 11, 10, 12),
            -0.2082,
            -0.3590,
            0.08286665027813787,
            (),
        ),
    ]

    for active, leader, challenger, expected_bound, expected_next in rounds:
        leader_coefficients, _ = ei_action_coefficients(
            leader, points, ELL, precision
        )
        challenger_coefficients, _ = ei_action_coefficients(
            challenger, points, ELL, precision
        )
        footprint = log_exponential_block_footprint(
            challenger_coefficients, leader_coefficients, blocks, BETA
        )
        active_mask = active_set_mask(N_FACTORS, active)
        load = omitted_factor_load(active_mask, GAMMA, TAU)
        bound = structural_bound(matrix, footprint, load)
        np.testing.assert_allclose(bound, expected_bound, rtol=2e-12, atol=2e-12)

        if expected_next:
            scores = ranked_omitted_contributions(
                matrix, footprint, active_mask, GAMMA, TAU
            )
            selected = tuple(np.argsort(scores)[-3:][::-1])
            assert selected == expected_next


def test_log_action_tilt_interpolation_identity() -> None:
    # Tensor Gauss-Hermite integration makes the check deterministic.  The
    # derivative in r is the covariance used to justify the archived notebook's
    # second (action-tilt) interpolation.
    nodes, one_dimensional_weights = np.polynomial.hermite.hermgauss(31)
    left, right = np.meshgrid(np.sqrt(2.0) * nodes, np.sqrt(2.0) * nodes)
    latent = np.column_stack((left.ravel(), right.ravel()))
    reference_weights = np.outer(
        one_dimensional_weights, one_dimensional_weights
    ).ravel() / np.pi

    factor = np.array(
        [symmetry_logcosh_energy(value, GAMMA, TAU) for value in latent]
    )
    interpolation_s = 0.4
    action_hat = np.array([0.75, -0.10])
    action_x = np.array([-0.20, 0.65])
    action_difference = BETA * (action_x - action_hat)

    def tilted_moments(r: float) -> tuple[float, float]:
        action = (1.0 - r) * action_hat + r * action_x
        log_tilt = -interpolation_s * factor + BETA * (latent @ action)
        weights = reference_weights * np.exp(log_tilt - log_tilt.max())
        weights /= weights.sum()
        linear_observable = latent @ action_difference
        factor_mean = float(weights @ factor)
        covariance = float(
            weights @ ((factor - factor_mean) * linear_observable)
        )
        return factor_mean, covariance

    evaluation_point = 0.37
    finite_difference_step = 2e-5
    plus_mean, _ = tilted_moments(evaluation_point + finite_difference_step)
    minus_mean, _ = tilted_moments(evaluation_point - finite_difference_step)
    finite_derivative = (plus_mean - minus_mean) / (
        2.0 * finite_difference_step
    )
    _, covariance_derivative = tilted_moments(evaluation_point)
    np.testing.assert_allclose(
        finite_derivative, covariance_derivative, rtol=2e-9, atol=2e-11
    )

    integration_nodes, integration_weights = np.polynomial.legendre.leggauss(32)
    r_values = 0.5 * (integration_nodes + 1.0)
    integrated_covariance = 0.5 * sum(
        weight * tilted_moments(float(r))[1]
        for r, weight in zip(r_values, integration_weights)
    )
    endpoint_difference = tilted_moments(1.0)[0] - tilted_moments(0.0)[0]
    np.testing.assert_allclose(
        endpoint_difference, integrated_covariance, rtol=2e-13, atol=2e-14
    )


def test_reflection_block_indexing() -> None:
    blocks = reflection_blocks(4)
    np.testing.assert_array_equal(blocks, [[3, 4], [2, 5], [1, 6], [0, 7]])


def test_committed_ei_validation_outputs_are_frozen_and_leakage_free() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    runner_path = repository_root / "experiments/symmetry/run_ei_validation.py"
    output_directory = (
        repository_root / "experiments/symmetry/outputs/t2b_ei_validation"
    )
    config_path = output_directory / "frozen_config.json"
    summary_path = output_directory / "summary.json"

    config = json.loads(config_path.read_text())
    summary = json.loads(summary_path.read_text())
    runner_namespace = runpy.run_path(str(runner_path))
    assert config == runner_namespace["FROZEN_CONFIG"]
    assert hashlib.sha256(config_path.read_bytes()).hexdigest() == summary[
        "config_sha256"
    ]

    assert summary["blocker_verdict"] == "PASS"
    assert summary["end_to_end_certificate"] is False
    assert summary["screening"][
        "no_omitted_factor_evaluated_during_screening"
    ]
    assert summary["screening"]["active_factors"] == 15
    assert summary["screening"]["omitted_factors_before_validation"] == 25
    assert summary["screening"]["optimistic_envelope"] < config["decision"][
        "epsilon_ei"
    ]
    assert summary["heldout_validation"]["within_epsilon_fraction"] == 1.0

    with (output_directory / "screening_history.csv").open(newline="") as handle:
        history = list(csv.DictReader(handle))
    assert history
    for row in history:
        assert int(row["factor_vectors_evaluated"]) == int(row["active_count"])
    assert history[-1]["stopped"] == "True"

    with (output_directory / "heldout_validation.csv").open(newline="") as handle:
        heldout = list(csv.DictReader(handle))
    assert len(heldout) == len(config["heldout_validation"]["seeds"])
    assert all(
        int(row["full_factor_vectors_evaluated_after_screening"]) == N_FACTORS
        for row in heldout
    )
