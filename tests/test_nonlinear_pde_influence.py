from __future__ import annotations

import hashlib
import json
import runpy
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

import conditioned_bo.nonlinear_pde_influence as pde
from conditioned_bo.nonlinear_pde_influence import (
    DEFAULT_PARAMETERS,
    analytic_interior_row_margin,
    build_nonlinear_pde_comparison,
    diagonal_dominance_threshold,
    ei_decision_footprint,
    extreme_eigenvalues,
    factor_gradient,
    factor_hessian,
    factor_hessian_terms,
    factor_supports,
    grid_neighbors,
    maximum_nonzeros_per_row,
    omitted_factor_load,
    residual_gradient,
    residual_hessian,
    residual_value,
    row_dominance_margins,
    solve_comparison,
    structural_screening_bound,
)


PARAMETERS = DEFAULT_PARAMETERS
GRID_SIZE = 24
LOCKED_STRUCTURAL_VALUE = 0.03874403301354687


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


def _comparison():
    return build_nonlinear_pde_comparison(GRID_SIZE, PARAMETERS)


def test_residual_gradient_matches_finite_difference() -> None:
    local = np.array([0.31, -0.22, 0.17, 0.44, -0.58])
    source = 0.19
    analytic = residual_gradient(
        local, PARAMETERS.coupling, PARAMETERS.nonlinearity
    )
    finite = _finite_gradient(
        lambda value: residual_value(
            value, source, PARAMETERS.coupling, PARAMETERS.nonlinearity
        ),
        local,
    )
    np.testing.assert_allclose(analytic, finite, rtol=2e-10, atol=2e-10)


def test_residual_hessian_matches_finite_difference() -> None:
    local = np.array([0.73, -0.22, 0.17, 0.44, -0.58])
    analytic = residual_hessian(local, PARAMETERS.nonlinearity)
    finite = _finite_jacobian(
        lambda value: residual_gradient(
            value, PARAMETERS.coupling, PARAMETERS.nonlinearity
        ),
        local,
    )
    np.testing.assert_allclose(analytic, finite, rtol=2e-10, atol=2e-11)


def test_factor_hessian_retains_both_chain_rule_terms() -> None:
    local = np.array([0.81, -0.35, 0.12, 0.47, -0.29])
    source = -0.14
    positive_term, nonlinear_term = factor_hessian_terms(
        local, source, PARAMETERS
    )
    analytic = factor_hessian(local, source, PARAMETERS)
    finite = _finite_jacobian(
        lambda value: factor_gradient(value, source, PARAMETERS), local
    )

    assert np.linalg.norm(positive_term) > 0.0
    assert np.linalg.norm(nonlinear_term) > 0.0
    np.testing.assert_allclose(analytic, positive_term + nonlinear_term)
    np.testing.assert_allclose(analytic, finite, rtol=3e-9, atol=2e-10)
    assert not np.allclose(finite, positive_term, rtol=1e-5, atol=1e-7)
    assert not np.allclose(finite, nonlinear_term, rtol=1e-5, atol=1e-7)


def test_worst_case_nonlinear_negative_curvature_bound() -> None:
    delta = PARAMETERS.negative_curvature
    center_projector = np.zeros((5, 5))
    center_projector[0, 0] = 1.0
    rng = np.random.default_rng(20260821)
    for local in rng.normal(size=(128, 5)):
        source = float(rng.normal())
        hessian = factor_hessian(local, source, PARAMETERS)
        assert np.linalg.eigvalsh(hessian + delta * center_projector).min() >= -2e-15

    local = np.array([np.pi / 2.0, 0.0, 0.0, 0.0, 0.0])
    source = local[0] + PARAMETERS.nonlinearity - 20.0 * PARAMETERS.tau
    _, nonlinear_term = factor_hessian_terms(local, source, PARAMETERS)
    np.testing.assert_allclose(nonlinear_term[0, 0], -delta, rtol=2e-15)


def test_logcosh_is_stable_for_large_residuals() -> None:
    values = np.array([-1.0e4, 0.0, 1.0e4])
    result = pde.stable_logcosh(values)
    assert np.all(np.isfinite(result))
    np.testing.assert_allclose(
        result,
        [1.0e4 - np.log(2.0), 0.0, 1.0e4 - np.log(2.0)],
        rtol=0.0,
        atol=1e-12,
    )


def test_corner_edge_and_interior_support_counts() -> None:
    _, derivative_bounds, _, _, _ = _comparison()
    factor_counts = np.diff(derivative_bounds.indptr)
    overlap_counts = np.asarray((derivative_bounds != 0).sum(axis=0)).ravel()
    corner = 0
    edge = GRID_SIZE // 2
    interior = (GRID_SIZE // 2) * GRID_SIZE + GRID_SIZE // 2
    assert factor_counts[corner] == 3
    assert factor_counts[edge] == 4
    assert factor_counts[interior] == 5
    assert overlap_counts[corner] == 3
    assert overlap_counts[edge] == 4
    assert overlap_counts[interior] == 5


def test_exact_three_nonzero_pair_coupling_classes() -> None:
    _, _, _, kappa, _ = _comparison()
    center = 12 * GRID_SIZE + 12
    nearest = 12 * GRID_SIZE + 13
    diagonal = 13 * GRID_SIZE + 13
    axial_two = 12 * GRID_SIZE + 14
    uncoupled = 12 * GRID_SIZE + 15
    np.testing.assert_allclose(kappa[center, nearest], 0.8666666666666667)
    np.testing.assert_allclose(kappa[center, diagonal], 0.0256)
    np.testing.assert_allclose(kappa[center, axial_two], 0.0128)
    np.testing.assert_allclose(kappa[center, uncoupled], 0.0)


def test_exact_corner_edge_and_interior_rho_values() -> None:
    _, _, rho, _, _ = _comparison()
    corner = 0
    edge = GRID_SIZE // 2
    interior = (GRID_SIZE // 2) * GRID_SIZE + GRID_SIZE // 2
    np.testing.assert_allclose(rho[corner], 4.633333333333334)
    np.testing.assert_allclose(rho[edge], 5.233333333333333)
    np.testing.assert_allclose(rho[interior], 5.833333333333333)


def test_n576_matrix_diagnostics() -> None:
    _, _, rho, kappa, matrix = _comparison()
    assert sparse.isspmatrix_csr(matrix)
    np.testing.assert_allclose(matrix.toarray(), matrix.toarray().T, atol=0.0)
    margins = row_dominance_margins(matrix)
    minimum_eigenvalue, maximum_eigenvalue = extreme_eigenvalues(matrix)

    # The handoff records these regression targets to ten decimal places, so
    # half a unit in that final recorded place is the tight comparison scale.
    rounded_tolerance = 5e-11
    np.testing.assert_allclose(rho.min(), 4.6333333333, rtol=0.0, atol=rounded_tolerance)
    np.testing.assert_allclose(
        kappa.data.max(), 0.8666666667, rtol=0.0, atol=rounded_tolerance
    )
    np.testing.assert_allclose(
        margins.min(), 2.2130666667, rtol=0.0, atol=rounded_tolerance
    )
    np.testing.assert_allclose(
        minimum_eigenvalue, 2.2366796511, rtol=0.0, atol=rounded_tolerance
    )
    np.testing.assert_allclose(
        maximum_eigenvalue, 9.1204570358, rtol=0.0, atol=rounded_tolerance
    )
    np.testing.assert_allclose(
        maximum_eigenvalue / minimum_eigenvalue,
        4.0776769402,
        rtol=0.0,
        atol=rounded_tolerance,
    )
    assert maximum_nonzeros_per_row(matrix) == 13


def test_clean_matrix_matches_literal_archived_notebook_construction() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    runner = runpy.run_path(
        str(repository_root / "experiments/nonlinear_pde/run_structural_validation.py")
    )
    archived_precision, archived_matrix = runner[
        "literal_archived_notebook_construction"
    ]()
    precision, _, _, _, matrix = _comparison()
    np.testing.assert_allclose(precision.toarray(), archived_precision, atol=0.0)
    np.testing.assert_allclose(matrix.toarray(), archived_matrix, rtol=0.0, atol=2e-15)


def test_menz_bounds_are_uniform_over_random_interpolation_coefficients() -> None:
    grid_size = 5
    precision, _, rho, kappa, _ = build_nonlinear_pde_comparison(
        grid_size, PARAMETERS
    )
    supports = factor_supports(grid_size)
    rng = np.random.default_rng(71)
    for _ in range(4):
        latent = rng.normal(size=grid_size * grid_size)
        source = rng.normal(size=grid_size * grid_size)
        interpolation = rng.uniform(0.0, 1.0, size=grid_size * grid_size)
        hessian = precision.toarray()
        for factor_index, support in enumerate(supports):
            hessian[np.ix_(support, support)] += interpolation[
                factor_index
            ] * factor_hessian(latent[support], source[factor_index], PARAMETERS)

        assert np.all(np.diag(hessian) >= rho - 2e-14)
        off_diagonal = hessian.copy()
        np.fill_diagonal(off_diagonal, 0.0)
        assert np.all(np.abs(off_diagonal) <= kappa.toarray() + 2e-14)
        globally_convex_remainder = (
            hessian
            - precision.toarray()
            + PARAMETERS.negative_curvature * np.eye(grid_size * grid_size)
        )
        assert np.linalg.eigvalsh(globally_convex_remainder).min() >= -2e-14


def test_omitted_load_matches_closed_form() -> None:
    _, derivative_bounds, _, _, _ = _comparison()
    omitted = np.zeros(GRID_SIZE * GRID_SIZE, dtype=bool)
    omitted[[0, 1, 24, 300, 301, 325]] = True
    actual = omitted_factor_load(
        derivative_bounds, omitted, PARAMETERS.gamma, PARAMETERS.tau
    )
    expected = np.zeros_like(actual)
    for site in range(expected.size):
        centered_load = PARAMETERS.center_derivative_bound * omitted[site]
        neighbor_load = PARAMETERS.coupling * sum(
            omitted[neighbor] for neighbor in grid_neighbors(GRID_SIZE, site)
        )
        expected[site] = PARAMETERS.gradient_scale * (
            centered_load + neighbor_load
        )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-16)


def test_ei_footprint_is_supported_on_the_two_action_coordinates() -> None:
    action = 9 * GRID_SIZE + 12
    leader = 14 * GRID_SIZE + 12
    footprint = ei_decision_footprint(GRID_SIZE**2, action, leader)
    assert footprint.sum() == 2.0
    assert footprint[action] == 1.0
    assert footprint[leader] == 1.0
    assert np.count_nonzero(footprint) == 2
    np.testing.assert_array_equal(
        ei_decision_footprint(GRID_SIZE**2, leader, leader),
        np.zeros(GRID_SIZE**2),
    )


def test_structural_screen_does_not_evaluate_omitted_factor_energies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, derivative_bounds, _, _, matrix = _comparison()
    omitted = np.ones(GRID_SIZE * GRID_SIZE, dtype=bool)

    def forbidden_energy(*_args, **_kwargs):
        raise AssertionError("factor energy must not be evaluated during screening")

    monkeypatch.setattr(pde, "factor_energy", forbidden_energy)
    bound = structural_screening_bound(
        matrix,
        derivative_bounds,
        omitted,
        9 * GRID_SIZE + 12,
        14 * GRID_SIZE + 12,
        PARAMETERS.gamma,
        PARAMETERS.tau,
    )
    assert np.isfinite(bound) and bound > 0.0


def test_sparse_solve_and_locked_structural_replay() -> None:
    _, derivative_bounds, _, _, matrix = _comparison()
    active = np.asarray(
        [
            179, 202, 203, 204, 205, 225, 226, 227, 228, 229,
            250, 251, 252, 253, 275, 298, 299, 300, 301, 322,
            323, 324, 325, 326, 345, 346, 347, 348, 349, 350,
            369, 370, 371, 372, 373, 374, 394, 395, 396, 397,
        ],
        dtype=int,
    )
    omitted = np.ones(GRID_SIZE * GRID_SIZE, dtype=bool)
    omitted[active] = False
    load = omitted_factor_load(
        derivative_bounds, omitted, PARAMETERS.gamma, PARAMETERS.tau
    )
    solution = solve_comparison(matrix, load)
    np.testing.assert_allclose(matrix @ solution, load, rtol=2e-14, atol=2e-14)
    bound = structural_screening_bound(
        matrix,
        derivative_bounds,
        omitted,
        9 * GRID_SIZE + 12,
        14 * GRID_SIZE + 12,
        PARAMETERS.gamma,
        PARAMETERS.tau,
    )
    np.testing.assert_allclose(bound, LOCKED_STRUCTURAL_VALUE, rtol=2e-15)


def test_expanding_domain_row_margin_is_domain_independent() -> None:
    expected_margin = analytic_interior_row_margin(PARAMETERS)
    np.testing.assert_allclose(
        diagonal_dominance_threshold(PARAMETERS), 1.2869333333333333
    )
    for grid_size in (6, 12, 24, 40):
        *_, matrix = build_nonlinear_pde_comparison(grid_size, PARAMETERS)
        np.testing.assert_allclose(
            row_dominance_margins(matrix).min(),
            expected_margin,
            rtol=2e-15,
            atol=2e-15,
        )


def test_committed_structural_validation_outputs_are_frozen() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    runner_path = (
        repository_root / "experiments/nonlinear_pde/run_structural_validation.py"
    )
    output_directory = (
        repository_root
        / "experiments/nonlinear_pde/outputs/t2b_structural_validation"
    )
    config_path = output_directory / "frozen_config.json"
    summary_path = output_directory / "summary.json"
    runner = runpy.run_path(str(runner_path))
    config = json.loads(config_path.read_text())
    summary = json.loads(summary_path.read_text())

    assert config == runner["FROZEN_CONFIG"]
    assert hashlib.sha256(config_path.read_bytes()).hexdigest() == summary[
        "config_sha256"
    ]
    prototype = (
        repository_root
        / "notebooks/prototypes/DEC_Nonlinear_PDE_BO_Demo.ipynb"
    )
    assert hashlib.sha256(prototype.read_bytes()).hexdigest() == config[
        "provenance"
    ]["prototype_sha256"]
    assert summary["verdict"] == "PASS"
    assert summary["t4_status"] == "PROVED_FOR_THIS_FAMILY"
    assert summary["end_to_end_finite_sample_certificate"] is False
    assert summary["inference_error_certification"] == "OPEN_BLOCKER"
    assert summary["checks"]["no_factor_energies_evaluated"]
    assert summary["notebook_comparison"][
        "clean_equals_archived_to_tolerance"
    ]
    np.testing.assert_allclose(
        summary["structural_replay"]["clean_structural_value"],
        LOCKED_STRUCTURAL_VALUE,
        rtol=2e-15,
    )
