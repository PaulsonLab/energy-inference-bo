from __future__ import annotations

from decimal import Decimal
import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy import sparse

from conditioned_bo.normalized_pbe_model import (
    WEIGHT,
    build_strict_pair_bank,
    chunked_influence_diagnostics,
    farthest_point_support,
    preference_objective_and_gradient,
    weighted_adjacency,
    weighted_graph_diagnostics,
    weighted_logistic_energy,
    weighted_logistic_gradient,
    weighted_logistic_hessian,
)
from conditioned_bo.pbe_factor_theory import (
    LegacyNode,
    build_comparison_matrix,
    factorize_sparse,
    observation_pattern_diagnostics,
    selected_inverse_rows,
    validate_comparison_matrix,
)


ROOT = Path(__file__).resolve().parents[1]
DRIVER_PATH = ROOT / "experiments/sun_oxide/normalized_pbe_model.py"
CONFIG_PATH = ROOT / "experiments/sun_oxide/configs/normalized_pbe_model.json"
CORE_PATH = ROOT / "src/conditioned_bo/normalized_pbe_model.py"


def _node(index: int, key: str, gap: str) -> LegacyNode:
    return LegacyNode(
        node_index=index,
        composition_key=key,
        normalized_formula=f"X{index}",
        pbe_band_gap_text=gap,
        pbe_band_gap=Decimal(gap),
    )


def _finite_gradient(function, value: np.ndarray, step: float = 1e-6) -> np.ndarray:
    result = np.empty_like(value)
    for coordinate in range(value.size):
        delta = np.zeros_like(value)
        delta[coordinate] = step
        result[coordinate] = (function(value + delta) - function(value - delta)) / (2 * step)
    return result


def _finite_jacobian(function, value: np.ndarray, step: float = 2e-5) -> np.ndarray:
    result = np.empty((value.size, value.size), dtype=np.float64)
    for coordinate in range(value.size):
        delta = np.zeros_like(value)
        delta[coordinate] = step
        result[:, coordinate] = (function(value + delta) - function(value - delta)) / (2 * step)
    return result


def test_farthest_point_support_is_deterministic_includes_initial_and_breaks_ties() -> None:
    points = np.asarray([[0.0], [1.0], [2.0], [3.0], [4.0]])
    keys = ["start", "key-z", "key-a", "far", "other-start"]
    initial = np.asarray([0, 4], dtype=np.int64)
    first = farthest_point_support(points, keys, initial, support_count=4)
    second = farthest_point_support(points, keys, initial[::-1], support_count=4)

    np.testing.assert_array_equal(first.selected_indices, [0, 2, 3, 4])
    np.testing.assert_array_equal(first.additional_indices, [2, 3])
    np.testing.assert_array_equal(first.selected_indices, second.selected_indices)
    np.testing.assert_array_equal(first.additional_indices, second.additional_indices)
    assert set(initial).issubset(first.selected_indices)
    assert first.exact_tie_step_count == 1


def test_complete_strict_pair_bank_omits_exact_ties_and_has_stable_order() -> None:
    nodes = (
        _node(0, "key-d", "2.0"),
        _node(1, "key-b", "1.00"),
        _node(2, "key-a", "1.0"),
        _node(3, "key-c", "3.0"),
    )
    first = build_strict_pair_bank([3, 0, 2, 1], nodes)
    second = build_strict_pair_bank([1, 2, 0, 3], nodes)
    np.testing.assert_array_equal(first.support_indices, [0, 1, 2, 3])
    np.testing.assert_array_equal(first.node_endpoint_pairs, second.node_endpoint_pairs)
    np.testing.assert_array_equal(first.signs, second.signs)
    assert first.strict_factor_count == 5
    assert first.omitted_exact_tie_pair_count == 1
    assert first.exact_tie_group_count == 1
    assert first.exact_tie_group_size_histogram == {"2": 1}
    assert not any(set(pair) == {1, 2} for pair in first.node_endpoint_pairs.tolist())


def test_global_weight_and_weighted_row_sum_bound() -> None:
    assert WEIGHT == 1.0 / 499.0
    pairs = np.asarray([[0, 1], [0, 2], [1, 2]], dtype=np.int64)
    graph = weighted_adjacency(4, pairs, WEIGHT)
    diagnostics = weighted_graph_diagnostics(graph)
    np.testing.assert_allclose(graph.data, WEIGHT, rtol=0.0, atol=0.0)
    assert diagnostics["maximum_weighted_row_sum"] == 2 * WEIGHT
    assert diagnostics["maximum_weighted_row_sum"] <= 1.0
    assert diagnostics["spectral_norm_numerical"] <= 1.0


def test_weighted_logistic_derivatives_hessian_psd_and_mixed_bound() -> None:
    for sign in (-1, 1):
        for local in (np.asarray([-1.2, 0.7]), np.asarray([0.0, 0.0]), np.asarray([2.1, -0.4])):
            gradient = weighted_logistic_gradient(local, (0, 1), sign)
            hessian = weighted_logistic_hessian(local, (0, 1), sign)
            np.testing.assert_allclose(
                gradient,
                _finite_gradient(lambda value: weighted_logistic_energy(value, (0, 1), sign), local),
                rtol=2e-8,
                atol=2e-11,
            )
            np.testing.assert_allclose(
                hessian,
                _finite_jacobian(lambda value: weighted_logistic_gradient(value, (0, 1), sign), local),
                rtol=3e-9,
                atol=3e-12,
            )
            assert np.max(np.abs(gradient)) <= WEIGHT
            assert np.linalg.eigvalsh(hessian).min() >= -1e-15
            assert abs(hessian[0, 1]) <= WEIGHT / 4.0
    np.testing.assert_allclose(
        abs(weighted_logistic_hessian([0.0, 0.0], (0, 1), 1)[0, 1]),
        WEIGHT / 4.0,
        rtol=0.0,
        atol=0.0,
    )


def test_composite_objective_gradient_finite_difference() -> None:
    q0 = sparse.csr_matrix(np.asarray([[1.2, -0.1, 0.0], [-0.1, 1.3, -0.2], [0.0, -0.2, 1.4]]))
    pairs = np.asarray([[0, 1], [0, 2], [1, 2]], dtype=np.int64)
    signs = np.asarray([-1, 1, -1], dtype=np.int8)
    latent = np.asarray([0.3, -0.7, 1.1])
    objective, gradient = preference_objective_and_gradient(
        latent, q0, pairs, signs, WEIGHT, chunk_size=2
    )
    finite = _finite_gradient(
        lambda value: preference_objective_and_gradient(
            value, q0, pairs, signs, WEIGHT, chunk_size=2
        )[0],
        latent,
    )
    assert np.isfinite(objective)
    np.testing.assert_allclose(gradient, finite, rtol=2e-9, atol=2e-10)


def test_comparison_identity_spd_floor_and_sparse_solve_fixture() -> None:
    pairs = np.asarray([[0, 1], [0, 2], [1, 2]], dtype=np.int64)
    graph = weighted_adjacency(4, pairs, WEIGHT)
    q0 = sparse.eye(4, format="csr")
    a0 = build_comparison_matrix(q0, graph)
    difference = (a0 - (q0 - 0.25 * graph)).tocsr()
    assert difference.nnz == 0
    diagnostics = validate_comparison_matrix(
        q0,
        graph,
        a0,
        q0_analytic_eigenvalue_floor=1.0,
        factor_graph_norm_upper_bound=1.0,
        symmetry_tolerance=1e-12,
        sign_tolerance=1e-14,
        eigenvalue_tolerance=1e-8,
    )
    assert diagnostics["analytic_smallest_eigenvalue_lower_bound"] == 0.75
    assert diagnostics["smallest_eigenvalue_numerical"] >= 0.75 - 1e-8
    factorization, _ = factorize_sparse(a0)
    selected, solve = selected_inverse_rows(a0, factorization, [0, 3])
    assert selected.shape == (2, 4)
    assert solve["relative_frobenius_residual"] <= 1e-10


def test_observation_monotonicity_uses_normalized_factor_bound() -> None:
    graph = weighted_adjacency(4, np.asarray([[0, 1], [1, 2], [2, 3]]), WEIGHT)
    a0 = build_comparison_matrix(sparse.eye(4, format="csr"), graph)
    factorization, _ = factorize_sparse(a0)
    minimum = float(np.linalg.eigvalsh(a0.toarray()).min())
    result = observation_pattern_diagnostics(
        a0,
        factorization,
        [{"name": "fixture", "node_indices": [1, 3], "diagonal_precision": 0.5, "a0_smallest_eigenvalue": minimum}],
        [0, 2],
        eigenvalue_tolerance=1e-10,
        identity_tolerance=1e-12,
        nonnegative_tolerance=1e-12,
        factor_mixed_curvature_bound=WEIGHT / 4.0,
    )[0]
    assert result["factor_mixed_curvature_bound"] == WEIGHT / 4.0
    assert result["sampled_inverse_monotonicity"]
    assert result["resolvent_identity_max_abs_error"] <= 1e-12


def test_normalized_factor_influence_formula() -> None:
    matrix = sparse.csr_matrix(
        np.asarray([[2.0, -0.2, 0.0], [-0.2, 2.0, -0.1], [0.0, -0.1, 2.0]])
    )
    inverse = np.linalg.inv(matrix.toarray())
    endpoints = np.asarray([[0, 1], [1, 2]], dtype=np.int64)
    pairs = [
        {"action_position_x": 0, "action_position_xhat": 1, "distance_quartile": quartile}
        for quartile in range(1, 5)
    ]
    rows, _ = chunked_influence_diagnostics(
        inverse[[0, 2]], endpoints, pairs, [0.5], weight=WEIGHT, chunk_size=1, nonnegative_tolerance=1e-12
    )
    expected = WEIGHT * np.asarray(
        [inverse[0, a] + inverse[0, b] + inverse[2, a] + inverse[2, b] for a, b in endpoints]
    )
    np.testing.assert_allclose(rows[0]["total_structural_influence"], expected.sum())


def test_normalized_model_interface_has_no_target_oracle_dependency() -> None:
    forbidden_file = "gw_oracle.csv"
    forbidden_column = "gw_band_gap_ev"
    for path in (CORE_PATH, DRIVER_PATH, CONFIG_PATH):
        source = path.read_text(encoding="utf-8")
        assert forbidden_file not in source
        assert forbidden_column not in source

    specification = importlib.util.spec_from_file_location("normalized_pbe_driver_test", DRIVER_PATH)
    assert specification and specification.loader
    driver = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(driver)
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    inputs = driver._verified_inputs(ROOT, config)
    assert set(inputs) == {
        "action_node_mapping",
        "adjacent_factor_bank",
        "descriptor_matrix",
        "legacy_pbe",
        "nlr_data_use_notice",
        "q0",
    }
    assert forbidden_file not in {path.name for path in inputs.values()}
