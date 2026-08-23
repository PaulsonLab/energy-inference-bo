from __future__ import annotations

from decimal import Decimal
import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy import sparse

from conditioned_bo.pbe_factor_theory import (
    LegacyNode,
    build_comparison_matrix,
    build_factor_bank,
    factor_adjacency,
    factor_endpoint_array,
    factor_graph_diagnostics,
    factor_influence_contributions,
    factorize_sparse,
    logistic_order_energy,
    logistic_order_gradient,
    logistic_order_hessian,
    observation_pattern_diagnostics,
    selected_inverse_rows,
    validate_comparison_matrix,
)


ROOT = Path(__file__).resolve().parents[1]
DRIVER_PATH = ROOT / "experiments/sun_oxide/pbe_factor_theory.py"
CONFIG_PATH = ROOT / "experiments/sun_oxide/configs/pbe_factor_theory.json"
CORE_PATH = ROOT / "src/conditioned_bo/pbe_factor_theory.py"


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


def test_deterministic_factor_construction_exact_ties_and_stable_ordering() -> None:
    nodes = (
        _node(0, "key-d", "2.0"),
        _node(1, "key-b", "1.0"),
        _node(2, "key-a", "1.00"),
        _node(3, "key-c", "1.5"),
        _node(4, "key-e", "3.0"),
    )
    ordered, factors, ties = build_factor_bank(nodes)
    reverse = build_factor_bank(tuple(reversed(nodes)))

    assert [node.composition_key for node in ordered] == [
        "key-a",
        "key-b",
        "key-c",
        "key-d",
        "key-e",
    ]
    assert (ordered, factors, ties) == reverse
    assert [(factor.composition_key_a, factor.composition_key_b) for factor in factors] == [
        ("key-b", "key-c"),
        ("key-c", "key-d"),
        ("key-d", "key-e"),
    ]
    assert all(Decimal(factor.pbe_gap_difference_ev) > 0 for factor in factors)
    assert len(ties) == 1
    assert ties[0]["size"] == 2
    assert ties[0]["composition_keys"] == ["key-a", "key-b"]


def test_logistic_gradient_hessian_psd_and_mixed_curvature_finite_differences() -> None:
    for local in (
        np.asarray([-1.2, 0.7]),
        np.asarray([0.0, 0.0]),
        np.asarray([1.8, -0.4]),
    ):
        gradient = logistic_order_gradient(local)
        hessian = logistic_order_hessian(local)
        np.testing.assert_allclose(
            gradient,
            _finite_gradient(logistic_order_energy, local),
            rtol=2e-9,
            atol=2e-10,
        )
        np.testing.assert_allclose(
            hessian,
            _finite_jacobian(logistic_order_gradient, local),
            rtol=3e-9,
            atol=3e-10,
        )
        assert np.max(np.abs(gradient)) <= 1.0
        assert np.linalg.eigvalsh(hessian).min() >= -1e-15
        assert abs(hessian[0, 1]) <= 0.25
    np.testing.assert_allclose(abs(logistic_order_hessian([0.0, 0.0])[0, 1]), 0.25)
    assert np.isfinite(logistic_order_energy([1.0e300, -1.0e300]))
    assert logistic_order_energy([-1.0e300, 1.0e300]) == 0.0


def test_factor_graph_is_a_path_subgraph_with_maximum_degree_two() -> None:
    nodes = tuple(_node(index, f"key-{index}", str(index // 2)) for index in range(8))
    _, factors, _ = build_factor_bank(nodes)
    adjacency = factor_adjacency(len(nodes), factor_endpoint_array(factors))
    diagnostics = factor_graph_diagnostics(adjacency)
    assert diagnostics["path_subgraph"]
    assert diagnostics["maximum_degree"] <= 2
    assert diagnostics["spectral_norm_numerical"] <= 2.0 + 1e-12


def test_comparison_identity_and_spd_lower_bound_fixture() -> None:
    endpoint_pairs = np.asarray([[0, 1], [1, 2], [3, 4]], dtype=np.int64)
    graph = factor_adjacency(5, endpoint_pairs)
    q0 = sparse.eye(5, format="csr")
    a0 = build_comparison_matrix(q0, graph)
    expected = q0 - 0.25 * graph
    difference = (a0 - expected).tocsr()
    assert difference.nnz == 0
    diagnostics = validate_comparison_matrix(
        q0,
        graph,
        a0,
        q0_analytic_eigenvalue_floor=1.0,
        factor_graph_norm_upper_bound=2.0,
        symmetry_tolerance=1e-12,
        sign_tolerance=1e-14,
        eigenvalue_tolerance=1e-8,
    )
    assert diagnostics["analytic_smallest_eigenvalue_lower_bound"] == 0.5
    assert diagnostics["smallest_eigenvalue_numerical"] >= 0.5 - 1e-8
    assert diagnostics["positive_definite"]


def test_observation_diagonal_monotonicity_and_resolvent_identity() -> None:
    graph = factor_adjacency(5, np.asarray([[0, 1], [1, 2], [2, 3], [3, 4]]))
    a0 = build_comparison_matrix(sparse.eye(5, format="csr"), graph)
    factorization, _ = factorize_sparse(a0)
    minimum = float(np.linalg.eigvalsh(a0.toarray()).min())
    diagnostics = observation_pattern_diagnostics(
        a0,
        factorization,
        [
            {
                "name": "small_fixture",
                "node_indices": [1, 3],
                "diagonal_precision": 0.75,
                "a0_smallest_eigenvalue": minimum,
            }
        ],
        [0, 2, 4],
        eigenvalue_tolerance=1e-10,
        identity_tolerance=1e-12,
        nonnegative_tolerance=1e-12,
    )[0]
    assert diagnostics["positive_definite"]
    assert diagnostics["same_factor_hessian_bound"]
    assert diagnostics["sampled_inverse_columns_entrywise_nonnegative"]
    assert diagnostics["sampled_inverse_monotonicity"]
    assert diagnostics["resolvent_identity_max_abs_error"] <= 1e-12


def test_factor_influence_formula_uses_only_selected_inverse_rows() -> None:
    matrix = sparse.csr_matrix(
        np.asarray(
            [
                [2.0, -0.2, 0.0, 0.0],
                [-0.2, 2.0, -0.1, 0.0],
                [0.0, -0.1, 2.0, -0.3],
                [0.0, 0.0, -0.3, 2.0],
            ]
        )
    )
    factorization, _ = factorize_sparse(matrix)
    selected, solve = selected_inverse_rows(matrix, factorization, [0, 3])
    endpoints = np.asarray([[0, 1], [2, 3]], dtype=np.int64)
    actual = factor_influence_contributions(selected, endpoints, 0, 1)
    inverse = np.linalg.inv(matrix.toarray())
    expected = np.asarray(
        [
            inverse[0, a] + inverse[0, b] + inverse[3, a] + inverse[3, b]
            for a, b in endpoints
        ]
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-15, atol=2e-15)
    assert solve["relative_frobenius_residual"] <= 1e-14


def test_factor_interface_is_target_blind_and_has_no_oracle_dependency() -> None:
    forbidden_file = "gw_oracle.csv"
    forbidden_column = "gw_band_gap_ev"
    for path in (CORE_PATH, DRIVER_PATH, CONFIG_PATH):
        source = path.read_text(encoding="utf-8")
        assert forbidden_file not in source
        assert forbidden_column not in source

    specification = importlib.util.spec_from_file_location("pbe_factor_driver_test", DRIVER_PATH)
    assert specification and specification.loader
    driver = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(driver)
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    inputs = driver._verified_inputs(ROOT, config)
    assert set(inputs) == {
        "action_node_mapping",
        "legacy_pbe",
        "nlr_data_use_notice",
        "q0",
    }
    assert forbidden_file not in {path.name for path in inputs.values()}
