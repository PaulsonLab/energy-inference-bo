from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

import conditioned_bo.adaptive_pbe as adaptive_module
from conditioned_bo.adaptive_pbe import (
    ActiveFactorState,
    AdaptiveSettings,
    PBEContext,
    adaptive_pbe_decision,
    apply_menz_observation_updates,
    construct_menz_support_reference,
    exact_activation_batch,
    full_pbe_context,
    omitted_factor_contributions,
    omitted_structural_load,
    sherman_morrison_observation_update,
    stable_descending_factor_ranking,
    structural_influence_vector,
)
from conditioned_bo.bo_value import (
    WEIGHT,
    construct_gaussian_reference,
    exact_support_marginal,
    precompute_exact_support_reference,
    update_exact_support_reference,
)


def _q_fixture() -> np.ndarray:
    return np.asarray(
        [
            [2.5, -0.4, -0.2, 0.0],
            [-0.4, 2.2, -0.3, -0.1],
            [-0.2, -0.3, 2.0, -0.4],
            [0.0, -0.1, -0.4, 1.8],
        ],
        dtype=np.float64,
    )


def test_j_h0_exact_construction_and_old_marginal_equivalence() -> None:
    q0 = sparse.csc_matrix(_q_fixture())
    support = np.asarray([0, 2, 3], dtype=np.int64)
    reference = precompute_exact_support_reference(q0, support)
    expected_covariance = np.linalg.inv(q0.toarray())[np.ix_(support, support)]
    np.testing.assert_allclose(reference.covariance, expected_covariance, atol=1e-13)
    np.testing.assert_allclose(
        reference.precision, np.linalg.inv(expected_covariance), atol=1e-13
    )

    observed_support = np.asarray([0, 2], dtype=np.int64)
    observed_nodes = support[observed_support]
    observed_z = np.asarray([0.35, -0.8])
    optimized = update_exact_support_reference(
        reference, observed_support, observed_z, sigma_obs=0.05
    )
    old_full_state = construct_gaussian_reference(
        q0, observed_nodes, observed_z, sigma_obs=0.05
    )
    old_marginal = exact_support_marginal(old_full_state, support)
    np.testing.assert_allclose(optimized.mean, old_marginal.mean, atol=2e-13)
    np.testing.assert_allclose(
        optimized.covariance, old_marginal.covariance, atol=2e-13
    )
    np.testing.assert_allclose(
        optimized.precision, old_marginal.precision, atol=2e-12
    )
    assert not optimized.diagnostics["repeated_500_rhs_marginalization"]


def test_diagonal_online_support_update_is_exact() -> None:
    q0 = sparse.csc_matrix(_q_fixture())
    support = np.asarray([0, 2, 3], dtype=np.int64)
    reference = precompute_exact_support_reference(q0, support)
    observed = np.asarray([0, 2], dtype=np.int64)
    values = np.asarray([0.3, -0.5])
    state = update_exact_support_reference(reference, observed, values)
    expected_precision = reference.precision.copy()
    expected_precision[observed, observed] += 400.0
    expected_information = np.zeros(3)
    expected_information[observed] = 400.0 * values
    np.testing.assert_allclose(state.precision, expected_precision)
    np.testing.assert_allclose(
        state.mean, np.linalg.solve(expected_precision, expected_information)
    )


def test_c_h0_extraction_matches_direct_dense_inverse() -> None:
    q0 = sparse.csc_matrix(_q_fixture())
    support = np.asarray([0, 2, 3], dtype=np.int64)
    pairs = np.asarray([[0, 1], [0, 2], [1, 2]], dtype=np.int64)
    weight = 0.08
    observed = construct_menz_support_reference(
        q0, support, pairs, weight=weight
    )
    full_pairs = support[pairs]
    adjacency = np.zeros(q0.shape, dtype=np.float64)
    for left, right in full_pairs:
        adjacency[left, right] += weight
        adjacency[right, left] += weight
    a0 = q0.toarray() - 0.25 * adjacency
    expected = np.linalg.inv(a0)[np.ix_(support, support)]
    np.testing.assert_allclose(observed.covariance, expected, atol=2e-13)
    assert not observed.diagnostics["dense_full_inverse_formed"]


def test_sherman_morrison_influence_update_matches_direct_full_solve() -> None:
    q0 = sparse.csc_matrix(_q_fixture())
    support = np.asarray([0, 2, 3], dtype=np.int64)
    pairs = np.asarray([[0, 1], [0, 2], [1, 2]], dtype=np.int64)
    weight = 0.08
    base = construct_menz_support_reference(q0, support, pairs, weight=weight)
    updated = apply_menz_observation_updates(base.covariance, [0, 2], delta=400.0)

    full_pairs = support[pairs]
    row = np.concatenate((full_pairs[:, 0], full_pairs[:, 1]))
    col = np.concatenate((full_pairs[:, 1], full_pairs[:, 0]))
    adjacency = sparse.coo_matrix(
        (np.full(row.size, weight), (row, col)), shape=q0.shape
    ).tocsc()
    a_t = q0 - 0.25 * adjacency
    diagonal = np.zeros(q0.shape[0])
    diagonal[support[[0, 2]]] = 400.0
    a_t = sparse.csc_matrix(a_t + sparse.diags(diagonal))
    rhs = np.zeros((q0.shape[0], support.size))
    rhs[support, np.arange(support.size)] = 1.0
    direct = splu(a_t).solve(rhs)[support]
    np.testing.assert_allclose(updated, direct, atol=2e-13)
    once = sherman_morrison_observation_update(base.covariance, 0)
    np.testing.assert_allclose(
        once,
        np.linalg.inv(np.linalg.inv(base.covariance) + np.diag([400.0, 0.0, 0.0])),
        atol=2e-13,
    )


def test_omitted_degree_load_maintenance_and_cumulative_active_reuse() -> None:
    pairs = np.asarray([[0, 1], [0, 2], [1, 2], [2, 3]], dtype=np.int64)
    state = ActiveFactorState.empty(pairs, 4)
    np.testing.assert_array_equal(state.omitted_endpoint_degree, [2, 2, 3, 1])
    state.activate([1, 3], pairs)
    np.testing.assert_array_equal(state.omitted_endpoint_degree, [1, 2, 1, 0])
    np.testing.assert_array_equal(state.active_indices(), [1, 3])
    state.activate([0], pairs)
    np.testing.assert_array_equal(state.active_indices(), [0, 1, 3])
    np.testing.assert_array_equal(state.omitted_endpoint_degree, [0, 1, 1, 0])
    np.testing.assert_allclose(
        omitted_structural_load(state.omitted_endpoint_degree),
        WEIGHT * np.asarray([0, 1, 1, 0]),
    )


def test_structural_bound_equals_sum_of_omitted_factor_contributions() -> None:
    covariance = np.asarray(
        [
            [1.2, 0.3, 0.2],
            [0.3, 1.1, 0.4],
            [0.2, 0.4, 1.0],
        ]
    )
    pairs = np.asarray([[0, 1], [0, 2], [1, 2]], dtype=np.int64)
    state = ActiveFactorState.empty(pairs, 3)
    state.activate([1], pairs)
    omitted = state.omitted_indices()
    influence = structural_influence_vector(
        covariance, state.omitted_endpoint_degree
    )
    leader, challenger = 0, 2
    bound = influence[challenger] + influence[leader]
    contributions = omitted_factor_contributions(
        covariance, pairs, omitted, challenger, leader
    )
    np.testing.assert_allclose(np.sum(contributions), bound, atol=1e-15)


def test_stable_worst_pair_factor_ranking_uses_bank_order_for_ties() -> None:
    omitted = np.asarray([1, 4, 7, 9], dtype=np.int64)
    contributions = np.asarray([0.2, 0.5, 0.5, 0.1])
    np.testing.assert_array_equal(
        stable_descending_factor_ranking(omitted, contributions), [4, 7, 1, 9]
    )


def test_exact_batch_selection_uses_smallest_top_prefix() -> None:
    omitted = np.arange(5, dtype=np.int64)
    contributions = np.asarray([0.04, 0.03, 0.02, 0.01, 0.005])
    # gap=-0.01, target=.03, remaining must be <=.024.  Removing .04+.03+.02
    # leaves .015, while removing only the first two leaves .035.
    batch = exact_activation_batch(
        omitted, contributions, active_gap=-0.01, epsilon_struct=0.02, rho=0.8
    )
    np.testing.assert_array_equal(batch, [0, 1, 2])


def _fake_context(
    selected: int,
    ei: np.ndarray,
    map_value: np.ndarray,
    warm_source: str,
    active_count: int,
) -> PBEContext:
    return PBEContext(
        selected=selected,
        ei=ei,
        map=map_value,
        laplace=None,
        diagnostics={
            "active_factor_count": active_count,
            "warm_start_source": warm_source,
            "factor_energy_gradient_calls": 0,
            "factor_energy_gradient_element_work": 0,
            "factor_hessian_calls": 0,
            "factor_hessian_element_work": 0,
            "stage_conditioning_seconds": 0.0,
        },
    )


def test_structural_stopping_and_map_warm_start_across_activation_stages(
    monkeypatch,
) -> None:
    calls: list[tuple[np.ndarray | None, str, int]] = []

    def fake_stage(*args, **kwargs):
        active_indices = args[3]
        initial_map = args[8]
        warm_source = kwargs["warm_start_source"]
        calls.append(
            (
                None if initial_map is None else np.asarray(initial_map).copy(),
                warm_source,
                int(active_indices.size),
            )
        )
        map_value = np.asarray([3.0, 4.0]) if len(calls) == 1 else np.asarray([5.0, 6.0])
        return _fake_context(0, np.asarray([1.0, 0.99]), map_value, warm_source, int(active_indices.size))

    monkeypatch.setattr(adaptive_module, "_laplace_ei_context", fake_stage)
    reference = SimpleNamespace(
        mean=np.zeros(2), covariance=np.eye(2), precision=np.eye(2)
    )
    pairs = np.asarray([[0, 1]], dtype=np.int64)
    state = ActiveFactorState.empty(pairs, 2)
    result = adaptive_pbe_decision(
        reference,
        10.0 * np.eye(2),
        pairs,
        [1],
        [0, 1],
        [],
        ["a", "b"],
        0.0,
        state,
        None,
        AdaptiveSettings(),
    )
    assert result.structurally_certified and not result.full_bank_fallback
    assert result.adaptive_stages == 1
    assert calls[0][0] is None
    np.testing.assert_array_equal(calls[1][0], [3.0, 4.0])
    assert calls[1][1] == "preceding_adaptive_stage_MAP"
    assert result.diagnostics["warm_start_across_stages_used"]


def test_map_warm_start_across_bo_iterations(monkeypatch) -> None:
    calls: list[np.ndarray | None] = []

    def fake_stage(*args, **kwargs):
        initial_map = args[8]
        calls.append(None if initial_map is None else np.asarray(initial_map).copy())
        return _fake_context(
            0,
            np.asarray([1.0, 0.0]),
            np.asarray([0.1, 0.2]),
            kwargs["warm_start_source"],
            int(args[3].size),
        )

    monkeypatch.setattr(adaptive_module, "_laplace_ei_context", fake_stage)
    reference = SimpleNamespace(
        mean=np.zeros(2), covariance=np.eye(2), precision=np.eye(2)
    )
    pairs = np.asarray([[0, 1]], dtype=np.int64)
    state = ActiveFactorState.empty(pairs, 2)
    state.activate([0], pairs)
    previous = np.asarray([7.0, 8.0])
    result = adaptive_pbe_decision(
        reference,
        np.eye(2),
        pairs,
        [1],
        [0, 1],
        [],
        ["a", "b"],
        0.0,
        state,
        previous,
        AdaptiveSettings(),
    )
    np.testing.assert_array_equal(calls[0], previous)
    assert result.diagnostics["warm_start_across_bo_used"]


def test_explicit_full_bank_fallback_after_eight_stages(monkeypatch) -> None:
    def fake_stage(*args, **kwargs):
        return _fake_context(
            0,
            np.asarray([0.0, 0.0]),
            np.asarray([0.1, 0.2]),
            kwargs["warm_start_source"],
            int(args[3].size),
        )

    def one_factor_batch(omitted, contributions, **kwargs):
        del contributions, kwargs
        return np.asarray([int(omitted[0])], dtype=np.int64)

    monkeypatch.setattr(adaptive_module, "_laplace_ei_context", fake_stage)
    monkeypatch.setattr(adaptive_module, "exact_activation_batch", one_factor_batch)
    reference = SimpleNamespace(
        mean=np.zeros(2), covariance=np.eye(2), precision=np.eye(2)
    )
    pairs = np.tile(np.asarray([[0, 1]], dtype=np.int64), (9, 1))
    state = ActiveFactorState.empty(pairs, 2)
    result = adaptive_pbe_decision(
        reference,
        100.0 * np.eye(2),
        pairs,
        np.ones(9, dtype=np.int8),
        [0, 1],
        [],
        ["a", "b"],
        0.0,
        state,
        None,
        AdaptiveSettings(),
    )
    assert result.full_bank_fallback and not result.structurally_certified
    assert result.adaptive_stages == 8
    assert result.active_count == 9 and result.active_fraction == 1.0
    assert result.diagnostics["stop_reason"] == "explicit_full_bank_fallback"


def test_full_timing_and_factor_element_work_are_internally_consistent() -> None:
    precision = np.asarray([[1.8, -0.2], [-0.2, 1.5]])
    covariance = np.linalg.inv(precision)
    reference = SimpleNamespace(
        mean=np.asarray([0.1, -0.2]),
        covariance=covariance,
        precision=precision,
    )
    pairs = np.asarray([[0, 1], [0, 1], [0, 1]], dtype=np.int64)
    context = full_pbe_context(
        reference,
        pairs,
        [1, -1, 1],
        [0, 1],
        [],
        ["a", "b"],
        0.0,
        None,
        AdaptiveSettings(chunk_size=2),
    )
    diagnostics = context.diagnostics
    assert diagnostics["pbe_conditioning_seconds"] > 0.0
    assert diagnostics["factor_energy_gradient_element_work"] == (
        3 * diagnostics["factor_energy_gradient_calls"]
    )
    assert diagnostics["factor_hessian_element_work"] == (
        3 * diagnostics["factor_hessian_calls"]
    )
    assert diagnostics["factor_hessian_calls"] == 1
