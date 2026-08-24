from __future__ import annotations

import numpy as np
import pytest

import conditioned_bo.nonlinear_pde_locality as locality
from conditioned_bo.nonlinear_pde_influence import (
    DEFAULT_PARAMETERS,
    build_nonlinear_pde_comparison,
    ranked_omitted_contributions,
)


def _tiny_problem_and_states():
    problem = locality.build_problem(
        5,
        2026082401,
        source_perturbation_scale=0.04,
    )
    states = locality.build_common_bo_states(
        problem,
        initialization_size=2,
        total_queries=4,
        checkpoint_queries=(2, 3, 4),
        observation_noise_variance=0.0025,
        reference_sample_count=64,
        trajectory_seed=731,
        incumbent=0.55,
    )
    return problem, states


def _run(method, state, problem, **overrides):
    arguments = {
        "epsilon": -1.0,
        "batch_size": 2,
        "maximum_refinement_stages": 2,
        "incumbent": 0.55,
        "delta_mc": 0.05,
        "minimum_reference_ess_fraction": 0.0,
        "laplace_sample_count": 32,
        "proposal_seed": 91,
        "proposal_inflation": 1.10,
        "gradient_tolerance": 1e-7,
        "maximum_laplace_iterations": 8,
    }
    arguments.update(overrides)
    return locality.run_selective_method(method, state, problem, **arguments)


def test_factor_support_distance_uses_actual_residual_support() -> None:
    # Center 12 on a 5x5 grid supports {12, 7, 17, 11, 13}; the decision at
    # 14 is distance one from support site 13 even though centers are distance 2.
    problem = locality.build_problem(5, 19, source_perturbation_scale=0.0)
    assert locality.factor_support_distance(5, problem.supports[12], (14,)) == 1
    assert locality.factor_support_distance(5, problem.supports[12], (13,)) == 0


def test_geometric_ranking_breaks_distance_ties_by_factor_index() -> None:
    problem = locality.build_problem(5, 19, source_perturbation_scale=0.0)
    omitted = np.ones(25, dtype=bool)
    first = locality.geometric_ranking(5, problem.supports, omitted, 12, 12)
    distances = [
        locality.factor_support_distance(5, problem.supports[int(j)], (12, 12))
        for j in first
    ]
    for distance in sorted(set(distances)):
        tied = first[np.asarray(distances) == distance]
        np.testing.assert_array_equal(tied, np.sort(tied))


def test_tiny_sparse_comparison_agrees_with_dense_solve() -> None:
    _, derivative_bounds, _, _, matrix = build_nonlinear_pde_comparison(5)
    omitted = np.ones(25, dtype=bool)
    scores = ranked_omitted_contributions(
        matrix,
        derivative_bounds,
        omitted,
        7,
        17,
        DEFAULT_PARAMETERS.gamma,
        DEFAULT_PARAMETERS.tau,
    )
    footprint = np.zeros(25)
    footprint[[7, 17]] = 1.0
    dense_solution = np.linalg.solve(matrix.toarray(), footprint)
    expected = np.asarray(
        (DEFAULT_PARAMETERS.gamma / DEFAULT_PARAMETERS.tau)
        * derivative_bounds
        @ dense_solution
    ).ravel()
    np.testing.assert_allclose(scores, expected, rtol=2e-14, atol=2e-14)


def test_m1_first_ranking_matches_existing_influence_helper() -> None:
    problem, states = _tiny_problem_and_states()
    state = states[0]
    result = _run(
        "ADAPTIVE_INFLUENCE",
        state,
        problem,
        maximum_refinement_stages=1,
    )
    _, derivative_bounds, _, _, base_matrix = build_nonlinear_pde_comparison(5)
    base_precision = build_nonlinear_pde_comparison(5)[0]
    matrix = base_matrix + np.diag(
        state.reference_precision.diagonal() - base_precision.diagonal()
    )
    omitted = np.ones(25, dtype=bool)
    first_stage = result.stages[0]
    scores = ranked_omitted_contributions(
        matrix,
        derivative_bounds,
        omitted,
        first_stage.challenger_index,
        first_stage.leader_index,
        DEFAULT_PARAMETERS.gamma,
        DEFAULT_PARAMETERS.tau,
    )
    indices = np.arange(25)
    expected = indices[np.lexsort((indices, -scores))][:2]
    np.testing.assert_array_equal(first_stage.activated_indices, expected)


def test_m2_never_consults_influence_scores(monkeypatch: pytest.MonkeyPatch) -> None:
    problem, states = _tiny_problem_and_states()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("M2 factor selection consulted influence scores")

    monkeypatch.setattr(locality, "ranked_omitted_contributions", forbidden)
    result = _run("DYNAMIC_GEOMETRIC_SHELL", states[0], problem)
    assert result.audit["m2_selection_consulted_influence_scores"] is False
    assert result.audit["influence_selection_calls"] == 0
    assert result.audit["geometric_selection_calls"] > 0


def test_factor_mask_resets_between_states_and_activation_is_cumulative() -> None:
    problem, states = _tiny_problem_and_states()
    first = _run("ADAPTIVE_INFLUENCE", states[0], problem)
    second = _run("ADAPTIVE_INFLUENCE", states[1], problem)
    assert first.audit["initial_active_count"] == 0
    assert second.audit["initial_active_count"] == 0
    assert first.audit["state_fingerprint"] != second.audit["state_fingerprint"]
    counts = [stage.active_count for stage in first.stages]
    assert counts == sorted(counts)


def test_full_fallback_and_factor_work_accounting() -> None:
    problem, states = _tiny_problem_and_states()
    result = _run("ADAPTIVE_INFLUENCE", states[0], problem)
    assert result.full_fallback
    assert len(result.final_active_indices) == 25
    assert result.stages[-1].stopped
    assert result.stages[-1].active_count == 25
    assert result.work["factor_energy_evaluations"] > 0
    assert result.work["sparse_comparison_solves"] > 0


def test_paired_methods_receive_byte_identical_bo_state() -> None:
    problem, states = _tiny_problem_and_states()
    first = _run("ADAPTIVE_INFLUENCE", states[0], problem, epsilon=1e9)
    second = _run("DYNAMIC_GEOMETRIC_SHELL", states[0], problem, epsilon=1e9)
    assert first.audit["state_fingerprint"] == second.audit["state_fingerprint"]


def test_prospective_seed_derivation_is_deterministic_and_literal() -> None:
    expected = [4215109622, 1083605379, 4045758625]
    assert [locality.derive_prospective_seed(index) for index in range(3)] == expected
    assert len(set(expected)) == 3


def test_oracle_diagnostic_is_one_way_and_never_feeds_deployable_method() -> None:
    problem, states = _tiny_problem_and_states()
    state = states[0]
    full = _run("FULL", state, problem)
    full_inference = locality.evaluate_fixed_subset(
        state,
        problem,
        np.arange(25),
        incumbent=0.55,
        delta_mc=0.05,
    )
    order = np.argsort(-full_inference.acquisition, kind="stable")
    diagnostic = locality.oracle_geometric_prefix(
        state,
        problem,
        full_action_index=full.action_index,
        full_challenger_index=int(state.action_indices[order[1]]),
        full_acquisition=full_inference.acquisition,
        batch_size=2,
        regret_threshold=0.01,
        incumbent=0.55,
        delta_mc=0.05,
    )
    assert diagnostic["deployable"] is False
    assert diagnostic["uses_full_information"] is True
    assert diagnostic["feeds_deployable_methods"] is False
    deployable = _run("ADAPTIVE_INFLUENCE", state, problem, epsilon=1e9)
    assert deployable.audit["full_shadow_information_used"] is False


def test_no_dense_inverse_is_used(monkeypatch: pytest.MonkeyPatch) -> None:
    problem, states = _tiny_problem_and_states()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("a dense inverse was requested")

    monkeypatch.setattr(np.linalg, "inv", forbidden)
    result = _run("ADAPTIVE_INFLUENCE", states[0], problem, epsilon=1e9)
    assert np.isfinite(result.stages[0].structural_bound)
