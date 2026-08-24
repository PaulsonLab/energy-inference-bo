"""Development-only decision-sparsity diagnostics for frozen Sun-oxide E3.

This module has no oracle interface.  It evaluates cumulative subsets of the
already-frozen 500-support normalized PBE factor bank against a caller-supplied
FULL Laplace acquisition vector.
"""

from __future__ import annotations

import hashlib
import time
from typing import Any, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import spearmanr

from conditioned_bo.bo_value import (
    NumericalFailure,
    gaussian_expected_improvement,
    select_unobserved_action,
)
from conditioned_bo.full_bank_scaling import (
    CompactActiveState,
    CompactPairBank,
    FixedReferenceState,
    ImplicitMenzSystem,
    ResourceGuard,
    TimedCompactFactorBank,
    fit_compact_laplace,
    peak_rss_bytes,
)


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]


def target_factor_counts(fractions: Sequence[float], factor_count: int) -> IntArray:
    """Apply the frozen ``round(fraction * N)`` rule with deterministic clipping."""

    values = np.asarray(fractions, dtype=np.float64)
    if factor_count < 1 or values.ndim != 1 or values.size == 0:
        raise ValueError("Fractions and factor count must be nonempty and positive")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("Fractions must be finite and lie in [0, 1]")
    if np.any(np.diff(values) <= 0.0):
        raise ValueError("Fractions must be strictly increasing")
    counts = np.asarray(
        [min(factor_count, max(0, round(float(value) * factor_count))) for value in values],
        dtype=np.int64,
    )
    if np.any(np.diff(counts) <= 0):
        raise ValueError("Rounded checkpoint counts must be strictly increasing")
    return counts


def deterministic_random_subset(
    factor_count: int,
    subset_count: int,
    *,
    base_seed: int,
    state_index: int,
    fraction_index: int,
    replicate: int,
) -> IntArray:
    """Draw an order-independent, exactly matched random subset."""

    if not 0 <= subset_count <= factor_count:
        raise ValueError("Random subset count lies outside the factor bank")
    if min(base_seed, state_index, fraction_index, replicate) < 0:
        raise ValueError("Random-seed coordinates must be nonnegative")
    sequence = np.random.SeedSequence(
        [int(base_seed), int(state_index), int(fraction_index), int(replicate)]
    )
    rng = np.random.default_rng(sequence)
    return np.sort(
        rng.choice(factor_count, size=subset_count, replace=False).astype(np.int64)
    )


def factor_set_sha256(factor_indices: ArrayLike) -> str:
    indices = np.asarray(factor_indices, dtype="<i8")
    return hashlib.sha256(indices.tobytes()).hexdigest()


def _available_mask(action_count: int, observed_positions: ArrayLike) -> BoolArray:
    result = np.ones(action_count, dtype=bool)
    result[np.asarray(observed_positions, dtype=np.int64)] = False
    if np.count_nonzero(result) < 2:
        raise ValueError("At least two unobserved actions are required")
    return result


def _stable_maximum(values: FloatArray, mask: BoolArray, keys: Sequence[str]) -> int:
    maximum = float(np.max(values[mask]))
    tied = np.flatnonzero(mask & (values == maximum))
    return min((int(index) for index in tied), key=lambda index: keys[index])


def fit_active_acquisition(
    reference: FixedReferenceState,
    bank: CompactPairBank,
    active_indices: ArrayLike,
    action_support_positions: ArrayLike,
    observed_action_positions: ArrayLike,
    action_keys: Sequence[str],
    incumbent: float,
    initial_map: ArrayLike | None,
    *,
    chunk_size: int,
    laplace_settings: dict[str, Any],
    guard: ResourceGuard,
    phase_name: str,
) -> tuple[FloatArray, int, FloatArray, dict[str, Any]]:
    """Fit one exact active Laplace target, or the exact empty Gaussian target."""

    active = np.asarray(active_indices, dtype=np.int64)
    actions = np.asarray(action_support_positions, dtype=np.int64)
    started = time.perf_counter()
    if active.size == 0:
        acquisition = gaussian_expected_improvement(
            reference.mean[actions], reference.action_variances, incumbent
        )
        leader = select_unobserved_action(
            acquisition, observed_action_positions, action_keys
        )
        return (
            np.asarray(acquisition, dtype=np.float64),
            int(leader),
            reference.mean.copy(),
            {
                "inference_kind": "exact_gaussian_reference",
                "optimizer_iterations": 0,
                "optimizer_function_evaluations": 0,
                "gradient_infinity_norm": 0.0,
                "factor_energy_gradient_calls": 0,
                "factor_energy_gradient_element_work": 0,
                "factor_energy_gradient_seconds": 0.0,
                "factor_hessian_calls": 0,
                "factor_hessian_element_work": 0,
                "factor_hessian_seconds": 0.0,
                "hessian_construction_seconds": 0.0,
                "dense_cholesky_seconds": 0.0,
                "selected_variance_solve_seconds": 0.0,
                "laplace_solve_relative_residual": 0.0,
                "single_active_fit_conditioning_seconds": time.perf_counter()
                - started,
                "peak_rss_bytes": peak_rss_bytes(),
            },
        )

    factor_bank = TimedCompactFactorBank(bank, active, chunk_size=chunk_size)
    first = (
        reference.mean
        if initial_map is None
        else np.asarray(initial_map, dtype=np.float64)
    )
    laplace = fit_compact_laplace(
        reference,
        factor_bank,
        first,
        actions,
        retry_map=np.zeros(bank.support_count, dtype=np.float64),
        guard=guard,
        phase_name=phase_name,
        **laplace_settings,
    )
    acquisition = gaussian_expected_improvement(
        laplace.map[actions], laplace.action_variances, incumbent
    )
    leader = select_unobserved_action(
        acquisition, observed_action_positions, action_keys
    )
    return (
        np.asarray(acquisition, dtype=np.float64),
        int(leader),
        laplace.map.copy(),
        {
            "inference_kind": "laplace",
            **laplace.diagnostics,
            "single_active_fit_conditioning_seconds": laplace.diagnostics[
                "total_conditioning_seconds"
            ],
        },
    )


def evaluate_influence_checkpoint(
    reference: FixedReferenceState,
    menz: ImplicitMenzSystem,
    bank: CompactPairBank,
    active_state: CompactActiveState,
    action_support_positions: ArrayLike,
    observed_action_positions: ArrayLike,
    action_keys: Sequence[str],
    incumbent: float,
    full_ei: ArrayLike,
    full_leader: int,
    initial_map: ArrayLike | None,
    *,
    epsilon_struct: float,
    chunk_size: int,
    laplace_settings: dict[str, Any],
    guard: ResourceGuard,
    phase_name: str,
) -> tuple[dict[str, Any], FloatArray]:
    """Fit and compare one influence-ranked active set to shadow FULL."""

    actions = np.asarray(action_support_positions, dtype=np.int64)
    observed = np.asarray(observed_action_positions, dtype=np.int64)
    shadow = np.asarray(full_ei, dtype=np.float64)
    available = _available_mask(len(action_keys), observed)
    acquisition, leader, map_value, diagnostics = fit_active_acquisition(
        reference,
        bank,
        active_state.active_indices(),
        actions,
        observed,
        action_keys,
        incumbent,
        initial_map,
        chunk_size=chunk_size,
        laplace_settings=laplace_settings,
        guard=guard,
        phase_name=phase_name,
    )

    load = bank.weight * active_state.omitted_endpoint_degree.astype(np.float64)
    rhs = np.zeros(menz.q0.shape[0], dtype=np.float64)
    rhs[bank.support_nodes] = load
    influence, load_solve = menz.solve(rhs)
    leader_support = int(actions[leader])
    psi = np.full(len(action_keys), -np.inf, dtype=np.float64)
    psi[available] = (
        acquisition[available]
        - acquisition[leader]
        + influence[bank.support_nodes[actions[available]]]
        + influence[int(bank.support_nodes[leader_support])]
    )
    challenger_mask = available.copy()
    challenger_mask[leader] = False
    challenger = _stable_maximum(psi, challenger_mask, action_keys)
    structural_bound = float(
        influence[int(bank.support_nodes[actions[challenger]])]
        + influence[int(bank.support_nodes[leader_support])]
    )
    max_psi = float(psi[challenger])

    regret = float(shadow[int(full_leader)] - shadow[leader])
    if regret < -1e-12:
        raise NumericalFailure("Shadow FULL leader has negative acquisition regret")
    regret = max(regret, 0.0)
    active_available = acquisition[available]
    full_available = shadow[available]
    correlation = float(spearmanr(active_available, full_available).statistic)
    if not np.isfinite(correlation):
        raise NumericalFailure("Active/FULL acquisition rank correlation is non-finite")
    absolute_difference = np.abs(active_available - full_available)
    active_indices = active_state.active_indices()
    record = {
        "active_factor_count": active_state.active_count,
        "active_factor_fraction": active_state.active_count / bank.factor_count,
        "active_set_sha256": factor_set_sha256(active_indices),
        "active_laplace_leader": int(leader),
        "full_shadow_leader": int(full_leader),
        "action_agreement": bool(leader == int(full_leader)),
        "full_laplace_ei_regret": regret,
        "active_full_ei_spearman": correlation,
        "maximum_absolute_ei_difference": float(np.max(absolute_difference)),
        "median_absolute_ei_difference": float(np.median(absolute_difference)),
        "worst_challenger": int(challenger),
        "active_ei_gap": float(acquisition[challenger] - acquisition[leader]),
        "b_struct": structural_bound,
        "max_psi": max_psi,
        "epsilon_struct": float(epsilon_struct),
        "theorem_certificate_passed": bool(max_psi <= epsilon_struct),
        "map_iterations": int(diagnostics["optimizer_iterations"]),
        "map_function_evaluations": int(
            diagnostics["optimizer_function_evaluations"]
        ),
        "map_gradient_infinity_norm": float(
            diagnostics["gradient_infinity_norm"]
        ),
        "factor_energy_gradient_calls": int(
            diagnostics["factor_energy_gradient_calls"]
        ),
        "factor_energy_gradient_work": int(
            diagnostics["factor_energy_gradient_element_work"]
        ),
        "factor_hessian_calls": int(diagnostics["factor_hessian_calls"]),
        "factor_hessian_work": int(
            diagnostics["factor_hessian_element_work"]
        ),
        "single_active_fit_conditioning_seconds": float(
            diagnostics["single_active_fit_conditioning_seconds"]
        ),
        "factor_energy_gradient_seconds": float(
            diagnostics["factor_energy_gradient_seconds"]
        ),
        "hessian_construction_seconds": float(
            diagnostics["hessian_construction_seconds"]
        ),
        "dense_cholesky_seconds": float(diagnostics["dense_cholesky_seconds"]),
        "selected_variance_solve_seconds": float(
            diagnostics["selected_variance_solve_seconds"]
        ),
        "laplace_solve_relative_residual": float(
            diagnostics["laplace_solve_relative_residual"]
        ),
        "load_solve_seconds": float(load_solve["seconds"]),
        "load_solve_iterations": int(load_solve["iterations"]),
        "load_solve_relative_residual": float(load_solve["relative_residual"]),
        "peak_rss_bytes": int(peak_rss_bytes()),
    }
    return record, map_value


def rank_omitted_factors(
    menz: ImplicitMenzSystem,
    bank: CompactPairBank,
    active_state: CompactActiveState,
    action_support_positions: ArrayLike,
    leader: int,
    challenger: int,
) -> tuple[IntArray, dict[str, Any]]:
    """Recompute the frozen stable influence ranking for one leader/challenger."""

    omitted = active_state.omitted_indices()
    if omitted.size == 0:
        return omitted, {
            "pair_solve_seconds": 0.0,
            "pair_solve_iterations": 0,
            "pair_solve_relative_residual": 0.0,
            "contribution_sum_error": 0.0,
        }
    actions = np.asarray(action_support_positions, dtype=np.int64)
    rhs = np.zeros(menz.q0.shape[0], dtype=np.float64)
    rhs[int(bank.support_nodes[actions[int(challenger)]])] += 1.0
    rhs[int(bank.support_nodes[actions[int(leader)]])] += 1.0
    influence_row, solve = menz.solve(rhs)
    pairs = bank.endpoint_pairs[omitted]
    contributions = bank.weight * (
        influence_row[bank.support_nodes[pairs[:, 0]]]
        + influence_row[bank.support_nodes[pairs[:, 1]]]
    )
    if float(np.min(contributions)) < -1e-10:
        raise NumericalFailure("A per-factor structural contribution is negative")
    order = np.argsort(-np.maximum(contributions, 0.0), kind="stable")
    return np.asarray(omitted[order], dtype=np.int64), {
        "pair_solve_seconds": float(solve["seconds"]),
        "pair_solve_iterations": int(solve["iterations"]),
        "pair_solve_relative_residual": float(solve["relative_residual"]),
        "contribution_sum": float(np.sum(contributions)),
    }


def run_influence_path(
    path_name: str,
    fractions: Sequence[float],
    reference: FixedReferenceState,
    menz: ImplicitMenzSystem,
    bank: CompactPairBank,
    action_support_positions: ArrayLike,
    observed_action_positions: ArrayLike,
    action_keys: Sequence[str],
    incumbent: float,
    full_ei: ArrayLike,
    full_leader: int,
    *,
    epsilon_struct: float,
    chunk_size: int,
    laplace_settings: dict[str, Any],
    guard: ResourceGuard,
    phase_name: str,
) -> list[dict[str, Any]]:
    """Evaluate either the one-shot static prefix or reranked fine path."""

    if path_name not in {"STATIC_INFLUENCE_PREFIX", "RERANKED_FINE_PATH"}:
        raise ValueError("Unknown influence path")
    counts = target_factor_counts(fractions, bank.factor_count)
    state = CompactActiveState.empty(bank)
    warm_map: FloatArray | None = None
    fixed_ranking: IntArray | None = None
    records: list[dict[str, Any]] = []

    for checkpoint_index, (fraction, count) in enumerate(zip(fractions, counts, strict=True)):
        if state.active_count != int(count):
            raise NumericalFailure("Influence path reached the wrong checkpoint count")
        record, warm_map = evaluate_influence_checkpoint(
            reference,
            menz,
            bank,
            state,
            action_support_positions,
            observed_action_positions,
            action_keys,
            incumbent,
            full_ei,
            full_leader,
            warm_map,
            epsilon_struct=epsilon_struct,
            chunk_size=chunk_size,
            laplace_settings=laplace_settings,
            guard=guard,
            phase_name=phase_name,
        )
        record.update(
            {
                "path": path_name,
                "checkpoint_index": checkpoint_index,
                "requested_fraction": float(fraction),
                "target_factor_count": int(count),
                "factors_added_to_next_checkpoint": 0,
                "ranking_solve_seconds": 0.0,
                "ranking_solve_iterations": 0,
                "ranking_solve_relative_residual": 0.0,
                "contribution_sum_error": 0.0,
            }
        )
        records.append(record)
        if checkpoint_index == len(counts) - 1:
            continue

        if path_name == "STATIC_INFLUENCE_PREFIX" and fixed_ranking is not None:
            ranking = fixed_ranking
            ranking_diagnostics: dict[str, Any] | None = None
        else:
            ranking, ranking_diagnostics = rank_omitted_factors(
                menz,
                bank,
                state,
                action_support_positions,
                int(record["active_laplace_leader"]),
                int(record["worst_challenger"]),
            )
            contribution_error = abs(
                float(ranking_diagnostics["contribution_sum"])
                - float(record["b_struct"])
            )
            if contribution_error > 1e-8 * max(1.0, abs(float(record["b_struct"]))):
                raise NumericalFailure(
                    "Ranked contributions do not sum to the structural bound"
                )
            record.update(
                {
                    "ranking_solve_seconds": ranking_diagnostics[
                        "pair_solve_seconds"
                    ],
                    "ranking_solve_iterations": ranking_diagnostics[
                        "pair_solve_iterations"
                    ],
                    "ranking_solve_relative_residual": ranking_diagnostics[
                        "pair_solve_relative_residual"
                    ],
                    "contribution_sum_error": contribution_error,
                }
            )
            if path_name == "STATIC_INFLUENCE_PREFIX":
                fixed_ranking = ranking.copy()

        next_count = int(counts[checkpoint_index + 1])
        addition_count = next_count - state.active_count
        if path_name == "STATIC_INFLUENCE_PREFIX":
            if fixed_ranking is None:
                raise NumericalFailure("Static ranking was not initialized")
            additions = fixed_ranking[state.active_count:next_count]
        else:
            additions = ranking[:addition_count]
        if additions.size != addition_count:
            raise NumericalFailure("Influence ranking did not supply the exact batch")
        state.activate(additions, bank)
        record["factors_added_to_next_checkpoint"] = int(additions.size)

    if records[-1]["active_factor_count"] != bank.factor_count:
        raise NumericalFailure("Influence path did not end at the full factor bank")
    return records


def stabilization_quantities(
    records: Sequence[dict[str, Any]], *, small_regret_threshold: float = 0.01
) -> dict[str, float]:
    """Compute the frozen first/stable agreement, regret, and certificate fractions."""

    if not records:
        raise ValueError("At least one path checkpoint is required")
    fractions = [float(record["requested_fraction"]) for record in records]
    if fractions != sorted(fractions) or fractions[-1] != 1.0:
        raise ValueError("Path checkpoints must be ordered and end at 1.0")
    agreement = [bool(record["action_agreement"]) for record in records]
    regrets = [float(record["full_laplace_ei_regret"]) for record in records]
    certificates = [bool(record["theorem_certificate_passed"]) for record in records]

    first_agreement = next(
        (fraction for fraction, flag in zip(fractions, agreement, strict=True) if flag),
        1.0,
    )
    first_stable = next(
        (
            fractions[index]
            for index in range(len(fractions))
            if all(agreement[index:])
        ),
        1.0,
    )
    first_small_regret = next(
        (
            fractions[index]
            for index in range(len(fractions))
            if all(value <= small_regret_threshold for value in regrets[index:])
        ),
        1.0,
    )
    certificate = next(
        (
            fraction
            for fraction, flag in zip(fractions, certificates, strict=True)
            if flag
        ),
        1.0,
    )
    return {
        "first_agreement_fraction": float(first_agreement),
        "first_stable_agreement_fraction": float(first_stable),
        "first_small_regret_fraction": float(first_small_regret),
        "certificate_fraction": float(certificate),
        "stable_to_certificate_gap": float(certificate - first_stable),
    }


def classify_development(
    reranked_state_summaries: Sequence[dict[str, Any]],
) -> str:
    """Apply the prospectively frozen terminal classification literally."""

    if len(reranked_state_summaries) != 3:
        raise ValueError("Classification requires exactly three reranked states")
    stable = np.asarray(
        [item["first_stable_agreement_fraction"] for item in reranked_state_summaries],
        dtype=np.float64,
    )
    gaps = np.asarray(
        [item["stable_to_certificate_gap"] for item in reranked_state_summaries],
        dtype=np.float64,
    )
    regret_suffix_pass = all(
        bool(item["stable_suffix_regret_at_most_0_01"])
        for item in reranked_state_summaries
    )
    if np.all(stable <= 0.40) and regret_suffix_pass and np.all(gaps >= 0.30):
        return "STRONG_BOUND_CONSERVATISM"
    if float(np.median(stable)) <= 0.50 and float(np.max(stable)) <= 0.70:
        return "MIXED_DECISION_SPARSITY"
    return "DECISION_DENSE"


__all__ = [
    "classify_development",
    "deterministic_random_subset",
    "evaluate_influence_checkpoint",
    "factor_set_sha256",
    "fit_active_acquisition",
    "rank_omitted_factors",
    "run_influence_path",
    "stabilization_quantities",
    "target_factor_counts",
]
