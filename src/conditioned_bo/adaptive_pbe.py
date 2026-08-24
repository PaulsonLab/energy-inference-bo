"""Exact shared-reference and adaptive PBE conditioning for Sun-oxide E3.

The routines in this module implement the frozen engineering handoff.  They do
not inspect GW targets; callers supply only the observations already acquired.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import sparse
from scipy.sparse.linalg import splu

from conditioned_bo.bo_value import (
    ExactSupportMarginal,
    LaplaceState,
    NumericalFailure,
    TimedFactorBank,
    WEIGHT,
    fit_laplace_approximation,
    gaussian_expected_improvement,
    select_unobserved_action,
)


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class MenzSupportReference:
    """The exact support block ``(A0^-1)[H,H]`` and precompute diagnostics."""

    covariance: FloatArray
    diagnostics: dict[str, Any]


@dataclass
class ActiveFactorState:
    """Cumulative active mask and incrementally maintained omitted degrees."""

    active_mask: BoolArray
    omitted_endpoint_degree: IntArray

    @classmethod
    def empty(cls, endpoint_pairs: ArrayLike, dimension: int) -> ActiveFactorState:
        pairs = _validated_pairs(endpoint_pairs, dimension)
        degree = np.bincount(pairs.ravel(), minlength=dimension).astype(np.int64)
        return cls(
            active_mask=np.zeros(pairs.shape[0], dtype=bool),
            omitted_endpoint_degree=degree,
        )

    @property
    def active_count(self) -> int:
        return int(np.count_nonzero(self.active_mask))

    @property
    def factor_count(self) -> int:
        return int(self.active_mask.size)

    def active_indices(self) -> IntArray:
        return np.flatnonzero(self.active_mask).astype(np.int64)

    def omitted_indices(self) -> IntArray:
        return np.flatnonzero(~self.active_mask).astype(np.int64)

    def activate(self, factor_indices: ArrayLike, endpoint_pairs: ArrayLike) -> None:
        indices = np.asarray(factor_indices, dtype=np.int64)
        pairs = _validated_pairs(endpoint_pairs, self.omitted_endpoint_degree.size)
        if pairs.shape[0] != self.factor_count:
            raise ValueError("Factor state and endpoint bank do not align")
        if indices.ndim != 1 or len(set(indices.tolist())) != indices.size:
            raise ValueError("Activated factor indices must be unique")
        if indices.size == 0:
            return
        if indices.min() < 0 or indices.max() >= self.factor_count:
            raise ValueError("Activated factor index lies outside the bank")
        if np.any(self.active_mask[indices]):
            raise ValueError("A factor cannot be activated twice")
        selected = pairs[indices]
        decrement = np.bincount(
            selected.ravel(), minlength=self.omitted_endpoint_degree.size
        ).astype(np.int64)
        updated = self.omitted_endpoint_degree - decrement
        if np.any(updated < 0):
            raise NumericalFailure("Incremental omitted endpoint degree became negative")
        self.active_mask[indices] = True
        self.omitted_endpoint_degree = updated


@dataclass(frozen=True)
class AdaptiveSettings:
    epsilon_struct: float = 0.02
    rho: float = 0.8
    max_stages: int = 8
    weight: float = WEIGHT
    chunk_size: int = 16384
    map_gradient_tolerance: float = 1e-5
    optimizer_gradient_tolerance: float = 1e-8
    function_tolerance: float = 1e-15
    maximum_iterations: int = 2000
    solve_residual_tolerance: float = 1e-9

    def validate(self) -> None:
        if self.epsilon_struct != 0.02:
            raise ValueError("Frozen structural tolerance changed")
        if self.rho != 0.8:
            raise ValueError("Frozen activation safety factor changed")
        if self.max_stages != 8:
            raise ValueError("Frozen maximum adaptive stages changed")
        if self.weight != WEIGHT:
            raise ValueError("Frozen factor weight changed")


@dataclass(frozen=True)
class PBEContext:
    selected: int
    ei: FloatArray
    map: FloatArray
    laplace: LaplaceState | None
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class AdaptiveDecision:
    selected: int
    ei: FloatArray
    map: FloatArray
    laplace: LaplaceState | None
    active_count: int
    active_fraction: float
    adaptive_stages: int
    full_bank_fallback: bool
    structurally_certified: bool
    structural_envelope: float
    worst_challenger: int | None
    diagnostics: dict[str, Any]


def _validated_pairs(endpoint_pairs: ArrayLike, dimension: int) -> IntArray:
    pairs = np.asarray(endpoint_pairs, dtype=np.int64)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("endpoint_pairs must have shape (n_factors, 2)")
    if pairs.size and (pairs.min() < 0 or pairs.max() >= dimension):
        raise ValueError("Factor endpoint lies outside the support")
    return pairs


def construct_menz_support_reference(
    q0: sparse.spmatrix,
    support_nodes: ArrayLike,
    endpoint_pairs: ArrayLike,
    *,
    weight: float = WEIGHT,
    residual_tolerance: float = 1e-9,
    nonnegative_tolerance: float = 1e-11,
) -> MenzSupportReference:
    """Compute ``C_H0=(A0^-1)[H,H]`` by one sparse factorization and 500 RHS."""

    reference = sparse.csc_matrix(q0, dtype=np.float64)
    support = np.asarray(support_nodes, dtype=np.int64)
    if reference.shape[0] != reference.shape[1]:
        raise ValueError("Q0 must be square")
    if support.ndim != 1 or len(set(support.tolist())) != support.size:
        raise ValueError("Support nodes must be a unique vector")
    if support.size and (support.min() < 0 or support.max() >= reference.shape[0]):
        raise ValueError("Support node lies outside Q0")
    pairs = _validated_pairs(endpoint_pairs, support.size)
    if weight <= 0.0:
        raise ValueError("Factor weight must be positive")

    full_pairs = support[pairs]
    row = np.concatenate((full_pairs[:, 0], full_pairs[:, 1]))
    col = np.concatenate((full_pairs[:, 1], full_pairs[:, 0]))
    data = np.full(row.size, weight, dtype=np.float64)
    weighted_adjacency = sparse.coo_matrix(
        (data, (row, col)), shape=reference.shape
    ).tocsc()
    weighted_adjacency.sum_duplicates()
    weighted_adjacency.sort_indices()
    a0 = sparse.csc_matrix(reference - 0.25 * weighted_adjacency)
    a0.sort_indices()

    total_started = time.perf_counter()
    factor_started = time.perf_counter()
    factorization = splu(a0, permc_spec="COLAMD")
    factorization_seconds = time.perf_counter() - factor_started
    rhs = np.zeros((a0.shape[0], support.size), dtype=np.float64)
    rhs[support, np.arange(support.size)] = 1.0
    solve_started = time.perf_counter()
    columns = np.asarray(factorization.solve(rhs), dtype=np.float64)
    solve_seconds = time.perf_counter() - solve_started
    residual = a0 @ columns - rhs
    relative_residual = float(np.linalg.norm(residual) / np.linalg.norm(rhs))
    if relative_residual > residual_tolerance:
        raise NumericalFailure(
            f"Menz support solve residual {relative_residual} exceeds "
            f"{residual_tolerance}"
        )
    covariance = np.asarray(columns[support, :], dtype=np.float64)
    covariance = 0.5 * (covariance + covariance.T)
    minimum_entry = float(np.min(covariance))
    if minimum_entry < -nonnegative_tolerance:
        raise NumericalFailure(
            f"Menz support inverse has negative entry {minimum_entry}"
        )
    return MenzSupportReference(
        covariance=covariance,
        diagnostics={
            "precompute_kind": "exact_state_zero_menz_support_inverse",
            "definition": "C_H0=(A0^-1)[H,H], A0=Q0-0.25*W_pbe",
            "full_dimension": int(a0.shape[0]),
            "support_dimension": int(support.size),
            "factor_count": int(pairs.shape[0]),
            "weight": float(weight),
            "a0_nnz": int(a0.nnz),
            "sparse_factorization_seconds": factorization_seconds,
            "support_rhs_solve_seconds": solve_seconds,
            "support_rhs_count": int(support.size),
            "support_solve_relative_residual": relative_residual,
            "support_symmetry_max_abs_error": float(
                np.max(np.abs(covariance - covariance.T))
            ),
            "minimum_support_inverse_entry": minimum_entry,
            "total_seconds": time.perf_counter() - total_started,
            "dense_full_inverse_formed": False,
        },
    )


def sherman_morrison_observation_update(
    covariance: ArrayLike,
    support_position: int,
    *,
    delta: float = 400.0,
) -> FloatArray:
    """Apply the exact support-block update for one diagonal observation."""

    matrix = np.asarray(covariance, dtype=np.float64)
    position = int(support_position)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Menz covariance must be square")
    if not 0 <= position < matrix.shape[0] or delta <= 0.0:
        raise ValueError("Invalid observation support position or precision")
    column = matrix[:, position].copy()
    denominator = 1.0 + delta * float(matrix[position, position])
    if denominator <= 0.0:
        raise NumericalFailure("Sherman-Morrison denominator is nonpositive")
    updated = matrix - (delta / denominator) * np.outer(column, column)
    return np.asarray(0.5 * (updated + updated.T), dtype=np.float64)


def apply_menz_observation_updates(
    base_covariance: ArrayLike,
    observed_support_positions: ArrayLike,
    *,
    delta: float = 400.0,
) -> FloatArray:
    result = np.asarray(base_covariance, dtype=np.float64).copy()
    positions = np.asarray(observed_support_positions, dtype=np.int64)
    if positions.ndim != 1 or len(set(positions.tolist())) != positions.size:
        raise ValueError("Observed support positions must be a unique vector")
    for position in positions.tolist():
        result = sherman_morrison_observation_update(
            result, position, delta=delta
        )
    return result


def omitted_structural_load(
    omitted_endpoint_degree: ArrayLike, *, weight: float = WEIGHT
) -> FloatArray:
    degree = np.asarray(omitted_endpoint_degree, dtype=np.int64)
    if degree.ndim != 1 or np.any(degree < 0) or weight <= 0.0:
        raise ValueError("Invalid omitted endpoint degree or factor weight")
    return np.asarray(weight * degree, dtype=np.float64)


def structural_influence_vector(
    menz_covariance: ArrayLike,
    omitted_endpoint_degree: ArrayLike,
    *,
    weight: float = WEIGHT,
) -> FloatArray:
    covariance = np.asarray(menz_covariance, dtype=np.float64)
    load = omitted_structural_load(omitted_endpoint_degree, weight=weight)
    if covariance.shape != (load.size, load.size):
        raise ValueError("Menz covariance and omitted load do not align")
    return np.asarray(covariance @ load, dtype=np.float64)


def omitted_factor_contributions(
    menz_covariance: ArrayLike,
    endpoint_pairs: ArrayLike,
    omitted_factor_indices: ArrayLike,
    challenger_support_position: int,
    leader_support_position: int,
    *,
    weight: float = WEIGHT,
) -> FloatArray:
    covariance = np.asarray(menz_covariance, dtype=np.float64)
    pairs = _validated_pairs(endpoint_pairs, covariance.shape[0])
    omitted = np.asarray(omitted_factor_indices, dtype=np.int64)
    if omitted.ndim != 1 or (omitted.size and np.any(np.diff(omitted) <= 0)):
        raise ValueError("Omitted factor IDs must be strictly increasing")
    if omitted.size and (omitted.min() < 0 or omitted.max() >= pairs.shape[0]):
        raise ValueError("Omitted factor ID lies outside the bank")
    challenger = int(challenger_support_position)
    leader = int(leader_support_position)
    if not 0 <= challenger < covariance.shape[0] or not 0 <= leader < covariance.shape[0]:
        raise ValueError("Leader or challenger lies outside the support")
    selected = pairs[omitted]
    return np.asarray(
        weight
        * (
            covariance[challenger, selected[:, 0]]
            + covariance[challenger, selected[:, 1]]
            + covariance[leader, selected[:, 0]]
            + covariance[leader, selected[:, 1]]
        ),
        dtype=np.float64,
    )


def stable_descending_factor_ranking(
    omitted_factor_indices: ArrayLike, contributions: ArrayLike
) -> IntArray:
    omitted = np.asarray(omitted_factor_indices, dtype=np.int64)
    values = np.asarray(contributions, dtype=np.float64)
    if omitted.ndim != 1 or values.shape != omitted.shape:
        raise ValueError("Omitted factor IDs and contributions must align")
    if omitted.size and np.any(np.diff(omitted) <= 0):
        raise ValueError("Omitted factor IDs must be in stable bank order")
    if not np.all(np.isfinite(values)):
        raise ValueError("Factor contributions must be finite")
    order = np.argsort(-values, kind="stable")
    return omitted[order]


def exact_activation_batch(
    omitted_factor_indices: ArrayLike,
    contributions: ArrayLike,
    *,
    active_gap: float,
    epsilon_struct: float = 0.02,
    rho: float = 0.8,
) -> IntArray:
    """Return the smallest stable top batch satisfying the frozen rule."""

    omitted = np.asarray(omitted_factor_indices, dtype=np.int64)
    values = np.asarray(contributions, dtype=np.float64)
    if omitted.ndim != 1 or values.shape != omitted.shape or omitted.size == 0:
        raise ValueError("A nonempty aligned omitted-factor bank is required")
    if np.any(values < -1e-12) or not np.all(np.isfinite(values)):
        raise ValueError("Structural factor contributions must be nonnegative")
    gap = float(active_gap)
    target = float(epsilon_struct - gap)
    if gap > 1e-12 or target <= 0.0 or not 0.0 < rho < 1.0:
        raise ValueError("Invalid active gap, structural tolerance, or safety factor")
    order = np.argsort(-values, kind="stable")
    ordered_values = values[order]
    required_removal = float(np.sum(ordered_values) - rho * target)
    if required_removal <= 0.0:
        raise ValueError("Activation batch requested although the safety target holds")
    cumulative = np.cumsum(ordered_values)
    count = int(np.searchsorted(cumulative, required_removal, side="left") + 1)
    return np.asarray(omitted[order[:count]], dtype=np.int64)


def _available_mask(action_count: int, observed_positions: ArrayLike) -> BoolArray:
    observed = np.asarray(observed_positions, dtype=np.int64)
    available = np.ones(action_count, dtype=bool)
    available[observed] = False
    return available


def _stable_maximum_position(
    values: FloatArray, mask: BoolArray, action_keys: Sequence[str]
) -> int:
    maximum = float(np.max(values[mask]))
    tied = np.flatnonzero(mask & (values == maximum))
    return min((int(index) for index in tied), key=lambda index: action_keys[index])


def _laplace_ei_context(
    reference: ExactSupportMarginal,
    endpoint_pairs: IntArray,
    signs: IntArray,
    active_indices: IntArray,
    action_support_positions: IntArray,
    observed_action_positions: ArrayLike,
    action_keys: Sequence[str],
    incumbent: float,
    initial_map: FloatArray | None,
    settings: AdaptiveSettings,
    *,
    warm_start_source: str,
) -> PBEContext:
    started = time.perf_counter()
    if active_indices.size == 0:
        ei = gaussian_expected_improvement(
            reference.mean[action_support_positions],
            np.maximum(np.diag(reference.covariance)[action_support_positions], 0.0),
            incumbent,
        )
        selected = select_unobserved_action(
            ei, observed_action_positions, action_keys
        )
        return PBEContext(
            selected=selected,
            ei=ei,
            map=np.asarray(reference.mean, dtype=np.float64).copy(),
            laplace=None,
            diagnostics={
                "active_factor_count": 0,
                "inference_kind": "exact_gaussian_reference",
                "warm_start_source": "not_applicable_empty_active_set",
                "factor_energy_gradient_calls": 0,
                "factor_energy_gradient_element_work": 0,
                "factor_hessian_calls": 0,
                "factor_hessian_element_work": 0,
                "stage_conditioning_seconds": time.perf_counter() - started,
            },
        )

    bank = TimedFactorBank(
        endpoint_pairs[active_indices],
        signs[active_indices],
        dimension=reference.mean.size,
        weight=settings.weight,
        chunk_size=settings.chunk_size,
    )
    first = (
        np.zeros(reference.mean.size, dtype=np.float64)
        if initial_map is None
        else np.asarray(initial_map, dtype=np.float64)
    )
    laplace = fit_laplace_approximation(
        reference.mean,
        reference.precision,
        bank,
        first,
        retry_map=np.zeros(reference.mean.size, dtype=np.float64),
        gradient_tolerance=settings.map_gradient_tolerance,
        optimizer_gradient_tolerance=settings.optimizer_gradient_tolerance,
        function_tolerance=settings.function_tolerance,
        maximum_iterations=settings.maximum_iterations,
        residual_tolerance=settings.solve_residual_tolerance,
    )
    ei = gaussian_expected_improvement(
        laplace.map[action_support_positions],
        np.maximum(np.diag(laplace.covariance)[action_support_positions], 0.0),
        incumbent,
    )
    selected = select_unobserved_action(ei, observed_action_positions, action_keys)
    return PBEContext(
        selected=selected,
        ei=ei,
        map=np.asarray(laplace.map, dtype=np.float64),
        laplace=laplace,
        diagnostics={
            "active_factor_count": int(active_indices.size),
            "inference_kind": "laplace",
            "warm_start_source": warm_start_source,
            **laplace.diagnostics,
            "stage_conditioning_seconds": time.perf_counter() - started,
        },
    )


def full_pbe_context(
    reference: ExactSupportMarginal,
    endpoint_pairs: ArrayLike,
    signs: ArrayLike,
    action_support_positions: ArrayLike,
    observed_action_positions: ArrayLike,
    action_keys: Sequence[str],
    incumbent: float,
    previous_map: ArrayLike | None,
    settings: AdaptiveSettings,
) -> PBEContext:
    """Fit the strong all-factor optimized baseline on the shared reference."""

    settings.validate()
    pairs = _validated_pairs(endpoint_pairs, reference.mean.size)
    directions = np.asarray(signs, dtype=np.int8)
    action_support = np.asarray(action_support_positions, dtype=np.int64)
    if directions.shape != (pairs.shape[0],):
        raise ValueError("Factor endpoints and signs do not align")
    initial = None if previous_map is None else np.asarray(previous_map, dtype=np.float64)
    context = _laplace_ei_context(
        reference,
        pairs,
        directions,
        np.arange(pairs.shape[0], dtype=np.int64),
        action_support,
        observed_action_positions,
        action_keys,
        incumbent,
        initial,
        settings,
        warm_start_source=(
            "zero_fallback_first_state"
            if previous_map is None
            else "preceding_FULL_PBE_OPT_MAP"
        ),
    )
    return PBEContext(
        selected=context.selected,
        ei=context.ei,
        map=context.map,
        laplace=context.laplace,
        diagnostics={
            **context.diagnostics,
            "method": "FULL_PBE_OPT",
            "pbe_conditioning_seconds": context.diagnostics[
                "stage_conditioning_seconds"
            ],
            "active_factor_fraction": 1.0,
        },
    )


def adaptive_pbe_decision(
    reference: ExactSupportMarginal,
    menz_covariance: ArrayLike,
    endpoint_pairs: ArrayLike,
    signs: ArrayLike,
    action_support_positions: ArrayLike,
    observed_action_positions: ArrayLike,
    action_keys: Sequence[str],
    incumbent: float,
    factor_state: ActiveFactorState,
    previous_map: ArrayLike | None,
    settings: AdaptiveSettings,
) -> AdaptiveDecision:
    """Run the frozen cumulative adaptive activation and stopping algorithm."""

    settings.validate()
    started = time.perf_counter()
    pairs = _validated_pairs(endpoint_pairs, reference.mean.size)
    directions = np.asarray(signs, dtype=np.int8)
    action_support = np.asarray(action_support_positions, dtype=np.int64)
    covariance = np.asarray(menz_covariance, dtype=np.float64)
    if directions.shape != (pairs.shape[0],) or factor_state.factor_count != pairs.shape[0]:
        raise ValueError("Factor arrays and cumulative state do not align")
    if covariance.shape != (reference.mean.size, reference.mean.size):
        raise ValueError("Menz covariance and support reference do not align")
    if action_support.shape != (len(action_keys),):
        raise ValueError("Action support positions and keys do not align")

    previous = None if previous_map is None else np.asarray(previous_map, dtype=np.float64)
    activation_count = 0
    stage_records: list[dict[str, Any]] = []
    energy_gradient_work = 0
    hessian_work = 0
    energy_gradient_calls = 0
    hessian_calls = 0
    maximum_contribution_sum_error = 0.0
    available = _available_mask(len(action_keys), observed_action_positions)

    while True:
        active_indices = factor_state.active_indices()
        if activation_count > 0:
            warm_source = "preceding_adaptive_stage_MAP"
        elif previous is not None:
            warm_source = "preceding_ADAPTIVE_PBE_MAP"
        else:
            warm_source = "exact_gaussian_stage_MAP"
        context = _laplace_ei_context(
            reference,
            pairs,
            directions,
            active_indices,
            action_support,
            observed_action_positions,
            action_keys,
            incumbent,
            previous,
            settings,
            warm_start_source=warm_source,
        )
        stage_diag = context.diagnostics
        energy_gradient_work += int(stage_diag["factor_energy_gradient_element_work"])
        hessian_work += int(stage_diag["factor_hessian_element_work"])
        energy_gradient_calls += int(stage_diag["factor_energy_gradient_calls"])
        hessian_calls += int(stage_diag["factor_hessian_calls"])
        previous = context.map

        influence = structural_influence_vector(
            covariance,
            factor_state.omitted_endpoint_degree,
            weight=settings.weight,
        )
        leader = int(context.selected)
        leader_support = int(action_support[leader])
        psi = np.full(len(action_keys), -np.inf, dtype=np.float64)
        psi[available] = (
            context.ei[available]
            - context.ei[leader]
            + influence[action_support[available]]
            + influence[leader_support]
        )
        challenger_mask = available.copy()
        challenger_mask[leader] = False
        challenger = _stable_maximum_position(psi, challenger_mask, action_keys)
        envelope = float(psi[challenger])
        record = {
            "stage": activation_count,
            "active_factor_count": factor_state.active_count,
            "active_factor_fraction": factor_state.active_count / factor_state.factor_count,
            "leader_action_position": leader,
            "worst_challenger_action_position": challenger,
            "active_ei_gap": float(context.ei[challenger] - context.ei[leader]),
            "structural_bound": float(
                influence[int(action_support[challenger])] + influence[leader_support]
            ),
            "structural_envelope": envelope,
            "inference": stage_diag,
            "activated_factor_count_after_stage": 0,
        }
        stage_records.append(record)
        if envelope <= settings.epsilon_struct:
            return AdaptiveDecision(
                selected=leader,
                ei=context.ei,
                map=context.map,
                laplace=context.laplace,
                active_count=factor_state.active_count,
                active_fraction=factor_state.active_count / factor_state.factor_count,
                adaptive_stages=activation_count,
                full_bank_fallback=False,
                structurally_certified=True,
                structural_envelope=envelope,
                worst_challenger=challenger,
                diagnostics={
                    "method": "ADAPTIVE_PBE",
                    "stop_reason": "structural_certificate",
                    "pbe_conditioning_seconds": time.perf_counter() - started,
                    "factor_energy_gradient_calls": energy_gradient_calls,
                    "factor_energy_gradient_element_work": energy_gradient_work,
                    "factor_hessian_calls": hessian_calls,
                    "factor_hessian_element_work": hessian_work,
                    "adaptive_stages": activation_count,
                    "stage_records": stage_records,
                    "maximum_contribution_sum_error": maximum_contribution_sum_error,
                    "warm_start_across_bo_used": previous_map is not None,
                    "warm_start_across_stages_used": activation_count > 0,
                },
            )

        if activation_count >= settings.max_stages:
            omitted = factor_state.omitted_indices()
            factor_state.activate(omitted, pairs)
            fallback = _laplace_ei_context(
                reference,
                pairs,
                directions,
                factor_state.active_indices(),
                action_support,
                observed_action_positions,
                action_keys,
                incumbent,
                previous,
                settings,
                warm_start_source="preceding_adaptive_stage_MAP_full_bank_fallback",
            )
            fallback_diag = fallback.diagnostics
            energy_gradient_work += int(
                fallback_diag["factor_energy_gradient_element_work"]
            )
            hessian_work += int(fallback_diag["factor_hessian_element_work"])
            energy_gradient_calls += int(fallback_diag["factor_energy_gradient_calls"])
            hessian_calls += int(fallback_diag["factor_hessian_calls"])
            fallback_available = _available_mask(
                len(action_keys), observed_action_positions
            )
            fallback_challengers = fallback_available.copy()
            fallback_challengers[fallback.selected] = False
            full_gap = fallback.ei - fallback.ei[fallback.selected]
            fallback_challenger = _stable_maximum_position(
                full_gap, fallback_challengers, action_keys
            )
            fallback_envelope = float(full_gap[fallback_challenger])
            stage_records.append(
                {
                    "stage": "full_bank_fallback",
                    "active_factor_count": factor_state.active_count,
                    "active_factor_fraction": 1.0,
                    "leader_action_position": int(fallback.selected),
                    "worst_challenger_action_position": fallback_challenger,
                    "active_ei_gap": fallback_envelope,
                    "structural_bound": 0.0,
                    "structural_envelope": fallback_envelope,
                    "inference": fallback_diag,
                    "activated_factor_count_after_stage": 0,
                }
            )
            return AdaptiveDecision(
                selected=int(fallback.selected),
                ei=fallback.ei,
                map=fallback.map,
                laplace=fallback.laplace,
                active_count=factor_state.active_count,
                active_fraction=1.0,
                adaptive_stages=activation_count,
                full_bank_fallback=True,
                structurally_certified=False,
                structural_envelope=fallback_envelope,
                worst_challenger=fallback_challenger,
                diagnostics={
                    "method": "ADAPTIVE_PBE",
                    "stop_reason": "explicit_full_bank_fallback",
                    "pbe_conditioning_seconds": time.perf_counter() - started,
                    "factor_energy_gradient_calls": energy_gradient_calls,
                    "factor_energy_gradient_element_work": energy_gradient_work,
                    "factor_hessian_calls": hessian_calls,
                    "factor_hessian_element_work": hessian_work,
                    "adaptive_stages": activation_count,
                    "stage_records": stage_records,
                    "maximum_contribution_sum_error": maximum_contribution_sum_error,
                    "warm_start_across_bo_used": previous_map is not None,
                    "warm_start_across_stages_used": True,
                },
            )

        omitted = factor_state.omitted_indices()
        contributions = omitted_factor_contributions(
            covariance,
            pairs,
            omitted,
            int(action_support[challenger]),
            leader_support,
            weight=settings.weight,
        )
        structural_bound = float(record["structural_bound"])
        contribution_sum_error = abs(float(np.sum(contributions)) - structural_bound)
        maximum_contribution_sum_error = max(
            maximum_contribution_sum_error, contribution_sum_error
        )
        if contribution_sum_error > 1e-9 * max(1.0, abs(structural_bound)):
            raise NumericalFailure(
                "Omitted-factor contributions do not sum to the structural bound: "
                f"error={contribution_sum_error}"
            )
        gap = float(context.ei[challenger] - context.ei[leader])
        batch = exact_activation_batch(
            omitted,
            contributions,
            active_gap=gap,
            epsilon_struct=settings.epsilon_struct,
            rho=settings.rho,
        )
        record["activated_factor_count_after_stage"] = int(batch.size)
        record["activated_factor_first_ids"] = batch[:10].tolist()
        factor_state.activate(batch, pairs)
        activation_count += 1


__all__ = [
    "ActiveFactorState",
    "AdaptiveDecision",
    "AdaptiveSettings",
    "MenzSupportReference",
    "PBEContext",
    "adaptive_pbe_decision",
    "apply_menz_observation_updates",
    "construct_menz_support_reference",
    "exact_activation_batch",
    "full_pbe_context",
    "omitted_factor_contributions",
    "omitted_structural_load",
    "sherman_morrison_observation_update",
    "stable_descending_factor_ranking",
    "structural_influence_vector",
]
