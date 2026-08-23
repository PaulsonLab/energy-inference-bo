"""Target-blind normalized dense PBE ranking model for the Sun oxide E3 case."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
import hashlib
import math
import time
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import sparse
from scipy.optimize import minimize
from scipy.sparse import csgraph
from scipy.special import expit
from scipy.stats import spearmanr

from conditioned_bo.pbe_factor_theory import LegacyNode
from conditioned_bo.preference_influence import (
    logistic_preference_energy,
    logistic_preference_gradient,
    logistic_preference_hessian,
)


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
SUPPORT_NAME = "PBE_SUPPORT_500_V1"
FACTOR_BANK_NAME = "NORMALIZED_ALL_PAIRS_PBE_500_V1"
SUPPORT_COUNT = 500
ACTION_COUNT = 191
ADDITIONAL_COUNT = 309
WEIGHT = 1.0 / 499.0
TEMPERATURE = 1.0


@dataclass(frozen=True)
class FarthestPointResult:
    selected_indices: IntArray
    additional_indices: IntArray
    additional_selection_distances: FloatArray
    initial_nearest_distances: FloatArray
    final_nearest_distances: FloatArray
    exact_tie_step_count: int
    maximum_exact_tie_candidates: int


@dataclass(frozen=True)
class PairBank:
    support_indices: IntArray
    support_endpoint_pairs: IntArray
    node_endpoint_pairs: IntArray
    signs: NDArray[np.int8]
    strict_factor_count: int
    omitted_exact_tie_pair_count: int
    exact_tie_group_count: int
    exact_tie_group_size_histogram: dict[str, int]


def float_array_sha256(values: ArrayLike) -> str:
    canonical = np.ascontiguousarray(values, dtype="<f8")
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def standardize_descriptor_space(raw_descriptors: ArrayLike) -> tuple[FloatArray, dict[str, Any]]:
    """Reproduce the committed StandardScaler transform without refitting metadata."""

    values = np.asarray(raw_descriptors, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or not np.all(np.isfinite(values)):
        raise ValueError("Descriptor matrix must be finite and two-dimensional")
    mean = np.mean(values, axis=0)
    variance = np.var(values, axis=0)
    sample_count = values.shape[0]
    epsilon = np.finfo(np.float64).eps
    constant = variance <= (
        sample_count * epsilon * variance
        + (sample_count * mean * epsilon) ** 2
    )
    scale = np.sqrt(variance)
    scale[constant] = 1.0
    standardized = np.asarray((values - mean) / scale, dtype=np.float64)
    if not np.all(np.isfinite(standardized)):
        raise ValueError("Descriptor standardization produced a non-finite value")
    return standardized, {
        "class": "sklearn.preprocessing.StandardScaler",
        "with_mean": True,
        "with_std": True,
        "mean_sha256": float_array_sha256(mean),
        "scale_sha256": float_array_sha256(scale),
        "zero_variance_feature_count": int(np.count_nonzero(constant)),
        "zero_variance_feature_indices": np.flatnonzero(constant).tolist(),
    }


def _squared_distance_to_point(points: FloatArray, point_index: int) -> FloatArray:
    differences = points - points[int(point_index)]
    return np.einsum("ij,ij->i", differences, differences, optimize=True)


def farthest_point_support(
    standardized_descriptors: ArrayLike,
    composition_keys: Sequence[str],
    initial_indices: ArrayLike,
    support_count: int = SUPPORT_COUNT,
) -> FarthestPointResult:
    """Select descriptor-diverse support with exact stable-key tie breaking."""

    points = np.asarray(standardized_descriptors, dtype=np.float64)
    keys = tuple(str(key) for key in composition_keys)
    initial = np.asarray(initial_indices, dtype=np.int64)
    if points.ndim != 2 or points.shape[0] != len(keys):
        raise ValueError("Descriptor rows and composition keys do not align")
    if not np.all(np.isfinite(points)):
        raise ValueError("Standardized descriptors contain non-finite values")
    if len(set(keys)) != len(keys):
        raise ValueError("Composition keys are not unique")
    if initial.ndim != 1 or len(set(initial.tolist())) != initial.size:
        raise ValueError("Initial support indices must be a unique vector")
    if initial.size and (initial.min() < 0 or initial.max() >= points.shape[0]):
        raise ValueError("Initial support index is outside descriptor rows")
    if not initial.size <= support_count <= points.shape[0]:
        raise ValueError("Invalid support count")

    selected = np.zeros(points.shape[0], dtype=bool)
    selected[initial] = True
    nearest_squared = np.full(points.shape[0], np.inf, dtype=np.float64)
    for point_index in initial:
        nearest_squared = np.minimum(
            nearest_squared,
            _squared_distance_to_point(points, int(point_index)),
        )
    nearest_squared[selected] = 0.0
    initial_nearest = np.sqrt(nearest_squared)

    additional: list[int] = []
    selection_distances: list[float] = []
    exact_tie_steps = 0
    maximum_tie_candidates = 1
    while int(np.count_nonzero(selected)) < support_count:
        eligible = np.flatnonzero(~selected)
        maximum = float(np.max(nearest_squared[eligible]))
        tied = eligible[nearest_squared[eligible] == maximum]
        if tied.size > 1:
            exact_tie_steps += 1
            maximum_tie_candidates = max(maximum_tie_candidates, int(tied.size))
        chosen = min((int(index) for index in tied), key=lambda index: keys[index])
        additional.append(chosen)
        selection_distances.append(math.sqrt(maximum))
        selected[chosen] = True
        nearest_squared = np.minimum(
            nearest_squared,
            _squared_distance_to_point(points, chosen),
        )
        nearest_squared[selected] = 0.0

    return FarthestPointResult(
        selected_indices=np.flatnonzero(selected).astype(np.int64),
        additional_indices=np.asarray(additional, dtype=np.int64),
        additional_selection_distances=np.asarray(selection_distances, dtype=np.float64),
        initial_nearest_distances=initial_nearest,
        final_nearest_distances=np.sqrt(nearest_squared),
        exact_tie_step_count=exact_tie_steps,
        maximum_exact_tie_candidates=maximum_tie_candidates,
    )


def _distance_summary(values: ArrayLike) -> dict[str, float]:
    distances = np.asarray(values, dtype=np.float64)
    if distances.ndim != 1 or not distances.size or not np.all(np.isfinite(distances)):
        raise ValueError("Distances must be a nonempty finite vector")
    return {
        "min": float(np.min(distances)),
        "mean": float(np.mean(distances)),
        "q25": float(np.quantile(distances, 0.25)),
        "median": float(np.quantile(distances, 0.50)),
        "q75": float(np.quantile(distances, 0.75)),
        "p90": float(np.quantile(distances, 0.90)),
        "p95": float(np.quantile(distances, 0.95)),
        "max": float(np.max(distances)),
    }


def support_coverage_diagnostics(
    result: FarthestPointResult,
    initial_action_indices: ArrayLike,
    node_count: int,
) -> dict[str, Any]:
    actions = np.asarray(initial_action_indices, dtype=np.int64)
    selected_mask = np.zeros(node_count, dtype=bool)
    selected_mask[result.selected_indices] = True
    initial_mask = np.ones(node_count, dtype=bool)
    initial_mask[actions] = False
    nonsupport_mask = ~selected_mask
    return {
        "support_count": int(result.selected_indices.size),
        "action_count": int(actions.size),
        "additional_pbe_only_count": int(result.additional_indices.size),
        "all_actions_included": bool(
            set(actions.tolist()).issubset(set(result.selected_indices.tolist()))
        ),
        "initial_action_cover_over_pbe_only_candidates": _distance_summary(
            result.initial_nearest_distances[initial_mask]
        ),
        "final_support_cover_over_all_nodes": _distance_summary(
            result.final_nearest_distances
        ),
        "final_support_cover_over_nonsupport_nodes": _distance_summary(
            result.final_nearest_distances[nonsupport_mask]
        ),
        "additional_selection_distance": _distance_summary(
            result.additional_selection_distances
        ),
        "initial_covering_radius": float(np.max(result.initial_nearest_distances)),
        "final_covering_radius": float(np.max(result.final_nearest_distances)),
        "covering_radius_ratio_final_to_initial": float(
            np.max(result.final_nearest_distances)
            / np.max(result.initial_nearest_distances)
        ),
        "exact_farthest_distance_tie_steps": result.exact_tie_step_count,
        "maximum_exact_tie_candidates": result.maximum_exact_tie_candidates,
        "pbe_values_used_for_support_selection": False,
        "target_values_used_for_support_selection": False,
    }


def build_strict_pair_bank(
    support_indices: ArrayLike,
    legacy_nodes: Sequence[LegacyNode],
) -> PairBank:
    """Create every strict support pair in canonical latent-node order."""

    support = np.sort(np.asarray(support_indices, dtype=np.int64))
    if support.ndim != 1 or len(set(support.tolist())) != support.size:
        raise ValueError("Support indices must be unique")
    if support.size and (support.min() < 0 or support.max() >= len(legacy_nodes)):
        raise ValueError("Support index is outside legacy nodes")
    if support.size < 2:
        raise ValueError("At least two support nodes are required")
    support_left, support_right = np.triu_indices(support.size, k=1)
    node_left = support[support_left]
    node_right = support[support_right]
    gap_values = [legacy_nodes[int(index)].pbe_band_gap for index in support]

    keep = np.empty(support_left.size, dtype=bool)
    signs = np.empty(support_left.size, dtype=np.int8)
    for pair_index, (left, right) in enumerate(
        zip(support_left.tolist(), support_right.tolist(), strict=True)
    ):
        difference = gap_values[left] - gap_values[right]
        keep[pair_index] = difference != Decimal(0)
        signs[pair_index] = 1 if difference > 0 else -1

    gap_counts = Counter(gap_values)
    tied_sizes = sorted(count for count in gap_counts.values() if count > 1)
    histogram = {
        str(size): count
        for size, count in sorted(Counter(tied_sizes).items())
    }
    support_pairs = np.column_stack((support_left[keep], support_right[keep])).astype(
        np.int64
    )
    node_pairs = np.column_stack((node_left[keep], node_right[keep])).astype(np.int64)
    return PairBank(
        support_indices=support,
        support_endpoint_pairs=support_pairs,
        node_endpoint_pairs=node_pairs,
        signs=signs[keep],
        strict_factor_count=int(np.count_nonzero(keep)),
        omitted_exact_tie_pair_count=int(np.count_nonzero(~keep)),
        exact_tie_group_count=len(tied_sizes),
        exact_tie_group_size_histogram=histogram,
    )


def weighted_adjacency(
    node_count: int,
    endpoint_pairs: ArrayLike,
    weight: float = WEIGHT,
) -> sparse.csr_matrix:
    pairs = np.asarray(endpoint_pairs, dtype=np.int64)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("endpoint_pairs must have shape (n_factors, 2)")
    if weight <= 0.0:
        raise ValueError("weight must be positive")
    if pairs.size and (pairs.min() < 0 or pairs.max() >= node_count):
        raise ValueError("Pair endpoint is outside node set")
    canonical = np.sort(pairs, axis=1)
    if np.any(canonical[:, 0] == canonical[:, 1]):
        raise ValueError("Pair endpoints must be distinct")
    if len({tuple(pair) for pair in canonical.tolist()}) != canonical.shape[0]:
        raise ValueError("Pair endpoints contain a duplicate edge")
    rows = canonical.ravel()
    cols = canonical[:, ::-1].ravel()
    data = np.full(rows.size, weight, dtype=np.float64)
    matrix = sparse.csr_matrix((data, (rows, cols)), shape=(node_count, node_count))
    matrix.sort_indices()
    return matrix


def weighted_graph_diagnostics(matrix: sparse.spmatrix) -> dict[str, Any]:
    values = sparse.csr_matrix(matrix, dtype=np.float64)
    difference = (values - values.T).tocsr()
    symmetry_error = float(np.max(np.abs(difference.data))) if difference.nnz else 0.0
    row_sums = np.asarray(values.sum(axis=1)).ravel()
    from scipy.sparse.linalg import eigsh

    spectral_norm = float(
        abs(eigsh(values, k=1, which="LM", return_eigenvectors=False, tol=1e-12)[0])
    )
    return {
        "symmetric": symmetry_error == 0.0,
        "symmetry_max_abs_error": symmetry_error,
        "nonnegative": bool(not values.nnz or np.min(values.data) >= 0.0),
        "nnz": int(values.nnz),
        "edge_count": int(values.nnz // 2),
        "maximum_weighted_row_sum": float(np.max(row_sums)),
        "minimum_positive_weighted_row_sum": float(np.min(row_sums[row_sums > 0.0])),
        "support_row_sum_distribution": _distance_summary(row_sums[row_sums > 0.0]),
        "spectral_norm_numerical": spectral_norm,
    }


def weighted_logistic_energy(
    latent: ArrayLike,
    endpoint_pair: Sequence[int],
    sign: int,
    weight: float = WEIGHT,
) -> float:
    return float(
        weight
        * logistic_preference_energy(latent, endpoint_pair, sign, TEMPERATURE)
    )


def weighted_logistic_gradient(
    latent: ArrayLike,
    endpoint_pair: Sequence[int],
    sign: int,
    weight: float = WEIGHT,
) -> FloatArray:
    return np.asarray(
        weight
        * logistic_preference_gradient(latent, endpoint_pair, sign, TEMPERATURE),
        dtype=np.float64,
    )


def weighted_logistic_hessian(
    latent: ArrayLike,
    endpoint_pair: Sequence[int],
    sign: int,
    weight: float = WEIGHT,
) -> FloatArray:
    return np.asarray(
        weight
        * logistic_preference_hessian(latent, endpoint_pair, sign, TEMPERATURE),
        dtype=np.float64,
    )


def preference_objective_and_gradient(
    latent: ArrayLike,
    q0: sparse.spmatrix,
    endpoint_pairs: ArrayLike,
    signs: ArrayLike,
    weight: float,
    *,
    chunk_size: int,
) -> tuple[float, FloatArray]:
    values = np.asarray(latent, dtype=np.float64)
    precision = sparse.csr_matrix(q0, dtype=np.float64)
    pairs = np.asarray(endpoint_pairs, dtype=np.int64)
    directions = np.asarray(signs, dtype=np.int8)
    if precision.shape != (values.size, values.size):
        raise ValueError("Q0 shape does not match latent")
    if pairs.ndim != 2 or pairs.shape[1] != 2 or directions.shape != (pairs.shape[0],):
        raise ValueError("Pair endpoints and signs do not align")
    if weight <= 0.0 or chunk_size < 1:
        raise ValueError("Invalid preference weight or chunk size")
    precision_times_values = np.asarray(precision @ values, dtype=np.float64)
    objective = 0.5 * float(values @ precision_times_values)
    gradient = precision_times_values.copy()
    for start in range(0, pairs.shape[0], chunk_size):
        stop = min(start + chunk_size, pairs.shape[0])
        left = pairs[start:stop, 0]
        right = pairs[start:stop, 1]
        sign = directions[start:stop].astype(np.float64)
        signed_margin = sign * (values[left] - values[right])
        objective += weight * float(np.sum(np.logaddexp(0.0, -signed_margin)))
        probability = expit(-signed_margin)
        coefficient = -weight * sign * probability
        gradient += np.bincount(left, weights=coefficient, minlength=values.size)
        gradient += np.bincount(right, weights=-coefficient, minlength=values.size)
    return objective, gradient


def solve_preference_map(
    q0: sparse.spmatrix,
    endpoint_pairs: ArrayLike,
    signs: ArrayLike,
    weight: float,
    *,
    chunk_size: int,
    gradient_tolerance: float,
    function_tolerance: float,
    maximum_iterations: int,
) -> tuple[FloatArray, dict[str, Any]]:
    precision = sparse.csr_matrix(q0, dtype=np.float64)
    def objective(values: FloatArray) -> tuple[float, FloatArray]:
        return preference_objective_and_gradient(
            values,
            precision,
            endpoint_pairs,
            signs,
            weight,
            chunk_size=chunk_size,
        )

    started = time.perf_counter()
    result = minimize(
        objective,
        np.zeros(precision.shape[0], dtype=np.float64),
        method="L-BFGS-B",
        jac=True,
        options={
            "gtol": gradient_tolerance,
            "ftol": function_tolerance,
            "maxiter": maximum_iterations,
            "maxls": 50,
        },
    )
    wall_seconds = time.perf_counter() - started
    latent = np.asarray(result.x, dtype=np.float64)
    objective, gradient = preference_objective_and_gradient(
        latent,
        precision,
        endpoint_pairs,
        signs,
        weight,
        chunk_size=chunk_size,
    )
    gradient_inf = float(np.linalg.norm(gradient, ord=np.inf))
    return latent, {
        "optimizer": "scipy.optimize.minimize/L-BFGS-B",
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "iterations": int(result.nit),
        "function_evaluations": int(result.nfev),
        "objective": objective,
        "gradient_infinity_norm": gradient_inf,
        "wall_seconds": wall_seconds,
        "latent_sha256": float_array_sha256(latent),
    }


def _strict_order_accuracy(
    pbe_values: Sequence[Decimal],
    latent: FloatArray,
) -> tuple[float, int]:
    left, right = np.triu_indices(len(pbe_values), k=1)
    keep: list[bool] = []
    signs: list[int] = []
    for first, second in zip(left.tolist(), right.tolist(), strict=True):
        difference = pbe_values[first] - pbe_values[second]
        keep.append(difference != Decimal(0))
        signs.append(1 if difference > 0 else -1)
    strict = np.asarray(keep, dtype=bool)
    direction = np.asarray(signs, dtype=np.int8)[strict]
    margins = direction * (latent[left[strict]] - latent[right[strict]])
    return float(np.mean(margins > 0.0)), int(margins.size)


def signal_diagnostics(
    legacy_nodes: Sequence[LegacyNode],
    latent: ArrayLike,
    evaluation_indices: ArrayLike,
) -> dict[str, Any]:
    values = np.asarray(latent, dtype=np.float64)
    indices = np.asarray(evaluation_indices, dtype=np.int64)
    pbe_decimal = [legacy_nodes[int(index)].pbe_band_gap for index in indices]
    pbe_float = np.asarray([float(value) for value in pbe_decimal], dtype=np.float64)
    evaluated = values[indices]
    correlation = float(spearmanr(pbe_float, evaluated).statistic)
    accuracy, strict_pairs = _strict_order_accuracy(pbe_decimal, evaluated)
    ordered_positions = sorted(
        range(indices.size),
        key=lambda position: (
            pbe_decimal[position],
            legacy_nodes[int(indices[position])].composition_key,
        ),
    )
    decile_count = int(math.ceil(indices.size / 10.0))
    bottom = np.asarray(ordered_positions[:decile_count], dtype=np.int64)
    top = np.asarray(ordered_positions[-decile_count:], dtype=np.int64)
    return {
        "node_count": int(indices.size),
        "spearman_pbe_rank_vs_map_rank": correlation,
        "strict_pair_ordering_accuracy": accuracy,
        "strict_pair_count": strict_pairs,
        "map_standard_deviation": float(np.std(evaluated, ddof=0)),
        "map_range": float(np.ptp(evaluated)),
        "decile_count": decile_count,
        "top_decile_minus_bottom_decile_map_contrast": float(
            np.mean(evaluated[top]) - np.mean(evaluated[bottom])
        ),
    }


def descriptor_graph_action_pairs(
    q0: sparse.spmatrix,
    actions: Sequence[Mapping[str, Any]],
    *,
    pair_count: int,
    pairs_per_quartile: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select stable-key action pairs from four distance-rank quartiles."""

    if pair_count != 4 * pairs_per_quartile:
        raise ValueError("pair_count must equal four times pairs_per_quartile")
    precision = sparse.csr_matrix(q0, dtype=np.float64)
    graph = precision.copy()
    graph.setdiag(0.0)
    graph.eliminate_zeros()
    graph.data = np.ones_like(graph.data)
    action_nodes = np.asarray([action["node_index"] for action in actions], dtype=np.int64)
    raw_distances = np.asarray(
        csgraph.shortest_path(
            graph, directed=False, unweighted=True, indices=action_nodes
        ),
        dtype=np.float64,
    )
    if not np.all(np.isfinite(raw_distances)):
        raise ValueError("Descriptor graph is disconnected from an action")
    distances = raw_distances.astype(np.int64)
    candidates: list[dict[str, Any]] = []
    for left in range(len(actions) - 1):
        for right in range(left + 1, len(actions)):
            endpoint_keys = sorted(
                (str(actions[left]["composition_key"]), str(actions[right]["composition_key"]))
            )
            candidates.append(
                {
                    "action_position_x": left,
                    "action_position_xhat": right,
                    "node_x": int(action_nodes[left]),
                    "node_xhat": int(action_nodes[right]),
                    "action_key_x": actions[left]["action_key"],
                    "action_key_xhat": actions[right]["action_key"],
                    "stable_pair_key": "|".join(endpoint_keys),
                    "descriptor_graph_distance": int(
                        distances[left, int(action_nodes[right])]
                    ),
                }
            )
    ordered = sorted(
        candidates,
        key=lambda row: (row["descriptor_graph_distance"], row["stable_pair_key"]),
    )
    quartile_indices = np.array_split(np.arange(len(ordered), dtype=np.int64), 4)
    selected: list[dict[str, Any]] = []
    quartile_summary: list[dict[str, Any]] = []
    for quartile, positions in enumerate(quartile_indices, start=1):
        bucket = [ordered[int(position)] for position in positions]
        chosen = sorted(bucket, key=lambda row: row["stable_pair_key"])[
            :pairs_per_quartile
        ]
        if len(chosen) != pairs_per_quartile:
            raise ValueError(f"Distance quartile {quartile} has too few pairs")
        for row in chosen:
            row["distance_quartile"] = quartile
        selected.extend(chosen)
        bucket_distances = np.asarray(
            [row["descriptor_graph_distance"] for row in bucket], dtype=np.float64
        )
        chosen_distances = np.asarray(
            [row["descriptor_graph_distance"] for row in chosen], dtype=np.float64
        )
        quartile_summary.append(
            {
                "quartile": quartile,
                "candidate_pair_count": len(bucket),
                "selected_pair_count": len(chosen),
                "candidate_distance": _distance_summary(bucket_distances),
                "selected_distance": _distance_summary(chosen_distances),
            }
        )
    if len({row["stable_pair_key"] for row in selected}) != pair_count:
        raise ValueError("Diagnostic action pairs are not distinct")
    return selected, {
        "all_action_pair_count": len(candidates),
        "diagnostic_pair_count": len(selected),
        "pairs_per_distance_quartile": pairs_per_quartile,
        "quartile_definition": (
            "sort all action pairs by unweighted descriptor-graph distance then "
            "stable pair key; split rank order into four contiguous buckets"
        ),
        "within_quartile_selection": "lexicographically smallest stable pair keys",
        "quartiles": quartile_summary,
        "all_pair_distance_histogram": {
            str(distance): count
            for distance, count in sorted(
                Counter(row["descriptor_graph_distance"] for row in candidates).items()
            )
        },
    }


def _summary(values: ArrayLike) -> dict[str, float]:
    data = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(data)),
        "q25": float(np.quantile(data, 0.25)),
        "median": float(np.quantile(data, 0.50)),
        "q75": float(np.quantile(data, 0.75)),
        "p90": float(np.quantile(data, 0.90)),
        "max": float(np.max(data)),
    }


def chunked_influence_diagnostics(
    selected_inverse_rows: ArrayLike,
    node_endpoint_pairs: ArrayLike,
    diagnostic_pairs: Sequence[Mapping[str, Any]],
    thresholds: Sequence[float],
    *,
    weight: float,
    chunk_size: int,
    nonnegative_tolerance: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = np.asarray(selected_inverse_rows, dtype=np.float64)
    endpoints = np.asarray(node_endpoint_pairs, dtype=np.int64)
    threshold_values = tuple(float(value) for value in thresholds)
    if rows.ndim != 2 or endpoints.ndim != 2 or endpoints.shape[1] != 2:
        raise ValueError("Invalid inverse rows or factor endpoints")
    if any(value <= 0.0 or value >= 1.0 for value in threshold_values):
        raise ValueError("Influence thresholds must lie between zero and one")
    if weight <= 0.0 or chunk_size < 1:
        raise ValueError("Invalid weight or chunk size")

    pair_rows: list[dict[str, Any]] = []
    fractions: dict[float, list[float]] = {value: [] for value in threshold_values}
    totals: list[float] = []
    fractions_by_quartile: dict[int, dict[float, list[float]]] = {
        quartile: {value: [] for value in threshold_values}
        for quartile in range(1, 5)
    }
    totals_by_quartile: dict[int, list[float]] = {
        quartile: [] for quartile in range(1, 5)
    }
    minimum_unclipped = math.inf
    for pair in diagnostic_pairs:
        left_position = int(pair["action_position_x"])
        right_position = int(pair["action_position_xhat"])
        contributions = np.empty(endpoints.shape[0], dtype=np.float64)
        for start in range(0, endpoints.shape[0], chunk_size):
            stop = min(start + chunk_size, endpoints.shape[0])
            left = endpoints[start:stop, 0]
            right = endpoints[start:stop, 1]
            chunk = weight * (
                rows[left_position, left]
                + rows[left_position, right]
                + rows[right_position, left]
                + rows[right_position, right]
            )
            minimum_unclipped = min(minimum_unclipped, float(np.min(chunk)))
            if np.min(chunk) < -nonnegative_tolerance:
                raise ValueError("A factor influence contribution is negative")
            contributions[start:stop] = np.maximum(chunk, 0.0)
        total = float(np.sum(contributions))
        if not math.isfinite(total) or total <= 0.0:
            raise ValueError("Total structural influence is not positive and finite")
        ranked = np.sort(contributions)[::-1]
        cumulative = np.cumsum(ranked)
        record = dict(pair)
        record["total_structural_influence"] = total
        quartile = int(pair["distance_quartile"])
        totals.append(total)
        totals_by_quartile[quartile].append(total)
        for threshold in threshold_values:
            count = int(np.searchsorted(cumulative, threshold * total, side="left") + 1)
            fraction = float(count / contributions.size)
            label = str(int(round(100 * threshold)))
            record[f"factors_for_{label}_percent"] = count
            record[f"factor_fraction_for_{label}_percent"] = fraction
            fractions[threshold].append(fraction)
            fractions_by_quartile[quartile][threshold].append(fraction)
        pair_rows.append(record)
    return pair_rows, {
        "diagnostic_action_pair_count": len(pair_rows),
        "factor_count": int(endpoints.shape[0]),
        "chunk_size": chunk_size,
        "factor_fractions_required": {
            f"{int(round(100 * threshold))}_percent": _summary(fractions[threshold])
            for threshold in threshold_values
        },
        "total_structural_influence": _summary(totals),
        "by_distance_quartile": {
            str(quartile): {
                "pair_count": len(totals_by_quartile[quartile]),
                "total_structural_influence": _summary(
                    totals_by_quartile[quartile]
                ),
                "factor_fractions_required": {
                    f"{int(round(100 * threshold))}_percent": _summary(
                        fractions_by_quartile[quartile][threshold]
                    )
                    for threshold in threshold_values
                },
            }
            for quartile in range(1, 5)
        },
        "minimum_unclipped_factor_contribution": minimum_unclipped,
        "diagnostic_not_gate": True,
    }


__all__ = [
    "ACTION_COUNT",
    "ADDITIONAL_COUNT",
    "FACTOR_BANK_NAME",
    "SUPPORT_COUNT",
    "SUPPORT_NAME",
    "TEMPERATURE",
    "WEIGHT",
    "FarthestPointResult",
    "PairBank",
    "build_strict_pair_bank",
    "chunked_influence_diagnostics",
    "descriptor_graph_action_pairs",
    "farthest_point_support",
    "float_array_sha256",
    "preference_objective_and_gradient",
    "signal_diagnostics",
    "solve_preference_map",
    "standardize_descriptor_space",
    "support_coverage_diagnostics",
    "weighted_adjacency",
    "weighted_graph_diagnostics",
    "weighted_logistic_energy",
    "weighted_logistic_gradient",
    "weighted_logistic_hessian",
]
