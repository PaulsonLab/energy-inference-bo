"""Frozen PBE-order factors and sparse Menz diagnostics for the Sun oxide case.

This module is deliberately target blind.  Its only tabular inputs are the
legacy PBE side-information table and the committed action-to-node mapping.
It performs no posterior inference and no Bayesian optimization.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
import csv
import hashlib
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import sparse
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh, splu
from scipy.special import expit


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
FACTOR_BANK_NAME = "ADJACENT_STRICT_PBE_ORDER_V1"
TEMPERATURE = 1.0
MAX_MIXED_CURVATURE = 0.25


@dataclass(frozen=True)
class LegacyNode:
    node_index: int
    composition_key: str
    normalized_formula: str
    pbe_band_gap_text: str
    pbe_band_gap: Decimal


@dataclass(frozen=True)
class PBEOrderFactor:
    factor_index: int
    sorted_position_a: int
    sorted_position_b: int
    node_a: int
    node_b: int
    composition_key_a: str
    composition_key_b: str
    normalized_formula_a: str
    normalized_formula_b: str
    pbe_band_gap_a_ev: str
    pbe_band_gap_b_ev: str
    pbe_gap_difference_ev: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_selected_csv(path: Path, columns: Sequence[str]) -> list[dict[str, str]]:
    """Read only named columns through an explicit, narrow interface."""

    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"Empty CSV: {path}") from exc
        missing = [column for column in columns if column not in header]
        if missing:
            raise ValueError(f"Missing columns in {path}: {missing}")
        indices = [header.index(column) for column in columns]
        rows: list[dict[str, str]] = []
        for line_number, raw in enumerate(reader, start=2):
            if len(raw) != len(header):
                raise ValueError(f"Malformed row {line_number} in {path}")
            rows.append(
                {
                    column: raw[index]
                    for column, index in zip(columns, indices, strict=True)
                }
            )
    return rows


def load_legacy_nodes(path: Path, expected_rows: int) -> tuple[LegacyNode, ...]:
    rows = _read_selected_csv(
        path,
        ["composition_key", "normalized_formula", "pbe_band_gap_ev"],
    )
    if len(rows) != expected_rows:
        raise ValueError(f"Legacy row count {len(rows)} != {expected_rows}")
    keys = [row["composition_key"] for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("Legacy composition keys are not unique")
    nodes: list[LegacyNode] = []
    for node_index, row in enumerate(rows):
        gap_text = row["pbe_band_gap_ev"]
        try:
            gap = Decimal(gap_text)
        except InvalidOperation as exc:
            raise ValueError(f"Invalid PBE gap at node {node_index}: {gap_text!r}") from exc
        if not gap.is_finite():
            raise ValueError(f"Non-finite PBE gap at node {node_index}")
        if not row["composition_key"] or not row["normalized_formula"]:
            raise ValueError(f"Blank legacy identifier at node {node_index}")
        nodes.append(
            LegacyNode(
                node_index=node_index,
                composition_key=row["composition_key"],
                normalized_formula=row["normalized_formula"],
                pbe_band_gap_text=gap_text,
                pbe_band_gap=gap,
            )
        )
    return tuple(nodes)


def load_action_mapping(path: Path, expected_actions: int, node_count: int) -> tuple[dict[str, Any], ...]:
    rows = _read_selected_csv(
        path,
        ["action_key", "composition_key", "node_index", "normalized_formula"],
    )
    if len(rows) != expected_actions:
        raise ValueError(f"Action row count {len(rows)} != {expected_actions}")
    action_keys = [row["action_key"] for row in rows]
    composition_keys = [row["composition_key"] for row in rows]
    indices = [int(row["node_index"]) for row in rows]
    if len(set(action_keys)) != len(rows):
        raise ValueError("Action keys are not unique")
    if len(set(composition_keys)) != len(rows):
        raise ValueError("Action composition keys are not unique")
    if len(set(indices)) != len(rows):
        raise ValueError("Action node indices are not unique")
    if any(index < 0 or index >= node_count for index in indices):
        raise ValueError("Action node index is outside the legacy node set")
    return tuple(
        {
            "action_key": row["action_key"],
            "composition_key": row["composition_key"],
            "node_index": int(row["node_index"]),
            "normalized_formula": row["normalized_formula"],
        }
        for row in rows
    )


def _decimal_text(value: Decimal) -> str:
    return format(value, "f")


def build_factor_bank(
    nodes: Sequence[LegacyNode],
) -> tuple[tuple[LegacyNode, ...], tuple[PBEOrderFactor, ...], tuple[dict[str, Any], ...]]:
    """Sort by exact decimal PBE gap/key and retain strict adjacent relations."""

    if not nodes:
        raise ValueError("At least one legacy node is required")
    if len({node.node_index for node in nodes}) != len(nodes):
        raise ValueError("Legacy node indices are not unique")
    if len({node.composition_key for node in nodes}) != len(nodes):
        raise ValueError("Legacy composition keys are not unique")
    ordered = tuple(sorted(nodes, key=lambda node: (node.pbe_band_gap, node.composition_key)))

    tie_groups: list[dict[str, Any]] = []
    group_start = 0
    while group_start < len(ordered):
        group_end = group_start + 1
        while (
            group_end < len(ordered)
            and ordered[group_end].pbe_band_gap == ordered[group_start].pbe_band_gap
        ):
            group_end += 1
        if group_end - group_start > 1:
            group = ordered[group_start:group_end]
            tie_groups.append(
                {
                    "pbe_band_gap_ev": _decimal_text(group[0].pbe_band_gap),
                    "size": len(group),
                    "sorted_positions": list(range(group_start, group_end)),
                    "node_indices": [node.node_index for node in group],
                    "composition_keys": [node.composition_key for node in group],
                }
            )
        group_start = group_end

    factors: list[PBEOrderFactor] = []
    for sorted_position_a, (node_a, node_b) in enumerate(zip(ordered, ordered[1:])):
        if node_b.pbe_band_gap == node_a.pbe_band_gap:
            continue
        if node_b.pbe_band_gap < node_a.pbe_band_gap:
            raise AssertionError("PBE ordering is not nondecreasing")
        factors.append(
            PBEOrderFactor(
                factor_index=len(factors),
                sorted_position_a=sorted_position_a,
                sorted_position_b=sorted_position_a + 1,
                node_a=node_a.node_index,
                node_b=node_b.node_index,
                composition_key_a=node_a.composition_key,
                composition_key_b=node_b.composition_key,
                normalized_formula_a=node_a.normalized_formula,
                normalized_formula_b=node_b.normalized_formula,
                pbe_band_gap_a_ev=node_a.pbe_band_gap_text,
                pbe_band_gap_b_ev=node_b.pbe_band_gap_text,
                pbe_gap_difference_ev=_decimal_text(
                    node_b.pbe_band_gap - node_a.pbe_band_gap
                ),
            )
        )
    return ordered, tuple(factors), tuple(tie_groups)


def factor_endpoint_array(factors: Sequence[PBEOrderFactor]) -> IntArray:
    endpoints = np.asarray([(factor.node_a, factor.node_b) for factor in factors], dtype=np.int64)
    if endpoints.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    return endpoints.reshape(-1, 2)


def factor_adjacency(node_count: int, endpoints: ArrayLike) -> sparse.csr_matrix:
    pairs = np.asarray(endpoints, dtype=np.int64)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("endpoints must have shape (n_factors, 2)")
    if pairs.size and (pairs.min() < 0 or pairs.max() >= node_count):
        raise ValueError("Factor endpoint is outside the node set")
    canonical = np.sort(pairs, axis=1)
    if np.any(canonical[:, 0] == canonical[:, 1]):
        raise ValueError("Factor endpoints must be distinct")
    if len({tuple(pair) for pair in canonical.tolist()}) != len(canonical):
        raise ValueError("Duplicate factor edge")
    rows = canonical.ravel()
    cols = canonical[:, ::-1].ravel()
    data = np.ones(rows.size, dtype=np.float64)
    result = sparse.csr_matrix((data, (rows, cols)), shape=(node_count, node_count))
    result.sum_duplicates()
    result.sort_indices()
    return result


def factor_graph_diagnostics(adjacency: sparse.spmatrix) -> dict[str, Any]:
    matrix = sparse.csr_matrix(adjacency, dtype=np.float64)
    difference = (matrix - matrix.T).tocsr()
    if difference.nnz:
        raise ValueError("Factor adjacency is not symmetric")
    if matrix.diagonal().any() or (matrix.nnz and np.any(matrix.data != 1.0)):
        raise ValueError("Factor adjacency is not binary with zero diagonal")
    degrees = np.asarray(matrix.sum(axis=1)).ravel().astype(np.int64)
    component_count, labels = csgraph.connected_components(matrix, directed=False)
    component_sizes = np.bincount(labels, minlength=component_count)
    nontrivial_sizes = component_sizes[component_sizes > 1]
    maximum_component_size = int(nontrivial_sizes.max()) if nontrivial_sizes.size else 1
    path_formula_norm = (
        float(2.0 * math.cos(math.pi / (maximum_component_size + 1)))
        if maximum_component_size > 1
        else 0.0
    )
    numerical_norm = (
        float(abs(eigsh(matrix, k=1, which="LM", return_eigenvectors=False, tol=1e-12)[0]))
        if matrix.nnz
        else 0.0
    )
    degree_distribution = {
        str(value): int(count)
        for value, count in zip(*np.unique(degrees, return_counts=True), strict=True)
    }
    return {
        "symmetric": True,
        "binary": True,
        "edge_count": int(matrix.nnz // 2),
        "maximum_degree": int(degrees.max(initial=0)),
        "degree_distribution": degree_distribution,
        "incident_node_count": int(np.count_nonzero(degrees)),
        "incident_node_fraction": float(np.count_nonzero(degrees) / matrix.shape[0]),
        "connected_component_count_including_isolates": int(component_count),
        "nontrivial_component_count": int(np.count_nonzero(component_sizes > 1)),
        "maximum_nontrivial_component_size": maximum_component_size,
        "path_subgraph": bool(degrees.max(initial=0) <= 2),
        "spectral_norm_path_formula": path_formula_norm,
        "spectral_norm_numerical": numerical_norm,
        "spectral_norm_formula_eigensolver_abs_difference": abs(
            path_formula_norm - numerical_norm
        ),
    }


def logistic_order_energy(local_values: ArrayLike) -> float:
    """Evaluate ``softplus(z_a - z_b)`` for the strict relation ``b > a``."""

    values = np.asarray(local_values, dtype=np.float64)
    if values.shape != (2,):
        raise ValueError("local_values must be [z_a, z_b]")
    return float(np.logaddexp(0.0, values[0] - values[1]))


def logistic_order_gradient(local_values: ArrayLike) -> FloatArray:
    values = np.asarray(local_values, dtype=np.float64)
    if values.shape != (2,):
        raise ValueError("local_values must be [z_a, z_b]")
    probability = float(expit(values[0] - values[1]))
    return np.asarray([probability, -probability], dtype=np.float64)


def logistic_order_hessian(local_values: ArrayLike) -> FloatArray:
    values = np.asarray(local_values, dtype=np.float64)
    if values.shape != (2,):
        raise ValueError("local_values must be [z_a, z_b]")
    probability = float(expit(values[0] - values[1]))
    curvature = probability * (1.0 - probability)
    return curvature * np.asarray([[1.0, -1.0], [-1.0, 1.0]])


def build_comparison_matrix(
    q0: sparse.spmatrix,
    factor_graph: sparse.spmatrix,
    mixed_curvature_bound: float = MAX_MIXED_CURVATURE,
) -> sparse.csr_matrix:
    reference = sparse.csr_matrix(q0, dtype=np.float64)
    adjacency = sparse.csr_matrix(factor_graph, dtype=np.float64)
    if reference.shape[0] != reference.shape[1] or adjacency.shape != reference.shape:
        raise ValueError("Q0 and factor graph must be square with equal shape")
    if mixed_curvature_bound < 0.0:
        raise ValueError("mixed_curvature_bound must be nonnegative")
    result = (reference - mixed_curvature_bound * adjacency).tocsr()
    result.eliminate_zeros()
    result.sort_indices()
    return result


def validate_comparison_matrix(
    q0: sparse.spmatrix,
    factor_graph: sparse.spmatrix,
    a0: sparse.spmatrix,
    *,
    q0_analytic_eigenvalue_floor: float,
    factor_graph_norm_upper_bound: float,
    symmetry_tolerance: float,
    sign_tolerance: float,
    eigenvalue_tolerance: float,
) -> dict[str, Any]:
    reference = sparse.csr_matrix(q0, dtype=np.float64)
    adjacency = sparse.csr_matrix(factor_graph, dtype=np.float64)
    comparison = sparse.csr_matrix(a0, dtype=np.float64)
    expected = build_comparison_matrix(reference, adjacency)
    identity_difference = (comparison - expected).tocsr()
    identity_error = (
        float(np.max(np.abs(identity_difference.data)))
        if identity_difference.nnz
        else 0.0
    )
    symmetry_difference = (comparison - comparison.T).tocsr()
    symmetry_error = (
        float(np.max(np.abs(symmetry_difference.data)))
        if symmetry_difference.nnz
        else 0.0
    )
    off_diagonal = comparison.copy()
    off_diagonal.setdiag(0.0)
    off_diagonal.eliminate_zeros()
    maximum_off_diagonal = (
        float(np.max(off_diagonal.data)) if off_diagonal.nnz else 0.0
    )
    minimum_eigenvalue = float(
        eigsh(
            comparison,
            k=1,
            which="SA",
            return_eigenvectors=False,
            tol=1e-12,
            maxiter=300_000,
        )[0]
    )
    analytic_floor = float(
        q0_analytic_eigenvalue_floor
        - MAX_MIXED_CURVATURE * factor_graph_norm_upper_bound
    )
    checks = {
        "identity_max_abs_error": identity_error,
        "symmetric": symmetry_error <= symmetry_tolerance,
        "symmetry_max_abs_error": symmetry_error,
        "off_diagonals_nonpositive": maximum_off_diagonal <= sign_tolerance,
        "maximum_off_diagonal": maximum_off_diagonal,
        "smallest_eigenvalue_numerical": minimum_eigenvalue,
        "positive_definite": minimum_eigenvalue > 0.0,
        "q0_analytic_smallest_eigenvalue_lower_bound": q0_analytic_eigenvalue_floor,
        "factor_graph_analytic_spectral_norm_upper_bound": factor_graph_norm_upper_bound,
        "analytic_smallest_eigenvalue_lower_bound": analytic_floor,
        "numerical_floor_check": minimum_eigenvalue >= analytic_floor - eigenvalue_tolerance,
        "nnz": int(comparison.nnz),
    }
    if identity_error != 0.0:
        raise ValueError(f"A0 identity error: {identity_error}")
    if not checks["symmetric"]:
        raise ValueError(f"A0 symmetry error: {symmetry_error}")
    if not checks["off_diagonals_nonpositive"]:
        raise ValueError(f"A0 has positive off-diagonal: {maximum_off_diagonal}")
    if analytic_floor < 0.5:
        raise ValueError(f"Analytic A0 eigenvalue floor is {analytic_floor} < 0.5")
    if not checks["positive_definite"]:
        raise ValueError(f"A0 is not SPD: lambda_min={minimum_eigenvalue}")
    if not checks["numerical_floor_check"]:
        raise ValueError(
            f"A0 numerical eigenvalue {minimum_eigenvalue} is below {analytic_floor}"
        )
    return checks


def factorize_sparse(matrix: sparse.spmatrix) -> tuple[Any, float]:
    started = time.perf_counter()
    factorization = splu(sparse.csc_matrix(matrix), permc_spec="COLAMD")
    return factorization, time.perf_counter() - started


def selected_inverse_rows(
    matrix: sparse.spmatrix,
    factorization: Any,
    selected_indices: ArrayLike,
) -> tuple[FloatArray, dict[str, float]]:
    values = sparse.csr_matrix(matrix, dtype=np.float64)
    indices = np.asarray(selected_indices, dtype=np.int64)
    if indices.ndim != 1 or len(set(indices.tolist())) != indices.size:
        raise ValueError("selected_indices must be a unique vector")
    if indices.size and (indices.min() < 0 or indices.max() >= values.shape[0]):
        raise ValueError("selected inverse index is outside the matrix")
    right_hand_sides = np.zeros((values.shape[0], indices.size), dtype=np.float64)
    right_hand_sides[indices, np.arange(indices.size)] = 1.0
    started = time.perf_counter()
    columns = np.asarray(factorization.solve(right_hand_sides), dtype=np.float64)
    solve_seconds = time.perf_counter() - started
    residual = values @ columns - right_hand_sides
    relative_residual = float(np.linalg.norm(residual) / np.linalg.norm(right_hand_sides))
    rows = columns.T
    selected_symmetry_error = float(
        np.max(np.abs(rows[:, indices] - rows[:, indices].T))
    )
    return rows, {
        "multiple_rhs_solve_seconds": solve_seconds,
        "relative_frobenius_residual": relative_residual,
        "selected_submatrix_symmetry_max_abs_error": selected_symmetry_error,
        "minimum_selected_inverse_entry": float(rows.min()),
    }


def factor_action_loads(selected_rows: ArrayLike, endpoints: ArrayLike) -> FloatArray:
    rows = np.asarray(selected_rows, dtype=np.float64)
    pairs = np.asarray(endpoints, dtype=np.int64)
    if rows.ndim != 2 or pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("Invalid selected rows or factor endpoints")
    if pairs.size and (pairs.min() < 0 or pairs.max() >= rows.shape[1]):
        raise ValueError("Factor endpoint is outside inverse rows")
    return np.asarray(rows[:, pairs[:, 0]] + rows[:, pairs[:, 1]], dtype=np.float64)


def factor_influence_contributions(
    selected_rows: ArrayLike,
    endpoints: ArrayLike,
    action_position: int,
    leader_position: int,
) -> FloatArray:
    """Return ``C[x,a]+C[x,b]+C[xhat,a]+C[xhat,b]`` for all factors."""

    loads = factor_action_loads(selected_rows, endpoints)
    if not (0 <= action_position < loads.shape[0]) or not (
        0 <= leader_position < loads.shape[0]
    ):
        raise ValueError("Action position is outside selected inverse rows")
    return np.asarray(loads[action_position] + loads[leader_position], dtype=np.float64)


def reference_graph_distances(
    q0: sparse.spmatrix, action_node_indices: ArrayLike
) -> IntArray:
    reference = sparse.csr_matrix(q0, dtype=np.float64)
    graph = reference.copy()
    graph.setdiag(0.0)
    graph.eliminate_zeros()
    graph.data = np.ones_like(graph.data)
    distances = csgraph.shortest_path(
        graph,
        directed=False,
        unweighted=True,
        indices=np.asarray(action_node_indices, dtype=np.int64),
    )
    if not np.all(np.isfinite(distances)):
        raise ValueError("Reference graph is disconnected from an action")
    return np.asarray(distances, dtype=np.int64)


def _set_digest(indices: IntArray) -> str:
    canonical = np.sort(np.asarray(indices, dtype="<i8"))
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def _quantile_summary(values: FloatArray) -> dict[str, float]:
    return {
        "q25": float(np.quantile(values, 0.25)),
        "median": float(np.quantile(values, 0.50)),
        "q75": float(np.quantile(values, 0.75)),
        "p90": float(np.quantile(values, 0.90)),
    }


def summarize_pair_influence(
    selected_rows: ArrayLike,
    endpoints: ArrayLike,
    actions: Sequence[Mapping[str, Any]],
    thresholds: Sequence[float],
    action_to_node_distances: ArrayLike,
    *,
    nonnegative_tolerance: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = np.asarray(selected_rows, dtype=np.float64)
    pairs = np.asarray(endpoints, dtype=np.int64)
    distances = np.asarray(action_to_node_distances, dtype=np.int64)
    threshold_values = tuple(float(value) for value in thresholds)
    if not threshold_values or any(value <= 0.0 or value >= 1.0 for value in threshold_values):
        raise ValueError("Influence thresholds must lie strictly between zero and one")
    if rows.shape[0] != len(actions) or distances.shape != rows.shape:
        raise ValueError("Action rows, metadata, and graph distances do not align")
    loads = factor_action_loads(rows, pairs)
    minimum_load = float(loads.min())
    if minimum_load < -nonnegative_tolerance:
        raise ValueError(f"Selected inverse factor load is negative: {minimum_load}")
    loads = np.maximum(loads, 0.0)

    pair_rows: list[dict[str, Any]] = []
    fractions: dict[float, list[float]] = {value: [] for value in threshold_values}
    set_digests: dict[float, Counter[str]] = {
        value: Counter() for value in threshold_values
    }
    top_factor_counts: Counter[int] = Counter()
    graph_distance_counts: Counter[int] = Counter()
    for left in range(len(actions) - 1):
        for right in range(left + 1, len(actions)):
            contributions = loads[left] + loads[right]
            total = float(contributions.sum())
            if not math.isfinite(total) or total <= 0.0:
                raise ValueError(f"Nonpositive total influence for action pair {left}, {right}")
            order = np.lexsort(
                (np.arange(contributions.size, dtype=np.int64), -contributions)
            )
            cumulative = np.cumsum(contributions[order])
            top_factor = int(order[0])
            top_factor_counts[top_factor] += 1
            endpoint_a, endpoint_b = pairs[top_factor]
            graph_distance = int(
                min(
                    distances[left, endpoint_a],
                    distances[left, endpoint_b],
                    distances[right, endpoint_a],
                    distances[right, endpoint_b],
                )
            )
            graph_distance_counts[graph_distance] += 1
            record: dict[str, Any] = {
                "action_position_x": left,
                "action_position_xhat": right,
                "action_key_x": actions[left]["action_key"],
                "action_key_xhat": actions[right]["action_key"],
                "node_x": int(actions[left]["node_index"]),
                "node_xhat": int(actions[right]["node_index"]),
                "total_structural_influence": total,
                "highest_influence_factor_index": top_factor,
                "highest_influence_factor_graph_distance": graph_distance,
            }
            for threshold in threshold_values:
                count = int(np.searchsorted(cumulative, threshold * total, side="left") + 1)
                fraction = float(count / contributions.size)
                selected = order[:count]
                digest = _set_digest(selected)
                label = f"{int(round(100 * threshold))}"
                record[f"factors_for_{label}_percent"] = count
                record[f"factor_fraction_for_{label}_percent"] = fraction
                record[f"factor_set_sha256_for_{label}_percent"] = digest
                fractions[threshold].append(fraction)
                set_digests[threshold][digest] += 1
            pair_rows.append(record)

    factor_fraction_summary: dict[str, Any] = {}
    factor_set_change_summary: dict[str, Any] = {}
    for threshold in threshold_values:
        label = f"{int(round(100 * threshold))}_percent"
        values = np.asarray(fractions[threshold], dtype=np.float64)
        counts = set_digests[threshold]
        mode_digest, mode_count = counts.most_common(1)[0]
        factor_fraction_summary[label] = _quantile_summary(values)
        factor_set_change_summary[label] = {
            "distinct_factor_sets": len(counts),
            "pair_count": len(pair_rows),
            "distinct_set_fraction": float(len(counts) / len(pair_rows)),
            "modal_set_sha256": mode_digest,
            "modal_set_pair_count": mode_count,
            "modal_set_pair_fraction": float(mode_count / len(pair_rows)),
        }
    top_factor, top_factor_mode_count = top_factor_counts.most_common(1)[0]
    graph_distance_values = np.repeat(
        np.asarray(list(graph_distance_counts.keys()), dtype=np.int64),
        np.asarray(list(graph_distance_counts.values()), dtype=np.int64),
    ).astype(np.float64)
    return pair_rows, {
        "unordered_action_pair_count": len(pair_rows),
        "factor_count": int(pairs.shape[0]),
        "factor_fraction_required": factor_fraction_summary,
        "top_factor_set_variation": factor_set_change_summary,
        "highest_influence_factor_variation": {
            "distinct_factor_count": len(top_factor_counts),
            "modal_factor_index": top_factor,
            "modal_factor_pair_count": top_factor_mode_count,
            "modal_factor_pair_fraction": float(top_factor_mode_count / len(pair_rows)),
        },
        "highest_influence_factor_reference_graph_distance": {
            **_quantile_summary(graph_distance_values),
            "min": int(graph_distance_values.min()),
            "max": int(graph_distance_values.max()),
            "histogram": {
                str(distance): count
                for distance, count in sorted(graph_distance_counts.items())
            },
        },
        "minimum_unclipped_action_factor_load": minimum_load,
        "set_definition": (
            "minimal deterministically ranked factor-index set reaching the stated "
            "fraction; ties break by factor index and sets are hashed after index sorting"
        ),
        "distance_definition": (
            "minimum unweighted committed reference-graph distance from either action "
            "node to either endpoint of the single highest-ranked factor"
        ),
    }


def observation_pattern_diagnostics(
    a0: sparse.spmatrix,
    a0_factorization: Any,
    patterns: Sequence[Mapping[str, Any]],
    probe_indices: ArrayLike,
    *,
    eigenvalue_tolerance: float,
    identity_tolerance: float,
    nonnegative_tolerance: float,
    factor_mixed_curvature_bound: float = MAX_MIXED_CURVATURE,
) -> list[dict[str, Any]]:
    base = sparse.csr_matrix(a0, dtype=np.float64)
    probes = np.asarray(probe_indices, dtype=np.int64)
    rhs = np.zeros((base.shape[0], probes.size), dtype=np.float64)
    rhs[probes, np.arange(probes.size)] = 1.0
    base_columns = np.asarray(a0_factorization.solve(rhs), dtype=np.float64)
    diagnostics: list[dict[str, Any]] = []
    for pattern in patterns:
        indices = np.asarray(pattern["node_indices"], dtype=np.int64)
        precision = float(pattern["diagonal_precision"])
        if indices.ndim != 1 or len(set(indices.tolist())) != indices.size:
            raise ValueError(f"Invalid observation indices for {pattern['name']}")
        if indices.size and (indices.min() < 0 or indices.max() >= base.shape[0]):
            raise ValueError(f"Observation index outside A0 for {pattern['name']}")
        if precision < 0.0:
            raise ValueError("Observation diagonal precision must be nonnegative")
        diagonal = np.zeros(base.shape[0], dtype=np.float64)
        diagonal[indices] = precision
        at = (base + sparse.diags(diagonal, format="csr")).tocsr()
        minimum_eigenvalue = float(
            eigsh(at, k=1, which="SA", return_eigenvectors=False, tol=1e-11)[0]
        )
        at_factorization = splu(at.tocsc(), permc_spec="COLAMD")
        updated_columns = np.asarray(at_factorization.solve(rhs), dtype=np.float64)
        identity_right = np.asarray(
            a0_factorization.solve(diagonal[:, None] * updated_columns),
            dtype=np.float64,
        )
        identity_left = base_columns - updated_columns
        identity_error = float(np.max(np.abs(identity_left - identity_right)))
        minimum_base = float(base_columns.min())
        minimum_updated = float(updated_columns.min())
        minimum_difference = float(identity_left.min())
        if minimum_eigenvalue <= 0.0:
            raise ValueError(f"At is not SPD for {pattern['name']}")
        if minimum_eigenvalue < float(pattern["a0_smallest_eigenvalue"]) - eigenvalue_tolerance:
            raise ValueError(f"At eigenvalue monotonicity failed for {pattern['name']}")
        if identity_error > identity_tolerance:
            raise ValueError(f"Resolvent identity failed for {pattern['name']}: {identity_error}")
        if min(minimum_base, minimum_updated, minimum_difference) < -nonnegative_tolerance:
            raise ValueError(f"M-matrix inverse monotonicity failed for {pattern['name']}")
        diagnostics.append(
            {
                "name": pattern["name"],
                "observation_count": int(indices.size),
                "diagonal_precision": precision,
                "smallest_eigenvalue": minimum_eigenvalue,
                "positive_definite": True,
                "factor_mixed_curvature_bound": factor_mixed_curvature_bound,
                "same_factor_hessian_bound": True,
                "probe_indices": probes.tolist(),
                "resolvent_identity_max_abs_error": identity_error,
                "minimum_a0_inverse_probe_entry": minimum_base,
                "minimum_at_inverse_probe_entry": minimum_updated,
                "minimum_a0_minus_at_inverse_probe_entry": minimum_difference,
                "sampled_inverse_columns_entrywise_nonnegative": True,
                "sampled_inverse_monotonicity": True,
                "dense_inverse_formed": False,
            }
        )
    return diagnostics


def factor_rows_for_csv(factors: Sequence[PBEOrderFactor]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for factor in factors:
        row = asdict(factor)
        row.update(
            {
                "factor_bank": FACTOR_BANK_NAME,
                "temperature": TEMPERATURE,
                "strict_relation": "z_b > z_a",
                "energy": "log(1 + exp(-(z_b - z_a)))",
            }
        )
        rows.append(row)
    return rows


__all__ = [
    "FACTOR_BANK_NAME",
    "MAX_MIXED_CURVATURE",
    "TEMPERATURE",
    "LegacyNode",
    "PBEOrderFactor",
    "build_comparison_matrix",
    "build_factor_bank",
    "factor_action_loads",
    "factor_adjacency",
    "factor_endpoint_array",
    "factor_graph_diagnostics",
    "factor_influence_contributions",
    "factor_rows_for_csv",
    "factorize_sparse",
    "load_action_mapping",
    "load_legacy_nodes",
    "logistic_order_energy",
    "logistic_order_gradient",
    "logistic_order_hessian",
    "observation_pattern_diagnostics",
    "reference_graph_distances",
    "selected_inverse_rows",
    "sha256_file",
    "summarize_pair_influence",
    "validate_comparison_matrix",
]
