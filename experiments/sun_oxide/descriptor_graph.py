#!/usr/bin/env python3
"""Descriptor and graph-Gaussian compatibility gate for the frozen oxide benchmark.

The module consumes composition keys and normalized formulas only. It contains
no preference-factor, influence, inference, or Bayesian-optimization code and
has no access path to GW target values.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import io
import json
import math
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
from typing import Any, Iterable, Sequence
import zipfile

import numpy as np
from packaging.requirements import Requirement
from scipy import sparse
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh, spsolve
from scipy.spatial.distance import pdist, squareform


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "experiments/sun_oxide/configs/descriptor_graph.json"
DEFAULT_OUTPUT = ROOT / "experiments/sun_oxide/outputs/descriptor_graph"
EXPECTED_OUTPUTS = [
    "RESULTS.md",
    "descriptor_matrix.npz",
    "descriptor_manifest.json",
    "graph_edges.csv",
    "graph_summary.json",
    "q0_sparse.npz",
    "action_node_map.csv",
    "environment_freeze.txt",
    "run_summary.json",
]
SAFE_BENCHMARK_HASH_CHECKS = [
    "current_nlr_legacy.csv",
    "current_nlr_gw_actions.csv",
    "NLR_DATA_USE_NOTICE.txt",
]


class GateFailure(RuntimeError):
    def __init__(self, verdict: str, message: str) -> None:
        super().__init__(message)
        self.verdict = verdict


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def write_json(path: Path, value: Any) -> None:
    path.write_bytes(canonical_json_bytes(value))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_string_sequence_sha256(values: Sequence[str]) -> str:
    return sha256_bytes(("\n".join(values) + "\n").encode("utf-8"))


def descriptor_matrix_sha256(matrix: np.ndarray) -> str:
    canonical = np.ascontiguousarray(matrix, dtype="<f8")
    return sha256_bytes(canonical.tobytes(order="C"))


def _npy_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.lib.format.write_array(buffer, np.asarray(array), allow_pickle=False)
    return buffer.getvalue()


def write_deterministic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Write a stable, pickle-free NPZ with fixed entry timestamps."""
    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in sorted(arrays):
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
                raise ValueError(f"Unsafe NPZ array name: {name!r}")
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, _npy_bytes(np.asarray(arrays[name])), compress_type=zipfile.ZIP_DEFLATED)


def write_sparse_npz(path: Path, matrix: sparse.spmatrix) -> None:
    csr = sparse.csr_matrix(matrix)
    csr.sort_indices()
    write_deterministic_npz(
        path,
        {
            "data": csr.data,
            "format": np.asarray(b"csr"),
            "indices": csr.indices,
            "indptr": csr.indptr,
            "shape": np.asarray(csr.shape, dtype=np.int64),
        },
    )


def _read_csv_columns(path: Path, required: Sequence[str]) -> list[dict[str, str]]:
    """Read and retain only explicitly permitted columns from a CSV."""
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"Empty CSV: {path}") from exc
        missing = [name for name in required if name not in header]
        if missing:
            raise ValueError(f"Missing required columns in {path}: {missing}")
        indices = [header.index(name) for name in required]
        rows: list[dict[str, str]] = []
        for line_number, raw in enumerate(reader, start=2):
            if len(raw) != len(header):
                raise ValueError(f"Malformed row {line_number} in {path}")
            rows.append({name: raw[index] for name, index in zip(required, indices, strict=True)})
        return rows


def load_legacy_nodes(path: Path, expected_rows: int) -> list[dict[str, str]]:
    rows = _read_csv_columns(path, ["composition_key", "normalized_formula"])
    if len(rows) != expected_rows:
        raise ValueError(f"Legacy row count {len(rows)} != {expected_rows}")
    keys = [row["composition_key"] for row in rows]
    formulas = [row["normalized_formula"] for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("Legacy composition keys are not unique")
    if len(set(formulas)) != len(formulas):
        raise ValueError("Legacy normalized formulas are not unique")
    if any(not key or not formula for key, formula in zip(keys, formulas, strict=True)):
        raise ValueError("Blank legacy key or normalized formula")
    return rows


def verify_frozen_benchmark(repository_root: Path, config: dict[str, Any]) -> dict[str, Any]:
    benchmark = config["benchmark"]
    manifest_path = repository_root / benchmark["manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["benchmark_name"] != benchmark["name"]:
        raise ValueError(f"Unexpected benchmark name: {manifest['benchmark_name']}")
    if manifest["counts"]["legacy_compositions"] != benchmark["expected_legacy_rows"]:
        raise ValueError("Frozen legacy count changed")
    if manifest["counts"]["strict_gw_actions"] != benchmark["expected_actions"]:
        raise ValueError("Frozen action count changed")
    benchmark_dir = manifest_path.parent
    checked: dict[str, str] = {}
    for name in SAFE_BENCHMARK_HASH_CHECKS:
        path = benchmark_dir / name
        observed = sha256_file(path)
        expected = manifest["artifacts"][name]["sha256"]
        if observed != expected:
            raise ValueError(f"Frozen benchmark artifact hash mismatch: {name}")
        checked[name] = observed
    return {
        "benchmark_name": manifest["benchmark_name"],
        "legacy_rows": manifest["counts"]["legacy_compositions"],
        "actions": manifest["counts"]["strict_gw_actions"],
        "manifest_sha256": sha256_file(manifest_path),
        "checked_artifact_sha256": checked,
        "gw_values_read": False,
    }


def validate_feature_labels(labels: Sequence[str], expected_count: int) -> list[str]:
    normalized = [str(label) for label in labels]
    if len(normalized) != expected_count:
        raise ValueError(f"Magpie feature count {len(normalized)} != {expected_count}")
    if len(set(normalized)) != len(normalized):
        raise ValueError("Magpie feature labels are not unique")
    if any(not label for label in normalized):
        raise ValueError("Blank Magpie feature label")
    return normalized


def featurize_compositions(
    formulas: Sequence[str], expected_feature_count: int, *, progress_every: int = 100
) -> tuple[np.ndarray, list[str]]:
    """Generate raw Magpie descriptors in committed row order."""
    from matminer.featurizers.composition import ElementProperty
    from pymatgen.core import Composition

    featurizer = ElementProperty.from_preset(preset_name="magpie", impute_nan=True)
    labels = validate_feature_labels(featurizer.feature_labels(), expected_feature_count)
    matrix = np.empty((len(formulas), expected_feature_count), dtype=np.float64)
    for index, formula in enumerate(formulas):
        values = np.asarray(featurizer.featurize(Composition(formula)), dtype=np.float64)
        if values.shape != (expected_feature_count,):
            raise ValueError(f"Descriptor shape for row {index} is {values.shape}")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Non-finite descriptor value at row {index}")
        matrix[index] = values
        if (index + 1) % progress_every == 0 or index + 1 == len(formulas):
            print(f"DESCRIPTOR_PROGRESS {index + 1}/{len(formulas)}", flush=True)
    return matrix, labels


def standardize_for_graph(matrix: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    standardized = np.asarray(scaler.fit_transform(matrix), dtype=np.float64)
    if standardized.shape != matrix.shape or not np.all(np.isfinite(standardized)):
        raise ValueError("StandardScaler produced invalid graph descriptors")
    return standardized, {
        "class": "sklearn.preprocessing.StandardScaler",
        "with_mean": bool(scaler.with_mean),
        "with_std": bool(scaler.with_std),
        "mean_sha256": descriptor_matrix_sha256(np.asarray(scaler.mean_, dtype=np.float64)),
        "scale_sha256": descriptor_matrix_sha256(np.asarray(scaler.scale_, dtype=np.float64)),
    }


def pairwise_euclidean(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] < 2 or not np.all(np.isfinite(points)):
        raise ValueError("Invalid points for Euclidean graph")
    distances = squareform(pdist(points, metric="euclidean"))
    if not np.all(np.isfinite(distances)) or np.any(distances < 0):
        raise ValueError("Invalid Euclidean distance matrix")
    return distances


def _key_ranks(composition_keys: Sequence[str]) -> np.ndarray:
    if len(set(composition_keys)) != len(composition_keys):
        raise ValueError("Graph composition keys are not unique")
    ranks = np.empty(len(composition_keys), dtype=np.int64)
    for rank, index in enumerate(sorted(range(len(composition_keys)), key=lambda i: composition_keys[i])):
        ranks[index] = rank
    return ranks


def deterministic_knn_edges(
    distances: np.ndarray, composition_keys: Sequence[str], k: int
) -> tuple[set[tuple[int, int]], dict[str, int]]:
    distances = np.asarray(distances, dtype=np.float64)
    node_count = len(composition_keys)
    if distances.shape != (node_count, node_count):
        raise ValueError("Distance matrix shape does not match graph keys")
    if not np.allclose(distances, distances.T, rtol=0.0, atol=1e-12):
        raise ValueError("Distance matrix is not symmetric")
    if np.max(np.abs(np.diag(distances))) > 1e-12:
        raise ValueError("Distance matrix diagonal is not zero")
    if not 1 <= k < node_count:
        raise ValueError(f"Invalid k={k} for {node_count} nodes")
    ranks = _key_ranks(composition_keys)
    all_indices = np.arange(node_count, dtype=np.int64)
    edges: set[tuple[int, int]] = set()
    tie_adjacencies = 0
    boundary_tie_nodes = 0
    boundary_unselected_ties = 0
    for node in range(node_count):
        candidates = all_indices[all_indices != node]
        candidate_distances = distances[node, candidates]
        order = np.lexsort((ranks[candidates], candidate_distances))
        sorted_candidates = candidates[order]
        sorted_distances = candidate_distances[order]
        tie_adjacencies += int(np.count_nonzero(sorted_distances[1:] == sorted_distances[:-1]))
        boundary_distance = sorted_distances[k - 1]
        equal_boundary = int(np.count_nonzero(candidate_distances == boundary_distance))
        selected_equal_boundary = int(np.count_nonzero(sorted_distances[:k] == boundary_distance))
        if equal_boundary > 1:
            boundary_tie_nodes += 1
            boundary_unselected_ties += equal_boundary - selected_equal_boundary
        for neighbor in sorted_candidates[:k]:
            left, right = sorted((node, int(neighbor)))
            edges.add((left, right))
    return edges, {
        "directed_choice_count": node_count * k,
        "undirected_edge_count": len(edges),
        "exact_distance_tie_adjacencies": tie_adjacencies,
        "k_boundary_tie_nodes": boundary_tie_nodes,
        "k_boundary_unselected_tied_candidates": boundary_unselected_ties,
    }


class _DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            parent = self.parent[value]
            self.parent[value] = root
            value = parent
        return root

    def union(self, left: int, right: int) -> bool:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return False
        low, high = sorted((root_left, root_right))
        self.parent[high] = low
        return True


def deterministic_mst_edges(
    distances: np.ndarray, composition_keys: Sequence[str]
) -> tuple[set[tuple[int, int]], dict[str, int]]:
    """Kruskal MST with explicit distance/key lexicographic ordering."""
    distances = np.asarray(distances, dtype=np.float64)
    node_count = len(composition_keys)
    if distances.shape != (node_count, node_count):
        raise ValueError("Distance matrix shape does not match MST keys")
    ranks = _key_ranks(composition_keys)
    left, right = np.triu_indices(node_count, k=1)
    weights = distances[left, right]
    left_rank = ranks[left]
    right_rank = ranks[right]
    first_rank = np.minimum(left_rank, right_rank)
    second_rank = np.maximum(left_rank, right_rank)
    order = np.lexsort((second_rank, first_rank, weights))
    sorted_weights = weights[order]
    tie_adjacencies = int(np.count_nonzero(sorted_weights[1:] == sorted_weights[:-1]))
    forest = _DisjointSet(node_count)
    edges: set[tuple[int, int]] = set()
    scanned = 0
    for edge_index in order:
        scanned += 1
        node_left = int(left[edge_index])
        node_right = int(right[edge_index])
        if forest.union(node_left, node_right):
            edges.add((node_left, node_right))
            if len(edges) == node_count - 1:
                break
    if len(edges) != node_count - 1:
        raise ValueError(f"MST has {len(edges)} edges for {node_count} nodes")
    return edges, {
        "edge_count": len(edges),
        "complete_candidate_edge_count": len(weights),
        "candidate_edges_scanned": scanned,
        "exact_distance_tie_adjacencies": tie_adjacencies,
    }


def adjacency_from_edges(node_count: int, edges: Iterable[tuple[int, int]]) -> sparse.csr_matrix:
    ordered = sorted(set(edges))
    if any(left < 0 or right >= node_count or left >= right for left, right in ordered):
        raise ValueError("Invalid undirected graph edge")
    rows = np.asarray([node for edge in ordered for node in edge], dtype=np.int64)
    cols = np.asarray([node for left, right in ordered for node in (right, left)], dtype=np.int64)
    data = np.ones(len(rows), dtype=np.float64)
    adjacency = sparse.csr_matrix((data, (rows, cols)), shape=(node_count, node_count))
    adjacency.sum_duplicates()
    adjacency.sort_indices()
    if adjacency.nnz != 2 * len(ordered) or np.any(adjacency.data != 1.0):
        raise ValueError("Adjacency is not binary and symmetric by construction")
    return adjacency


def graph_summary(
    adjacency: sparse.csr_matrix,
    *,
    knn_edges: set[tuple[int, int]],
    mst_edges: set[tuple[int, int]],
    knn_ties: dict[str, int],
    mst_ties: dict[str, int],
) -> dict[str, Any]:
    components, labels = csgraph.connected_components(adjacency, directed=False, return_labels=True)
    degrees = np.asarray(adjacency.sum(axis=1)).ravel().astype(np.int64)
    if labels.shape != (adjacency.shape[0],):
        raise ValueError("Invalid connected-component labels")
    final_edges = set(knn_edges) | set(mst_edges)
    return {
        "node_count": adjacency.shape[0],
        "knn_directed_choice_count": knn_ties["directed_choice_count"],
        "knn_edge_count": len(knn_edges),
        "mst_edge_count": len(mst_edges),
        "mst_edges_already_in_knn": len(knn_edges & mst_edges),
        "final_unique_edge_count": len(final_edges),
        "connected_component_count": int(components),
        "isolated_node_count": int(np.count_nonzero(degrees == 0)),
        "degree": {
            "min": int(degrees.min()),
            "median": float(np.median(degrees)),
            "mean": float(np.mean(degrees)),
            "max": int(degrees.max()),
        },
        "exact_distance_ties_encountered": (
            knn_ties["exact_distance_tie_adjacencies"]
            + mst_ties["exact_distance_tie_adjacencies"]
        ),
        "knn_tie_diagnostics": knn_ties,
        "mst_tie_diagnostics": mst_ties,
    }


def normalized_laplacian_reference(adjacency: sparse.csr_matrix) -> sparse.csr_matrix:
    adjacency = sparse.csr_matrix(adjacency, dtype=np.float64)
    degrees = np.asarray(adjacency.sum(axis=1)).ravel()
    if np.any(degrees <= 0):
        raise ValueError("Cannot construct normalized Laplacian with isolated nodes")
    inv_sqrt = 1.0 / np.sqrt(degrees)
    normalized_adjacency = sparse.diags(inv_sqrt) @ adjacency @ sparse.diags(inv_sqrt)
    identity = sparse.eye(adjacency.shape[0], format="csr", dtype=np.float64)
    q0 = (2.0 * identity - normalized_adjacency).tocsr()
    q0.eliminate_zeros()
    q0.sort_indices()
    return q0


def validate_q0(q0: sparse.csr_matrix, config: dict[str, Any]) -> dict[str, Any]:
    reference = config["reference"]
    difference = (q0 - q0.T).tocsr()
    symmetry_error = float(np.max(np.abs(difference.data))) if difference.nnz else 0.0
    if symmetry_error > reference["symmetry_tolerance"]:
        raise ValueError(f"Q0 symmetry error {symmetry_error}")
    off_diagonal = q0.copy()
    off_diagonal.setdiag(0.0)
    off_diagonal.eliminate_zeros()
    max_off_diagonal = float(np.max(off_diagonal.data)) if off_diagonal.nnz else 0.0
    if max_off_diagonal > reference["sign_tolerance"]:
        raise ValueError(f"Positive Q0 off-diagonal {max_off_diagonal}")
    smallest = float(
        eigsh(q0, k=1, which="SA", return_eigenvectors=False, tol=1e-12, maxiter=200000)[0]
    )
    largest = float(
        eigsh(q0, k=1, which="LA", return_eigenvectors=False, tol=1e-12, maxiter=200000)[0]
    )
    tolerance = reference["eigenvalue_tolerance"]
    if smallest < reference["smallest_eigenvalue_lower_bound"] - tolerance:
        raise ValueError(f"Q0 smallest eigenvalue {smallest} violates lower bound")
    if largest > reference["largest_eigenvalue_upper_bound"] + tolerance:
        raise ValueError(f"Q0 largest eigenvalue {largest} violates upper bound")
    if smallest <= 0:
        raise ValueError(f"Q0 is not positive definite: smallest eigenvalue {smallest}")
    return {
        "shape": list(q0.shape),
        "nnz": int(q0.nnz),
        "symmetric": True,
        "symmetry_max_abs_error": symmetry_error,
        "off_diagonal_nonpositive": True,
        "max_off_diagonal": max_off_diagonal,
        "smallest_eigenvalue": smallest,
        "largest_eigenvalue": largest,
        "positive_definite": True,
        "dense_inverse_formed": False,
    }


def deterministic_right_hand_sides(node_count: int) -> list[tuple[str, np.ndarray]]:
    index = np.arange(node_count, dtype=np.float64)
    return [
        ("ones", np.ones(node_count, dtype=np.float64)),
        ("linear_minus_one_to_one", np.linspace(-1.0, 1.0, node_count, dtype=np.float64)),
        ("sinusoid_index_times_sqrt_two", np.sin(index * math.sqrt(2.0))),
    ]


def run_sparse_solves(q0: sparse.csr_matrix, relative_residual_max: float) -> list[dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    for name, rhs in deterministic_right_hand_sides(q0.shape[0]):
        started = time.perf_counter()
        solution = spsolve(q0, rhs, permc_spec="COLAMD", use_umfpack=False)
        wall_seconds = time.perf_counter() - started
        residual = q0 @ solution - rhs
        relative = float(np.linalg.norm(residual) / np.linalg.norm(rhs))
        if not math.isfinite(relative) or relative > relative_residual_max:
            raise ValueError(f"Sparse solve {name} residual {relative}")
        diagnostics.append(
            {
                "rhs": name,
                "relative_residual": relative,
                "wall_seconds": wall_seconds,
            }
        )
    return diagnostics


def map_actions_to_nodes(
    action_path: Path,
    legacy_rows: Sequence[dict[str, str]],
    expected_actions: int,
) -> list[dict[str, Any]]:
    actions = _read_csv_columns(action_path, ["action_key", "composition_key", "normalized_formula"])
    if len(actions) != expected_actions:
        raise ValueError(f"Action count {len(actions)} != {expected_actions}")
    action_keys = [row["action_key"] for row in actions]
    composition_keys = [row["composition_key"] for row in actions]
    if len(set(action_keys)) != len(actions):
        raise ValueError("GW action keys are not unique")
    if len(set(composition_keys)) != len(actions):
        raise ValueError("GW action composition keys are not unique")
    node_by_key = {row["composition_key"]: index for index, row in enumerate(legacy_rows)}
    formula_by_key = {row["composition_key"]: row["normalized_formula"] for row in legacy_rows}
    mapped: list[dict[str, Any]] = []
    for action in actions:
        key = action["composition_key"]
        if key not in node_by_key:
            raise ValueError(f"Action composition key absent from legacy nodes: {key}")
        if action["normalized_formula"] != formula_by_key[key]:
            raise ValueError(f"Action formula mismatch for composition key {key}")
        mapped.append(
            {
                "action_key": action["action_key"],
                "composition_key": key,
                "node_index": node_by_key[key],
                "normalized_formula": action["normalized_formula"],
            }
        )
    return mapped


def _write_csv(path: Path, rows: Sequence[dict[str, Any]], columns: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(columns), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _locked_packages(lock_path: Path) -> dict[str, str]:
    packages: dict[str, str] = {}
    for line in lock_path.read_text(encoding="utf-8").splitlines():
        if not re.match(r"^[A-Za-z0-9_.-]+==", line):
            continue
        requirement = Requirement(line.rstrip(" \\"))
        if requirement.marker is not None and not requirement.marker.evaluate():
            continue
        specifier = str(requirement.specifier)
        if not specifier.startswith("==") or "," in specifier:
            raise ValueError(f"Requirement is not exactly pinned: {line}")
        packages[requirement.name] = specifier[2:]
    if not packages:
        raise ValueError("No pinned packages found in requirements lock")
    return packages


def environment_record(repository_root: Path, config: dict[str, Any]) -> tuple[dict[str, Any], str]:
    environment = config["environment"]
    if f"{sys.version_info.major}.{sys.version_info.minor}" != environment["python_major_minor"]:
        raise ValueError(f"Python {sys.version.split()[0]} is not the frozen Python 3.12 environment")
    lock_path = repository_root / environment["requirements_lock"]
    locked = _locked_packages(lock_path)
    installed = {name: importlib.metadata.version(name) for name in locked}
    mismatches = {
        name: {"locked": version, "installed": installed[name]}
        for name, version in locked.items()
        if installed[name] != version
    }
    if mismatches:
        raise ValueError(f"Installed environment differs from lock: {mismatches}")
    for name, expected in environment["required_versions"].items():
        if installed.get(name) != expected:
            raise ValueError(f"Required version mismatch for {name}: {installed.get(name)} != {expected}")
    record = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "requirements_lock_sha256": sha256_file(lock_path),
        "packages": dict(sorted(installed.items())),
    }
    lines = [f"python=={record['python']}", f"platform={record['platform']}"]
    lines.extend(f"{name}=={version}" for name, version in sorted(installed.items()))
    lines.append(f"requirements_lock_sha256={record['requirements_lock_sha256']}")
    return record, "\n".join(lines) + "\n"


def _current_git_sha(repository_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _validate_run_sha(run_sha: str, repository_root: Path) -> None:
    if not re.fullmatch(r"[0-9a-f]{40}", run_sha):
        raise ValueError(f"Invalid RUN_SHA: {run_sha!r}")
    observed = _current_git_sha(repository_root)
    if observed != run_sha:
        raise ValueError(f"RUN_SHA {run_sha} != checked-out HEAD {observed}")


def _graph_edge_rows(
    edges: set[tuple[int, int]],
    knn_edges: set[tuple[int, int]],
    mst_edges: set[tuple[int, int]],
    composition_keys: Sequence[str],
) -> list[dict[str, Any]]:
    return [
        {
            "node_i": left,
            "node_j": right,
            "composition_key_i": composition_keys[left],
            "composition_key_j": composition_keys[right],
            "in_knn": int((left, right) in knn_edges),
            "in_mst": int((left, right) in mst_edges),
        }
        for left, right in sorted(edges)
    ]


def _results_markdown(run_summary: dict[str, Any]) -> str:
    descriptor = run_summary["descriptor"]
    graph = run_summary["graph"]
    reference = run_summary["reference"]
    solve_lines = "\n".join(
        f"- `{item['rhs']}`: residual `{item['relative_residual']}`, wall time `{item['wall_seconds']}` s."
        for item in reference["sparse_solves"]
    )
    return f"""# CURRENT_NLR_PBE_GW_V1 descriptor/graph Colab result

Colab verdict: `PASS_DESCRIPTOR_GRAPH_COLAB`

This is a pending compatibility-gate result awaiting independent ZIP
verification. It is not a final `PASS_DESCRIPTOR_GRAPH` record and contains no
preference factors, influence calculation, inference, or BO result.

## Provenance and scope

- RUN_SHA: `{run_summary['run_sha']}`
- Frozen benchmark: `{run_summary['benchmark']['benchmark_name']}`
- Legacy/action counts: `{run_summary['benchmark']['legacy_rows']}` / `{run_summary['actions']['mapped_actions']}`
- GW values read: `{run_summary['target_isolation']['gw_values_read']}`

## Descriptor compatibility

- Raw descriptor shape: `{descriptor['shape']}`
- Non-finite values: `{descriptor['nonfinite_count']}`
- Zero-variance features: `{descriptor['zero_variance_feature_count']}`
- Matrix SHA-256: `{descriptor['matrix_sha256']}`
- Row-key SHA-256: `{descriptor['row_key_sha256']}`

## Graph compatibility

- 10-NN union edges: `{graph['knn_edge_count']}`
- MST edges: `{graph['mst_edge_count']}`
- Final unique edges: `{graph['final_unique_edge_count']}`
- Connected components / isolated nodes: `{graph['connected_component_count']}` / `{graph['isolated_node_count']}`
- Degree min / median / mean / max: `{graph['degree']['min']}` / `{graph['degree']['median']}` / `{graph['degree']['mean']}` / `{graph['degree']['max']}`
- Exact distance ties encountered: `{graph['exact_distance_ties_encountered']}`

## Graph-Gaussian reference

- Q0 smallest / largest eigenvalue: `{reference['q0']['smallest_eigenvalue']}` / `{reference['q0']['largest_eigenvalue']}`
- Q0 symmetric, nonpositive off-diagonals, positive definite: `{reference['q0']['symmetric']}`, `{reference['q0']['off_diagonal_nonpositive']}`, `{reference['q0']['positive_definite']}`

Sparse-solve diagnostics:

{solve_lines}
"""


def _artifact_manifest(output_dir: Path, run_sha: str) -> dict[str, Any]:
    files = []
    for name in EXPECTED_OUTPUTS:
        path = output_dir / name
        files.append({"path": name, "sha256": sha256_file(path), "size_bytes": path.stat().st_size})
    return {"schema_version": 1, "run_sha": run_sha, "files": files}


def create_deterministic_zip(output_dir: Path, zip_path: Path) -> None:
    names = EXPECTED_OUTPUTS + ["artifact_manifest.json"]
    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in names:
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, (output_dir / name).read_bytes(), compress_type=zipfile.ZIP_DEFLATED)


def run_gate(
    config_path: Path,
    repository_root: Path,
    output_dir: Path,
    run_sha: str,
    zip_path: Path,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["status"] != "PENDING_COLAB":
        raise GateFailure("IMPLEMENTATION_BLOCKED", "Descriptor/graph config is not pending Colab")
    _validate_run_sha(run_sha, repository_root)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise GateFailure("IMPLEMENTATION_BLOCKED", f"Refusing nonempty output directory: {output_dir}")
    if zip_path.exists():
        raise GateFailure("IMPLEMENTATION_BLOCKED", f"Refusing existing ZIP path: {zip_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        benchmark_record = verify_frozen_benchmark(repository_root, config)
        benchmark = config["benchmark"]
        legacy_path = repository_root / benchmark["legacy_table"]
        action_path = repository_root / benchmark["action_table"]
        legacy_rows = load_legacy_nodes(legacy_path, benchmark["expected_legacy_rows"])
        action_rows = map_actions_to_nodes(action_path, legacy_rows, benchmark["expected_actions"])
        environment, environment_text = environment_record(repository_root, config)
    except Exception as exc:
        raise GateFailure("IMPLEMENTATION_BLOCKED", str(exc)) from exc

    composition_keys = [row["composition_key"] for row in legacy_rows]
    formulas = [row["normalized_formula"] for row in legacy_rows]
    try:
        descriptor_config = config["descriptor"]
        matrix, feature_names = featurize_compositions(
            formulas, descriptor_config["expected_feature_count"]
        )
        expected_shape = tuple(descriptor_config["expected_shape"])
        if matrix.shape != expected_shape:
            raise ValueError(f"Descriptor shape {matrix.shape} != {expected_shape}")
        nonfinite_count = int(matrix.size - np.count_nonzero(np.isfinite(matrix)))
        if nonfinite_count:
            raise ValueError(f"Descriptor non-finite count is {nonfinite_count}")
        zero_variance = np.flatnonzero(np.ptp(matrix, axis=0) == 0.0)
        standardized, standardizer_record = standardize_for_graph(matrix)
    except Exception as exc:
        raise GateFailure("DESCRIPTOR_COMPATIBILITY_BLOCKED", str(exc)) from exc

    try:
        distances = pairwise_euclidean(standardized)
        graph_config = config["graph"]
        knn_edges, knn_diagnostics = deterministic_knn_edges(
            distances, composition_keys, graph_config["knn_k"]
        )
        mst_edges, mst_diagnostics = deterministic_mst_edges(distances, composition_keys)
        final_edges = knn_edges | mst_edges
        adjacency = adjacency_from_edges(len(legacy_rows), final_edges)
        graph_record = graph_summary(
            adjacency,
            knn_edges=knn_edges,
            mst_edges=mst_edges,
            knn_ties=knn_diagnostics,
            mst_ties=mst_diagnostics,
        )
        if graph_record["connected_component_count"] != graph_config["expected_connected_components"]:
            raise ValueError(f"Graph has {graph_record['connected_component_count']} components")
        if graph_record["isolated_node_count"]:
            raise ValueError(f"Graph has {graph_record['isolated_node_count']} isolated nodes")
        q0 = normalized_laplacian_reference(adjacency)
        q0_record = validate_q0(q0, config)
        solve_record = run_sparse_solves(q0, config["reference"]["solve_relative_residual_max"])
    except Exception as exc:
        raise GateFailure("GRAPH_COMPATIBILITY_BLOCKED", str(exc)) from exc

    descriptor_record = {
        "run_sha": run_sha,
        "config_sha256": sha256_file(config_path),
        "benchmark_manifest_sha256": benchmark_record["manifest_sha256"],
        "shape": list(matrix.shape),
        "dtype": str(matrix.dtype),
        "nonfinite_count": nonfinite_count,
        "zero_variance_feature_count": int(len(zero_variance)),
        "zero_variance_feature_indices": zero_variance.tolist(),
        "feature_names": feature_names,
        "feature_name_sha256": stable_string_sequence_sha256(feature_names),
        "matrix_sha256": descriptor_matrix_sha256(matrix),
        "row_key_sha256": stable_string_sequence_sha256(composition_keys),
        "hash_definitions": {
            "matrix_sha256": "contiguous little-endian float64 C-order descriptor bytes",
            "row_key_sha256": "UTF-8 composition keys joined by newline with one terminal newline",
            "feature_name_sha256": "UTF-8 feature labels joined by newline with one terminal newline",
        },
        "featurizer": {
            "class": descriptor_config["class"],
            "preset_name": descriptor_config["preset_name"],
            "impute_nan": descriptor_config["impute_nan"],
        },
        "standardization": standardizer_record,
        "package_versions": environment["packages"],
    }
    reference_record = {"q0": q0_record, "sparse_solves": solve_record}
    run_summary = {
        "schema_version": 1,
        "verdict": "PASS_DESCRIPTOR_GRAPH_COLAB",
        "run_sha": run_sha,
        "benchmark": benchmark_record,
        "descriptor": {
            key: descriptor_record[key]
            for key in (
                "shape",
                "dtype",
                "nonfinite_count",
                "zero_variance_feature_count",
                "matrix_sha256",
                "row_key_sha256",
            )
        },
        "graph": graph_record,
        "reference": reference_record,
        "actions": {
            "expected_actions": benchmark["expected_actions"],
            "mapped_actions": len(action_rows),
            "unique_action_keys": len({row["action_key"] for row in action_rows}),
            "unique_node_indices": len({row["node_index"] for row in action_rows}),
            "mapping_ambiguities": 0,
        },
        "environment": environment,
        "target_isolation": {
            "gw_values_read": False,
            "pbe_values_used_for_descriptors_or_graph": False,
        },
    }

    write_deterministic_npz(
        output_dir / "descriptor_matrix.npz",
        {
            "composition_keys": np.asarray(composition_keys, dtype=np.str_),
            "feature_names": np.asarray(feature_names, dtype=np.str_),
            "normalized_formulas": np.asarray(formulas, dtype=np.str_),
            "raw_descriptors": matrix,
        },
    )
    write_json(output_dir / "descriptor_manifest.json", descriptor_record)
    _write_csv(
        output_dir / "graph_edges.csv",
        _graph_edge_rows(final_edges, knn_edges, mst_edges, composition_keys),
        ["node_i", "node_j", "composition_key_i", "composition_key_j", "in_knn", "in_mst"],
    )
    write_json(output_dir / "graph_summary.json", graph_record)
    write_sparse_npz(output_dir / "q0_sparse.npz", q0)
    _write_csv(
        output_dir / "action_node_map.csv",
        action_rows,
        ["action_key", "composition_key", "node_index", "normalized_formula"],
    )
    (output_dir / "environment_freeze.txt").write_text(environment_text, encoding="utf-8")
    write_json(output_dir / "run_summary.json", run_summary)
    (output_dir / "RESULTS.md").write_text(_results_markdown(run_summary), encoding="utf-8")
    write_json(output_dir / "artifact_manifest.json", _artifact_manifest(output_dir, run_sha))
    create_deterministic_zip(output_dir, zip_path)
    return run_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--repository-root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run-sha", required=True)
    parser.add_argument("--zip-path", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = run_gate(
            args.config,
            args.repository_root,
            args.output_dir,
            args.run_sha,
            args.zip_path,
        )
    except GateFailure as exc:
        print(f"{exc.verdict}: {exc}", file=sys.stderr, flush=True)
        return 2
    except Exception as exc:
        print(f"IMPLEMENTATION_BLOCKED: {exc}", file=sys.stderr, flush=True)
        return 2
    print(
        json.dumps(
            {
                "run_sha": summary["run_sha"],
                "descriptor_shape": summary["descriptor"]["shape"],
                "graph_edges": summary["graph"]["final_unique_edge_count"],
                "connected_components": summary["graph"]["connected_component_count"],
                "mapped_actions": summary["actions"]["mapped_actions"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
