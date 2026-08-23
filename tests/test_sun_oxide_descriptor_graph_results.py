from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse import csgraph


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments/sun_oxide/outputs/descriptor_graph"
BENCHMARK = ROOT / "experiments/sun_oxide/benchmark"
MODULE_PATH = ROOT / "experiments/sun_oxide/descriptor_graph.py"
RUN_SHA = "843b2173454b70cf12a6199b4d1a32740e60315e"
EXPECTED_SCIENTIFIC_OUTPUTS = [
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
SPEC = importlib.util.spec_from_file_location("sunoxide_descriptor_graph_results", MODULE_PATH)
assert SPEC and SPEC.loader
GRAPH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GRAPH)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _permitted_csv_columns(path: Path, columns: list[str]) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        indices = [header.index(column) for column in columns]
        return [
            {column: raw[index] for column, index in zip(columns, indices, strict=True)}
            for raw in reader
        ]


def test_committed_colab_artifact_manifest_and_terminal_record() -> None:
    artifact_manifest = json.loads((OUTPUT / "artifact_manifest.json").read_text(encoding="utf-8"))
    assert artifact_manifest["run_sha"] == RUN_SHA
    assert [item["path"] for item in artifact_manifest["files"]] == EXPECTED_SCIENTIFIC_OUTPUTS
    assert sorted(path.name for path in OUTPUT.iterdir()) == sorted(
        EXPECTED_SCIENTIFIC_OUTPUTS
        + ["artifact_manifest.json", "NLR_DATA_USE_NOTICE.txt", "VERIFICATION.md"]
    )
    for item in artifact_manifest["files"]:
        path = OUTPUT / item["path"]
        assert path.stat().st_size == item["size_bytes"]
        assert _sha256(path) == item["sha256"]
    assert _sha256(OUTPUT / "NLR_DATA_USE_NOTICE.txt") == _sha256(
        BENCHMARK / "NLR_DATA_USE_NOTICE.txt"
    )

    summary = json.loads((OUTPUT / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["verdict"] == "PASS_DESCRIPTOR_GRAPH_COLAB"
    assert summary["run_sha"] == RUN_SHA
    assert summary["benchmark"]["benchmark_name"] == "CURRENT_NLR_PBE_GW_V1"
    assert summary["benchmark"]["legacy_rows"] == 2142
    assert summary["actions"]["mapped_actions"] == 191
    assert summary["target_isolation"] == {
        "gw_values_read": False,
        "pbe_values_used_for_descriptors_or_graph": False,
    }
    verification = (OUTPUT / "VERIFICATION.md").read_text(encoding="utf-8")
    assert "Terminal verdict: `PASS_DESCRIPTOR_GRAPH`." in verification


def test_committed_descriptor_matrix_matches_frozen_row_order() -> None:
    descriptor_manifest = json.loads(
        (OUTPUT / "descriptor_manifest.json").read_text(encoding="utf-8")
    )
    legacy = _permitted_csv_columns(
        BENCHMARK / "current_nlr_legacy.csv", ["composition_key", "normalized_formula"]
    )
    with np.load(OUTPUT / "descriptor_matrix.npz", allow_pickle=False) as arrays:
        assert arrays.files == [
            "composition_keys",
            "feature_names",
            "normalized_formulas",
            "raw_descriptors",
        ]
        keys = arrays["composition_keys"]
        names = arrays["feature_names"]
        formulas = arrays["normalized_formulas"]
        matrix = arrays["raw_descriptors"]

    assert matrix.shape == (2142, 132)
    assert matrix.dtype == np.dtype("float64")
    assert np.all(np.isfinite(matrix))
    assert names.shape == (132,)
    assert len(set(names.tolist())) == 132
    assert np.array_equal(keys, np.asarray([row["composition_key"] for row in legacy]))
    assert np.array_equal(formulas, np.asarray([row["normalized_formula"] for row in legacy]))
    assert GRAPH.descriptor_matrix_sha256(matrix) == descriptor_manifest["matrix_sha256"]
    assert GRAPH.stable_string_sequence_sha256(keys.tolist()) == descriptor_manifest["row_key_sha256"]
    assert GRAPH.stable_string_sequence_sha256(names.tolist()) == descriptor_manifest["feature_name_sha256"]
    assert np.flatnonzero(np.ptp(matrix, axis=0) == 0.0).tolist() == descriptor_manifest[
        "zero_variance_feature_indices"
    ]
    assert descriptor_manifest["zero_variance_feature_count"] == 15


def test_committed_graph_q0_solves_and_action_mapping() -> None:
    with np.load(OUTPUT / "descriptor_matrix.npz", allow_pickle=False) as arrays:
        keys = arrays["composition_keys"].tolist()
        formulas = arrays["normalized_formulas"].tolist()

    edges: list[tuple[int, int]] = []
    knn_edges: set[tuple[int, int]] = set()
    mst_edges: set[tuple[int, int]] = set()
    with (OUTPUT / "graph_edges.csv").open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == [
            "node_i",
            "node_j",
            "composition_key_i",
            "composition_key_j",
            "in_knn",
            "in_mst",
        ]
        for row in reader:
            left, right = int(row["node_i"]), int(row["node_j"])
            edge = (left, right)
            assert 0 <= left < right < 2142
            assert row["composition_key_i"] == keys[left]
            assert row["composition_key_j"] == keys[right]
            assert row["in_knn"] in {"0", "1"}
            assert row["in_mst"] in {"0", "1"}
            assert row["in_knn"] == "1" or row["in_mst"] == "1"
            edges.append(edge)
            if row["in_knn"] == "1":
                knn_edges.add(edge)
            if row["in_mst"] == "1":
                mst_edges.add(edge)
    assert edges == sorted(edges)
    assert len(edges) == len(set(edges)) == 14072
    assert len(knn_edges) == 14063
    assert len(mst_edges) == 2141

    adjacency = GRAPH.adjacency_from_edges(2142, edges)
    components, _ = csgraph.connected_components(adjacency, directed=False, return_labels=True)
    degrees = np.asarray(adjacency.sum(axis=1)).ravel()
    assert components == 1
    assert np.count_nonzero(degrees == 0) == 0
    assert (degrees.min(), np.median(degrees), np.mean(degrees), degrees.max()) == (
        10,
        13.0,
        13.139122315592903,
        26,
    )

    q0 = sparse.load_npz(OUTPUT / "q0_sparse.npz").tocsr()
    expected_q0 = GRAPH.normalized_laplacian_reference(adjacency)
    difference = (q0 - expected_q0).tocsr()
    assert difference.nnz == 0 or np.max(np.abs(difference.data)) <= 1e-15
    q0_checks = GRAPH.validate_q0(
        q0,
        json.loads(
            (ROOT / "experiments/sun_oxide/configs/descriptor_graph.json").read_text(
                encoding="utf-8"
            )
        ),
    )
    assert q0_checks["smallest_eigenvalue"] >= 1.0 - 1e-8
    assert q0_checks["largest_eigenvalue"] <= 3.0 + 1e-8
    solve_checks = GRAPH.run_sparse_solves(q0, 1e-10)
    assert all(item["relative_residual"] <= 1e-10 for item in solve_checks)

    action_source = _permitted_csv_columns(
        BENCHMARK / "current_nlr_gw_actions.csv",
        ["action_key", "composition_key", "normalized_formula"],
    )
    saved_actions = _permitted_csv_columns(
        OUTPUT / "action_node_map.csv",
        ["action_key", "composition_key", "node_index", "normalized_formula"],
    )
    node_by_key = {key: index for index, key in enumerate(keys)}
    assert len(saved_actions) == len(action_source) == 191
    for saved, source in zip(saved_actions, action_source, strict=True):
        assert saved["action_key"] == source["action_key"]
        assert saved["composition_key"] == source["composition_key"]
        assert saved["normalized_formula"] == source["normalized_formula"]
        assert formulas[int(saved["node_index"])] == source["normalized_formula"]
        assert int(saved["node_index"]) == node_by_key[source["composition_key"]]
    assert len({row["action_key"] for row in saved_actions}) == 191
    assert len({row["composition_key"] for row in saved_actions}) == 191
    assert len({row["node_index"] for row in saved_actions}) == 191

    source = MODULE_PATH.read_text(encoding="utf-8")
    assert "gw_oracle.csv" not in source
    assert "gw_band_gap_ev" not in source
    assert "pbe_band_gap_ev" not in source
    assert all(math.isfinite(item["wall_seconds"]) for item in json.loads(
        (OUTPUT / "run_summary.json").read_text(encoding="utf-8")
    )["reference"]["sparse_solves"])
