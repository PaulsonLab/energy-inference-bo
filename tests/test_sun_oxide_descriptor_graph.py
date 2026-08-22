from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "experiments/sun_oxide/descriptor_graph.py"
CONFIG_PATH = ROOT / "experiments/sun_oxide/configs/descriptor_graph.json"
LOCK_PATH = ROOT / "experiments/sun_oxide/requirements-colab-graph.txt"
NOTEBOOK_PATH = ROOT / "experiments/sun_oxide/colab_descriptor_graph.ipynb"
SPEC = importlib.util.spec_from_file_location("sun_oxide_descriptor_graph", MODULE_PATH)
assert SPEC and SPEC.loader
GRAPH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GRAPH)
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _write_csv(path: Path, columns: list[str], rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(columns)
        writer.writerows(rows)


def test_descriptor_feature_count_contract() -> None:
    labels = [f"magpie_{index:03d}" for index in range(132)]
    assert GRAPH.validate_feature_labels(labels, 132) == labels
    with pytest.raises(ValueError, match="feature count"):
        GRAPH.validate_feature_labels(labels[:-1], 132)
    with pytest.raises(ValueError, match="not unique"):
        GRAPH.validate_feature_labels(labels[:-1] + [labels[-2]], 132)
    assert CONFIG["descriptor"]["expected_shape"] == [2142, 132]


def test_stable_legacy_row_order_is_preserved(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy.csv"
    _write_csv(
        legacy,
        ["composition_key", "normalized_formula", "unused_value"],
        [["key-z", "O2 Zr", "9"], ["key-a", "Ag2 O", "1"], ["key-m", "O Mg", "4"]],
    )
    assert GRAPH.load_legacy_nodes(legacy, 3) == [
        {"composition_key": "key-z", "normalized_formula": "O2 Zr"},
        {"composition_key": "key-a", "normalized_formula": "Ag2 O"},
        {"composition_key": "key-m", "normalized_formula": "O Mg"},
    ]


def test_deterministic_knn_tie_breaks_by_composition_key() -> None:
    points = np.asarray([[0.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [0.0, 2.0]])
    keys = ["key-z", "key-b", "key-a", "key-c"]
    distances = GRAPH.pairwise_euclidean(points)
    edges, diagnostics = GRAPH.deterministic_knn_edges(distances, keys, k=1)
    assert edges == {(0, 1), (0, 2), (0, 3)}
    assert diagnostics["directed_choice_count"] == 4
    assert diagnostics["k_boundary_tie_nodes"] >= 1


def test_deterministic_mst_ties_are_invariant_to_row_permutation() -> None:
    points = np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    keys = ["key-d", "key-a", "key-c", "key-b"]
    edges_a, diagnostics_a = GRAPH.deterministic_mst_edges(GRAPH.pairwise_euclidean(points), keys)
    key_edges_a = {tuple(sorted((keys[left], keys[right]))) for left, right in edges_a}

    permutation = np.asarray([2, 0, 3, 1])
    points_b = points[permutation]
    keys_b = [keys[index] for index in permutation]
    edges_b, diagnostics_b = GRAPH.deterministic_mst_edges(
        GRAPH.pairwise_euclidean(points_b), keys_b
    )
    key_edges_b = {tuple(sorted((keys_b[left], keys_b[right]))) for left, right in edges_b}
    assert key_edges_a == key_edges_b
    assert len(key_edges_a) == 3
    assert diagnostics_a["exact_distance_tie_adjacencies"] > 0
    assert diagnostics_b["exact_distance_tie_adjacencies"] > 0


def _path_reference():
    adjacency = GRAPH.adjacency_from_edges(3, {(0, 1), (1, 2)})
    q0 = GRAPH.normalized_laplacian_reference(adjacency)
    return adjacency, q0


def test_normalized_laplacian_and_q0_calculation() -> None:
    _, q0 = _path_reference()
    expected = np.asarray(
        [
            [2.0, -1.0 / np.sqrt(2.0), 0.0],
            [-1.0 / np.sqrt(2.0), 2.0, -1.0 / np.sqrt(2.0)],
            [0.0, -1.0 / np.sqrt(2.0), 2.0],
        ]
    )
    assert np.allclose(q0.toarray(), expected, rtol=0.0, atol=1e-15)
    summary = GRAPH.validate_q0(q0, CONFIG)
    assert summary["symmetric"]
    assert summary["off_diagonal_nonpositive"]
    assert summary["positive_definite"]
    assert summary["smallest_eigenvalue"] >= 1.0 - 1e-8
    assert summary["largest_eigenvalue"] <= 3.0 + 1e-8
    assert not summary["dense_inverse_formed"]


def test_graph_connectivity_fixture() -> None:
    adjacency, _ = _path_reference()
    summary = GRAPH.graph_summary(
        adjacency,
        knn_edges={(0, 1)},
        mst_edges={(0, 1), (1, 2)},
        knn_ties={
            "directed_choice_count": 3,
            "exact_distance_tie_adjacencies": 0,
        },
        mst_ties={"exact_distance_tie_adjacencies": 0},
    )
    assert summary["connected_component_count"] == 1
    assert summary["isolated_node_count"] == 0
    assert summary["degree"] == {"min": 1, "median": 1.0, "mean": 4 / 3, "max": 2}


def test_sparse_solve_residual_fixture() -> None:
    _, q0 = _path_reference()
    diagnostics = GRAPH.run_sparse_solves(q0, 1e-10)
    assert [item["rhs"] for item in diagnostics] == [
        "ones",
        "linear_minus_one_to_one",
        "sinusoid_index_times_sqrt_two",
    ]
    assert all(item["relative_residual"] <= 1e-10 for item in diagnostics)


def test_deterministic_npz_and_sparse_npz_are_reloadable(tmp_path: Path) -> None:
    array_path_a = tmp_path / "array-a.npz"
    array_path_b = tmp_path / "array-b.npz"
    arrays = {"values": np.asarray([[1.0, 2.0], [3.0, 4.0]]), "labels": np.asarray(["a", "b"])}
    GRAPH.write_deterministic_npz(array_path_a, arrays)
    GRAPH.write_deterministic_npz(array_path_b, arrays)
    assert array_path_a.read_bytes() == array_path_b.read_bytes()
    with np.load(array_path_a, allow_pickle=False) as loaded:
        assert np.array_equal(loaded["values"], arrays["values"])
        assert np.array_equal(loaded["labels"], arrays["labels"])

    sparse_path = tmp_path / "sparse.npz"
    expected = sparse.csr_matrix(np.asarray([[2.0, -0.5], [-0.5, 2.0]]))
    GRAPH.write_sparse_npz(sparse_path, expected)
    observed = sparse.load_npz(sparse_path)
    assert np.array_equal(observed.toarray(), expected.toarray())


def test_action_mapping_is_unique_and_exact(tmp_path: Path) -> None:
    actions = tmp_path / "actions.csv"
    _write_csv(
        actions,
        ["action_key", "composition_key", "normalized_formula", "unused_metadata"],
        [["action-1", "key-b", "O2 Ti", "x"], ["action-2", "key-a", "Ag2 O", "y"]],
    )
    legacy = [
        {"composition_key": "key-a", "normalized_formula": "Ag2 O"},
        {"composition_key": "key-b", "normalized_formula": "O2 Ti"},
        {"composition_key": "key-c", "normalized_formula": "O Zn"},
    ]
    assert GRAPH.map_actions_to_nodes(actions, legacy, 2) == [
        {
            "action_key": "action-1",
            "composition_key": "key-b",
            "node_index": 1,
            "normalized_formula": "O2 Ti",
        },
        {
            "action_key": "action-2",
            "composition_key": "key-a",
            "node_index": 0,
            "normalized_formula": "Ag2 O",
        },
    ]


def test_frozen_benchmark_and_all_191_action_keys_are_compatible() -> None:
    record = GRAPH.verify_frozen_benchmark(ROOT, CONFIG)
    assert record["benchmark_name"] == "CURRENT_NLR_PBE_GW_V1"
    assert record["legacy_rows"] == 2142
    assert record["actions"] == 191
    assert record["gw_values_read"] is False
    legacy = GRAPH.load_legacy_nodes(
        ROOT / CONFIG["benchmark"]["legacy_table"],
        CONFIG["benchmark"]["expected_legacy_rows"],
    )
    mapped = GRAPH.map_actions_to_nodes(
        ROOT / CONFIG["benchmark"]["action_table"],
        legacy,
        CONFIG["benchmark"]["expected_actions"],
    )
    assert len(mapped) == 191
    assert len({row["action_key"] for row in mapped}) == 191
    assert len({row["node_index"] for row in mapped}) == 191


def test_descriptor_module_is_explicitly_gw_oracle_isolated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    forbidden_name = "gw_oracle.csv"
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert forbidden_name not in source
    assert "gw_band_gap_ev" not in source
    assert "pbe_band_gap_ev" not in source

    benchmark_dir = tmp_path / "benchmark"
    benchmark_dir.mkdir()
    safe_contents = {
        "current_nlr_legacy.csv": b"legacy\n",
        "current_nlr_gw_actions.csv": b"actions\n",
        "NLR_DATA_USE_NOTICE.txt": b"notice\n",
    }
    artifacts = {}
    for name, content in safe_contents.items():
        (benchmark_dir / name).write_bytes(content)
        artifacts[name] = {"sha256": hashlib.sha256(content).hexdigest()}
    (benchmark_dir / forbidden_name).write_text("must not be opened", encoding="utf-8")
    artifacts[forbidden_name] = {"sha256": "not-consulted"}
    manifest = {
        "benchmark_name": "CURRENT_NLR_PBE_GW_V1",
        "counts": {"legacy_compositions": 2142, "strict_gw_actions": 191},
        "artifacts": artifacts,
    }
    (benchmark_dir / "benchmark_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    opened: list[str] = []
    original_open = Path.open

    def guarded_open(path: Path, *args, **kwargs):
        opened.append(path.name)
        if path.name == forbidden_name:
            raise AssertionError("forbidden GW oracle access")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    config = {
        "benchmark": {
            "name": "CURRENT_NLR_PBE_GW_V1",
            "manifest": "benchmark/benchmark_manifest.json",
            "expected_legacy_rows": 2142,
            "expected_actions": 191,
        }
    }
    record = GRAPH.verify_frozen_benchmark(tmp_path, config)
    assert not record["gw_values_read"]
    assert forbidden_name not in opened


def test_lock_is_fully_pinned_with_required_transitives() -> None:
    packages = GRAPH._locked_packages(LOCK_PATH)
    assert len(packages) >= 40
    assert packages["matminer"] == "0.10.1"
    assert packages["numpy"] == "2.3.5"
    assert packages["pandas"] == "2.3.3"
    assert packages["scipy"] == "1.18.0"
    assert packages["scikit-learn"] == "1.9.0"
    for name in ("monty", "pymatgen", "pymatgen-core", "sympy"):
        assert name in packages


def test_colab_notebook_pins_sha_and_uses_only_the_isolated_venv() -> None:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    code_cells = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
    code = "\n".join(code_cells)
    assert all(not cell.get("outputs") and cell.get("execution_count") is None for cell in notebook["cells"] if cell["cell_type"] == "code")
    assert "git', 'clone', '--branch', 'main'" in code
    assert code.index("RUN_SHA = subprocess.run") < code.index("'checkout', '--detach', RUN_SHA")
    assert "Path('/content/sunoxide_graph_venv')" in code
    assert "str(VENV_PYTHON), '-m', 'pip', 'install', '--require-hashes', '--no-deps'" in code
    assert "str(VENV_PYTHON), '-m', 'pip', 'check'" in code
    assert "str(VENV_PYTHON),\n    str(REPOSITORY_DIR / 'experiments/sun_oxide/descriptor_graph.py')" in code
    assert "gw_oracle.csv" not in code
    assert code_cells[-1].rstrip().endswith("print('PASS_DESCRIPTOR_GRAPH_COLAB')")
