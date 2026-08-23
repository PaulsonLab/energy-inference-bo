from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import sparse

from conditioned_bo.normalized_pbe_model import (
    WEIGHT,
    build_strict_pair_bank,
    farthest_point_support,
    standardize_descriptor_space,
    weighted_adjacency,
)
from conditioned_bo.pbe_factor_theory import (
    build_comparison_matrix,
    load_action_mapping,
    load_legacy_nodes,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments/sun_oxide/outputs/normalized_pbe_model"
BENCHMARK = ROOT / "experiments/sun_oxide/benchmark"
GRAPH_OUTPUT = ROOT / "experiments/sun_oxide/outputs/descriptor_graph"
STARTING_SHA = "810d3c29ac3579a1159ad7eba301df4069db8d66"
SCIENTIFIC_OUTPUTS = [
    "RESULTS.md",
    "pbe_support_500.csv",
    "normalized_pbe_factor_bank.csv",
    "model_summary.json",
    "theory_summary.json",
    "pbe_signal_summary.json",
    "influence_summary.json",
    "influence_pair_summary.csv",
    "NLR_DATA_USE_NOTICE.txt",
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_committed_normalized_outputs_are_complete_and_immutable() -> None:
    manifest = json.loads((OUTPUT / "artifact_manifest.json").read_text(encoding="utf-8"))
    assert manifest["starting_sha"] == STARTING_SHA
    assert manifest["verdict"] == "PASS_NORMALIZED_PBE_MODEL"
    assert [item["path"] for item in manifest["files"]] == SCIENTIFIC_OUTPUTS
    assert sorted(path.name for path in OUTPUT.iterdir()) == sorted(
        SCIENTIFIC_OUTPUTS + ["artifact_manifest.json"]
    )
    for item in manifest["files"]:
        path = OUTPUT / item["path"]
        assert path.stat().st_size == item["size_bytes"]
        assert _sha256(path) == item["sha256"]
    assert _sha256(OUTPUT / "NLR_DATA_USE_NOTICE.txt") == _sha256(
        BENCHMARK / "NLR_DATA_USE_NOTICE.txt"
    )
    results = (OUTPUT / "RESULTS.md").read_text(encoding="utf-8")
    assert "Terminal verdict: `PASS_NORMALIZED_PBE_MODEL`." in results
    assert "GW oracle read: `False`" in results


def test_committed_support_and_complete_strict_pair_bank_reproduce() -> None:
    nodes = load_legacy_nodes(BENCHMARK / "current_nlr_legacy.csv", 2142)
    actions = load_action_mapping(GRAPH_OUTPUT / "action_node_map.csv", 191, 2142)
    with np.load(GRAPH_OUTPUT / "descriptor_matrix.npz", allow_pickle=False) as archive:
        standardized, _ = standardize_descriptor_space(archive["raw_descriptors"])
    action_nodes = np.asarray([action["node_index"] for action in actions], dtype=np.int64)
    support = farthest_point_support(
        standardized,
        [node.composition_key for node in nodes],
        action_nodes,
        support_count=500,
    )
    expected = build_strict_pair_bank(support.selected_indices, nodes)

    with (OUTPUT / "pbe_support_500.csv").open("r", encoding="utf-8", newline="") as stream:
        support_rows = list(csv.DictReader(stream))
    assert len(support_rows) == 500
    np.testing.assert_array_equal(
        [int(row["node_index"]) for row in support_rows], expected.support_indices
    )
    assert {int(row["node_index"]) for row in support_rows if row["role"] == "action"} == set(action_nodes)
    assert sum(row["role"] == "pbe_only_fps" for row in support_rows) == 309

    with (OUTPUT / "normalized_pbe_factor_bank.csv").open(
        "r", encoding="utf-8", newline=""
    ) as stream:
        factor_rows = list(csv.DictReader(stream))
    assert len(factor_rows) == expected.strict_factor_count == 124718
    np.testing.assert_array_equal(
        [[int(row["node_i"]), int(row["node_j"])] for row in factor_rows],
        expected.node_endpoint_pairs,
    )
    np.testing.assert_array_equal(
        [int(row["sign_s_ij"]) for row in factor_rows], expected.signs
    )
    assert all(float(row["weight"]) == WEIGHT for row in factor_rows)
    assert expected.omitted_exact_tie_pair_count == 32


def test_committed_theory_signal_influence_and_isolation_checks() -> None:
    model = json.loads((OUTPUT / "model_summary.json").read_text(encoding="utf-8"))
    theory = json.loads((OUTPUT / "theory_summary.json").read_text(encoding="utf-8"))
    signal = json.loads((OUTPUT / "pbe_signal_summary.json").read_text(encoding="utf-8"))
    influence = json.loads((OUTPUT / "influence_summary.json").read_text(encoding="utf-8"))

    graph = model["weighted_factor_graph"]
    assert graph["maximum_weighted_row_sum"] <= 1.0 + 1e-12
    assert graph["spectral_norm_numerical"] <= 1.0 + 1e-8
    nodes = load_legacy_nodes(BENCHMARK / "current_nlr_legacy.csv", 2142)
    with (OUTPUT / "normalized_pbe_factor_bank.csv").open(
        "r", encoding="utf-8", newline=""
    ) as stream:
        endpoints = np.asarray(
            [[int(row["node_i"]), int(row["node_j"])] for row in csv.DictReader(stream)],
            dtype=np.int64,
        )
    weighted_graph = weighted_adjacency(len(nodes), endpoints, WEIGHT)
    q0 = sparse.load_npz(GRAPH_OUTPUT / "q0_sparse.npz").tocsr()
    a0 = build_comparison_matrix(q0, weighted_graph)
    expected = (q0 - 0.25 * weighted_graph).tocsr()
    assert (a0 - expected).nnz == 0

    comparison = theory["comparison"]
    assert comparison["analytic_smallest_eigenvalue_lower_bound"] == 0.75
    assert comparison["smallest_eigenvalue_numerical"] >= 0.75 - 1e-8
    assert comparison["positive_definite"]
    assert comparison["symmetric"] and comparison["off_diagonals_nonpositive"]
    assert theory["factor_calculus"]["mixed_hessian_absolute_bound"] == WEIGHT / 4.0
    assert not theory["dense_inverse_formed"]
    for observation in theory["full_size_observation_patterns"]:
        assert observation["positive_definite"]
        assert observation["sampled_inverse_monotonicity"]

    assert signal["diagnostic_not_gate"] and signal["target_blind"]
    assert set(signal) >= {"normalized_dense", "adjacent_chain_baseline"}
    assert influence["sparse_a0"]["action_solve_relative_frobenius_residual"] <= 1e-10
    assert influence["pair_diagnostics"]["diagnostic_action_pair_count"] == 256
    assert set(influence["pair_diagnostics"]["factor_fractions_required"]) == {
        "50_percent", "75_percent", "90_percent", "95_percent"
    }
    assert influence["target_isolation"]["target_values_read"] is False
    assert influence["target_isolation"]["target_statistics_computed"] is False
    assert not influence["sparse_a0"]["full_dense_inverse_formed"]

    with (OUTPUT / "influence_pair_summary.csv").open(
        "r", encoding="utf-8", newline=""
    ) as stream:
        pair_rows = list(csv.DictReader(stream))
    assert len(pair_rows) == 256
    assert {int(row["distance_quartile"]) for row in pair_rows} == {1, 2, 3, 4}
    assert all(sum(int(row["distance_quartile"]) == quartile for row in pair_rows) == 64 for quartile in range(1, 5))
