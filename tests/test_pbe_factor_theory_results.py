from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import sparse

from conditioned_bo.pbe_factor_theory import (
    build_comparison_matrix,
    build_factor_bank,
    factor_adjacency,
    factor_endpoint_array,
    load_legacy_nodes,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments/sun_oxide/outputs/pbe_factor_theory"
BENCHMARK = ROOT / "experiments/sun_oxide/benchmark"
GRAPH_OUTPUT = ROOT / "experiments/sun_oxide/outputs/descriptor_graph"
STARTING_SHA = "903f61930e8e46f5e124083fb728084b62912d51"
SCIENTIFIC_OUTPUTS = [
    "RESULTS.md",
    "pbe_factor_bank.csv",
    "factor_summary.json",
    "theory_summary.json",
    "influence_summary.json",
    "influence_pair_summary.csv",
    "NLR_DATA_USE_NOTICE.txt",
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_committed_factor_artifacts_are_complete_and_immutable() -> None:
    manifest = json.loads((OUTPUT / "artifact_manifest.json").read_text(encoding="utf-8"))
    assert manifest["starting_sha"] == STARTING_SHA
    assert manifest["verdict"] == "PASS_PBE_FACTOR_THEORY"
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
    assert "Terminal verdict: `PASS_PBE_FACTOR_THEORY`." in results
    assert "not Sun et al.'s all-pairs likelihood" in results
    assert "GW oracle read: `False`" in results


def test_committed_factor_bank_reproduces_exact_strict_adjacent_order() -> None:
    nodes = load_legacy_nodes(BENCHMARK / "current_nlr_legacy.csv", 2142)
    _, expected, expected_ties = build_factor_bank(nodes)
    with (OUTPUT / "pbe_factor_bank.csv").open(
        "r", encoding="utf-8", newline=""
    ) as stream:
        rows = list(csv.DictReader(stream))

    assert len(rows) == len(expected) == 1681
    for row, factor in zip(rows, expected, strict=True):
        assert int(row["factor_index"]) == factor.factor_index
        assert row["factor_bank"] == "ADJACENT_STRICT_PBE_ORDER_V1"
        assert float(row["temperature"]) == 1.0
        assert int(row["sorted_position_b"]) == int(row["sorted_position_a"]) + 1
        assert int(row["node_a"]) == factor.node_a
        assert int(row["node_b"]) == factor.node_b
        assert row["composition_key_a"] == factor.composition_key_a
        assert row["composition_key_b"] == factor.composition_key_b
        assert float(row["pbe_band_gap_b_ev"]) > float(row["pbe_band_gap_a_ev"])
        assert row["strict_relation"] == "z_b > z_a"

    summary = json.loads((OUTPUT / "factor_summary.json").read_text(encoding="utf-8"))
    assert summary["construction_deterministic"]
    assert summary["exact_ties_omitted"]
    assert summary["factor_bank_csv_sha256"] == _sha256(OUTPUT / "pbe_factor_bank.csv")
    assert summary["ties"]["exact_pbe_tie_group_count"] == len(expected_ties) == 369
    assert summary["ties"]["skipped_adjacent_ties"] == 460
    assert summary["ties"]["exact_pbe_tie_group_size_histogram"] == {
        "2": 296,
        "3": 57,
        "4": 14,
        "5": 2,
    }
    assert summary["factor_graph"]["maximum_degree"] <= 2
    assert summary["factor_graph"]["spectral_norm_numerical"] <= 2.0 + 1e-8


def test_committed_comparison_and_uniformity_diagnostics() -> None:
    nodes = load_legacy_nodes(BENCHMARK / "current_nlr_legacy.csv", 2142)
    _, factors, _ = build_factor_bank(nodes)
    graph = factor_adjacency(2142, factor_endpoint_array(factors))
    q0 = sparse.load_npz(GRAPH_OUTPUT / "q0_sparse.npz").tocsr()
    a0 = build_comparison_matrix(q0, graph)
    expected = (q0 - 0.25 * graph).tocsr()
    difference = (a0 - expected).tocsr()
    assert difference.nnz == 0

    summary = json.loads((OUTPUT / "theory_summary.json").read_text(encoding="utf-8"))
    comparison = summary["comparison"]
    assert comparison["identity_max_abs_error"] == 0.0
    assert comparison["symmetric"]
    assert comparison["off_diagonals_nonpositive"]
    assert comparison["analytic_smallest_eigenvalue_lower_bound"] == 0.5
    assert comparison["smallest_eigenvalue_numerical"] >= 0.5 - 1e-8
    assert comparison["positive_definite"]
    assert not summary["dense_inverse_formed"]
    assert not summary["new_covariance_theorem_introduced"]
    assert len(summary["full_size_observation_patterns"]) == 3
    for pattern in summary["full_size_observation_patterns"]:
        assert pattern["positive_definite"]
        assert pattern["same_factor_hessian_bound"]
        assert pattern["sampled_inverse_columns_entrywise_nonnegative"]
        assert pattern["sampled_inverse_monotonicity"]
        assert not pattern["dense_inverse_formed"]


def test_all_action_pair_influence_diagnostics_are_complete_and_target_blind() -> None:
    summary = json.loads((OUTPUT / "influence_summary.json").read_text(encoding="utf-8"))
    assert summary["sparse_a0"]["factorization_count"] == 1
    assert summary["sparse_a0"]["action_multiple_rhs_count"] == 191
    assert summary["sparse_a0"]["action_solve_relative_frobenius_residual"] <= 1e-10
    assert not summary["sparse_a0"]["full_dense_inverse_formed"]
    assert summary["diagnostic_not_gate"]
    assert summary["target_isolation"]["gw_values_read"] is False
    assert summary["target_isolation"]["gw_target_statistics_computed"] is False
    pair_summary = summary["pair_diagnostics"]
    assert pair_summary["unordered_action_pair_count"] == 191 * 190 // 2
    assert pair_summary["factor_count"] == 1681
    assert set(pair_summary["factor_fraction_required"]) == {
        "90_percent",
        "95_percent",
        "99_percent",
    }

    with (OUTPUT / "influence_pair_summary.csv").open(
        "r", encoding="utf-8", newline=""
    ) as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 191 * 190 // 2
    assert [(int(rows[0]["action_position_x"]), int(rows[0]["action_position_xhat"])),
            (int(rows[-1]["action_position_x"]), int(rows[-1]["action_position_xhat"]))] == [
        (0, 1),
        (189, 190),
    ]
    for row in rows:
        for label in ("90", "95", "99"):
            count = int(row[f"factors_for_{label}_percent"])
            fraction = float(row[f"factor_fraction_for_{label}_percent"])
            assert 1 <= count <= 1681
            np.testing.assert_allclose(fraction, count / 1681, rtol=0.0, atol=0.0)
