from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = (
    ROOT
    / "experiments/sun_oxide/outputs/decision_sparsity_diagnostic"
)
STARTING_SHA = "5f49f140acc3532b0231b1d1c446d22cd0e168d8"
IMPLEMENTATION_SHA = "2ac5dd3576c548f0c9999cbea3bdf7d6f626656d"
CONFIG_SHA256 = "281e9b173a234029563f8ed876b5d252befb6706849bf100cd61f281e65662e6"
GRID_COUNTS = [
    0,
    1247,
    2494,
    6236,
    12472,
    24944,
    37415,
    49887,
    62359,
    74831,
    87303,
    99774,
    112246,
    118482,
    124718,
]


def _json(name: str):
    return json.loads((OUTPUT / name).read_text(encoding="utf-8"))


def _csv(name: str) -> list[dict[str, str]]:
    with (OUTPUT / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_full_reference_reproduces_committed_seed0_leaders() -> None:
    result = _json("full_reference_summary.json")
    assert result["starting_main_sha"] == STARTING_SHA
    assert result["implementation_sha"] == IMPLEMENTATION_SHA
    assert result["model"] == {
        "action_count": 191,
        "model_name": "NORMALIZED_ALL_PAIRS_PBE_500_V1",
        "observation_precision": 400.0,
        "sigma_obs": 0.05,
        "strict_factor_count": 124718,
        "support_count": 500,
        "support_name": "PBE_SUPPORT_500_V1",
        "weight": pytest.approx(1 / 499),
        "weight_exact": "1/499",
    }
    expected = {
        "seed_0_initial": (8, 13),
        "seed_0_after_6_queries": (14, 134),
        "seed_0_after_12_queries": (20, 133),
    }
    for state in result["states"]:
        count, leader = expected[state["state"]]
        assert state["observation_count"] == count
        assert state["committed_full_leader"] == leader
        assert state["recomputed_full_leader"] == leader
        assert state["leader_agreement"]
        assert state["unauthorized_later_target_rows_read"] == 0
        assert state["diagnostics"]["gradient_infinity_norm"] <= 1e-5
        assert state["diagnostics"]["laplace_solve_relative_residual"] <= 1e-9


def test_influence_paths_cover_the_exact_grid_and_preserve_numerics() -> None:
    rows = _csv("influence_paths.csv")
    assert len(rows) == 3 * 2 * 15
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault((row["state"], row["path"]), []).append(row)
        assert int(row["unauthorized_later_target_rows_read"]) == 0
        assert float(row["map_gradient_infinity_norm"]) <= 1e-5
        assert float(row["laplace_solve_relative_residual"]) <= 1e-9
        for residual_name in (
            "load_solve_relative_residual",
            "ranking_solve_relative_residual",
        ):
            if row[residual_name]:
                assert float(row[residual_name]) <= 1e-9
        if row["contribution_sum_error"]:
            assert float(row["contribution_sum_error"]) <= 1e-9
    assert len(grouped) == 6
    for group in grouped.values():
        assert [int(row["active_factor_count"]) for row in group] == GRID_COUNTS
        final = group[-1]
        assert final["action_agreement"] == "True"
        assert float(final["full_laplace_ei_regret"]) <= 1e-8
        assert final["theorem_certificate_passed"] == "True"


def test_frozen_classification_and_stabilization_values() -> None:
    result = _json("decision_stabilization_summary.json")
    assert result["terminal_classification"] == "MIXED_DECISION_SPARSITY"
    assert result["classification_primary_path"] == "RERANKED_FINE_PATH"
    primary = {
        item["state"]: (
            item["first_agreement_fraction"],
            item["first_stable_agreement_fraction"],
            item["first_small_regret_fraction"],
            item["certificate_fraction"],
        )
        for item in result["path_summaries"]
        if item["path"] == "RERANKED_FINE_PATH"
    }
    assert primary == {
        "seed_0_initial": (0.7, 0.7, 0.0, 1.0),
        "seed_0_after_6_queries": (0.2, 0.2, 0.0, 1.0),
        "seed_0_after_12_queries": (0.0, 0.1, 0.0, 1.0),
    }


def test_random_baselines_are_complete_and_oracle_isolated() -> None:
    rows = _csv("random_baselines.csv")
    assert len(rows) == 3 * 3 * 20
    keys = [
        (row["state"], row["requested_fraction"], int(row["replicate"]))
        for row in rows
    ]
    assert len(set(keys)) == len(keys)
    counts = Counter((row["state"], row["requested_fraction"]) for row in rows)
    assert set(counts.values()) == {20}
    for row in rows:
        assert int(row["unauthorized_later_target_rows_read"]) == 0
        assert float(row["map_gradient_infinity_norm"]) <= 1e-5


def test_resource_guard_and_artifact_manifest() -> None:
    resource = _json("resource_usage.json")
    assert resource["development_only"]
    assert not resource["fresh_seeds_accessed"]
    assert not resource["colab_used"]
    assert not resource["local_resource_blocked"]
    assert resource["peak_rss_gb"] < resource["rss_limit_gb"] == 8.0

    manifest = _json("artifact_manifest.json")
    assert manifest["classification"] == "MIXED_DECISION_SPARSITY"
    assert manifest["implementation_sha"] == IMPLEMENTATION_SHA
    assert manifest["config_sha256"] == CONFIG_SHA256
    assert not manifest["fresh_seeds_accessed"]
    expected_names = {
        "RESULTS.md",
        "full_reference_summary.json",
        "influence_paths.csv",
        "random_baselines.csv",
        "decision_stabilization_summary.json",
        "resource_usage.json",
        "ei_regret_vs_active_fraction.png",
    }
    assert {item["path"] for item in manifest["files"]} == expected_names
    for item in manifest["files"]:
        data = (OUTPUT / item["path"]).read_bytes()
        assert len(data) == item["size_bytes"]
        assert hashlib.sha256(data).hexdigest() == item["sha256"]
