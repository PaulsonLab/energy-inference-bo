from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments/sun_oxide/outputs/bo_value_pilot"
BENCHMARK = ROOT / "experiments/sun_oxide/benchmark"
CONFIG = ROOT / "experiments/sun_oxide/configs/bo_value_pilot.json"
RUN_SHA = "44f58f100f41247afe0937e42eebe58055104225"
CONFIG_SHA256 = "6cc47d41dfbdbf88187d535d405ca6afd971e4b07f91932d55dbbbf5c101ef0f"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def test_committed_bo_value_archive_is_complete_and_immutable() -> None:
    manifest = json.loads((OUTPUT / "artifact_manifest.json").read_text())
    assert manifest["run_sha"] == RUN_SHA
    assert manifest["config_sha256"] == CONFIG_SHA256
    assert manifest["terminal_state"] == "PASS_PBE_VALUE_COLAB"
    declared = [item["path"] for item in manifest["files"]]
    actual = sorted(
        str(path.relative_to(OUTPUT))
        for path in OUTPUT.rglob("*")
        if path.is_file()
    )
    assert actual == sorted(declared + ["VERIFICATION.md", "artifact_manifest.json"])
    for item in manifest["files"]:
        path = OUTPUT / item["path"]
        assert path.stat().st_size == item["size_bytes"]
        assert _sha256(path) == item["sha256"]
    assert _sha256(OUTPUT / "frozen_config.json") == CONFIG_SHA256
    assert (OUTPUT / "frozen_config.json").read_bytes() == CONFIG.read_bytes()
    assert _sha256(OUTPUT / "NLR_DATA_USE_NOTICE.txt") == _sha256(
        BENCHMARK / "NLR_DATA_USE_NOTICE.txt"
    )

    provenance = json.loads((OUTPUT / "provenance.json").read_text())
    assert provenance["run_sha"] == RUN_SHA
    assert provenance["config_sha256"] == CONFIG_SHA256
    assert provenance["smoke_passed_before_oracle_open"]
    assert not provenance["smoke_report_records_oracle_file_opened"]


def test_committed_trajectories_reproduce_primary_value_metrics() -> None:
    rows = _csv_rows(OUTPUT / "trajectories.csv")
    assert len(rows) == 12 * 2 * 20
    oracle_rows = _csv_rows(BENCHMARK / "gw_oracle.csv")
    oracle = {row["action_key"]: float(row["gw_band_gap_ev"]) for row in oracle_rows}
    oracle_maximum = max(oracle.values())
    assert oracle_maximum == 10.349

    aurc: dict[str, list[float]] = {"NO_PBE": [], "FULL_PBE": []}
    finals: dict[str, list[float]] = {"NO_PBE": [], "FULL_PBE": []}
    optimum_counts = {"NO_PBE": 0, "FULL_PBE": 0}
    for seed in range(12):
        expected_initial = np.random.default_rng(seed).choice(191, size=8, replace=False)
        method_initial: dict[str, list[int]] = {}
        for method in ("NO_PBE", "FULL_PBE"):
            selected = [
                row
                for row in rows
                if int(row["seed"]) == seed and row["method"] == method
            ]
            initial = sorted(
                (row for row in selected if row["phase"] == "initial"),
                key=lambda row: int(row["initial_order"]),
            )
            sequential = sorted(
                (row for row in selected if row["phase"] == "sequential"),
                key=lambda row: int(row["query_index"]),
            )
            positions = [int(row["action_position"]) for row in initial]
            np.testing.assert_array_equal(positions, expected_initial)
            method_initial[method] = positions
            assert len(sequential) == 12
            assert len({row["action_key"] for row in selected}) == 20
            for row in selected:
                assert float(row["gw_band_gap_ev"]) == oracle[row["action_key"]]
            initial_values = np.asarray([float(row["gw_band_gap_ev"]) for row in initial])
            mean = float(initial_values.mean())
            scale = max(float(initial_values.std(ddof=1)), 0.25)
            for row in selected:
                expected_z = (float(row["gw_band_gap_ev"]) - mean) / scale
                assert np.isclose(float(row["standardized_observation"]), expected_z)
            best = float(initial_values.max())
            regrets = []
            for row in sequential:
                best = max(best, float(row["gw_band_gap_ev"]))
                regret = oracle_maximum - best
                regrets.append(regret)
                assert np.isclose(float(row["simple_regret_ev"]), regret)
            aurc[method].append(float(sum(regrets)))
            finals[method].append(regrets[-1])
            optimum_counts[method] += best == oracle_maximum
        assert method_initial["NO_PBE"] == method_initial["FULL_PBE"]

    summary = json.loads((OUTPUT / "run_summary.json").read_text())
    metrics = summary["metrics"]
    assert summary["scientific_verdict"] == "PASS_PBE_VALUE"
    assert summary["terminal_state"] == "PASS_PBE_VALUE_COLAB"
    assert summary["completed_seed_count"] == 12
    for method in ("NO_PBE", "FULL_PBE"):
        assert np.isclose(np.median(aurc[method]), metrics["median_aurc_ev"][method])
        assert np.isclose(
            np.median(finals[method]), metrics["median_final_simple_regret_ev"][method]
        )
        assert optimum_counts[method] == metrics["global_optimum_discovery_count"][method]
    differences = np.asarray(aurc["FULL_PBE"]) - np.asarray(aurc["NO_PBE"])
    np.testing.assert_allclose(
        differences, metrics["paired_aurc_difference_full_minus_no_ev"]
    )
    assert np.mean(differences < 0.0) == metrics["fraction_seeds_full_pbe_wins"]
    assert np.median(aurc["FULL_PBE"]) <= 0.90 * np.median(aurc["NO_PBE"])
    assert np.median(finals["FULL_PBE"]) <= np.median(finals["NO_PBE"])


def test_committed_inference_validations_and_oracle_isolation_pass() -> None:
    summary = json.loads((OUTPUT / "run_summary.json").read_text())
    expected = {
        "seed_0_initial": (0.9051015750899615, 13, 5),
        "seed_0_after_6_queries": (0.9053400461853587, 134, 179),
        "seed_0_after_12_queries": (0.906766084361209, 133, 4),
    }
    for name, (ess_fraction, laplace_action, is_action) in expected.items():
        report = json.loads((OUTPUT / f"is_validation_{name}.json").read_text())
        assert report["sample_count"] == 4096
        assert report["passed"] and summary["validation_passed"][name]
        assert np.isclose(report["ess_fraction"], ess_fraction)
        assert report["laplace_selected_action_position"] == laplace_action
        assert report["is_selected_action_position"] == is_action
        assert report["ess_fraction"] >= report["ess_fraction_minimum"]
        assert report["is_estimated_regret_standardized"] <= report[
            "decision_regret_threshold_standardized"
        ]
        assert not report["validation_alters_bo_decision"]
        assert not report["laplace_is_exact_conditioned_posterior"]
        assert not report["importance_validation_in_routine_decision_time"]

    access = json.loads((OUTPUT / "oracle_access_log.json").read_text())
    assert len(access) == 386
    assert all(item["access"] == "queried_action" for item in access[:384])
    assert access[-2:] == [
        {"access": "post_run_evaluation_unlocked"},
        {"access": "full_oracle_evaluation"},
    ]
    diagnostics = json.loads((OUTPUT / "inference_diagnostics.json").read_text())
    routine = [row for row in diagnostics if row.get("decision_kind") != "validation_only_after_12_queries"]
    assert len(routine) == 288
    assert all(not row["hyperparameters_fit_or_updated"] for row in routine)
    full = [row for row in routine if row["method"] == "FULL_PBE"]
    assert len(full) == 144
    assert all(row["optimizer_success"] for row in full)
    assert all(row["laplace_spd_cholesky_success"] for row in full)
    assert all(not row["principal_precision_submatrix_used"] for row in full)
    assert max(row["gradient_infinity_norm"] for row in full) <= 1e-5
    assert max(row["laplace_solve_relative_residual"] for row in full) <= 1e-9
