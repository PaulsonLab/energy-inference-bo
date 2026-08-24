from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments/sun_oxide/outputs/adaptive_e3_smoke"
IMPLEMENTATION_SHA = "7fbfb202268dd0fd92d35defbea2cc4990f089e2"
CONFIG_SHA256 = "aa327b3a0462c103a2dfbfed721bc30b7946acdb7b3c02032078001dc186b1a9"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def test_smoke_archive_manifest_is_complete_and_immutable() -> None:
    manifest = json.loads((OUTPUT / "artifact_manifest.json").read_text())
    assert manifest["implementation_sha"] == IMPLEMENTATION_SHA
    assert manifest["config_sha256"] == CONFIG_SHA256
    assert manifest["terminal_state"] == "ENGINEERING_SMOKE_PASS"
    declared = [item["path"] for item in manifest["files"]]
    actual = sorted(
        str(path.relative_to(OUTPUT))
        for path in OUTPUT.rglob("*")
        if path.is_file()
    )
    assert actual == sorted(declared + ["artifact_manifest.json"])
    for item in manifest["files"]:
        path = OUTPUT / item["path"]
        assert path.stat().st_size == item["size_bytes"]
        assert _sha256(path) == item["sha256"]


def test_smoke_full_regression_adaptive_termination_and_shadow_pass() -> None:
    summary = json.loads((OUTPUT / "run_summary.json").read_text())
    assert summary["terminal_state"] == "ENGINEERING_SMOKE_PASS"
    assert summary["mechanically_sound"]
    assert not summary["scientific_evidence"]
    assert summary["implementation_sha"] == IMPLEMENTATION_SHA
    assert summary["config_sha256"] == CONFIG_SHA256
    assert all(
        record["exact_match"]
        for record in summary["optimized_full_old_decision_regression"].values()
    )
    assert summary["all_adaptive_certified_or_fallback"]
    assert summary["adaptive_fallback_decision_count"] == 3
    assert summary["adaptive_shadow_full_action_agreement_fraction"] == 1.0
    assert summary["median_adaptive_active_factor_fraction"] == 1.0
    assert not summary["pathological_stop"]
    assert summary["median_adaptive_to_full_conditioning_time_ratio"] <= 1.25

    timing = _csv_rows(OUTPUT / "timing_and_work.csv")
    assert len(timing) == 72
    adaptive = [row for row in timing if row["method"] == "ADAPTIVE_PBE"]
    full = [row for row in timing if row["method"] == "FULL_PBE_OPT"]
    assert len(adaptive) == len(full) == 36
    assert all(
        row["structurally_certified"] == "True"
        or row["full_bank_fallback"] == "True"
        for row in adaptive
    )
    assert all(int(row["active_factor_count"]) == 124718 for row in adaptive)
    assert all(int(row["factor_hessian_element_work"]) >= 0 for row in timing)
    assert all(int(row["factor_energy_gradient_element_work"]) >= 0 for row in timing)
    assert all(float(row["pbe_conditioning_seconds"]) >= 0.0 for row in timing)
    shadow = _csv_rows(OUTPUT / "shadow_full.csv")
    assert len(shadow) == 36
    assert all(row["action_agreement"] == "True" for row in shadow)
    assert max(
        float(row["shadow_full_laplace_ei_regret_standardized"]) for row in shadow
    ) == 0.0


def test_smoke_oracle_access_is_exactly_the_authorized_trajectories() -> None:
    trajectories = _csv_rows(OUTPUT / "trajectories.csv")
    assert len(trajectories) == 3 * 2 * 20
    access = json.loads((OUTPUT / "oracle_access_log.json").read_text())
    assert len(access) == 3 * (8 + 12 + 12)
    assert all(row["access"] == "queried_action" for row in access)
    assert sum(row["method"] == "SHARED_INITIAL" for row in access) == 24
    assert sum(row["method"] == "FULL_PBE_OPT" for row in access) == 36
    assert sum(row["method"] == "ADAPTIVE_PBE" for row in access) == 36
    for seed in range(3):
        expected_initial = np.random.default_rng(seed).choice(191, size=8, replace=False)
        logged_initial = [
            row["action_position"]
            for row in access
            if row["method"] == "SHARED_INITIAL" and row["seed"] == seed
        ]
        np.testing.assert_array_equal(logged_initial, expected_initial)
        for method in ("FULL_PBE_OPT", "ADAPTIVE_PBE"):
            logged = [
                row["action_position"]
                for row in access
                if row["method"] == method and row["seed"] == seed
            ]
            saved = [
                int(row["action_position"])
                for row in sorted(
                    (
                        row
                        for row in trajectories
                        if int(row["seed"]) == seed
                        and row["method"] == method
                        and row["phase"] == "sequential"
                    ),
                    key=lambda row: int(row["query_index"]),
                )
            ]
            assert logged == saved

