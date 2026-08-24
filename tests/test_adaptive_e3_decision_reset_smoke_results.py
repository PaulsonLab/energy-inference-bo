from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments/sun_oxide/outputs/adaptive_e3_decision_reset_smoke"
IMPLEMENTATION_SHA = "1fab6fb6afd626381a87df55d2d6b348e7475584"
CONFIG_SHA256 = "01cabcda9c4d50ae6b6b498467fa570ea5167b7c89da4feb6a8ec890f6c33c4c"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def test_decision_reset_smoke_archive_is_complete_and_immutable() -> None:
    manifest = json.loads((OUTPUT / "artifact_manifest.json").read_text())
    assert manifest["implementation_sha"] == IMPLEMENTATION_SHA
    assert manifest["config_sha256"] == CONFIG_SHA256
    assert manifest["terminal_state"] == "ADAPTIVE_ENGINEERING_PATHOLOGICAL"
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


def test_decision_reset_smoke_records_the_engineering_blocker() -> None:
    summary = json.loads((OUTPUT / "run_summary.json").read_text())
    assert summary["terminal_state"] == "ADAPTIVE_ENGINEERING_PATHOLOGICAL"
    assert summary["mechanically_sound"]
    assert not summary["scientific_evidence"]
    assert summary["implementation_sha"] == IMPLEMENTATION_SHA
    assert summary["config_sha256"] == CONFIG_SHA256
    assert summary["pathological_stop"]
    assert summary["adaptive_fallback_decision_count"] == 36
    assert summary["all_adaptive_certified_or_fallback"]
    assert summary["adaptive_shadow_full_action_agreement_fraction"] == 1.0
    assert summary["median_adaptive_active_factor_fraction"] == 1.0
    assert summary["median_adaptive_to_full_conditioning_time_ratio"] > 1.25
    assert all(
        record["exact_match"]
        for record in summary["optimized_full_old_decision_regression"].values()
    )

    timing = _csv_rows(OUTPUT / "timing_and_work.csv")
    adaptive = [row for row in timing if row["method"] == "ADAPTIVE_PBE"]
    full = [row for row in timing if row["method"] == "FULL_PBE_OPT"]
    assert len(adaptive) == len(full) == 36
    assert all(int(row["active_factor_count"]) == 124718 for row in adaptive)
    assert all(int(row["adaptive_stages"]) == 8 for row in adaptive)
    assert all(row["full_bank_fallback"] == "True" for row in adaptive)
    assert all(
        row["active_set_retained_from_previous_bo_iteration"] == "False"
        for row in adaptive
    )
    later = [row for row in adaptive if int(row["query_index"]) >= 2]
    assert len(later) == 33
    assert all(row["warm_start_across_bo_used"] == "True" for row in later)
    assert sum(float(row["pbe_conditioning_seconds"]) for row in adaptive) > sum(
        float(row["pbe_conditioning_seconds"]) for row in full
    )
    assert sum(int(row["factor_energy_gradient_element_work"]) for row in adaptive) > sum(
        int(row["factor_energy_gradient_element_work"]) for row in full
    )
    assert sum(int(row["factor_hessian_element_work"]) for row in adaptive) > sum(
        int(row["factor_hessian_element_work"]) for row in full
    )

    shadow = _csv_rows(OUTPUT / "shadow_full.csv")
    assert len(shadow) == 36
    assert all(row["action_agreement"] == "True" for row in shadow)
    assert max(
        float(row["shadow_full_laplace_ei_regret_standardized"]) for row in shadow
    ) == 0.0


def test_decision_reset_smoke_did_not_spend_fresh_seeds() -> None:
    access = json.loads((OUTPUT / "oracle_access_log.json").read_text())
    assert len(access) == 3 * (8 + 12 + 12)
    assert {row["seed"] for row in access} == {0, 1, 2}
    assert all(row["access"] == "queried_action" for row in access)
    assert {row["method"] for row in access} == {
        "SHARED_INITIAL",
        "FULL_PBE_OPT",
        "ADAPTIVE_PBE",
    }
