from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments/sun_oxide/outputs/full_bank_scaling_probe"


def _json(name: str):
    return json.loads((OUTPUT / name).read_text(encoding="utf-8"))


def _csv(name: str) -> list[dict[str, str]]:
    with (OUTPUT / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_full_bank_scaling_result_is_development_only_and_complete() -> None:
    resource = _json("resource_usage.json")
    assert resource["classification"] == "FULL_ARCHIVE_NOT_HELPFUL"
    assert resource["development_only"]
    assert not resource["fresh_seeds_accessed"]
    assert not resource["local_resource_blocked"]
    assert resource["peak_rss_bytes"] < resource["rss_limit_bytes"]

    support = _json("support_summary.json")
    expected = {
        "500": (124718, 32),
        "1000": (499361, 139),
        "2142": (2292440, 571),
    }
    for size, (factor_count, ties) in expected.items():
        model = support["models"][size]
        assert model["strict_factor_count"] == factor_count
        assert model["omitted_exact_tie_pair_count"] == ties
        assert model["maximum_weighted_incident_degree"] <= 1.0
        assert not model["pbe_magnitude_used_for_selection"]
        assert not model["gw_values_used_for_selection"]


def test_full_archive_theory_signal_and_implicit_residuals_pass() -> None:
    theory = _json("theory_summary.json")
    full_theory = theory["models"]["2142"]
    assert full_theory["a0_smallest_eigenvalue"] >= 0.75
    assert full_theory["a0_eigen_residual"] <= 1e-9
    assert not full_theory["dense_inverse_formed"]
    regression = theory["implicit_500_regression"]
    assert regression["w_max_abs_error"] <= 1e-8
    assert regression["v_max_abs_error"] <= 1e-8

    signal = _json("pbe_signal_summary.json")["models"]["2142"]
    assert signal["actions"]["spearman_pbe_rank_vs_map_rank"] > 0.97
    assert signal["support"]["spearman_pbe_rank_vs_map_rank"] > 0.97
    assert signal["optimization"]["gradient_infinity_norm"] <= 1e-7

    stages = _csv("adaptive_stage_scaling.csv")
    for field in ("load_solve_relative_residual", "pair_solve_relative_residual"):
        residuals = [float(row[field]) for row in stages if row[field]]
        assert max(residuals) <= 1e-9


def test_primary_probe_falls_back_without_oracle_leakage() -> None:
    full = _csv("full_runtime_scaling.csv")
    assert len(full) == 9
    assert {int(row["authorized_target_rows_read"]) for row in full} == {8, 14, 20}
    assert all(int(row["unauthorized_later_target_rows_read"]) == 0 for row in full)
    leaders_500 = [
        int(row["full_leader"]) for row in full if row["support_count"] == "500"
    ]
    assert leaders_500 == [13, 134, 133]

    stages = _csv("adaptive_stage_scaling.csv")
    fallbacks = [row for row in stages if row["record_kind"] == "fallback"]
    assert len(fallbacks) == 9
    assert all(row["full_bank_fallback"] == "True" for row in fallbacks)
    assert not any(row["certified"] == "True" for row in stages)

    summary = _json("support_summary.json")["structural_scaling_summary"]
    assert summary["2142"]["median_pre_fallback_active_factor_count"] == 2230019
    assert summary["2142"]["median_pre_fallback_active_factor_fraction"] > 0.97
    assert summary["scaling"]["active_count_power_vs_support_size"] > 2.0


def test_artifact_manifest_hashes_match() -> None:
    manifest = _json("artifact_manifest.json")
    assert manifest["implementation_sha"] == (
        "aa39cb19b1198bd433a70d63e00dc0dc39ec1fb1"
    )
    assert manifest["config_sha256"] == (
        "046c9358f06d16377e790cf4af112be37d5c9458669c0c1b7a31b3a23b9af392"
    )
    for artifact in manifest["files"]:
        path = OUTPUT / artifact["path"]
        assert path.stat().st_size == artifact["size_bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact["sha256"]
