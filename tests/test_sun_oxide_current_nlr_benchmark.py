from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
BUILDER_PATH = ROOT / "experiments/sun_oxide/current_nlr_benchmark.py"
BENCHMARK_DIR = ROOT / "experiments/sun_oxide/benchmark"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BUILDER = _load("sun_oxide_current_nlr_benchmark_test", BUILDER_PATH)


def _row(mident: int, formula: str, energy: str, family: int | None = None) -> dict[str, object]:
    return {
        "mident": mident,
        "sorted_formula": formula,
        "icsd_space_group": "",
        "final_space_group": "1",
        "total_energy_per_atom_ev": energy,
        "standards": "['fere']",
        "parents": "",
        "family_mident": family,
    }


def _gw_row(mident: int, formula: str, parent: int, family: int) -> dict[str, object]:
    return {
        "mident": mident,
        "sorted_formula": formula,
        "icsd_space_group": "",
        "final_space_group": "1",
        "icsd_id": "",
        "standards": "['gwvd']",
        "parents": f"[{parent}]",
        "family_mident": family,
    }


def _wave_row(mident: int, formula: str, energy: str, family: int) -> dict[str, object]:
    return {
        "mident": mident,
        "sorted_formula": formula,
        "icsd_space_group": "",
        "final_space_group": "1",
        "total_energy_per_atom_ev": energy,
        "standards": "['wave']",
        "parents": f"[{family}]",
        "family_mident": family,
    }


def test_pbe_canonical_selection_uses_energy_then_stable_id() -> None:
    rows = [
        _row(30, "O2 Ti", "-4.0"),
        _row(20, "O2 Ti", "-5.0"),
        _row(10, "O2 Ti", "-5.0"),
        _row(40, "O Sn", "-3.0"),
    ]
    selected_a, summary_a = BUILDER.select_canonical_pbe_rows(rows)
    selected_b, summary_b = BUILDER.select_canonical_pbe_rows(list(reversed(rows)))
    assert [row["mident"] for row in selected_a] == [40, 10]
    assert [row["mident"] for row in selected_b] == [40, 10]
    assert summary_a == summary_b
    assert summary_a["exact_displayed_energy_tie_groups"] == 1


def test_gw_selection_uses_only_wave_parent_energy_and_stable_id() -> None:
    gw_rows = [
        _gw_row(300, "O2 Ti", 301, 100),
        _gw_row(200, "O2 Ti", 201, 101),
        _gw_row(100, "O Sn", 101, 102),
    ]
    wave_rows = [
        _wave_row(301, "O2 Ti", "-3.0", 100),
        _wave_row(201, "O2 Ti", "-4.0", 101),
        _wave_row(101, "O Sn", "-2.0", 102),
    ]
    selected, summary = BUILDER.select_gw_rows(gw_rows, wave_rows)
    assert [row["mident"] for row in selected] == [100, 200]
    assert summary == {
        "raw_rows": 3,
        "unique_compositions": 2,
        "single_row_compositions": 1,
        "multiple_row_compositions": 1,
        "multiple_resolved_without_tie": 1,
        "exact_energy_tie_groups": 0,
    }


def test_selection_rejects_any_gw_target_field() -> None:
    gw = _gw_row(300, "O2 Ti", 301, 100)
    gw["gw_band_gap_ev"] = "unused"
    with pytest.raises(ValueError, match="forbidden target fields"):
        BUILDER.select_gw_rows([gw], [_wave_row(301, "O2 Ti", "-3.0", 100)])


def test_committed_benchmark_counts_mapping_and_isolated_oracle() -> None:
    result = BUILDER.validate_outputs(BENCHMARK_DIR, require_oracle=False)
    manifest = result["manifest"]
    assert result["legacy_rows"] == 2142
    assert result["action_rows"] == 191
    assert manifest["counts"] == {
        "gw_exact_energy_ties": 0,
        "gw_multiple_compositions": 28,
        "gw_multiple_resolved_without_tie": 28,
        "gw_raw_rows": 244,
        "gw_single_compositions": 166,
        "initial_gw_compositions": 194,
        "legacy_compositions": 2142,
        "pbe_raw_rows": 5604,
        "strict_gw_actions": 191,
    }
    assert [row["display_formula"] for row in manifest["strict_mapping"]["excluded"]] == [
        "CdO",
        "Ga2O3",
        "Sb2O3",
    ]
    assert manifest["target_isolation"]["gw_target_used_for_candidate_selection"] is False
    assert manifest["target_isolation"]["gw_target_used_for_strict_mapping"] is False
    assert manifest["target_isolation"]["oracle_columns"] == [
        "action_key",
        "gw_band_gap_ev",
    ]


def test_new_selection_code_cannot_read_gw_magnitudes() -> None:
    builder_source = BUILDER_PATH.read_text(encoding="utf-8")
    assert "gw_band_gap_ev" not in builder_source
    assert "gw_oracle.csv" not in builder_source
    config = json.loads(
        (ROOT / "experiments/sun_oxide/configs/current_nlr_pbe_gw_v1.json").read_text(encoding="utf-8")
    )
    assert config["queries"]["gw_metadata"]["form"][3] == ["qrestrictExpr", ""]
    assert all("gap" not in field.lower() for field in config["queries"]["gw_metadata"]["fields"])
    with (BENCHMARK_DIR / "current_nlr_gw_actions.csv").open(
        "r", encoding="utf-8", newline=""
    ) as stream:
        action_columns = next(csv.reader(stream))
    assert "gw_band_gap_ev" not in action_columns
    manifest = json.loads((BENCHMARK_DIR / "benchmark_manifest.json").read_text())
    assert manifest["target_isolation"]["oracle_columns"] == [
        "action_key",
        "gw_band_gap_ev",
    ]


def test_historical_source_recovery_outputs_remain_immutable() -> None:
    expected = {
        "SOURCE_AUDIT.md": "c01613d8b906e56f269c65ce95d1c40a697ebec85139d5e292da7cb8a412cbcd",
        "descriptor_manifest.json": "be8e1ab7aab40ea3f9261781bd0e9cf5e849d54bc83ef72e35f02dbca07a7b81",
        "normalized_schema.json": "c03e352038e61eca4cb209ba21257f060d71ffa5815db112048f3283425e2034",
        "reconstruction_summary.json": "4d8991dc2836b6d051cf14e28c7051718a7172bbc5d29e2564cd8d20f6e8493c",
        "source_manifest.json": "5a7a9b106f38fbff790c882ae52bc8313cab125d12bd64413d530160838c356a",
    }
    recovery = ROOT / "experiments/sun_oxide/outputs/source_recovery"
    actual = {name: hashlib.sha256((recovery / name).read_bytes()).hexdigest() for name in expected}
    assert actual == expected
