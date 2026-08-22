#!/usr/bin/env python3
"""Freeze the target-blind CURRENT_NLR_PBE_GW_V1 benchmark tables.

This module selects canonical PBE/FERE rows and GW actions using only source
metadata and wave-parent energies. It deliberately cannot read the GW oracle.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "experiments/sun_oxide/configs/current_nlr_pbe_gw_v1.json"
DEFAULT_METADATA_CACHE = ROOT / "data/external/sun_oxide/current_nlr_pbe_gw_v1"
DEFAULT_SOURCE_CACHE = ROOT / "data/external/sun_oxide/source_recovery"
DEFAULT_OUTPUT = ROOT / "experiments/sun_oxide/benchmark"

LEGACY_COLUMNS = [
    "composition_key",
    "normalized_formula",
    "pbe_mident",
    "pbe_family_mident",
    "pbe_parent_ids",
    "pbe_icsd_space_group",
    "pbe_final_space_group",
    "pbe_total_energy_per_atom_ev",
    "pbe_band_gap_ev",
    "selection_rule",
    "source_query",
]

ACTION_COLUMNS = [
    "action_key",
    "composition_key",
    "normalized_formula",
    "gw_mident",
    "gw_family_mident",
    "gw_parent_ids",
    "gw_wave_parent_mident",
    "gw_icsd_id",
    "gw_icsd_space_group",
    "gw_final_space_group",
    "wave_parent_family_mident",
    "wave_parent_parent_ids",
    "wave_parent_icsd_space_group",
    "wave_parent_final_space_group",
    "wave_parent_total_energy_per_atom_ev",
    "strict_pbe_family_anchor_mident",
    "canonical_pbe_mident",
    "selection_rule",
    "source_query",
]

DATA_USE_NOTICE = """DATA USE DISCLAIMER AGREEMENT ("Agreement")

These data ("Data") are provided by the National Laboratory of the Rockies
("NLR"), which is operated by the Alliance for Sustainable Energy, LLC
("ALLIANCE") for the U.S. Department Of Energy ("DOE").

Access to and use of these Data shall impose the following obligations on the
user, as set forth in this Agreement. The user is granted the right, without
any fee or cost, to use, copy, and distribute these Data for any purpose
whatsoever, provided that this entire notice appears in all copies of the Data.
Further, the user agrees to credit DOE/NLR/ALLIANCE in any publication that
results from the use of these Data. The names DOE/NLR/ALLIANCE, however, may
not be used in any advertising or publicity to endorse or promote any products
or commercial entities unless specific written permission is obtained from
DOE/NLR/ALLIANCE. The user also understands that DOE/NLR/ALLIANCE is not
obligated to provide the user with any support, consulting, training or
assistance of any kind with regard to the use of these Data or to provide the
user with any updates, revisions or new versions of these Data.

YOU AGREE TO INDEMNIFY DOE/NLR/ALLIANCE, AND ITS SUBSIDIARIES, AFFILIATES,
OFFICERS, AGENTS, AND EMPLOYEES AGAINST ANY CLAIM OR DEMAND, INCLUDING
REASONABLE ATTORNEYS' FEES, RELATED TO YOUR USE OF THESE DATA. THESE DATA ARE
PROVIDED BY DOE/NLR/ALLIANCE "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
INCLUDING BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL
DOE/NLR/ALLIANCE BE LIABLE FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES
OR ANY DAMAGES WHATSOEVER, INCLUDING BUT NOT LIMITED TO CLAIMS ASSOCIATED WITH
THE LOSS OF DATA OR PROFITS, WHICH MAY RESULT FROM AN ACTION IN CONTRACT,
NEGLIGENCE OR OTHER TORTIOUS CLAIM THAT ARISES OUT OF OR IN CONNECTION WITH
THE ACCESS, USE OR PERFORMANCE OF THESE DATA.

Source: https://materials.nlr.gov/help?hname=disclaimer
Captured: 2026-08-22
"""


def _load_source_audit() -> Any:
    path = ROOT / "experiments/sun_oxide/source_audit.py"
    spec = importlib.util.spec_from_file_location("sun_oxide_source_audit_for_benchmark", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load source audit helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


AUDIT = _load_source_audit()


def sha256_file(path: Path) -> str:
    return AUDIT.sha256_file(path)


def canonical_json_bytes(value: Any) -> bytes:
    return AUDIT.canonical_json_bytes(value)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.write_bytes(canonical_json_bytes(value))


def _relative_to_root(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _manifest_entry(
    path: Path,
    *,
    rows: int | None = None,
    columns: list[str] | None = None,
    logical_path: str | None = None,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "path": logical_path or _relative_to_root(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }
    if rows is not None:
        entry["rows"] = rows
    if columns is not None:
        entry["columns"] = columns
    return entry


def _parse_int_list(value: str, *, field: str, mident: int) -> tuple[int, ...]:
    if not value.strip():
        return ()
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"Invalid {field} for MatDB {mident}: {value!r}") from exc
    if not isinstance(parsed, list) or any(not isinstance(item, int) for item in parsed):
        raise ValueError(f"Invalid {field} for MatDB {mident}: {value!r}")
    return tuple(parsed)


def _finite_float(value: Any, *, field: str, mident: int) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Non-numeric {field} for MatDB {mident}") from exc
    if not math.isfinite(result):
        raise ValueError(f"Non-finite {field} for MatDB {mident}")
    return result


def _format_float(value: float) -> str:
    return format(value, ".15g")


def _format_optional(value: Any) -> str:
    return "" if value is None or str(value).strip() == "" else str(value).strip()


def _format_int_list(values: Iterable[int]) -> str:
    return json.dumps(list(values), separators=(",", ":"))


def composition_key(formula: str) -> str:
    normalized = AUDIT.normalized_formula_key(formula)
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return f"current-nlr-pbe-gw-v1:composition:{digest}"


def action_key(mident: int) -> str:
    return f"current-nlr-pbe-gw-v1:gw:matdb:{mident:09d}"


def _assert_target_free(rows: Iterable[dict[str, Any]], *, label: str) -> None:
    forbidden = {"target", "oracle", "band_gap", "bandgap"}
    for row in rows:
        bad = sorted(key for key in row if any(token in key.lower() for token in forbidden))
        if bad:
            raise ValueError(f"{label} selection metadata contains forbidden target fields: {bad}")


def fetch_metadata(config_path: Path, cache_dir: Path) -> dict[str, Any]:
    """Fetch only target-free selection metadata and pin every page hash."""
    config = _load_json(config_path)
    manifest_path = cache_dir / "cache_manifest.json"
    if manifest_path.exists():
        manifest = _load_json(manifest_path)
        if manifest["config_sha256"] != sha256_file(config_path):
            raise ValueError("Metadata cache config hash does not match the frozen benchmark config")
        for entry in manifest["files"]:
            AUDIT.verify_cache_entry(cache_dir / entry["local_filename"], entry["sha256"])
        return manifest
    if cache_dir.exists() and any(cache_dir.iterdir()):
        raise FileExistsError(f"Refusing incomplete or mixed metadata cache without manifest: {cache_dir}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    for query_name in ("pbe_metadata", "gw_metadata", "wave_metadata"):
        query_entries = AUDIT._fetch_query(config, cache_dir, query_name)
        for entry in query_entries:
            entry.update(
                {
                    "role": f"current_nlr_benchmark_{query_name}_page",
                    "retrieval_date": config["freeze_date"],
                    "source_version": config["sources"]["nrel_matdb"]["source_version"],
                    "license": config["sources"]["nrel_matdb"]["terms"],
                }
            )
        entries.extend(query_entries)
    manifest = {
        "schema_version": 1,
        "benchmark_name": config["benchmark_name"],
        "download_date": config["freeze_date"],
        "config_sha256": sha256_file(config_path),
        "target_free": True,
        "files": sorted(entries, key=lambda item: item["local_filename"]),
    }
    _write_json(manifest_path, manifest)
    return manifest


def _verified_manifest(cache_dir: Path) -> dict[str, Any]:
    manifest_path = cache_dir / "cache_manifest.json"
    manifest = _load_json(manifest_path)
    for entry in manifest["files"]:
        AUDIT.verify_cache_entry(cache_dir / entry["local_filename"], entry["sha256"])
    return manifest


def _load_metadata_rows(
    config: dict[str, Any], cache_dir: Path, query_name: str
) -> tuple[int, list[dict[str, Any]]]:
    query = config["queries"][query_name]
    paths = sorted(cache_dir.glob(f"{query['asset_prefix']}_page_*.html"))
    if not paths:
        raise FileNotFoundError(f"No cached pages for {query_name}")
    totals: set[int] = set()
    rows: list[dict[str, Any]] = []
    for path in paths:
        total, page_rows = AUDIT.parse_matdb_page(path, query["fields"])
        totals.add(total)
        rows.extend(page_rows)
    if len(totals) != 1:
        raise ValueError(f"Inconsistent totals for {query_name}: {sorted(totals)}")
    total = totals.pop()
    if total != len(rows):
        raise ValueError(f"{query_name} total {total} != parsed row count {len(rows)}")
    return total, AUDIT.deterministic_order(rows)


def select_canonical_pbe_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Select PBE rows without receiving any target-valued columns."""
    _assert_target_free(rows, label="PBE")
    groups = AUDIT.group_by_formula(rows)
    selected: list[dict[str, Any]] = []
    tie_groups = 0
    duplicate_groups = 0
    for formula in sorted(groups):
        candidates = groups[formula]
        if len(candidates) > 1:
            duplicate_groups += 1
        energies = {
            int(row["mident"]): _finite_float(
                row["total_energy_per_atom_ev"], field="PBE total energy per atom", mident=int(row["mident"])
            )
            for row in candidates
        }
        minimum = min(energies.values())
        tied = [row for row in candidates if energies[int(row["mident"])] == minimum]
        tie_groups += len(tied) > 1
        chosen = min(tied, key=lambda row: int(row["mident"]))
        selected.append({**chosen, "_selection_energy": energies[int(chosen["mident"])]})
    return selected, {
        "raw_rows": len(rows),
        "unique_compositions": len(groups),
        "duplicate_composition_groups": duplicate_groups,
        "exact_displayed_energy_tie_groups": tie_groups,
    }


def select_gw_rows(
    gw_rows: list[dict[str, Any]], wave_rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Select GW rows using only authoritative parent metadata and energy."""
    _assert_target_free(gw_rows, label="GW")
    _assert_target_free(wave_rows, label="wave")
    wave_by_id = {int(row["mident"]): row for row in wave_rows}
    if len(wave_by_id) != len(wave_rows):
        raise ValueError("Duplicate MatDB IDs in wave metadata")
    prepared: list[dict[str, Any]] = []
    for gw in gw_rows:
        gw_id = int(gw["mident"])
        parents = _parse_int_list(gw["parents"], field="GW parents", mident=gw_id)
        if len(parents) != 1:
            raise ValueError(f"GW MatDB {gw_id} has {len(parents)} parents; expected one wave parent")
        wave = wave_by_id.get(parents[0])
        if wave is None:
            raise ValueError(f"GW MatDB {gw_id} wave parent {parents[0]} is absent")
        if AUDIT.normalized_formula_key(gw["sorted_formula"]) != AUDIT.normalized_formula_key(
            wave["sorted_formula"]
        ):
            raise ValueError(f"GW MatDB {gw_id} and wave parent {parents[0]} have different compositions")
        wave_standards = _parse_string_list(wave["standards"], field="wave standards", mident=parents[0])
        if "wave" not in wave_standards:
            raise ValueError(f"GW MatDB {gw_id} parent {parents[0]} is not authoritative wave metadata")
        family = gw["family_mident"]
        wave_family = wave["family_mident"]
        wave_parents = _parse_int_list(wave["parents"], field="wave parents", mident=parents[0])
        if family is None or wave_family != family or wave_parents != (family,):
            raise ValueError(f"GW MatDB {gw_id} has inconsistent GW/wave family lineage")
        energy = _finite_float(
            wave["total_energy_per_atom_ev"], field="wave total energy per atom", mident=parents[0]
        )
        prepared.append(
            {
                **gw,
                "_gw_parents": parents,
                "_wave": wave,
                "_wave_parents": wave_parents,
                "_selection_energy": energy,
            }
        )
    groups = AUDIT.group_by_formula(prepared)
    selected: list[dict[str, Any]] = []
    single = 0
    multiple = 0
    exact_ties = 0
    for formula in sorted(groups):
        candidates = groups[formula]
        if len(candidates) == 1:
            single += 1
            selected.append(candidates[0])
            continue
        multiple += 1
        minimum = min(row["_selection_energy"] for row in candidates)
        tied = [row for row in candidates if row["_selection_energy"] == minimum]
        exact_ties += len(tied) > 1
        selected.append(min(tied, key=lambda row: int(row["mident"])))
    return selected, {
        "raw_rows": len(gw_rows),
        "unique_compositions": len(groups),
        "single_row_compositions": single,
        "multiple_row_compositions": multiple,
        "multiple_resolved_without_tie": multiple - exact_ties,
        "exact_energy_tie_groups": exact_ties,
    }


def _parse_string_list(value: str, *, field: str, mident: int) -> tuple[str, ...]:
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"Invalid {field} for MatDB {mident}: {value!r}") from exc
    if not isinstance(parsed, list) or any(not isinstance(item, str) for item in parsed):
        raise ValueError(f"Invalid {field} for MatDB {mident}: {value!r}")
    return tuple(parsed)


def strict_map_gw_rows(
    selected_gw: list[dict[str, Any]], pbe_rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Map by exact composition and the authoritative family anchor."""
    _assert_target_free(selected_gw, label="selected GW")
    _assert_target_free(pbe_rows, label="PBE")
    pbe_groups = AUDIT.group_by_formula(pbe_rows)
    retained: list[dict[str, Any]] = []
    excluded: list[dict[str, str]] = []
    for gw in selected_gw:
        formula = AUDIT.normalized_formula_key(gw["sorted_formula"])
        candidates = pbe_groups.get(formula, [])
        if not candidates:
            excluded.append(
                {
                    "normalized_formula": formula,
                    "reason_code": "COMPOSITION_ABSENT_FROM_PBE_FERE",
                }
            )
            continue
        family = int(gw["family_mident"])
        anchors = [row for row in candidates if int(row["mident"]) == family]
        if not anchors:
            excluded.append(
                {
                    "normalized_formula": formula,
                    "reason_code": "SELECTED_GW_FAMILY_ABSENT_FROM_PBE_FERE",
                }
            )
            continue
        if len(anchors) != 1:
            raise ValueError(f"GW composition {formula} maps to {len(anchors)} PBE family anchors")
        retained.append({**gw, "_strict_pbe_family_anchor": anchors[0]})
    return retained, excluded


def _load_pbe_values(source_cache_dir: Path) -> tuple[list[dict[str, Any]], dict[int, str]]:
    """Mechanically attach PBE legacy values after metadata-only selection."""
    source_config = _load_json(ROOT / "experiments/sun_oxide/configs/source_recovery.json")
    _, rows = AUDIT._load_query_rows(source_config, source_cache_dir, "pbe")
    values: dict[int, str] = {}
    for row in rows:
        mident = int(row["mident"])
        value = _finite_float(row["pbe_band_gap_ev"], field="PBE band gap", mident=mident)
        values[mident] = _format_float(value)
    if len(values) != len(rows):
        raise ValueError("Duplicate MatDB IDs in PBE value source")
    return rows, values


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label}: observed {actual!r}, expected {expected!r}")


def _source_entries(manifest: dict[str, Any], *, query_name: str) -> list[dict[str, Any]]:
    return [
        {
            "local_filename": entry["local_filename"],
            "sha256": entry["sha256"],
            "size_bytes": entry["size_bytes"],
        }
        for entry in manifest["files"]
        if entry.get("query_name") == query_name
    ]


def build_benchmark(
    config_path: Path, metadata_cache_dir: Path, source_cache_dir: Path, output_dir: Path
) -> dict[str, Any]:
    config = _load_json(config_path)
    expected = config["expected"]
    metadata_manifest = _verified_manifest(metadata_cache_dir)
    _require_equal(metadata_manifest.get("target_free"), True, "metadata cache target isolation")
    source_manifest = _verified_manifest(source_cache_dir)

    pbe_total, pbe_rows = _load_metadata_rows(config, metadata_cache_dir, "pbe_metadata")
    gw_total, gw_rows = _load_metadata_rows(config, metadata_cache_dir, "gw_metadata")
    wave_total, wave_rows = _load_metadata_rows(config, metadata_cache_dir, "wave_metadata")
    _assert_target_free(pbe_rows, label="PBE")
    _assert_target_free(gw_rows, label="GW")
    _assert_target_free(wave_rows, label="wave")

    canonical_pbe, pbe_summary = select_canonical_pbe_rows(pbe_rows)
    selected_gw, gw_summary = select_gw_rows(gw_rows, wave_rows)
    retained_gw, exclusions = strict_map_gw_rows(selected_gw, pbe_rows)

    _require_equal(pbe_total, expected["pbe_raw_rows"], "PBE raw row count")
    _require_equal(pbe_summary["unique_compositions"], expected["pbe_unique_compositions"], "PBE count")
    _require_equal(gw_total, expected["gw_raw_rows"], "GW raw row count")
    _require_equal(wave_total, expected["gw_raw_rows"], "wave raw row count")
    _require_equal(gw_summary["unique_compositions"], expected["gw_unique_compositions"], "GW count")
    _require_equal(gw_summary["single_row_compositions"], expected["gw_single_compositions"], "GW singles")
    _require_equal(
        gw_summary["multiple_row_compositions"], expected["gw_multiple_compositions"], "GW multiples"
    )
    _require_equal(
        gw_summary["multiple_resolved_without_tie"],
        expected["gw_multiple_resolved_without_tie"],
        "GW uniquely resolved multiples",
    )
    _require_equal(gw_summary["exact_energy_tie_groups"], expected["gw_exact_energy_ties"], "GW ties")
    _require_equal(len(retained_gw), expected["strict_gw_actions"], "strict GW actions")

    expected_exclusions = [
        {"normalized_formula": item["normalized_formula"], "reason_code": item["reason_code"]}
        for item in expected["strict_mapping_exclusions"]
    ]
    _require_equal(exclusions, expected_exclusions, "strict mapping exclusions")

    pbe_value_rows, pbe_values = _load_pbe_values(source_cache_dir)
    pbe_metadata_by_id = {int(row["mident"]): row for row in pbe_rows}
    _require_equal(set(pbe_values), set(pbe_metadata_by_id), "PBE metadata/value MatDB IDs")
    for row in pbe_value_rows:
        metadata = pbe_metadata_by_id[int(row["mident"])]
        _require_equal(
            AUDIT.normalized_formula_key(row["sorted_formula"]),
            AUDIT.normalized_formula_key(metadata["sorted_formula"]),
            f"PBE formula for MatDB {row['mident']}",
        )

    canonical_by_formula = {
        AUDIT.normalized_formula_key(row["sorted_formula"]): row for row in canonical_pbe
    }
    pbe_rule = config["selection_rules"]["pbe"]
    gw_rule = config["selection_rules"]["gw"]
    legacy_rows: list[dict[str, Any]] = []
    for selected in canonical_pbe:
        formula = AUDIT.normalized_formula_key(selected["sorted_formula"])
        mident = int(selected["mident"])
        parents = _parse_int_list(selected["parents"], field="PBE parents", mident=mident)
        legacy_rows.append(
            {
                "composition_key": composition_key(formula),
                "normalized_formula": formula,
                "pbe_mident": mident,
                "pbe_family_mident": _format_optional(selected["family_mident"]),
                "pbe_parent_ids": _format_int_list(parents),
                "pbe_icsd_space_group": _format_optional(selected["icsd_space_group"]),
                "pbe_final_space_group": _format_optional(selected["final_space_group"]),
                "pbe_total_energy_per_atom_ev": _format_float(selected["_selection_energy"]),
                "pbe_band_gap_ev": pbe_values[mident],
                "selection_rule": pbe_rule,
                "source_query": "pbe_metadata + source_recovery:pbe",
            }
        )

    action_rows: list[dict[str, Any]] = []
    for selected in retained_gw:
        formula = AUDIT.normalized_formula_key(selected["sorted_formula"])
        gw_id = int(selected["mident"])
        wave = selected["_wave"]
        wave_id = int(wave["mident"])
        canonical = canonical_by_formula[formula]
        action_rows.append(
            {
                "action_key": action_key(gw_id),
                "composition_key": composition_key(formula),
                "normalized_formula": formula,
                "gw_mident": gw_id,
                "gw_family_mident": int(selected["family_mident"]),
                "gw_parent_ids": _format_int_list(selected["_gw_parents"]),
                "gw_wave_parent_mident": wave_id,
                "gw_icsd_id": _format_optional(selected["icsd_id"]),
                "gw_icsd_space_group": _format_optional(selected["icsd_space_group"]),
                "gw_final_space_group": _format_optional(selected["final_space_group"]),
                "wave_parent_family_mident": int(wave["family_mident"]),
                "wave_parent_parent_ids": _format_int_list(selected["_wave_parents"]),
                "wave_parent_icsd_space_group": _format_optional(wave["icsd_space_group"]),
                "wave_parent_final_space_group": _format_optional(wave["final_space_group"]),
                "wave_parent_total_energy_per_atom_ev": _format_float(selected["_selection_energy"]),
                "strict_pbe_family_anchor_mident": int(selected["_strict_pbe_family_anchor"]["mident"]),
                "canonical_pbe_mident": int(canonical["mident"]),
                "selection_rule": gw_rule,
                "source_query": "gw_metadata + wave_metadata",
            }
        )

    legacy_rows.sort(key=lambda row: (row["normalized_formula"], int(row["pbe_mident"])))
    action_rows.sort(key=lambda row: (row["normalized_formula"], int(row["gw_mident"])))
    _require_equal(len({row["composition_key"] for row in legacy_rows}), len(legacy_rows), "legacy keys")
    _require_equal(len({row["action_key"] for row in action_rows}), len(action_rows), "action keys")
    _require_equal(
        len({row["composition_key"] for row in action_rows}), len(action_rows), "one action per legacy key"
    )
    legacy_keys = {row["composition_key"] for row in legacy_rows}
    _require_equal(
        sum(row["composition_key"] in legacy_keys for row in action_rows),
        len(action_rows),
        "GW-to-legacy mapping",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    legacy_path = output_dir / "current_nlr_legacy.csv"
    actions_path = output_dir / "current_nlr_gw_actions.csv"
    notice_path = output_dir / "NLR_DATA_USE_NOTICE.txt"
    _write_csv(legacy_path, legacy_rows, LEGACY_COLUMNS)
    _write_csv(actions_path, action_rows, ACTION_COLUMNS)
    notice_path.write_text(DATA_USE_NOTICE, encoding="utf-8")

    metadata_manifest_path = metadata_cache_dir / "cache_manifest.json"
    source_manifest_path = source_cache_dir / "cache_manifest.json"
    exclusion_records = []
    expected_by_formula = {
        item["normalized_formula"]: item for item in expected["strict_mapping_exclusions"]
    }
    for exclusion in exclusions:
        exclusion_records.append({**expected_by_formula[exclusion["normalized_formula"]]})

    manifest = {
        "schema_version": 1,
        "benchmark_name": config["benchmark_name"],
        "description": config["description"],
        "freeze_date": config["freeze_date"],
        "verdict": "PENDING_ISOLATED_ORACLE_WRITE",
        "source_of_truth": config["source_of_truth"],
        "selection_rules": config["selection_rules"],
        "counts": {
            "pbe_raw_rows": pbe_total,
            "legacy_compositions": len(legacy_rows),
            "gw_raw_rows": gw_total,
            "initial_gw_compositions": gw_summary["unique_compositions"],
            "gw_single_compositions": gw_summary["single_row_compositions"],
            "gw_multiple_compositions": gw_summary["multiple_row_compositions"],
            "gw_multiple_resolved_without_tie": gw_summary["multiple_resolved_without_tie"],
            "gw_exact_energy_ties": gw_summary["exact_energy_tie_groups"],
            "strict_gw_actions": len(action_rows),
        },
        "duplicate_and_tie_summary": {
            "pbe_duplicate_composition_groups": pbe_summary["duplicate_composition_groups"],
            "pbe_exact_displayed_energy_tie_groups": pbe_summary["exact_displayed_energy_tie_groups"],
            "gw_multiple_composition_groups": gw_summary["multiple_row_compositions"],
            "gw_extra_polymorph_rows": gw_total - gw_summary["unique_compositions"],
            "gw_exact_parent_energy_tie_groups": gw_summary["exact_energy_tie_groups"],
        },
        "strict_mapping": {
            "retained": len(action_rows),
            "excluded": exclusion_records,
            "all_retained_map_to_exactly_one_legacy_composition_key": True,
            "all_retained_have_exactly_one_pbe_family_anchor": True,
        },
        "target_isolation": {
            "selection_metadata_queries_return_gw_target_magnitudes": False,
            "gw_target_used_for_candidate_selection": False,
            "gw_target_used_for_strict_mapping": False,
            "oracle_status": "PENDING_MECHANICAL_WRITER",
        },
        "provenance": {
            "query_endpoint": config["sources"]["nrel_matdb"]["query_url"],
            "query_date": config["freeze_date"],
            "source_version": config["sources"]["nrel_matdb"]["source_version"],
            "benchmark_config": _manifest_entry(config_path),
            "benchmark_generator": _manifest_entry(Path(__file__)),
            "metadata_cache_manifest": _manifest_entry(metadata_manifest_path),
            "source_recovery_cache_manifest": _manifest_entry(source_manifest_path),
            "metadata_cache_files": metadata_manifest["files"],
            "pbe_value_cache_files": _source_entries(source_manifest, query_name="pbe"),
        },
        "artifacts": {
            "current_nlr_legacy.csv": _manifest_entry(
                legacy_path,
                rows=len(legacy_rows),
                columns=LEGACY_COLUMNS,
                logical_path="experiments/sun_oxide/benchmark/current_nlr_legacy.csv",
            ),
            "current_nlr_gw_actions.csv": _manifest_entry(
                actions_path,
                rows=len(action_rows),
                columns=ACTION_COLUMNS,
                logical_path="experiments/sun_oxide/benchmark/current_nlr_gw_actions.csv",
            ),
            "NLR_DATA_USE_NOTICE.txt": _manifest_entry(
                notice_path,
                logical_path="experiments/sun_oxide/benchmark/NLR_DATA_USE_NOTICE.txt",
            ),
        },
        "validation": {
            "deterministic_ordering": True,
            "stable_keys_unique": True,
            "mapping_one_to_one": True,
            "selection_target_free": True,
            "oracle_complete": False,
        },
    }
    _write_json(output_dir / "benchmark_manifest.json", manifest)
    validate_outputs(output_dir, require_oracle=False)
    return manifest


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        return list(reader.fieldnames or []), rows


def validate_outputs(output_dir: Path, *, require_oracle: bool = False) -> dict[str, Any]:
    """Validate selection artifacts without opening the isolated oracle."""
    manifest = _load_json(output_dir / "benchmark_manifest.json")
    legacy_columns, legacy = _read_csv(output_dir / "current_nlr_legacy.csv")
    action_columns, actions = _read_csv(output_dir / "current_nlr_gw_actions.csv")
    _require_equal(legacy_columns, LEGACY_COLUMNS, "legacy columns")
    _require_equal(action_columns, ACTION_COLUMNS, "action columns")
    _require_equal(len(legacy), 2142, "legacy row count")
    _require_equal(len(actions), 191, "action row count")
    _require_equal(
        legacy,
        sorted(legacy, key=lambda row: (row["normalized_formula"], int(row["pbe_mident"]))),
        "legacy deterministic order",
    )
    _require_equal(
        actions,
        sorted(actions, key=lambda row: (row["normalized_formula"], int(row["gw_mident"]))),
        "action deterministic order",
    )
    legacy_keys = {row["composition_key"] for row in legacy}
    _require_equal(len(legacy_keys), len(legacy), "unique legacy composition keys")
    _require_equal(len({row["action_key"] for row in actions}), len(actions), "unique action keys")
    _require_equal(
        len({row["composition_key"] for row in actions}), len(actions), "one action per legacy key"
    )
    _require_equal(
        all(row["composition_key"] in legacy_keys for row in actions), True, "action-to-legacy mapping"
    )
    for name in ("current_nlr_legacy.csv", "current_nlr_gw_actions.csv", "NLR_DATA_USE_NOTICE.txt"):
        _require_equal(
            sha256_file(output_dir / name), manifest["artifacts"][name]["sha256"], f"artifact hash {name}"
        )
    _require_equal(
        manifest["target_isolation"]["gw_target_used_for_candidate_selection"], False, "target isolation"
    )
    if require_oracle:
        _require_equal(manifest["verdict"], "PASS_CURRENT_NLR_BENCHMARK", "terminal verdict")
        _require_equal(manifest["validation"]["oracle_complete"], True, "oracle completion")
    return {"legacy_rows": len(legacy), "action_rows": len(actions), "manifest": manifest}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    fetch = subparsers.add_parser("fetch")
    fetch.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    fetch.add_argument("--cache-dir", type=Path, default=DEFAULT_METADATA_CACHE)
    build = subparsers.add_parser("build")
    build.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    build.add_argument("--metadata-cache-dir", type=Path, default=DEFAULT_METADATA_CACHE)
    build.add_argument("--source-cache-dir", type=Path, default=DEFAULT_SOURCE_CACHE)
    build.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "fetch":
        manifest = fetch_metadata(args.config, args.cache_dir)
        print(f"TARGET_FREE_METADATA_READY files={len(manifest['files'])}")
    elif args.command == "build":
        build_benchmark(args.config, args.metadata_cache_dir, args.source_cache_dir, args.output_dir)
        print("BENCHMARK_SELECTION_READY")
    else:
        validate_outputs(args.output_dir, require_oracle=False)
        print("BENCHMARK_SELECTION_VALID")
    return 0


if __name__ == "__main__":
    sys.exit(main())
