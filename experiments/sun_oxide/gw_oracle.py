#!/usr/bin/env python3
"""Mechanical, isolated GW oracle writer for CURRENT_NLR_PBE_GW_V1.

This is the only new module allowed to read GW target magnitudes. It performs
no candidate selection, ordering by target, summary statistics, or modeling.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "experiments/sun_oxide/configs/current_nlr_pbe_gw_v1.json"
DEFAULT_SOURCE_CACHE = ROOT / "data/external/sun_oxide/source_recovery"
DEFAULT_OUTPUT = ROOT / "experiments/sun_oxide/benchmark"
ORACLE_COLUMNS = ["action_key", "gw_band_gap_ev"]


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


AUDIT = _load_module("sun_oxide_source_audit_for_oracle", ROOT / "experiments/sun_oxide/source_audit.py")
BENCHMARK = _load_module(
    "sun_oxide_current_nlr_benchmark_for_oracle",
    ROOT / "experiments/sun_oxide/current_nlr_benchmark.py",
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.write_bytes(AUDIT.canonical_json_bytes(value))


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        return list(reader.fieldnames or []), rows


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=ORACLE_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _finite_target(value: Any, *, mident: int) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Non-numeric GW oracle target for MatDB {mident}") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"Non-finite GW oracle target for MatDB {mident}")
    return format(numeric, ".15g")


def _source_entries(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "local_filename": entry["local_filename"],
            "sha256": entry["sha256"],
            "size_bytes": entry["size_bytes"],
        }
        for entry in manifest["files"]
        if entry.get("query_name") == "gw"
    ]


def _verified_source_manifest(source_cache_dir: Path) -> dict[str, Any]:
    manifest = _load_json(source_cache_dir / "cache_manifest.json")
    for entry in manifest["files"]:
        AUDIT.verify_cache_entry(source_cache_dir / entry["local_filename"], entry["sha256"])
    return manifest


def write_oracle(config_path: Path, source_cache_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Join already-selected action IDs to GW values without changing selection."""
    config = _load_json(config_path)
    BENCHMARK.validate_outputs(output_dir, require_oracle=False)
    action_columns, actions = _read_csv(output_dir / "current_nlr_gw_actions.csv")
    if action_columns != BENCHMARK.ACTION_COLUMNS:
        raise ValueError("Unexpected GW action schema")
    if len(actions) != config["expected"]["strict_gw_actions"]:
        raise ValueError("GW action count changed before oracle attachment")

    source_manifest = _verified_source_manifest(source_cache_dir)
    source_config = _load_json(ROOT / "experiments/sun_oxide/configs/source_recovery.json")
    _, target_rows = AUDIT._load_query_rows(source_config, source_cache_dir, "gw")
    targets: dict[int, tuple[str, str]] = {}
    for row in target_rows:
        mident = int(row["mident"])
        if mident in targets:
            raise ValueError(f"Duplicate GW target MatDB ID {mident}")
        targets[mident] = (
            AUDIT.normalized_formula_key(row["sorted_formula"]),
            _finite_target(row["gw_band_gap_ev"], mident=mident),
        )

    oracle_rows: list[dict[str, str]] = []
    for action in actions:
        mident = int(action["gw_mident"])
        if mident not in targets:
            raise ValueError(f"Selected GW MatDB {mident} has no oracle target")
        formula, target = targets[mident]
        if formula != action["normalized_formula"]:
            raise ValueError(f"GW target composition mismatch for MatDB {mident}")
        oracle_rows.append({"action_key": action["action_key"], "gw_band_gap_ev": target})

    oracle_path = output_dir / "gw_oracle.csv"
    _write_csv(oracle_path, oracle_rows)
    manifest_path = output_dir / "benchmark_manifest.json"
    manifest = _load_json(manifest_path)
    manifest["verdict"] = "PASS_CURRENT_NLR_BENCHMARK"
    manifest["target_isolation"].update(
        {
            "oracle_status": "WRITTEN_BY_ISOLATED_MECHANICAL_WRITER",
            "oracle_columns": ORACLE_COLUMNS,
            "oracle_contains_selection_metadata": False,
        }
    )
    manifest["provenance"].update(
        {
            "gw_oracle_writer": BENCHMARK._manifest_entry(Path(__file__)),
            "gw_value_cache_files": _source_entries(source_manifest),
        }
    )
    manifest["artifacts"]["gw_oracle.csv"] = {
        **BENCHMARK._manifest_entry(oracle_path, rows=len(oracle_rows), columns=ORACLE_COLUMNS),
        "path": "experiments/sun_oxide/benchmark/gw_oracle.csv",
    }
    manifest["validation"].update(
        {
            "oracle_complete": True,
            "oracle_rows": len(oracle_rows),
            "oracle_action_keys_match": True,
            "oracle_targets_finite": True,
        }
    )
    _write_json(manifest_path, manifest)
    validate_oracle(output_dir, require_pass=True)
    return manifest


def validate_oracle(output_dir: Path, *, require_pass: bool) -> dict[str, Any]:
    """Validate oracle shape/coverage without reporting any target magnitude."""
    BENCHMARK.validate_outputs(output_dir, require_oracle=require_pass)
    _, actions = _read_csv(output_dir / "current_nlr_gw_actions.csv")
    columns, oracle = _read_csv(output_dir / "gw_oracle.csv")
    if columns != ORACLE_COLUMNS:
        raise ValueError(f"Oracle columns {columns!r} != {ORACLE_COLUMNS!r}")
    if len(oracle) != len(actions):
        raise ValueError(f"Oracle row count {len(oracle)} != action count {len(actions)}")
    if [row["action_key"] for row in oracle] != [row["action_key"] for row in actions]:
        raise ValueError("Oracle action keys or ordering do not match frozen actions")
    for row in oracle:
        _finite_target(row["gw_band_gap_ev"], mident=-1)
    manifest = _load_json(output_dir / "benchmark_manifest.json")
    artifact = manifest["artifacts"]["gw_oracle.csv"]
    if AUDIT.sha256_file(output_dir / "gw_oracle.csv") != artifact["sha256"]:
        raise ValueError("GW oracle artifact hash mismatch")
    return {"oracle_rows": len(oracle), "action_keys_match": True, "finite": True}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    write = subparsers.add_parser("write")
    write.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    write.add_argument("--source-cache-dir", type=Path, default=DEFAULT_SOURCE_CACHE)
    write.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "write":
        write_oracle(args.config, args.source_cache_dir, args.output_dir)
    else:
        validate_oracle(args.output_dir, require_pass=True)
    print("PASS_CURRENT_NLR_BENCHMARK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
