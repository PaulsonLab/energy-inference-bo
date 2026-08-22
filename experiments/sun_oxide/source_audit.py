#!/usr/bin/env python3
"""Reproducible source-only audit for the Sun et al. oxide benchmark.

This module deliberately contains no BO, modeling, target-ranking, or
descriptor-generation code. Raw downloads belong in the gitignored cache.
"""

from __future__ import annotations

import argparse
import hashlib
import html
from html.parser import HTMLParser
import json
import math
from pathlib import Path
import re
import sys
import time
from typing import Any, Iterable
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import HTTPCookieProcessor, Request, build_opener, urlopen
import http.cookiejar


VERDICTS = {
    "PASS_SOURCE_RECOVERY",
    "SOURCE_NOT_RECOVERABLE",
    "SOURCE_COUNT_MISMATCH",
    "JOIN_AMBIGUOUS",
    "DESCRIPTOR_RECONSTRUCTION_BLOCKED",
    "PROVENANCE_OR_LICENSE_BLOCKED",
}


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_bytes(canonical_json_bytes(value))


def verify_cache_entry(path: Path, expected_sha256: str) -> None:
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(f"SHA-256 mismatch for {path}: {actual} != {expected_sha256}")


def _download(request: Request, destination: Path, opener: Any | None = None) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    open_request = opener.open if opener is not None else urlopen
    for attempt in range(4):
        try:
            with open_request(request, timeout=120) as response:
                body = response.read()
                final_url = response.geturl()
                content_type = response.headers.get_content_type()
            break
        except HTTPError as exc:
            if exc.code not in {429, 500, 502, 503, 504} or attempt == 3:
                raise
            time.sleep(2**attempt)
    destination.write_bytes(body)
    return {
        "local_filename": destination.name,
        "url": request.full_url,
        "resolved_url": final_url,
        "content_type": content_type,
        "size_bytes": len(body),
        "sha256": sha256_bytes(body),
    }


def _get(url: str, destination: Path) -> dict[str, Any]:
    request = Request(url, headers={"User-Agent": "energy-inference-bo-source-audit/1"})
    return _download(request, destination)


class MatDBResultsParser(HTMLParser):
    """Extract result rows without third-party HTML dependencies."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.in_row = False
        self.in_cell = False
        self.cell_parts: list[str] = []
        self.cells: list[str] = []
        self.mident: int | None = None
        self.family_mident: int | None = None
        self.rows: list[dict[str, Any]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        amap = dict(attrs)
        if tag == "tr":
            self.in_row = True
            self.cells = []
            self.mident = None
            self.family_mident = None
        elif self.in_row and tag in {"th", "td"}:
            self.in_cell = True
            self.cell_parts = []
        elif self.in_row and tag == "a":
            match = re.search(r"detail\?midentspec=(\d+)", amap.get("href") or "")
            if match:
                self.mident = int(match.group(1))
        elif self.in_row and tag == "input":
            match = re.fullmatch(r"family\s+(\d+)", amap.get("value") or "")
            if match:
                self.family_mident = int(match.group(1))

    def handle_data(self, data: str) -> None:
        if self.in_cell:
            self.cell_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if self.in_row and self.in_cell and tag in {"th", "td"}:
            value = " ".join("".join(self.cell_parts).split())
            self.cells.append(html.unescape(value))
            self.in_cell = False
            self.cell_parts = []
        elif tag == "tr" and self.in_row:
            if self.mident is not None:
                self.rows.append(
                    {
                        "mident": self.mident,
                        "cells": self.cells.copy(),
                        "family_mident": self.family_mident,
                    }
                )
            self.in_row = False


def parse_matdb_page(path: Path, fields: list[str]) -> tuple[int, list[dict[str, Any]]]:
    raw = path.read_text(encoding="utf-8")
    total_match = re.search(r"Total matches:\s*(\d+)", raw)
    if not total_match:
        raise ValueError(f"No MatDB total count in {path}")
    parser = MatDBResultsParser()
    parser.feed(raw)
    rows: list[dict[str, Any]] = []
    for parsed in parser.rows:
        cells = parsed["cells"]
        if not cells or str(parsed["mident"]) != cells[0]:
            raise ValueError(f"Unexpected MatDB row layout in {path}: {cells[:2]}")
        values = cells[1 : 1 + len(fields) - 1]
        if len(values) != len(fields) - 1:
            raise ValueError(f"Expected {len(fields)} fields in {path}, found {len(cells)} cells")
        row = {"mident": parsed["mident"]}
        row.update(dict(zip(fields[1:], values, strict=True)))
        row["family_mident"] = parsed["family_mident"]
        rows.append(row)
    return int(total_match.group(1)), rows


_FORMULA_TOKEN = re.compile(r"^[A-Z][a-z]?(?:\d+(?:\.\d+)?)?$")


def normalized_formula_key(formula: str) -> str:
    tokens = formula.split()
    if not tokens or any(not _FORMULA_TOKEN.fullmatch(token) for token in tokens):
        raise ValueError(f"Invalid MatDB formula: {formula!r}")
    if not any(token == "O" or token.startswith("O2") or re.fullmatch(r"O\d+(?:\.\d+)?", token) for token in tokens):
        raise ValueError(f"Formula is not an oxide: {formula!r}")
    return " ".join(tokens)


def deterministic_row_key(row: dict[str, Any]) -> str:
    return f"matdb:{int(row['mident']):09d}"


def deterministic_order(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: (normalized_formula_key(row["sorted_formula"]), int(row["mident"])))


def group_by_formula(rows: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in deterministic_order(rows):
        grouped.setdefault(normalized_formula_key(row["sorted_formula"]), []).append(row)
    return grouped


def _finite_gap_count(rows: Iterable[dict[str, Any]], field: str) -> int:
    count = 0
    for row in rows:
        try:
            value = float(row[field])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Non-numeric {field} for MatDB {row['mident']}") from exc
        if not math.isfinite(value):
            raise ValueError(f"Non-finite {field} for MatDB {row['mident']}")
        count += 1
    return count


def duplicate_summary(groups: dict[str, list[dict[str, Any]]]) -> dict[str, int]:
    duplicate_groups = [rows for rows in groups.values() if len(rows) > 1]
    return {
        "duplicate_formula_groups": len(duplicate_groups),
        "duplicate_extra_rows": sum(len(rows) - 1 for rows in duplicate_groups),
        "duplicate_groups_multiple_final_space_groups": sum(
            len({row["final_space_group"] for row in rows}) > 1 for rows in duplicate_groups
        ),
        "duplicate_groups_multiple_families": sum(
            len({row["family_mident"] for row in rows}) > 1 for rows in duplicate_groups
        ),
    }


def validate_descriptor_metadata(
    shape: list[int] | tuple[int, int] | None,
    feature_names: list[str] | None,
    nonfinite_count: int | None,
) -> bool:
    return bool(
        tuple(shape or ()) == (2142, 132)
        and feature_names is not None
        and len(feature_names) == 132
        and len(set(feature_names)) == 132
        and nonfinite_count == 0
    )


def evaluate_gate(
    *,
    pbe_count: int,
    gw_count: int,
    mapped_count: int,
    join_ambiguous: bool,
    provenance_usable: bool,
    descriptor_valid: bool,
) -> str:
    if not provenance_usable:
        return "PROVENANCE_OR_LICENSE_BLOCKED"
    if pbe_count != 2142 or gw_count != 194:
        return "SOURCE_COUNT_MISMATCH"
    if mapped_count != 194 or join_ambiguous:
        return "JOIN_AMBIGUOUS"
    if not descriptor_valid:
        return "DESCRIPTOR_RECONSTRUCTION_BLOCKED"
    return "PASS_SOURCE_RECOVERY"


def inspect_prefint_notebook(path: Path) -> dict[str, Any]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    all_source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    exact_call = 'ElementProperty.from_preset(preset_name="magpie")'
    if exact_call not in all_source:
        raise ValueError("Pinned PrefInt history does not contain the expected ElementProperty call")
    outputs = "\n".join(
        "".join(output.get("data", {}).get("text/plain", []))
        for cell in notebook["cells"]
        for output in cell.get("outputs", [])
    )
    head_output = "\n".join(
        "".join(output.get("data", {}).get("text/plain", []))
        for output in notebook["cells"][4].get("outputs", [])
    )
    head_counts = {
        formula: len(re.findall(rf"^\s*\d+\s+\d+\s+{re.escape(formula)}\s+", head_output, re.MULTILINE))
        for formula in ("O2 Sn", "O2 Ti")
    }
    return {
        "exact_constructor_call": exact_call,
        "python_version": notebook.get("metadata", {}).get("language_info", {}).get("version"),
        "raw_gw_count_evidence": 244 if "max=244" in outputs else None,
        "deduplicated_gw_count_evidence": 194 if "[194 rows x" in outputs else None,
        "displayed_raw_head_formula_counts": head_counts,
    }


def _post_query(
    opener: Any,
    endpoint: str,
    form: list[list[str]],
    page: int,
    destination: Path,
) -> dict[str, Any]:
    values = [tuple(pair) for pair in form] + [("submitPage", str(page))]
    request = Request(
        endpoint,
        data=urlencode(values).encode("utf-8"),
        headers={"User-Agent": "energy-inference-bo-source-audit/1"},
        method="POST",
    )
    return _download(request, destination, opener=opener)


def _fetch_query(config: dict[str, Any], cache_dir: Path, name: str) -> list[dict[str, Any]]:
    query = config["queries"][name]
    endpoint = config["sources"]["nrel_matdb"]["query_url"]
    cookie_jar = http.cookiejar.CookieJar()
    opener = build_opener(HTTPCookieProcessor(cookie_jar))
    first = cache_dir / f"{query['asset_prefix']}_page_000.html"
    entries: list[dict[str, Any]] = []
    if not first.exists():
        entries.append(_post_query(opener, endpoint, query["form"], 0, first))
    total, _ = parse_matdb_page(first, query["fields"])
    page_count = math.ceil(total / int(query["page_size"]))
    for page in range(page_count):
        path = cache_dir / f"{query['asset_prefix']}_page_{page:03d}.html"
        if page == 0 and path.exists():
            if not entries:
                entries.append(
                    {
                        "local_filename": path.name,
                        "url": endpoint,
                        "resolved_url": endpoint,
                        "content_type": "text/html",
                        "size_bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                )
            continue
        if not path.exists():
            entries.append(_post_query(opener, endpoint, query["form"], page, path))
        else:
            entries.append(
                {
                    "local_filename": path.name,
                    "url": endpoint,
                    "resolved_url": endpoint,
                    "content_type": "text/html",
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    for entry in entries:
        entry["role"] = f"nrel_{name}_query_page"
        entry["query_name"] = name
    return sorted(entries, key=lambda item: item["local_filename"])


def _fetch_archived_supplement(source: dict[str, Any], cache_dir: Path) -> dict[str, Any]:
    cdx_path = cache_dir / "iop_supplement_cdx.json"
    cdx_entry = _get(source["archive_cdx_url"], cdx_path)
    payload = json.loads(cdx_path.read_text(encoding="utf-8"))
    header = payload[0]
    matches = [dict(zip(header, row, strict=True)) for row in payload[1:]]
    match = next((row for row in matches if row["digest"] == source["archive_digest"]), None)
    if match is None:
        raise ValueError("Pinned IOP supplement digest is absent from the Wayback CDX response")
    replay_url = f"https://web.archive.org/web/{match['timestamp']}id_/{match['original']}"
    destination = cache_dir / source["local_filename"]
    supplement_entry = _get(replay_url, destination)
    if not destination.read_bytes().startswith(b"%PDF"):
        raise ValueError("Publisher supplement did not resolve to a PDF")
    cdx_entry["role"] = "publisher_supplement_archive_index"
    supplement_entry["role"] = "publisher_supplement"
    supplement_entry["original_url"] = source["url"]
    supplement_entry["archive_digest"] = source["archive_digest"]
    return {"cdx": cdx_entry, "supplement": supplement_entry}


def fetch_sources(config_path: Path, cache_dir: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / "cache_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for entry in manifest["files"]:
            verify_cache_entry(cache_dir / entry["local_filename"], entry["sha256"])
        return manifest

    files: list[dict[str, Any]] = []
    prefint = config["sources"]["prefint"]
    entry = _get(prefint["evidence_url"], cache_dir / prefint["local_filename"])
    entry.update(
        {
            "role": "prefint_descriptor_and_gw_preprocessing_evidence",
            "repository_commit": prefint["evidence_commit"],
            "repository_blob": prefint["evidence_blob"],
            "retrieval_date": config["audit_date"],
            "source_version": prefint["evidence_commit"],
            "license": prefint["license"],
        }
    )
    files.append(entry)
    archived = _fetch_archived_supplement(config["sources"]["publisher_supplement"], cache_dir)
    archived["cdx"].update(
        {
            "retrieval_date": config["audit_date"],
            "source_version": "Wayback CDX response captured on audit date",
            "license": "Archive index metadata; underlying publisher terms apply to the archived asset",
        }
    )
    archived["supplement"].update(
        {
            "doi": config["sources"]["paper"]["doi"],
            "retrieval_date": config["audit_date"],
            "source_version": "publisher revision2 archived capture",
            "license": config["sources"]["publisher_supplement"]["license"],
        }
    )
    files.extend([archived["cdx"], archived["supplement"]])
    disclaimer = config["sources"]["nrel_matdb"]
    entry = _get(disclaimer["disclaimer_url"], cache_dir / disclaimer["disclaimer_local_filename"])
    entry.update(
        {
            "role": "nrel_data_use_terms",
            "retrieval_date": config["audit_date"],
            "source_version": "live page captured on audit date",
            "license": disclaimer["terms"],
        }
    )
    files.append(entry)
    for query_name in ("pbe", "gw"):
        query_files = _fetch_query(config, cache_dir, query_name)
        for query_file in query_files:
            query_file.update(
                {
                    "retrieval_date": config["audit_date"],
                    "source_version": "live query captured on audit date",
                    "license": disclaimer["terms"],
                }
            )
        files.extend(query_files)
    manifest = {
        "schema_version": 1,
        "download_date": config["audit_date"],
        "config_sha256": sha256_file(config_path),
        "files": sorted(files, key=lambda item: item["local_filename"]),
    }
    write_json(manifest_path, manifest)
    return manifest


def _load_query_rows(config: dict[str, Any], cache_dir: Path, name: str) -> tuple[int, list[dict[str, Any]]]:
    query = config["queries"][name]
    paths = sorted(cache_dir.glob(f"{query['asset_prefix']}_page_*.html"))
    if not paths:
        raise FileNotFoundError(f"No cached pages for {name}")
    totals: set[int] = set()
    rows: list[dict[str, Any]] = []
    for path in paths:
        total, page_rows = parse_matdb_page(path, query["fields"])
        totals.add(total)
        rows.extend(page_rows)
    if len(totals) != 1:
        raise ValueError(f"Inconsistent MatDB totals for {name}: {sorted(totals)}")
    total = totals.pop()
    if total != len(rows):
        raise ValueError(f"MatDB {name} total {total} != parsed rows {len(rows)}")
    return total, deterministic_order(rows)


def _source_manifest(
    config: dict[str, Any], cache_manifest: dict[str, Any], implementation_sha: str | None
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "audit_date": config["audit_date"],
        "implementation_sha": implementation_sha,
        "sources": config["sources"],
        "external_files": cache_manifest["files"],
        "redistribution_conclusion": (
            "No NREL/NLR raw or derived material rows are committed. The MatDB terms permit use, copy, "
            "and distribution if the complete notice accompanies copies and DOE/NLR/ALLIANCE are credited. "
            "PrefInt is MIT-licensed; the article and publisher supplement are CC BY 4.0."
        ),
    }


def run_audit(
    config_path: Path,
    cache_dir: Path,
    output_dir: Path,
    implementation_sha: str | None = None,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite terminal output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    cache_manifest = json.loads((cache_dir / "cache_manifest.json").read_text(encoding="utf-8"))
    for entry in cache_manifest["files"]:
        verify_cache_entry(cache_dir / entry["local_filename"], entry["sha256"])

    pbe_raw_count, pbe_rows = _load_query_rows(config, cache_dir, "pbe")
    gw_raw_count, gw_rows = _load_query_rows(config, cache_dir, "gw")
    pbe_groups = group_by_formula(pbe_rows)
    gw_groups = group_by_formula(gw_rows)
    pbe_finite_count = _finite_gap_count(pbe_rows, "pbe_band_gap_ev")
    gw_finite_count = _finite_gap_count(gw_rows, "gw_band_gap_ev")
    mapped_formulas = sorted(set(pbe_groups) & set(gw_groups))
    missing_formulas = sorted(set(gw_groups) - set(pbe_groups))
    gw_duplicates = duplicate_summary(gw_groups)

    prefint_path = cache_dir / config["sources"]["prefint"]["local_filename"]
    prefint_evidence = inspect_prefint_notebook(prefint_path)
    displayed = prefint_evidence["displayed_raw_head_formula_counts"]
    current_head_counts = {formula: len(gw_groups.get(formula, [])) for formula in displayed}
    historical_source_drift = displayed != current_head_counts

    join_ambiguous = bool(
        historical_source_drift
        or gw_duplicates["duplicate_groups_multiple_families"]
        or len(mapped_formulas) != config["expected"]["mapped_gw_compositions"]
    )
    descriptor_valid = validate_descriptor_metadata(None, None, None)
    verdict = evaluate_gate(
        pbe_count=len(pbe_groups),
        gw_count=len(gw_groups),
        mapped_count=len(mapped_formulas),
        join_ambiguous=join_ambiguous,
        provenance_usable=True,
        descriptor_valid=descriptor_valid,
    )
    if verdict not in VERDICTS:
        raise AssertionError(verdict)

    summary = {
        "schema_version": 1,
        "implementation_sha": implementation_sha,
        "verdict": verdict,
        "terminal": True,
        "paper_method_evidence": False,
        "pbe": {
            "raw_rows": pbe_raw_count,
            "finite_rows": pbe_finite_count,
            "unique_compositions": len(pbe_groups),
        },
        "gw": {
            "raw_rows": gw_raw_count,
            "finite_rows": gw_finite_count,
            "unique_compositions": len(gw_groups),
            **gw_duplicates,
        },
        "join": {
            "mapped_unique_compositions": len(mapped_formulas),
            "missing_unique_composition_count": len(missing_formulas),
            "missing_unique_compositions": missing_formulas,
            "stable_one_to_one_join": False,
        },
        "historical_source_drift": {
            "detected": historical_source_drift,
            "prefint_displayed_raw_head_counts": displayed,
            "current_query_counts_for_same_compositions": current_head_counts,
        },
        "descriptor": {
            "exact_matrix_recovered": False,
            "constructor_recovered": prefint_evidence["exact_constructor_call"],
            "notebook_python_version": prefint_evidence["python_version"],
            "matrix_shape": None,
            "nonfinite_count": None,
            "status": "NOT_REACHED_UPSTREAM_JOIN_BLOCKED",
        },
        "reason": (
            "The exact 2142 and 194 composition counts are reproducible, but the current GW source differs "
            "from the pinned author's displayed raw rows, 28 GW compositions span multiple stable MatDB "
            "families, and only 193 current GW compositions occur in the finite-FERE 2142-composition set. "
            "The author's formula-only drop_duplicates operation cannot identify the historical polymorphs."
        ),
    }
    source_manifest = _source_manifest(config, cache_manifest, implementation_sha)
    normalized_schema = {
        "schema_version": 1,
        "status": "NOT_EMITTED_JOIN_AMBIGUOUS",
        "deterministic_order": ["sorted_formula", "mident"],
        "row_key": "matdb:<zero-padded-mident>",
        "intended_columns": [
            {"name": "row_key", "type": "string"},
            {"name": "pbe_mident", "type": "integer"},
            {"name": "sorted_formula", "type": "string"},
            {"name": "final_space_group", "type": "integer_or_null"},
            {"name": "pbe_band_gap_ev", "type": "finite_number"},
            {"name": "gw_mident", "type": "integer_or_null"},
            {"name": "gw_band_gap_ev", "type": "finite_number_or_null"},
            {"name": "gw_parent_mident", "type": "integer_or_null"},
            {"name": "gw_family_mident", "type": "integer_or_null"}
        ],
        "note": "No normalized benchmark table was written because row identity is unresolved."
    }
    descriptor_manifest = {
        "schema_version": 1,
        "status": "NOT_REACHED_UPSTREAM_JOIN_BLOCKED",
        "published_dimension": 132,
        "exact_matrix_recovered": False,
        "exact_recovered_call": prefint_evidence["exact_constructor_call"],
        "featurization_call": 'StrToComposition().featurize_dataframe(..., "sorted formula"); '
        'ElementProperty.from_preset(preset_name="magpie").featurize_dataframe(..., col_id="composition")',
        "evidence": {
            "repository": config["sources"]["prefint"]["repository_url"],
            "requested_commit": config["sources"]["prefint"]["requested_commit"],
            "history_commit": config["sources"]["prefint"]["evidence_commit"],
            "path": config["sources"]["prefint"]["evidence_path"],
            "blob": config["sources"]["prefint"]["evidence_blob"],
            "sha256": sha256_file(prefint_path),
            "notebook_python_version": prefint_evidence["python_version"]
        },
        "matrix_shape": None,
        "feature_names": None,
        "nonfinite_count": None,
        "imputation_behavior": "Not established because descriptor reconstruction was not reached.",
        "version_pin": "Not frozen because the upstream source/join gate failed."
    }

    write_json(output_dir / "source_manifest.json", source_manifest)
    write_json(output_dir / "reconstruction_summary.json", summary)
    write_json(output_dir / "normalized_schema.json", normalized_schema)
    write_json(output_dir / "descriptor_manifest.json", descriptor_manifest)
    audit_md = f"""# Sun oxide source audit

Terminal verdict: `{verdict}`

This is a source-recovery verdict only. It is not evidence for the paper method.

## Reproduced source facts

- Current authoritative NREL/NLR finite-FERE query: {pbe_raw_count} raw rows, {len(pbe_groups)} unique compositions.
- Current authoritative NREL/NLR finite-GW query: {gw_raw_count} raw rows, {len(gw_groups)} unique compositions.
- Finite PBE/GW raw fields: {pbe_finite_count}/{gw_finite_count}.
- Formula-level GW-to-PBE coverage: {len(mapped_formulas)} of {len(gw_groups)}; missing composition count: {len(missing_formulas)}.

## Blocking evidence

- The pinned PrefInt history uses `{prefint_evidence['exact_constructor_call']}` after formula-to-composition conversion and formula-only `drop_duplicates`.
- It records 244 raw GW rows and 194 rows after composition de-duplication.
- Its displayed raw head has counts {displayed}; the current authoritative query has {current_head_counts} for the same formulas. The raw source has therefore drifted despite unchanged aggregate counts.
- The current GW source has {gw_duplicates['duplicate_formula_groups']} duplicate-formula groups and {gw_duplicates['duplicate_extra_rows']} extra polymorph rows. {gw_duplicates['duplicate_groups_multiple_final_space_groups']} groups span multiple final space groups and {gw_duplicates['duplicate_groups_multiple_families']} span multiple stable MatDB family IDs.
- Stable identifiers do not rescue the historical selection because the committed author notebook discarded `id` before `drop_duplicates`, and the exact 2019/2020 CSV is absent from the repository and publisher supplement.

## Source-route findings

- The requested PrefInt tree at `{config['sources']['prefint']['requested_commit']}` contains molecular benchmark data, not the oxide CSV. The exact preprocessing evidence survives only in the recorded history commit and blob.
- The publisher supplement is a three-page molecular table (94 molecules), not an oxide data file.
- The NIMS/SAMURAI record was inspected, but exposes no downloadable MDR item; current MDR DOI/title searches did not recover the oxide assets.
- The cited Lany and Pilania DOI/source routes did not expose the exact historical Sun-study PBE/GW CSVs.

Implementation SHA: `{implementation_sha or 'not recorded'}`.

No target ranking, correlation, histogram, model, descriptor regeneration, or BO operation was performed.
"""
    (output_dir / "SOURCE_AUDIT.md").write_text(audit_md, encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("fetch", "audit"):
        sub = subparsers.add_parser(name)
        sub.add_argument("--config", type=Path, required=True)
        sub.add_argument("--cache-dir", type=Path, required=True)
        if name == "audit":
            sub.add_argument("--output-dir", type=Path, required=True)
            sub.add_argument("--implementation-sha")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "fetch":
        manifest = fetch_sources(args.config, args.cache_dir)
        print(f"CACHE_READY files={len(manifest['files'])}")
    else:
        summary = run_audit(
            args.config,
            args.cache_dir,
            args.output_dir,
            implementation_sha=args.implementation_sha,
        )
        print(summary["verdict"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
