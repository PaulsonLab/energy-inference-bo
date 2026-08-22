from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "experiments" / "sun_oxide" / "source_audit.py"
SPEC = importlib.util.spec_from_file_location("sun_oxide_source_audit", MODULE_PATH)
assert SPEC and SPEC.loader
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def _page(rows: str, total: int = 2) -> str:
    return f"""
    <html><body><b>Total matches: {total}</b><table>
    {rows}
    </table></body></html>
    """


def _row(mident: int, formula: str, space_group: int, gap: float, family: int) -> str:
    return f"""
    <tr><th><a href="detail?midentspec={mident}">{mident}</a></th>
    <td>{formula}</td><td></td><td>{space_group}</td><td>{gap}</td>
    <td>['gwvd', 'vexp']</td><td>[{family + 1}]</td>
    <td><input type="submit" name="submitFamily" value="family {family}"/></td></tr>
    """


def test_html_parser_and_duplicate_detection(tmp_path: Path) -> None:
    page = tmp_path / "page.html"
    page.write_text(
        _page(_row(10, "O2 Sn", 152, 4.5, 100) + _row(11, "O2 Sn", 141, 4.1, 101)),
        encoding="utf-8",
    )
    total, rows = AUDIT.parse_matdb_page(
        page,
        ["mident", "sorted_formula", "icsd_space_group", "final_space_group", "gw_band_gap_ev", "standards", "parents"],
    )
    assert total == 2
    assert [row["mident"] for row in rows] == [10, 11]
    summary = AUDIT.duplicate_summary(AUDIT.group_by_formula(rows))
    assert summary == {
        "duplicate_formula_groups": 1,
        "duplicate_extra_rows": 1,
        "duplicate_groups_multiple_final_space_groups": 1,
        "duplicate_groups_multiple_families": 1,
    }


def test_hash_verification_and_mismatch(tmp_path: Path) -> None:
    asset = tmp_path / "asset.bin"
    asset.write_bytes(b"frozen source")
    digest = AUDIT.sha256_file(asset)
    AUDIT.verify_cache_entry(asset, digest)
    asset.write_bytes(b"changed source")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        AUDIT.verify_cache_entry(asset, digest)


def test_deterministic_row_keys_and_order() -> None:
    rows = [
        {"mident": 20, "sorted_formula": "O2 Ti"},
        {"mident": 11, "sorted_formula": "O2 Sn"},
        {"mident": 10, "sorted_formula": "O2 Sn"},
    ]
    ordered = AUDIT.deterministic_order(rows)
    assert [AUDIT.deterministic_row_key(row) for row in ordered] == [
        "matdb:000000010",
        "matdb:000000011",
        "matdb:000000020",
    ]


def test_exact_count_and_join_gate() -> None:
    assert AUDIT.evaluate_gate(
        pbe_count=2142,
        gw_count=194,
        mapped_count=194,
        join_ambiguous=False,
        provenance_usable=True,
        descriptor_valid=True,
    ) == "PASS_SOURCE_RECOVERY"
    assert AUDIT.evaluate_gate(
        pbe_count=2142,
        gw_count=194,
        mapped_count=193,
        join_ambiguous=True,
        provenance_usable=True,
        descriptor_valid=False,
    ) == "JOIN_AMBIGUOUS"
    assert AUDIT.evaluate_gate(
        pbe_count=2141,
        gw_count=194,
        mapped_count=194,
        join_ambiguous=False,
        provenance_usable=True,
        descriptor_valid=True,
    ) == "SOURCE_COUNT_MISMATCH"


def test_descriptor_contract_is_exact_and_ordered() -> None:
    names = [f"magpie_{index:03d}" for index in range(132)]
    assert AUDIT.validate_descriptor_metadata((2142, 132), names, 0)
    assert not AUDIT.validate_descriptor_metadata((2142, 131), names[:-1], 0)
    assert not AUDIT.validate_descriptor_metadata((2142, 132), names[:-1] + [names[-2]], 0)
    assert not AUDIT.validate_descriptor_metadata((2142, 132), names, 1)


def test_cached_input_rerun_is_identical(tmp_path: Path) -> None:
    asset = tmp_path / "source.html"
    asset.write_text(_page(_row(7, "O2 Si", 136, 9.1, 70), total=1), encoding="utf-8")
    digest_before = AUDIT.sha256_file(asset)
    total_a, rows_a = AUDIT.parse_matdb_page(
        asset,
        ["mident", "sorted_formula", "icsd_space_group", "final_space_group", "gw_band_gap_ev", "standards", "parents"],
    )
    total_b, rows_b = AUDIT.parse_matdb_page(
        asset,
        ["mident", "sorted_formula", "icsd_space_group", "final_space_group", "gw_band_gap_ev", "standards", "parents"],
    )
    assert (total_a, rows_a) == (total_b, rows_b)
    assert AUDIT.sha256_file(asset) == digest_before
