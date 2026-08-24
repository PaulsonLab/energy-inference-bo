#!/usr/bin/env python3
"""Development-only Sun-oxide normalized full-bank scaling probe.

The probe never opens the GW oracle.  Its only target-bearing input is a small
fixture containing the already-consumed seed-0 FULL_PBE trajectory, and each
fixed state is read only through its authorized observation cutoff.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Sequence

import numpy as np
from scipy import sparse


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from conditioned_bo.bo_value import gaussian_expected_improvement, select_unobserved_action
from conditioned_bo.full_bank_scaling import (
    CompactPairBank,
    ImplicitMenzSystem,
    LocalResourceBlocked,
    ResourceGuard,
    TimedCompactFactorBank,
    build_compact_pair_bank,
    construct_support_precision_reference,
    fit_compact_laplace,
    fit_compact_map_only,
    implicit_theory_diagnostics,
    peak_rss_bytes,
    rank_signal_diagnostics,
    regression_against_dense_500,
    run_adaptive_stage_probe,
    update_fixed_reference_state,
)
from conditioned_bo.normalized_pbe_model import (
    farthest_point_support,
    standardize_descriptor_space,
)
from conditioned_bo.pbe_factor_theory import load_action_mapping, load_legacy_nodes


DEFAULT_CONFIG = ROOT / "experiments/sun_oxide/configs/full_bank_scaling_probe.json"
DEFAULT_OUTPUT = ROOT / "experiments/sun_oxide/outputs/full_bank_scaling_probe"
EXPECTED_OUTPUTS = (
    "RESULTS.md",
    "support_summary.json",
    "theory_summary.json",
    "pbe_signal_summary.csv",
    "pbe_signal_summary.json",
    "full_runtime_scaling.csv",
    "adaptive_stage_scaling.csv",
    "epsilon_sensitivity.csv",
    "resource_usage.json",
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_canonical_json(value), encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _head_sha(repository_root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _verified_inputs(
    repository_root: Path, config: dict[str, Any]
) -> dict[str, Path]:
    root = repository_root.resolve()
    result: dict[str, Path] = {}
    for name, specification in sorted(config["inputs"].items()):
        if set(specification) != {"path", "sha256"}:
            raise ValueError(f"Invalid input declaration for {name}")
        path = (root / specification["path"]).resolve()
        if path != root and root not in path.parents:
            raise ValueError(f"Input escapes repository root: {path}")
        observed = _sha256(path)
        if observed != specification["sha256"]:
            raise ValueError(f"Input hash mismatch for {name}: {observed}")
        result[name] = path
    return result


def _load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1 or not config.get("development_only"):
        raise ValueError("This script requires the development-only schema")
    if config.get("benchmark_name") != "CURRENT_NLR_PBE_GW_V1":
        raise ValueError("Benchmark changed")
    if [item["count"] for item in config["support_models"]] != [500, 1000, 2142]:
        raise ValueError("Support sizes must be exactly 500, 1000, 2142")
    if config["adaptive"] != {
        "epsilon_primary": 0.02,
        "epsilon_sensitivity_middle_state": [0.05, 0.1],
        "max_stages": 8,
        "rho": 0.8,
    }:
        raise ValueError("Development adaptive settings changed")
    if config["fresh_seed_policy"]["forbidden_seeds"] != list(range(12, 32)):
        raise ValueError("Fresh-seed guard changed")
    if config["fresh_seed_policy"]["scientific_preregistration_created"]:
        raise ValueError("This task must not create a scientific preregistration")
    return config


def _load_authorized_state(
    path: Path,
    *,
    observation_count: int,
    action_rows: Sequence[dict[str, Any]],
    expected_mean: float,
    expected_scale: float,
) -> dict[str, Any]:
    """Read exactly the authorized prefix and stop before later target rows."""

    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError("Development state fixture has no header")
        for row in reader:
            rows.append(row)
            if len(rows) == observation_count:
                break
    if len(rows) != observation_count:
        raise ValueError("Authorized development state is incomplete")
    orders = [int(row["observation_order"]) for row in rows]
    if orders != list(range(1, observation_count + 1)):
        raise ValueError("Development observation ordering changed")
    positions = np.asarray([int(row["action_position"]) for row in rows], dtype=np.int64)
    if len(set(positions.tolist())) != positions.size:
        raise ValueError("Development state repeats an action")
    for row, position in zip(rows, positions.tolist(), strict=True):
        action = action_rows[position]
        if row["action_key"] != action["action_key"]:
            raise ValueError("Development action key does not match mapping")
        if int(row["node_index"]) != int(action["node_index"]):
            raise ValueError("Development node does not match mapping")
    values_ev = np.asarray([float(row["gw_band_gap_ev"]) for row in rows])
    standardized = np.asarray(
        [float(row["standardized_observation"]) for row in rows]
    )
    expected = (values_ev - expected_mean) / expected_scale
    if not np.allclose(standardized, expected, atol=0.0, rtol=1e-15):
        raise ValueError("Committed seed-0 standardization changed")
    return {
        "observed_action_positions": positions,
        "observed_nodes": np.asarray(
            [int(row["node_index"]) for row in rows], dtype=np.int64
        ),
        "values_ev": values_ev,
        "standardized_observations": standardized,
        "incumbent_standardized": float(np.max(standardized)),
        "rows_read": observation_count,
        "later_rows_read": 0,
    }


def _frozen_support_nodes(path: Path) -> np.ndarray:
    rows = _read_csv(path)
    support = np.asarray([int(row["node_index"]) for row in rows], dtype=np.int32)
    if support.shape != (500,) or np.any(np.diff(support.astype(np.int64)) <= 0):
        raise ValueError("PBE_SUPPORT_500_V1 changed")
    return support


def _validate_frozen_bank_500(path: Path, bank: CompactPairBank) -> None:
    rows = _read_csv(path)
    endpoints = np.asarray(
        [[int(row["support_i"]), int(row["support_j"])] for row in rows],
        dtype=np.int32,
    )
    signs = np.asarray([int(row["sign_s_ij"]) for row in rows], dtype=np.int8)
    if not np.array_equal(endpoints, bank.endpoint_pairs):
        raise ValueError("Compact m=500 endpoints do not reproduce the frozen bank")
    if not np.array_equal(signs, bank.signs):
        raise ValueError("Compact m=500 signs do not reproduce the frozen bank")


def _build_supports(
    config: dict[str, Any],
    paths: dict[str, Path],
    nodes: Sequence[Any],
    action_nodes: np.ndarray,
) -> tuple[dict[int, np.ndarray], dict[int, dict[str, Any]]]:
    with np.load(paths["descriptor_matrix"], allow_pickle=False) as archive:
        descriptor_keys = archive["composition_keys"].astype(str)
        raw = np.asarray(archive["raw_descriptors"], dtype=np.float64)
    if descriptor_keys.tolist() != [node.composition_key for node in nodes]:
        raise ValueError("Descriptor and legacy node order differ")
    standardized, standardization = standardize_descriptor_space(raw)
    keys = [node.composition_key for node in nodes]
    frozen_500 = _frozen_support_nodes(paths["frozen_support_500"])
    supports: dict[int, np.ndarray] = {}
    summaries: dict[int, dict[str, Any]] = {}
    for specification in config["support_models"]:
        count = int(specification["count"])
        started = time.perf_counter()
        if count == 500:
            fps = farthest_point_support(standardized, keys, action_nodes, 500)
            if not np.array_equal(fps.selected_indices.astype(np.int32), frozen_500):
                raise ValueError("Farthest-point m=500 does not reproduce frozen support")
            support = frozen_500
            selection = "reused_exact_PBE_SUPPORT_500_V1"
            added_order_hash = hashlib.sha256(
                fps.additional_indices.astype("<i4").tobytes()
            ).hexdigest()
        elif count == 1000:
            fps = farthest_point_support(standardized, keys, action_nodes, 1000)
            support = fps.selected_indices.astype(np.int32)
            selection = "descriptor_farthest_point_actions_plus_809"
            added_order_hash = hashlib.sha256(
                fps.additional_indices.astype("<i4").tobytes()
            ).hexdigest()
        elif count == len(nodes):
            fps = None
            support = np.arange(len(nodes), dtype=np.int32)
            selection = "all_legacy_nodes_no_selection"
            added_order_hash = "not_applicable"
        else:
            raise ValueError("Unexpected support size")
        if not set(action_nodes.tolist()).issubset(set(support.tolist())):
            raise ValueError("A support omits an action")
        supports[count] = support
        summaries[count] = {
            "support_count": count,
            "support_name": specification["support_name"],
            "model_name": specification["name"],
            "selection": selection,
            "action_count": int(action_nodes.size),
            "additional_legacy_node_count": count - int(action_nodes.size),
            "descriptor_only_selection": True,
            "pbe_magnitude_used_for_selection": False,
            "gw_values_used_for_selection": False,
            "stable_composition_key_tie_breaking": True,
            "additional_selection_order_sha256": added_order_hash,
            "support_nodes_sha256": hashlib.sha256(
                support.astype("<i4").tobytes()
            ).hexdigest(),
            "selection_seconds": time.perf_counter() - started,
            "descriptor_standardization": standardization,
        }
    return supports, summaries


def _action_support_positions(
    support_nodes: np.ndarray, action_nodes: np.ndarray
) -> np.ndarray:
    lookup = {int(node): position for position, node in enumerate(support_nodes)}
    return np.asarray([lookup[int(node)] for node in action_nodes], dtype=np.int64)


def _laplace_settings(config: dict[str, Any]) -> dict[str, Any]:
    values = config["laplace"]
    return {
        "gradient_tolerance": values["map_gradient_infinity_norm_maximum"],
        "optimizer_gradient_tolerance": values["optimizer_gradient_tolerance"],
        "function_tolerance": values["function_tolerance"],
        "maximum_iterations": values["maximum_iterations"],
        "residual_tolerance": values["solve_relative_residual_maximum"],
    }


def _flush_partial(
    output_dir: Path,
    support_summary: dict[str, Any],
    theory_summary: dict[str, Any],
    signal_json: dict[str, Any],
    signal_rows: list[dict[str, Any]],
    full_rows: list[dict[str, Any]],
    adaptive_rows: list[dict[str, Any]],
    epsilon_rows: list[dict[str, Any]],
    resource: dict[str, Any],
) -> None:
    _write_json(output_dir / "support_summary.json", support_summary)
    _write_json(output_dir / "theory_summary.json", theory_summary)
    _write_json(output_dir / "pbe_signal_summary.json", signal_json)
    _write_csv(output_dir / "pbe_signal_summary.csv", signal_rows)
    _write_csv(output_dir / "full_runtime_scaling.csv", full_rows)
    _write_csv(output_dir / "adaptive_stage_scaling.csv", adaptive_rows)
    _write_csv(output_dir / "epsilon_sensitivity.csv", epsilon_rows)
    _write_json(output_dir / "resource_usage.json", resource)


def _median(values: Sequence[float]) -> float:
    return float(np.median(np.asarray(values, dtype=np.float64)))


def _structural_summary(
    support_summary: dict[str, Any],
    adaptive_rows: list[dict[str, Any]],
    full_rows: list[dict[str, Any]],
    outcomes: list[dict[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for support_count in (500, 1000, 2142):
        model_outcomes = [
            row
            for row in outcomes
            if row["support_count"] == support_count and row["epsilon"] == 0.02
        ]
        stages = [
            row
            for row in adaptive_rows
            if int(row["support_count"]) == support_count
            and float(row["epsilon"]) == 0.02
            and row["record_kind"] == "stage"
        ]
        last_by_state: list[dict[str, Any]] = []
        first_agreement_fractions: list[float] = []
        for state_name in (
            "seed_0_initial",
            "seed_0_after_6_queries",
            "seed_0_after_12_queries",
        ):
            state_stages = [row for row in stages if row["state"] == state_name]
            last_by_state.append(state_stages[-1])
            agreements = [
                float(row["active_factor_fraction"])
                for row in state_stages
                if bool(row["shadow_full_action_agreement"])
            ]
            first_agreement_fractions.append(min(agreements) if agreements else 1.0)
        model_full = [row for row in full_rows if int(row["support_count"]) == support_count]
        paired_time_ratios = [
            float(outcome["total_conditioning_seconds"])
            / float(next(row for row in model_full if row["state"] == outcome["state"])["total_full_conditioning_seconds"])
            for outcome in model_outcomes
        ]
        paired_work_ratios = [
            float(outcome["factor_energy_gradient_work"])
            / float(next(row for row in model_full if row["state"] == outcome["state"])["factor_energy_gradient_element_work"])
            for outcome in model_outcomes
        ]
        result[str(support_count)] = {
            "factor_count": support_summary["models"][str(support_count)]["strict_factor_count"],
            "median_pre_fallback_active_factor_count": _median(
                [float(row["pre_fallback_active_count"]) for row in model_outcomes]
            ),
            "median_pre_fallback_active_factor_fraction": _median(
                [float(row["pre_fallback_active_fraction"]) for row in model_outcomes]
            ),
            "median_max_psi_after_final_stage": _median(
                [float(row["max_psi"]) for row in last_by_state]
            ),
            "fraction_of_stages_active_leader_equals_full": float(
                np.mean([bool(row["shadow_full_action_agreement"]) for row in stages])
            ),
            "smallest_active_fraction_for_full_action_agreement_by_state": first_agreement_fractions,
            "median_smallest_active_fraction_for_full_action_agreement": _median(
                first_agreement_fractions
            ),
            "median_adaptive_to_full_conditioning_time_ratio": _median(
                paired_time_ratios
            ),
            "median_adaptive_to_full_factor_energy_gradient_work_ratio": _median(
                paired_work_ratios
            ),
            "certified_state_count": sum(bool(row["certified"]) for row in model_outcomes),
            "fallback_state_count": sum(bool(row["full_bank_fallback"]) for row in model_outcomes),
        }
    active_500 = result["500"]["median_pre_fallback_active_factor_count"]
    active_2142 = result["2142"]["median_pre_fallback_active_factor_count"]
    result["scaling"] = {
        "support_size_ratio_2142_to_500": 2142.0 / 500.0,
        "factor_count_ratio_2142_to_500": (
            result["2142"]["factor_count"] / result["500"]["factor_count"]
        ),
        "active_count_ratio_2142_to_500": active_2142 / active_500,
        "active_count_power_vs_support_size": math.log(active_2142 / active_500)
        / math.log(2142.0 / 500.0),
    }
    return result


def _classify(structural: dict[str, Any]) -> str:
    full = structural["2142"]
    pre_fraction = float(full["median_pre_fallback_active_factor_fraction"])
    first_agreement = float(
        full["median_smallest_active_fraction_for_full_action_agreement"]
    )
    active_power = float(structural["scaling"]["active_count_power_vs_support_size"])
    if pre_fraction <= 0.75 and active_power < 1.5:
        return "FULL_ARCHIVE_PROMISING"
    if first_agreement + 0.10 < pre_fraction:
        return "BOUND_CONSERVATISM_DIAGNOSED"
    return "FULL_ARCHIVE_NOT_HELPFUL"


def _results_markdown(
    implementation_sha: str,
    config_sha: str,
    classification: str,
    support_summary: dict[str, Any],
    theory_summary: dict[str, Any],
    signal_json: dict[str, Any],
    full_rows: list[dict[str, Any]],
    structural: dict[str, Any],
    epsilon_rows: list[dict[str, Any]],
    resource: dict[str, Any],
) -> str:
    lines = [
        "# Full-bank scaling probe — DEVELOPMENT ONLY",
        "",
        f"Development classification: `{classification}`.",
        "",
        "This is not scientific evidence or a fresh-seed preregistration. It used",
        "only the three fixed, already-consumed seed-0 FULL_PBE states. Seeds",
        "12--31 were not accessed, and the immutable failed adaptive smokes were",
        "not modified or reinterpreted.",
        "",
        f"- Implementation SHA: `{implementation_sha}`",
        f"- Config SHA-256: `{config_sha}`",
        f"- Peak process RSS: `{resource['peak_rss_gb']}` GB",
        "",
        "## Support, theory, and signal",
        "",
        "| m | strict factors | tie pairs omitted | omega | max weighted degree | lambda_min(A0) | action/support Spearman |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for count in (500, 1000, 2142):
        model = support_summary["models"][str(count)]
        theory = theory_summary["models"][str(count)]
        signal = signal_json["models"][str(count)]
        lines.append(
            f"| {count} | {model['strict_factor_count']} | "
            f"{model['omitted_exact_tie_pair_count']} | {model['omega']:.10g} | "
            f"{model['maximum_weighted_incident_degree']:.10g} | "
            f"{theory['a0_smallest_eigenvalue']:.10g} | "
            f"{signal['actions']['spearman_pbe_rank_vs_map_rank']:.6f} / "
            f"{signal['support']['spearman_pbe_rank_vs_map_rank']:.6f} |"
        )
    lines.extend(
        [
            "",
            "All three A0 operators remained SPD above the existing analytic 0.75",
            "floor. No dense Menz inverse was used in routine calculations.",
            "",
            "## FULL runtime scaling",
            "",
            "| m | state | FULL leader | MAP s | factor E/G s | Hessian s | Cholesky s | variance s | total s | peak RSS GB |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in full_rows:
        lines.append(
            f"| {row['support_count']} | {row['state']} | {row['full_leader']} | "
            f"{float(row['map_optimization_seconds']):.4f} | "
            f"{float(row['factor_energy_gradient_seconds']):.4f} | "
            f"{float(row['hessian_construction_seconds']):.4f} | "
            f"{float(row['dense_cholesky_seconds']):.4f} | "
            f"{float(row['selected_variance_solve_seconds']):.4f} | "
            f"{float(row['total_full_conditioning_seconds']):.4f} | "
            f"{float(row['peak_rss_gb']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Structural scaling at epsilon 0.02",
            "",
            "| m | median pre-fallback count | fraction | final max-Psi | stage leader agreement | first-agreement fraction | ADAPT/FULL time | ADAPT/FULL factor work | fallbacks |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for count in (500, 1000, 2142):
        row = structural[str(count)]
        lines.append(
            f"| {count} | {row['median_pre_fallback_active_factor_count']:.0f} | "
            f"{row['median_pre_fallback_active_factor_fraction']:.6f} | "
            f"{row['median_max_psi_after_final_stage']:.6f} | "
            f"{row['fraction_of_stages_active_leader_equals_full']:.6f} | "
            f"{row['median_smallest_active_fraction_for_full_action_agreement']:.6f} | "
            f"{row['median_adaptive_to_full_conditioning_time_ratio']:.4f} | "
            f"{row['median_adaptive_to_full_factor_energy_gradient_work_ratio']:.4f} | "
            f"{row['fallback_state_count']}/3 |"
        )
    scaling = structural["scaling"]
    lines.extend(
        [
            "",
            f"From m=500 to 2142, total factors grew by "
            f"`{scaling['factor_count_ratio_2142_to_500']:.4f}x`, while the median "
            f"pre-fallback active count grew by "
            f"`{scaling['active_count_ratio_2142_to_500']:.4f}x`. The implied "
            f"active-count power versus m is "
            f"`{scaling['active_count_power_vs_support_size']:.4f}`.",
            "",
            "## Epsilon sensitivity at the middle state",
            "",
            "| m | epsilon | active fraction | stages | fallback | shadow agreement | EI regret | conditioning s |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in epsilon_rows:
        lines.append(
            f"| {row['support_count']} | {float(row['epsilon']):.2f} | "
            f"{float(row['active_fraction_at_certification_or_fallback']):.6f} | "
            f"{row['stage_count']} | {row['full_bank_fallback']} | "
            f"{row['shadow_full_action_agreement']} | "
            f"{float(row['shadow_full_laplace_ei_regret']):.8g} | "
            f"{float(row['conditioning_seconds']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The tables above answer whether enlarging the archive separates decision",
            "complexity from total conditioning complexity. The classification is a",
            "development diagnosis, not a paper verdict or permission to spend fresh",
            "seeds. See the raw stage CSV for the exact point at which each active",
            "leader first matches shadow FULL and the remaining theorem-backed envelope.",
        ]
    )
    return "\n".join(lines) + "\n"


def _manifest(
    output_dir: Path,
    implementation_sha: str,
    config_sha: str,
    classification: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "development_only": True,
        "implementation_sha": implementation_sha,
        "config_sha256": config_sha,
        "classification": classification,
        "fresh_seeds_accessed": False,
        "files": [
            {
                "path": name,
                "sha256": _sha256(output_dir / name),
                "size_bytes": (output_dir / name).stat().st_size,
            }
            for name in EXPECTED_OUTPUTS
        ],
    }


def _blocked_notebook(config_path: Path) -> dict[str, Any]:
    relative_config = config_path.relative_to(ROOT).as_posix()
    return {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Full-bank scaling probe — DEVELOPMENT ONLY\n",
                    "Created only because the identical local probe hit `LOCAL_RESOURCE_BLOCKED`.\n",
                    "This notebook does not authorize fresh seeds or a scientific validation.\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "!python experiments/sun_oxide/full_bank_scaling_probe.py \\\n",
                    f"  --config {relative_config} \\\n",
                    "  --output-dir experiments/sun_oxide/outputs/full_bank_scaling_probe_colab \\\n",
                    "  --implementation-sha $(git rev-parse HEAD)\n",
                ],
            },
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def run_probe(
    config_path: Path,
    repository_root: Path,
    output_dir: Path,
    implementation_sha: str,
) -> str:
    if _head_sha(repository_root) != implementation_sha:
        raise ValueError("implementation_sha must equal repository HEAD")
    config = _load_config(config_path)
    paths = _verified_inputs(repository_root, config)
    config_sha = _sha256(config_path)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    guard = ResourceGuard.create(
        config["resource_guard"]["peak_rss_gb_maximum"],
        config["resource_guard"]["phase_time_limits_seconds"],
    )
    overall_started = time.perf_counter()
    resource_summary: dict[str, Any] = {
        "schema_version": 1,
        "development_only": True,
        "rss_limit_bytes": guard.rss_limit_bytes,
        "rss_limit_gb": config["resource_guard"]["peak_rss_gb_maximum"],
        "phase_time_limits_seconds": guard.phase_time_limits_seconds,
        "phases": guard.records,
        "local_resource_blocked": False,
    }
    support_summary: dict[str, Any] = {
        "schema_version": 1,
        "development_only": True,
        "models": {},
        "target_values_used_for_support_selection": False,
    }
    theory_summary: dict[str, Any] = {
        "schema_version": 1,
        "development_only": True,
        "models": {},
        "implicit_500_regression": {},
    }
    signal_json: dict[str, Any] = {
        "schema_version": 1,
        "development_only": True,
        "target_blind": True,
        "models": {},
    }
    signal_rows: list[dict[str, Any]] = []
    full_rows: list[dict[str, Any]] = []
    adaptive_rows: list[dict[str, Any]] = []
    epsilon_rows: list[dict[str, Any]] = []
    outcomes: list[dict[str, Any]] = []

    try:
        manifest = json.loads(
            paths["frozen_normalized_model_manifest"].read_text(encoding="utf-8")
        )
        if manifest.get("verdict") != "PASS_NORMALIZED_PBE_MODEL":
            raise ValueError("Frozen normalized model is not a committed PASS")
        nodes = load_legacy_nodes(paths["legacy_pbe"], 2142)
        actions = load_action_mapping(paths["action_node_mapping"], 191, 2142)
        action_nodes = np.asarray(
            [int(action["node_index"]) for action in actions], dtype=np.int64
        )
        action_keys = [str(action["action_key"]) for action in actions]
        q0 = sparse.load_npz(paths["q0"]).tocsr().astype(np.float64)
        if q0.shape != (2142, 2142):
            raise ValueError("Frozen Q0 dimension changed")
        supports, selection_summaries = _build_supports(
            config, paths, nodes, action_nodes
        )
        pbe_values = [node.pbe_band_gap for node in nodes]

        for model_specification in config["support_models"]:
            support_count = int(model_specification["count"])
            print(f"PROGRESS support={support_count} phase=pair_bank", flush=True)
            support_nodes = supports[support_count]
            bank_started = time.perf_counter()
            bank = build_compact_pair_bank(
                support_nodes,
                pbe_values,
                model_name=model_specification["name"],
            )
            if support_count == 500:
                _validate_frozen_bank_500(paths["frozen_factor_bank_500"], bank)
            model_summary = {
                **selection_summaries[support_count],
                "strict_factor_count": bank.factor_count,
                "complete_possible_pair_count": support_count * (support_count - 1) // 2,
                "omitted_exact_tie_pair_count": bank.omitted_tie_pair_count,
                "omega": bank.weight,
                "omega_exact": f"1/{support_count - 1}",
                "maximum_weighted_incident_degree": bank.maximum_weighted_incident_degree,
                "endpoint_dtype": str(bank.endpoint_pairs.dtype),
                "sign_dtype": str(bank.signs.dtype),
                "compact_factor_array_bytes": int(
                    bank.endpoint_pairs.nbytes + bank.signs.nbytes
                ),
                "pair_bank_construction_seconds": time.perf_counter() - bank_started,
            }
            support_summary["models"][str(support_count)] = model_summary

            print(f"PROGRESS support={support_count} phase=theory", flush=True)
            theory_started = time.perf_counter()
            theory = implicit_theory_diagnostics(
                q0,
                bank,
                eigensolver_tolerance=config["theory"]["eigensolver_tolerance"],
            )
            guard.record("theory", theory_started, support_count=support_count)
            if theory["maximum_weighted_incident_degree"] > 1.0 + 1e-12:
                raise ValueError("Weighted incident-degree bound failed")
            if theory["a0_smallest_eigenvalue"] < 0.75 - 1e-8:
                raise ValueError("A0 numerical eigenvalue fell below analytic bound")
            if max(
                theory["weighted_adjacency_eigen_residual"],
                theory["a0_eigen_residual"],
            ) > config["theory"]["eigensolver_residual_maximum"]:
                raise ValueError("Theory eigensolver residual failed")
            theory_summary["models"][str(support_count)] = theory
            if support_count == 500:
                theory_summary["implicit_500_regression"] = regression_against_dense_500(
                    q0,
                    bank,
                    action_nodes,
                    residual_tolerance=config["theory"][
                        "influence_solve_relative_residual_maximum"
                    ],
                )

            print(f"PROGRESS support={support_count} phase=support_reference", flush=True)
            reference_started = time.perf_counter()
            support_reference = construct_support_precision_reference(
                q0,
                support_nodes,
                residual_tolerance=config["laplace"][
                    "solve_relative_residual_maximum"
                ],
                guard=guard,
            )
            guard.record(
                "support_reference",
                reference_started,
                support_count=support_count,
            )
            model_summary["exact_support_reference"] = support_reference.diagnostics

            print(f"PROGRESS support={support_count} phase=pbe_signal", flush=True)
            pbe_signal_started = time.perf_counter()
            full_factor_bank = TimedCompactFactorBank(
                bank, None, chunk_size=config["factor_bank"]["chunk_size"]
            )
            pbe_map, optimization = fit_compact_map_only(
                support_reference.precision,
                full_factor_bank,
                gradient_tolerance=config["pbe_only_map"][
                    "map_gradient_infinity_norm_maximum"
                ],
                optimizer_gradient_tolerance=config["pbe_only_map"][
                    "optimizer_gradient_tolerance"
                ],
                function_tolerance=config["pbe_only_map"]["function_tolerance"],
                maximum_iterations=config["pbe_only_map"]["maximum_iterations"],
                maximum_retries=config["pbe_only_map"]["maximum_retries"],
                guard=guard,
            )
            action_support = _action_support_positions(support_nodes, action_nodes)
            support_pbe = [pbe_values[int(node)] for node in support_nodes]
            support_keys = [nodes[int(node)].composition_key for node in support_nodes]
            action_pbe = [pbe_values[int(node)] for node in action_nodes]
            action_composition_keys = [nodes[int(node)].composition_key for node in action_nodes]
            support_signal = rank_signal_diagnostics(
                support_pbe, pbe_map, support_keys
            )
            action_signal = rank_signal_diagnostics(
                action_pbe, pbe_map[action_support], action_composition_keys
            )
            signal_json["models"][str(support_count)] = {
                "model_name": bank.model_name,
                "optimization": optimization,
                "support": support_signal,
                "actions": action_signal,
            }
            for domain, diagnostics in (
                ("support", support_signal),
                ("actions", action_signal),
            ):
                signal_rows.append(
                    {
                        "support_count": support_count,
                        "model_name": bank.model_name,
                        "domain": domain,
                        **diagnostics,
                        "optimization_iterations": optimization["iterations"],
                        "factor_gradient_calls": optimization[
                            "factor_energy_gradient_calls"
                        ],
                        "optimization_wall_seconds": optimization["wall_seconds"],
                    }
                )
            guard.record(
                "pbe_signal", pbe_signal_started, support_count=support_count
            )

            full_contexts: dict[str, dict[str, Any]] = {}
            previous_full_map: np.ndarray | None = None
            for state_specification in config["fixed_states"]:
                state_name = state_specification["name"]
                state = _load_authorized_state(
                    paths["development_seed0_states"],
                    observation_count=state_specification["observation_count"],
                    action_rows=actions,
                    expected_mean=config["seed0_state_provenance"]["target_mean_ev"],
                    expected_scale=config["seed0_state_provenance"]["target_scale_ev"],
                )
                observed_support = action_support[state["observed_action_positions"]]
                fixed_reference = update_fixed_reference_state(
                    support_reference,
                    observed_support,
                    state["standardized_observations"],
                    action_support,
                    observation_precision=config["observation_precision"],
                    residual_tolerance=config["laplace"][
                        "solve_relative_residual_maximum"
                    ],
                )
                print(
                    f"PROGRESS support={support_count} state={state_name} phase=FULL",
                    flush=True,
                )
                full_started = time.perf_counter()
                timed_bank = TimedCompactFactorBank(
                    bank, None, chunk_size=config["factor_bank"]["chunk_size"]
                )
                full_laplace = fit_compact_laplace(
                    fixed_reference,
                    timed_bank,
                    fixed_reference.mean
                    if previous_full_map is None
                    else previous_full_map,
                    action_support,
                    retry_map=np.zeros(support_count, dtype=np.float64),
                    guard=guard,
                    phase_name="full_fixed_state",
                    **_laplace_settings(config),
                )
                full_ei = gaussian_expected_improvement(
                    full_laplace.map[action_support],
                    full_laplace.action_variances,
                    state["incumbent_standardized"],
                )
                full_leader = select_unobserved_action(
                    full_ei,
                    state["observed_action_positions"],
                    action_keys,
                )
                previous_full_map = full_laplace.map
                diagnostics = full_laplace.diagnostics
                full_row = {
                    "support_count": support_count,
                    "factor_count": bank.factor_count,
                    "state": state_name,
                    "observation_count": state_specification["observation_count"],
                    "authorized_target_rows_read": state["rows_read"],
                    "unauthorized_later_target_rows_read": state["later_rows_read"],
                    "full_leader": int(full_leader),
                    "full_leader_action_key": action_keys[full_leader],
                    "full_leader_ei": float(full_ei[full_leader]),
                    "reference_update_seconds": fixed_reference.diagnostics[
                        "total_reference_seconds"
                    ],
                    **diagnostics,
                    "total_full_conditioning_seconds": diagnostics[
                        "total_conditioning_seconds"
                    ],
                    "peak_rss_gb": peak_rss_bytes() / 1_000_000_000.0,
                }
                full_rows.append(full_row)
                full_contexts[state_name] = {
                    "state": state,
                    "reference": fixed_reference,
                    "ei": full_ei,
                    "leader": int(full_leader),
                    "map": full_laplace.map,
                }
                guard.record(
                    "full_fixed_state",
                    full_started,
                    support_count=support_count,
                    state=state_name,
                )
                _flush_partial(
                    output_dir,
                    support_summary,
                    theory_summary,
                    signal_json,
                    signal_rows,
                    full_rows,
                    adaptive_rows,
                    epsilon_rows,
                    resource_summary,
                )

            previous_adaptive_map: np.ndarray | None = None
            previous_map_before_middle: np.ndarray | None = None
            for state_index, state_specification in enumerate(config["fixed_states"]):
                state_name = state_specification["name"]
                context = full_contexts[state_name]
                state = context["state"]
                if state_index == 1:
                    previous_map_before_middle = previous_adaptive_map
                menz = ImplicitMenzSystem(
                    q0,
                    bank,
                    state["observed_nodes"],
                    observation_precision=config["observation_precision"],
                    residual_tolerance=config["theory"][
                        "influence_solve_relative_residual_maximum"
                    ],
                )
                print(
                    f"PROGRESS support={support_count} state={state_name} phase=ADAPTIVE eps=0.02",
                    flush=True,
                )
                adaptive_started = time.perf_counter()
                records, outcome = run_adaptive_stage_probe(
                    context["reference"],
                    menz,
                    bank,
                    action_support,
                    state["observed_action_positions"],
                    action_keys,
                    state["incumbent_standardized"],
                    context["ei"],
                    context["leader"],
                    previous_adaptive_map,
                    epsilon=config["adaptive"]["epsilon_primary"],
                    rho=config["adaptive"]["rho"],
                    max_stages=config["adaptive"]["max_stages"],
                    chunk_size=config["factor_bank"]["chunk_size"],
                    laplace_settings=_laplace_settings(config),
                    guard=guard,
                    phase_name="adaptive_fixed_state",
                )
                for record in records:
                    adaptive_rows.append(
                        {
                            "support_count": support_count,
                            "factor_count": bank.factor_count,
                            "state": state_name,
                            **record,
                        }
                    )
                outcomes.append(
                    {
                        "support_count": support_count,
                        "state": state_name,
                        "epsilon": 0.02,
                        **{key: value for key, value in outcome.items() if key != "map"},
                    }
                )
                previous_adaptive_map = np.asarray(outcome["map"], dtype=np.float64)
                guard.record(
                    "adaptive_fixed_state",
                    adaptive_started,
                    support_count=support_count,
                    state=state_name,
                    epsilon=0.02,
                )
                _flush_partial(
                    output_dir,
                    support_summary,
                    theory_summary,
                    signal_json,
                    signal_rows,
                    full_rows,
                    adaptive_rows,
                    epsilon_rows,
                    resource_summary,
                )

            middle_name = "seed_0_after_6_queries"
            middle = full_contexts[middle_name]
            for epsilon in config["adaptive"]["epsilon_sensitivity_middle_state"]:
                state = middle["state"]
                menz = ImplicitMenzSystem(
                    q0,
                    bank,
                    state["observed_nodes"],
                    observation_precision=config["observation_precision"],
                    residual_tolerance=config["theory"][
                        "influence_solve_relative_residual_maximum"
                    ],
                )
                print(
                    f"PROGRESS support={support_count} state={middle_name} phase=SENSITIVITY eps={epsilon}",
                    flush=True,
                )
                _, outcome = run_adaptive_stage_probe(
                    middle["reference"],
                    menz,
                    bank,
                    action_support,
                    state["observed_action_positions"],
                    action_keys,
                    state["incumbent_standardized"],
                    middle["ei"],
                    middle["leader"],
                    previous_map_before_middle,
                    epsilon=float(epsilon),
                    rho=config["adaptive"]["rho"],
                    max_stages=config["adaptive"]["max_stages"],
                    chunk_size=config["factor_bank"]["chunk_size"],
                    laplace_settings=_laplace_settings(config),
                    guard=guard,
                    phase_name="adaptive_fixed_state",
                )
                epsilon_rows.append(
                    {
                        "support_count": support_count,
                        "factor_count": bank.factor_count,
                        "state": middle_name,
                        "epsilon": epsilon,
                        "active_factor_count_at_certification_or_fallback": outcome[
                            "pre_fallback_active_count"
                        ],
                        "active_fraction_at_certification_or_fallback": outcome[
                            "pre_fallback_active_fraction"
                        ],
                        "stage_count": outcome["stage_count"],
                        "certified": outcome["certified"],
                        "full_bank_fallback": outcome["full_bank_fallback"],
                        "shadow_full_action_agreement": outcome[
                            "shadow_full_action_agreement"
                        ],
                        "shadow_full_laplace_ei_regret": outcome[
                            "shadow_full_laplace_ei_regret"
                        ],
                        "conditioning_seconds": outcome[
                            "total_conditioning_seconds"
                        ],
                        "factor_energy_gradient_work": outcome[
                            "factor_energy_gradient_work"
                        ],
                        "factor_hessian_work": outcome["factor_hessian_work"],
                        "peak_rss_bytes": peak_rss_bytes(),
                    }
                )

            _flush_partial(
                output_dir,
                support_summary,
                theory_summary,
                signal_json,
                signal_rows,
                full_rows,
                adaptive_rows,
                epsilon_rows,
                resource_summary,
            )
            del bank, support_reference, pbe_map, full_contexts
            gc.collect()

        structural = _structural_summary(
            support_summary, adaptive_rows, full_rows, outcomes
        )
        support_summary["structural_scaling_summary"] = structural
        classification = _classify(structural)
        resource_summary.update(
            {
                "overall_elapsed_seconds": time.perf_counter() - overall_started,
                "peak_rss_bytes": peak_rss_bytes(),
                "peak_rss_gb": peak_rss_bytes() / 1_000_000_000.0,
                "local_resource_blocked": False,
                "classification": classification,
                "fresh_seeds_accessed": False,
                "target_input": {
                    "path": str(paths["development_seed0_states"].relative_to(ROOT)),
                    "authorized_seed": 0,
                    "authorized_method": "FULL_PBE",
                    "maximum_authorized_observation_count": 20,
                    "fresh_oracle_opened": False,
                },
            }
        )
        _flush_partial(
            output_dir,
            support_summary,
            theory_summary,
            signal_json,
            signal_rows,
            full_rows,
            adaptive_rows,
            epsilon_rows,
            resource_summary,
        )
        (output_dir / "RESULTS.md").write_text(
            _results_markdown(
                implementation_sha,
                config_sha,
                classification,
                support_summary,
                theory_summary,
                signal_json,
                full_rows,
                structural,
                epsilon_rows,
                resource_summary,
            ),
            encoding="utf-8",
        )
        _write_json(
            output_dir / "artifact_manifest.json",
            _manifest(output_dir, implementation_sha, config_sha, classification),
        )
        print(classification, flush=True)
        return classification
    except LocalResourceBlocked as exc:
        classification = "LOCAL_RESOURCE_BLOCKED"
        resource_summary.update(
            {
                "overall_elapsed_seconds": time.perf_counter() - overall_started,
                "peak_rss_bytes": peak_rss_bytes(),
                "peak_rss_gb": peak_rss_bytes() / 1_000_000_000.0,
                "local_resource_blocked": True,
                "blocker": str(exc),
                "classification": classification,
                "fresh_seeds_accessed": False,
            }
        )
        _flush_partial(
            output_dir,
            support_summary,
            theory_summary,
            signal_json,
            signal_rows,
            full_rows,
            adaptive_rows,
            epsilon_rows,
            resource_summary,
        )
        (output_dir / "RESULTS.md").write_text(
            "# Full-bank scaling probe — DEVELOPMENT ONLY\n\n"
            "Development classification: `LOCAL_RESOURCE_BLOCKED`.\n\n"
            f"{exc}\n\nCompleted outputs were preserved. No fresh seed was accessed.\n",
            encoding="utf-8",
        )
        notebook_path = ROOT / "experiments/sun_oxide/colab_full_bank_scaling_probe.ipynb"
        _write_json(notebook_path, _blocked_notebook(config_path))
        _write_json(
            output_dir / "artifact_manifest.json",
            _manifest(output_dir, implementation_sha, config_sha, classification),
        )
        print(classification, flush=True)
        return classification


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--repository-root", type=Path, default=ROOT)
    parser.add_argument("--implementation-sha", required=True)
    return parser.parse_args()


def main() -> int:
    arguments = _parse_arguments()
    classification = run_probe(
        arguments.config.resolve(),
        arguments.repository_root.resolve(),
        arguments.output_dir.resolve(),
        arguments.implementation_sha,
    )
    return 0 if classification != "LOCAL_RESOURCE_BLOCKED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
