#!/usr/bin/env python3
"""Run the final development-only E3 decision-sparsity diagnostic.

The only target-bearing input is the committed seed-0 FULL_PBE observation
fixture.  Each state loader stops at its authorized prefix.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import time
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from scipy.stats import spearmanr

from conditioned_bo.bo_value import (
    NumericalFailure,
    gaussian_expected_improvement,
    select_unobserved_action,
)
from conditioned_bo.decision_sparsity import (
    classify_development,
    deterministic_random_subset,
    factor_set_sha256,
    fit_active_acquisition,
    run_influence_path,
    stabilization_quantities,
    target_factor_counts,
)
from conditioned_bo.full_bank_scaling import (
    CompactActiveState,
    ImplicitMenzSystem,
    ResourceGuard,
    TimedCompactFactorBank,
    build_compact_pair_bank,
    construct_support_precision_reference,
    fit_compact_laplace,
    peak_rss_bytes,
    update_fixed_reference_state,
)
from conditioned_bo.pbe_factor_theory import load_action_mapping, load_legacy_nodes


ROOT = Path(__file__).resolve().parents[2]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


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
        if root not in path.parents:
            raise ValueError(f"Input escapes repository root: {path}")
        observed = _sha256(path)
        if observed != specification["sha256"]:
            raise ValueError(f"Input hash mismatch for {name}: {observed}")
        result[name] = path
    return result


def _load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1 or not config.get("development_only"):
        raise ValueError("The diagnostic requires the development-only config")
    expected_model = {
        "support_name": "PBE_SUPPORT_500_V1",
        "model_name": "NORMALIZED_ALL_PAIRS_PBE_500_V1",
        "support_count": 500,
        "action_count": 191,
        "strict_factor_count": 124718,
        "weight": 1.0 / 499.0,
        "weight_exact": "1/499",
        "sigma_obs": 0.05,
        "observation_precision": 400.0,
    }
    if config.get("benchmark_name") != "CURRENT_NLR_PBE_GW_V1":
        raise ValueError("Frozen benchmark changed")
    if config.get("starting_main_sha") != (
        "5f49f140acc3532b0231b1d1c446d22cd0e168d8"
    ):
        raise ValueError("Starting GitHub main SHA changed")
    if config.get("model") != expected_model:
        raise ValueError("Frozen 500-support model changed")
    if config.get("fixed_states") != [
        {
            "name": "seed_0_initial",
            "observation_count": 8,
            "completed_full_pbe_queries": 0,
            "committed_full_leader": 13,
        },
        {
            "name": "seed_0_after_6_queries",
            "observation_count": 14,
            "completed_full_pbe_queries": 6,
            "committed_full_leader": 134,
        },
        {
            "name": "seed_0_after_12_queries",
            "observation_count": 20,
            "completed_full_pbe_queries": 12,
            "committed_full_leader": 133,
        },
    ]:
        raise ValueError("Frozen seed-0 states changed")
    expected_fractions = [
        0.0,
        0.01,
        0.02,
        0.05,
        0.10,
        0.20,
        0.30,
        0.40,
        0.50,
        0.60,
        0.70,
        0.80,
        0.90,
        0.95,
        1.00,
    ]
    if config.get("fraction_grid") != expected_fractions:
        raise ValueError("Frozen subset-fraction grid changed")
    if config.get("influence_paths") != [
        "STATIC_INFLUENCE_PREFIX",
        "RERANKED_FINE_PATH",
    ]:
        raise ValueError("Influence paths changed")
    if config.get("epsilon_struct") != 0.02:
        raise ValueError("Frozen epsilon_struct changed")
    random = config["random_baseline"]
    if random != {
        "fractions": [0.10, 0.20, 0.40],
        "replicates_per_state_fraction": 20,
        "base_seed": 20260823,
        "seed_scheme": (
            "numpy.SeedSequence([base_seed,state_index,fraction_index,replicate])"
        ),
    }:
        raise ValueError("Matched random baseline changed")
    rules = config["classification_rules"]
    if rules != {
        "primary_path": "RERANKED_FINE_PATH",
        "strong_bound_conservatism": {
            "maximum_each_first_stable_agreement_fraction": 0.4,
            "maximum_suffix_full_ei_regret": 0.01,
            "minimum_each_certificate_gap": 0.3,
        },
        "mixed_decision_sparsity": {
            "maximum_median_first_stable_agreement_fraction": 0.5,
            "maximum_each_first_stable_agreement_fraction": 0.7,
        },
        "otherwise": "DECISION_DENSE",
    }:
        raise ValueError("Prospective classification rules changed")
    if config["fresh_seed_policy"]["forbidden_seeds"] != list(range(12, 32)):
        raise ValueError("Fresh-seed guard changed")
    if config["fresh_seed_policy"]["scientific_preregistration_created"]:
        raise ValueError("A scientific preregistration is forbidden")
    if config["resource_guard"]["colab_fallback_allowed"]:
        raise ValueError("This diagnostic must remain local")
    target_factor_counts(expected_fractions, 124718)
    return config


def _load_authorized_state(
    path: Path,
    *,
    observation_count: int,
    action_rows: Sequence[dict[str, Any]],
    expected_mean: float,
    expected_scale: float,
) -> dict[str, Any]:
    """Parse exactly one authorized prefix and stop before the next row."""

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
    positions = np.asarray(
        [int(row["action_position"]) for row in rows], dtype=np.int64
    )
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
        "standardized_observations": standardized,
        "incumbent_standardized": float(np.max(standardized)),
        "rows_read": observation_count,
        "later_rows_read": 0,
    }


def _load_frozen_bank(
    paths: dict[str, Path], config: dict[str, Any]
) -> tuple[Any, np.ndarray, list[dict[str, Any]], np.ndarray, list[str]]:
    nodes = load_legacy_nodes(paths["legacy_pbe"], 2142)
    actions = load_action_mapping(paths["action_node_mapping"], 191, 2142)
    support_rows = _read_csv(paths["frozen_support_500"])
    support = np.asarray([int(row["node_index"]) for row in support_rows], dtype=np.int32)
    if support.shape != (500,) or np.any(np.diff(support.astype(np.int64)) <= 0):
        raise ValueError("PBE_SUPPORT_500_V1 changed")
    bank = build_compact_pair_bank(
        support,
        [node.pbe_band_gap for node in nodes],
        model_name=config["model"]["model_name"],
    )
    factor_rows = _read_csv(paths["frozen_factor_bank_500"])
    endpoints = np.asarray(
        [[int(row["support_i"]), int(row["support_j"])] for row in factor_rows],
        dtype=np.int32,
    )
    signs = np.asarray(
        [int(row["sign_s_ij"]) for row in factor_rows], dtype=np.int8
    )
    weights = {float(row["weight"]) for row in factor_rows}
    if not np.array_equal(endpoints, bank.endpoint_pairs):
        raise ValueError("Frozen factor endpoints changed")
    if not np.array_equal(signs, bank.signs):
        raise ValueError("Frozen factor signs changed")
    if weights != {1.0 / 499.0}:
        raise ValueError("Frozen factor weight changed")
    action_nodes = np.asarray(
        [int(action["node_index"]) for action in actions], dtype=np.int64
    )
    action_keys = [str(action["action_key"]) for action in actions]
    return bank, support, actions, action_nodes, action_keys


def _action_support_positions(
    support_nodes: np.ndarray, action_nodes: np.ndarray
) -> np.ndarray:
    lookup = {int(node): position for position, node in enumerate(support_nodes)}
    positions = np.asarray(
        [lookup[int(node)] for node in action_nodes], dtype=np.int64
    )
    if positions.shape != (191,):
        raise ValueError("Frozen action/support mapping changed")
    return positions


def _laplace_settings(config: dict[str, Any]) -> dict[str, Any]:
    values = config["laplace"]
    return {
        "gradient_tolerance": values["map_gradient_infinity_norm_maximum"],
        "optimizer_gradient_tolerance": values["optimizer_gradient_tolerance"],
        "function_tolerance": values["function_tolerance"],
        "maximum_iterations": values["maximum_iterations"],
        "residual_tolerance": values["solve_relative_residual_maximum"],
    }


def _random_comparison(
    active_ei: np.ndarray,
    active_leader: int,
    full_ei: np.ndarray,
    full_leader: int,
    available: np.ndarray,
) -> dict[str, Any]:
    regret = float(full_ei[full_leader] - full_ei[active_leader])
    if regret < -1e-12:
        raise NumericalFailure("Random active action has negative FULL regret")
    difference = np.abs(active_ei[available] - full_ei[available])
    correlation = float(spearmanr(active_ei[available], full_ei[available]).statistic)
    return {
        "active_laplace_leader": int(active_leader),
        "full_shadow_leader": int(full_leader),
        "action_agreement": bool(active_leader == full_leader),
        "full_laplace_ei_regret": max(regret, 0.0),
        "active_full_ei_spearman": correlation,
        "maximum_absolute_ei_difference": float(np.max(difference)),
        "median_absolute_ei_difference": float(np.median(difference)),
    }


def _summarize(
    config: dict[str, Any],
    influence_rows: list[dict[str, Any]],
    random_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    state_order = [item["name"] for item in config["fixed_states"]]
    path_summaries: list[dict[str, Any]] = []
    for state_name in state_order:
        for path_name in config["influence_paths"]:
            rows = [
                row
                for row in influence_rows
                if row["state"] == state_name and row["path"] == path_name
            ]
            quantities = stabilization_quantities(
                rows,
                small_regret_threshold=config["small_regret_threshold"],
            )
            stable_fraction = quantities["first_stable_agreement_fraction"]
            suffix = [
                row
                for row in rows
                if float(row["requested_fraction"]) >= stable_fraction
            ]
            path_summaries.append(
                {
                    "state": state_name,
                    "path": path_name,
                    **quantities,
                    "stable_suffix_regret_at_most_0_01": all(
                        float(row["full_laplace_ei_regret"])
                        <= config["small_regret_threshold"]
                        for row in suffix
                    ),
                    "maximum_regret_from_stable_fraction": max(
                        float(row["full_laplace_ei_regret"]) for row in suffix
                    ),
                }
            )

    reranked = [
        summary
        for summary in path_summaries
        if summary["path"] == "RERANKED_FINE_PATH"
    ]
    classification = classify_development(reranked)
    path_comparison: list[dict[str, Any]] = []
    for state_name in state_order:
        static = next(
            item
            for item in path_summaries
            if item["state"] == state_name
            and item["path"] == "STATIC_INFLUENCE_PREFIX"
        )
        fine = next(
            item
            for item in path_summaries
            if item["state"] == state_name and item["path"] == "RERANKED_FINE_PATH"
        )
        path_comparison.append(
            {
                "state": state_name,
                "static_first_stable_agreement_fraction": static[
                    "first_stable_agreement_fraction"
                ],
                "reranked_first_stable_agreement_fraction": fine[
                    "first_stable_agreement_fraction"
                ],
                "reranked_minus_static_first_stable_fraction": fine[
                    "first_stable_agreement_fraction"
                ]
                - static["first_stable_agreement_fraction"],
                "static_certificate_fraction": static["certificate_fraction"],
                "reranked_certificate_fraction": fine["certificate_fraction"],
            }
        )

    random_summary: list[dict[str, Any]] = []
    for state_name in state_order:
        for fraction in config["random_baseline"]["fractions"]:
            rows = [
                row
                for row in random_rows
                if row["state"] == state_name
                and float(row["requested_fraction"]) == float(fraction)
            ]
            regrets = np.asarray(
                [float(row["full_laplace_ei_regret"]) for row in rows],
                dtype=np.float64,
            )
            fine = next(
                row
                for row in influence_rows
                if row["state"] == state_name
                and row["path"] == "RERANKED_FINE_PATH"
                and float(row["requested_fraction"]) == float(fraction)
            )
            random_summary.append(
                {
                    "state": state_name,
                    "requested_fraction": float(fraction),
                    "active_factor_count": int(rows[0]["active_factor_count"]),
                    "random_replicate_count": len(rows),
                    "random_action_agreement_fraction": float(
                        np.mean([bool(row["action_agreement"]) for row in rows])
                    ),
                    "random_median_full_ei_regret": float(np.median(regrets)),
                    "random_q25_full_ei_regret": float(np.quantile(regrets, 0.25)),
                    "random_q75_full_ei_regret": float(np.quantile(regrets, 0.75)),
                    "reranked_action_agreement": bool(fine["action_agreement"]),
                    "reranked_full_ei_regret": float(
                        fine["full_laplace_ei_regret"]
                    ),
                }
            )
    return {
        "schema_version": 1,
        "development_only": True,
        "terminal_classification": classification,
        "classification_primary_path": "RERANKED_FINE_PATH",
        "small_regret_threshold": config["small_regret_threshold"],
        "epsilon_struct": config["epsilon_struct"],
        "path_summaries": path_summaries,
        "static_vs_reranked": path_comparison,
        "matched_random_summary": random_summary,
        "classification_rules": config["classification_rules"],
    }


def _plot(
    path: Path,
    config: dict[str, Any],
    influence_rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    state_order = [item["name"] for item in config["fixed_states"]]
    labels = {
        "seed_0_initial": "Initial 8 observations",
        "seed_0_after_6_queries": "After 6 FULL queries",
        "seed_0_after_12_queries": "After 12 FULL queries",
    }
    colors = {
        "STATIC_INFLUENCE_PREFIX": "#0072B2",
        "RERANKED_FINE_PATH": "#D55E00",
    }
    styles = {
        "STATIC_INFLUENCE_PREFIX": "--",
        "RERANKED_FINE_PATH": "-",
    }
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.7), sharey=True)
    for axis, state_name in zip(axes, state_order, strict=True):
        for path_name in config["influence_paths"]:
            rows = [
                row
                for row in influence_rows
                if row["state"] == state_name and row["path"] == path_name
            ]
            x = [float(row["requested_fraction"]) for row in rows]
            y = [float(row["full_laplace_ei_regret"]) for row in rows]
            label = "Static prefix" if path_name.startswith("STATIC") else "Reranked"
            axis.plot(
                x,
                y,
                marker="o",
                markersize=3.2,
                linewidth=1.5,
                linestyle=styles[path_name],
                color=colors[path_name],
                label=label,
            )
            certificate = next(
                item["certificate_fraction"]
                for item in summary["path_summaries"]
                if item["state"] == state_name and item["path"] == path_name
            )
            axis.axvline(
                float(certificate),
                color=colors[path_name],
                linestyle=styles[path_name],
                linewidth=1.0,
                alpha=0.7,
            )
        axis.axhline(
            config["small_regret_threshold"],
            color="#666666",
            linestyle=":",
            linewidth=1.0,
        )
        axis.set_title(labels[state_name], fontsize=10)
        axis.set_xlim(-0.01, 1.01)
        axis.grid(axis="y", alpha=0.25)
        axis.set_xlabel("Active factor fraction")
    axes[0].set_ylabel("FULL-PBE Laplace EI regret")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle(
        "Decision stabilization versus theorem certification",
        y=1.02,
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
        metadata={"Software": "energy-inference-bo"},
    )
    plt.close(fig)


def _results_markdown(
    implementation_sha: str,
    config_sha: str,
    full_summary: dict[str, Any],
    summary: dict[str, Any],
    resource: dict[str, Any],
) -> str:
    classification = summary["terminal_classification"]
    lines = [
        "# Decision-sparsity diagnostic — DEVELOPMENT ONLY",
        "",
        f"Terminal development classification: `{classification}`.",
        "",
        "This final diagnostic used only the frozen 500-support E3 model and",
        "the three already-consumed seed-0 FULL_PBE states. It is not a fresh",
        "scientific preregistration and does not treat the known value of PBE",
        "preferences for PBE-to-GW prediction as a novel paper contribution.",
        "",
        f"- Starting GitHub main SHA: `{full_summary['starting_main_sha']}`",
        f"- Implementation SHA: `{implementation_sha}`",
        f"- Config SHA-256: `{config_sha}`",
        f"- Peak process RSS: `{resource['peak_rss_gb']:.9f}` GB",
        "- Fresh seeds 12--31 accessed: `False`",
        "",
        "`PASS_PBE_VALUE`, `ADAPTIVE_ENGINEERING_PATHOLOGICAL`, and",
        "`FULL_ARCHIVE_NOT_HELPFUL` remain valid. The superseded Colab",
        "preregistration remains unauthorized and unrun.",
        "",
        "## FULL shadow reference",
        "",
        "| State | Observations | Committed leader | Recomputed leader | Agreement | FULL fit s |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for state in full_summary["states"]:
        lines.append(
            f"| {state['state']} | {state['observation_count']} | "
            f"{state['committed_full_leader']} | {state['recomputed_full_leader']} | "
            f"{state['leader_agreement']} | "
            f"{state['diagnostics']['total_conditioning_seconds']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Decision stabilization",
            "",
            "| State | Path | First agreement | First stable | First <=0.01 regret | Certificate | Gap |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for item in summary["path_summaries"]:
        lines.append(
            f"| {item['state']} | {item['path']} | "
            f"{item['first_agreement_fraction']:.2f} | "
            f"{item['first_stable_agreement_fraction']:.2f} | "
            f"{item['first_small_regret_fraction']:.2f} | "
            f"{item['certificate_fraction']:.2f} | "
            f"{item['stable_to_certificate_gap']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Matched random subsets",
            "",
            "| State | Fraction | Reranked agreement/regret | Random agreement | Random regret q25 / median / q75 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for item in summary["matched_random_summary"]:
        lines.append(
            f"| {item['state']} | {item['requested_fraction']:.2f} | "
            f"{item['reranked_action_agreement']} / "
            f"{item['reranked_full_ei_regret']:.8g} | "
            f"{item['random_action_agreement_fraction']:.2f} | "
            f"{item['random_q25_full_ei_regret']:.8g} / "
            f"{item['random_median_full_ei_regret']:.8g} / "
            f"{item['random_q75_full_ei_regret']:.8g} |"
        )
    lines.extend(
        [
            "",
            "## Frozen terminal interpretation",
            "",
            "`RERANKED_FINE_PATH` is the primary diagnostic. STATIC and",
            "RERANKED differences, random-subset comparisons, and the empirical",
            "stabilization-versus-certificate gap are reported above without",
            "relaxing `epsilon_struct=0.02` or changing any E3 model choice.",
            "",
        ]
    )
    if classification == "DECISION_DENSE":
        lines.extend(
            [
                "Close adaptive preferences as a main efficiency demonstration.",
                "Preserve E3 only as a limitation/example if useful; do not spend",
                "fresh seeds 12--31.",
            ]
        )
    elif classification == "STRONG_BOUND_CONSERVATISM":
        lines.extend(
            [
                "Do not spend fresh seeds yet. The next task is a mathematical audit",
                "of whether state-dependent/local gradient information can yield a",
                "valid tighter E3 structural bound.",
            ]
        )
    else:
        lines.extend(
            [
                "The result is mixed decision sparsity. Stop at these development",
                "numbers without creating another experiment plan.",
            ]
        )
    return "\n".join(lines) + "\n"


def _artifact_manifest(
    output_dir: Path,
    config_sha: str,
    implementation_sha: str,
    classification: str,
) -> dict[str, Any]:
    filenames = [
        "RESULTS.md",
        "full_reference_summary.json",
        "influence_paths.csv",
        "random_baselines.csv",
        "decision_stabilization_summary.json",
        "resource_usage.json",
        "ei_regret_vs_active_fraction.png",
    ]
    return {
        "schema_version": 1,
        "development_only": True,
        "classification": classification,
        "implementation_sha": implementation_sha,
        "config_sha256": config_sha,
        "fresh_seeds_accessed": False,
        "files": [
            {
                "path": name,
                "size_bytes": (output_dir / name).stat().st_size,
                "sha256": _sha256(output_dir / name),
            }
            for name in filenames
        ],
    }


def run_diagnostic(
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
    bank, support, actions, action_nodes, action_keys = _load_frozen_bank(paths, config)
    if bank.factor_count != config["model"]["strict_factor_count"]:
        raise ValueError("Frozen strict factor count changed")
    q0 = sparse.load_npz(paths["q0"]).tocsr().astype(np.float64)
    action_support = _action_support_positions(support, action_nodes)

    reference_started = time.perf_counter()
    support_reference = construct_support_precision_reference(
        q0,
        support,
        residual_tolerance=config["laplace"]["solve_relative_residual_maximum"],
        guard=guard,
    )
    guard.record("support_reference", reference_started, support_count=500)

    full_states: list[dict[str, Any]] = []
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
        reference = update_fixed_reference_state(
            support_reference,
            action_support[state["observed_action_positions"]],
            state["standardized_observations"],
            action_support,
            observation_precision=config["model"]["observation_precision"],
            residual_tolerance=config["laplace"]["solve_relative_residual_maximum"],
        )
        print(f"PROGRESS state={state_name} phase=FULL", flush=True)
        full_started = time.perf_counter()
        timed_bank = TimedCompactFactorBank(
            bank, None, chunk_size=config["laplace"]["factor_chunk_size"]
        )
        full = fit_compact_laplace(
            reference,
            timed_bank,
            reference.mean if previous_full_map is None else previous_full_map,
            action_support,
            retry_map=np.zeros(bank.support_count, dtype=np.float64),
            guard=guard,
            phase_name="full_reference",
            **_laplace_settings(config),
        )
        full_ei = gaussian_expected_improvement(
            full.map[action_support], full.action_variances, state["incumbent_standardized"]
        )
        full_leader = select_unobserved_action(
            full_ei, state["observed_action_positions"], action_keys
        )
        expected_leader = int(state_specification["committed_full_leader"])
        if full_leader != expected_leader:
            raise NumericalFailure(
                f"FULL leader changed at {state_name}: {full_leader} != {expected_leader}"
            )
        previous_full_map = full.map.copy()
        available = np.ones(len(action_keys), dtype=bool)
        available[state["observed_action_positions"]] = False
        full_states.append(
            {
                "state": state_name,
                "observation_count": state_specification["observation_count"],
                "authorized_target_rows_read": state["rows_read"],
                "unauthorized_later_target_rows_read": state["later_rows_read"],
                "committed_full_leader": expected_leader,
                "recomputed_full_leader": int(full_leader),
                "leader_agreement": True,
                "full_leader_action_key": action_keys[full_leader],
                "unobserved_action_positions": np.flatnonzero(available).tolist(),
                "unobserved_action_keys": [
                    action_keys[index] for index in np.flatnonzero(available)
                ],
                "full_laplace_ei_unobserved": full_ei[available].tolist(),
                "diagnostics": full.diagnostics,
            }
        )
        full_contexts[state_name] = {
            "state": state,
            "reference": reference,
            "full_ei": full_ei,
            "full_leader": int(full_leader),
        }
        guard.record("full_reference", full_started, state=state_name)

    full_summary = {
        "schema_version": 1,
        "development_only": True,
        "starting_main_sha": config["starting_main_sha"],
        "implementation_sha": implementation_sha,
        "model": config["model"],
        "exact_support_reference": support_reference.diagnostics,
        "states": full_states,
        "shadow_reference_only": True,
        "fresh_seeds_accessed": False,
    }
    _write_json(output_dir / "full_reference_summary.json", full_summary)

    influence_rows: list[dict[str, Any]] = []
    for state_specification in config["fixed_states"]:
        state_name = state_specification["name"]
        context = full_contexts[state_name]
        state = context["state"]
        menz = ImplicitMenzSystem(
            q0,
            bank,
            state["observed_nodes"],
            observation_precision=config["model"]["observation_precision"],
            residual_tolerance=config["laplace"]["solve_relative_residual_maximum"],
        )
        for path_name in config["influence_paths"]:
            print(f"PROGRESS state={state_name} phase={path_name}", flush=True)
            path_started = time.perf_counter()
            records = run_influence_path(
                path_name,
                config["fraction_grid"],
                context["reference"],
                menz,
                bank,
                action_support,
                state["observed_action_positions"],
                action_keys,
                state["incumbent_standardized"],
                context["full_ei"],
                context["full_leader"],
                epsilon_struct=config["epsilon_struct"],
                chunk_size=config["laplace"]["factor_chunk_size"],
                laplace_settings=_laplace_settings(config),
                guard=guard,
                phase_name="influence_path",
            )
            for record in records:
                influence_rows.append(
                    {
                        "state": state_name,
                        "observation_count": state_specification["observation_count"],
                        "authorized_target_rows_read": state["rows_read"],
                        "unauthorized_later_target_rows_read": state[
                            "later_rows_read"
                        ],
                        **record,
                    }
                )
            guard.record(
                "influence_path",
                path_started,
                state=state_name,
                path=path_name,
            )
            _write_csv(output_dir / "influence_paths.csv", influence_rows)

    random_rows: list[dict[str, Any]] = []
    random_specification = config["random_baseline"]
    for state_index, state_specification in enumerate(config["fixed_states"]):
        state_name = state_specification["name"]
        context = full_contexts[state_name]
        state = context["state"]
        available = np.ones(len(action_keys), dtype=bool)
        available[state["observed_action_positions"]] = False
        for fraction_index, fraction in enumerate(random_specification["fractions"]):
            subset_count = int(round(float(fraction) * bank.factor_count))
            for replicate in range(
                random_specification["replicates_per_state_fraction"]
            ):
                print(
                    f"PROGRESS state={state_name} phase=RANDOM "
                    f"fraction={fraction:.2f} replicate={replicate + 1}/20",
                    flush=True,
                )
                random_started = time.perf_counter()
                subset = deterministic_random_subset(
                    bank.factor_count,
                    subset_count,
                    base_seed=random_specification["base_seed"],
                    state_index=state_index,
                    fraction_index=fraction_index,
                    replicate=replicate,
                )
                acquisition, leader, _, diagnostics = fit_active_acquisition(
                    context["reference"],
                    bank,
                    subset,
                    action_support,
                    state["observed_action_positions"],
                    action_keys,
                    state["incumbent_standardized"],
                    context["reference"].mean,
                    chunk_size=config["laplace"]["factor_chunk_size"],
                    laplace_settings=_laplace_settings(config),
                    guard=guard,
                    phase_name="random_baseline",
                )
                random_rows.append(
                    {
                        "state": state_name,
                        "observation_count": state_specification["observation_count"],
                        "requested_fraction": float(fraction),
                        "active_factor_count": subset_count,
                        "active_factor_fraction": subset_count / bank.factor_count,
                        "replicate": replicate,
                        "base_seed": random_specification["base_seed"],
                        "state_index": state_index,
                        "fraction_index": fraction_index,
                        "subset_sha256": factor_set_sha256(subset),
                        **_random_comparison(
                            acquisition,
                            leader,
                            context["full_ei"],
                            context["full_leader"],
                            available,
                        ),
                        "map_iterations": diagnostics["optimizer_iterations"],
                        "map_gradient_infinity_norm": diagnostics[
                            "gradient_infinity_norm"
                        ],
                        "factor_energy_gradient_work": diagnostics[
                            "factor_energy_gradient_element_work"
                        ],
                        "factor_hessian_work": diagnostics[
                            "factor_hessian_element_work"
                        ],
                        "single_active_fit_conditioning_seconds": diagnostics[
                            "single_active_fit_conditioning_seconds"
                        ],
                        "peak_rss_bytes": peak_rss_bytes(),
                        "authorized_target_rows_read": state["rows_read"],
                        "unauthorized_later_target_rows_read": state[
                            "later_rows_read"
                        ],
                    }
                )
                guard.record(
                    "random_baseline",
                    random_started,
                    state=state_name,
                    fraction=float(fraction),
                    replicate=replicate,
                )
            _write_csv(output_dir / "random_baselines.csv", random_rows)

    summary = _summarize(config, influence_rows, random_rows)
    _write_json(output_dir / "decision_stabilization_summary.json", summary)
    _plot(
        output_dir / "ei_regret_vs_active_fraction.png",
        config,
        influence_rows,
        summary,
    )
    resource = {
        "schema_version": 1,
        "development_only": True,
        "terminal_classification": summary["terminal_classification"],
        "overall_elapsed_seconds": time.perf_counter() - overall_started,
        "peak_rss_bytes": peak_rss_bytes(),
        "peak_rss_gb": peak_rss_bytes() / 1_000_000_000.0,
        "rss_limit_bytes": guard.rss_limit_bytes,
        "rss_limit_gb": config["resource_guard"]["peak_rss_gb_maximum"],
        "local_resource_blocked": False,
        "colab_used": False,
        "fresh_seeds_accessed": False,
        "phases": guard.records,
        "target_input": {
            "path": str(paths["development_seed0_states"].relative_to(ROOT)),
            "authorized_seed": 0,
            "authorized_method": "FULL_PBE",
            "maximum_authorized_observation_count": 20,
            "fresh_target_source_opened": False,
        },
    }
    _write_json(output_dir / "resource_usage.json", resource)
    (output_dir / "RESULTS.md").write_text(
        _results_markdown(
            implementation_sha,
            config_sha,
            full_summary,
            summary,
            resource,
        ),
        encoding="utf-8",
    )
    _write_json(
        output_dir / "artifact_manifest.json",
        _artifact_manifest(
            output_dir,
            config_sha,
            implementation_sha,
            summary["terminal_classification"],
        ),
    )
    return str(summary["terminal_classification"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/sun_oxide/configs/decision_sparsity_diagnostic.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/sun_oxide/outputs/decision_sparsity_diagnostic"),
    )
    parser.add_argument("--repository-root", type=Path, default=ROOT)
    parser.add_argument("--implementation-sha", required=True)
    arguments = parser.parse_args()
    verdict = run_diagnostic(
        arguments.config.resolve(),
        arguments.repository_root.resolve(),
        arguments.output_dir.resolve(),
        arguments.implementation_sha,
    )
    print(verdict, flush=True)


if __name__ == "__main__":
    main()
