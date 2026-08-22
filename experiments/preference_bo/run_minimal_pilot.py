"""Run the preregistered minimal E3 preference-informed BO pilot."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import platform
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from conditioned_bo.preference_bo import (
    action_grid,
    active_set_turnover,
    evaluate_gates,
    gp_reference_posterior,
    load_frozen_config,
    numerical_settings_from_config,
    prepare_pilot_inputs,
    preference_influence_components,
    run_seed,
    validate_trajectory_rows,
)
from conditioned_bo.preference_influence import preference_graph_degrees


CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "minimal_pilot.json"
DEFAULT_OUTPUT_DIRECTORY = (
    Path(__file__).resolve().parent / "outputs" / "minimal_pilot"
)
DEFAULT_SMOKE_OUTPUT_DIRECTORY = (
    Path(__file__).resolve().parent / "outputs" / "minimal_pilot_smoke"
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_output(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty required CSV: {path.name}")
    fieldnames: list[str] = []
    for row in rows:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _preference_bank_rows(
    config: dict[str, Any], prepared_inputs, config_sha256: str
) -> list[dict[str, Any]]:
    grid = action_grid(config)
    pairs = np.asarray(
        config["preference_bank"]["endpoint_index_pairs"], dtype=int
    )
    degrees = preference_graph_degrees(pairs, grid.size)
    protocols = config["preference_bank"].get("protocols", {})
    protocol_by_factor = {
        int(factor_index): protocol
        for protocol, payload in protocols.items()
        for factor_index in payload["factor_indices"]
    }
    rows: list[dict[str, Any]] = []
    for seed_input in prepared_inputs:
        for factor_index, (left, right) in enumerate(pairs):
            rows.append(
                {
                    "seed": seed_input.seed,
                    "factor_index": factor_index,
                    "left_action_index": int(left),
                    "right_action_index": int(right),
                    "left_x": float(grid[left]),
                    "right_x": float(grid[right]),
                    "protocol": protocol_by_factor.get(factor_index),
                    "left_graph_degree": int(degrees[left]),
                    "right_graph_degree": int(degrees[right]),
                    "preference_sign": int(
                        seed_input.preference_signs[factor_index]
                    ),
                    "positive_sign_probability": float(
                        seed_input.preference_probabilities[factor_index]
                    ),
                    "preference_bank_sha256": seed_input.preference_bank_sha256,
                    "generated_before_methods": seed_input.generated_before_methods,
                    "config_sha256": config_sha256,
                }
            )
    return rows


def _scalar_noise_rows(
    config: dict[str, Any], prepared_inputs, config_sha256: str
) -> list[dict[str, Any]]:
    grid = action_grid(config)
    rows: list[dict[str, Any]] = []
    for seed_input in prepared_inputs:
        for action_index, noise in enumerate(seed_input.scalar_noise):
            rows.append(
                {
                    "seed": seed_input.seed,
                    "action_index": action_index,
                    "x": float(grid[action_index]),
                    "scalar_noise": float(noise),
                    "scalar_noise_sha256": seed_input.scalar_noise_sha256,
                    "generated_before_methods": seed_input.generated_before_methods,
                    "config_sha256": config_sha256,
                }
            )
    return rows


def _metrics_from_trajectory(
    trajectory_rows: list[dict[str, Any]], horizon: int, n_factors: int
) -> tuple[dict[str, list[int]], list[float], list[dict[str, Any]]]:
    seeds = sorted({int(row["seed"]) for row in trajectory_rows})
    t_values: dict[str, list[int]] = {
        "standard": [],
        "full": [],
        "adaptive": [],
    }
    adaptive_ratios: list[float] = []
    per_seed: list[dict[str, Any]] = []
    for seed in seeds:
        seed_summary: dict[str, Any] = {"seed": seed}
        for method in ("standard", "full", "adaptive"):
            rows = sorted(
                (
                    row
                    for row in trajectory_rows
                    if int(row["seed"]) == seed and row["method"] == method
                ),
                key=lambda row: int(row["bo_iteration"]),
            )
            if len(rows) != horizon:
                raise RuntimeError(
                    f"seed {seed} method {method} has {len(rows)} rows, expected {horizon}"
                )
            t_value = int(rows[0]["t_0_10"])
            t_values[method].append(t_value)
            seed_summary[f"t_0_10_{method}"] = t_value
            seed_summary[f"actions_{method}"] = [
                int(row["selected_action_index"]) for row in rows
            ]
        adaptive_rows = sorted(
            (
                row
                for row in trajectory_rows
                if int(row["seed"]) == seed and row["method"] == "adaptive"
            ),
            key=lambda row: int(row["bo_iteration"]),
        )
        active_sets = [
            json.loads(row["active_factor_indices"]) for row in adaptive_rows
        ]
        factor_sum = sum(int(row["M_t"]) for row in adaptive_rows)
        ratio = factor_sum / (horizon * n_factors)
        adaptive_ratios.append(float(ratio))
        seed_summary["adaptive_active_factor_counts"] = [
            int(row["M_t"]) for row in adaptive_rows
        ]
        seed_summary["adaptive_active_factor_sets"] = active_sets
        seed_summary["adaptive_cumulative_factor_use"] = factor_sum
        seed_summary["adaptive_factor_ratio"] = float(ratio)
        seed_summary["consecutive_active_set_turnover"] = active_set_turnover(
            active_sets
        )
        per_seed.append(seed_summary)
    return t_values, adaptive_ratios, per_seed


def _mechanical_checks(
    *,
    config: dict[str, Any],
    trajectory_rows: list[dict[str, Any]],
    refinement_rows: list[dict[str, Any]],
    acquisition_rows: list[dict[str, Any]],
    bank_rows: list[dict[str, Any]],
    noise_rows: list[dict[str, Any]],
    seed_count: int,
    horizon: int,
) -> dict[str, bool]:
    methods = {row["method"] for row in trajectory_rows}
    pairs = np.asarray(
        config["preference_bank"]["endpoint_index_pairs"], dtype=int
    )
    undirected_edges = {
        tuple(sorted((int(left), int(right)))) for left, right in pairs
    }
    graph_degrees = preference_graph_degrees(
        pairs, int(config["action_grid"]["count"])
    )
    expected_degrees = config["preference_bank"].get("expected_grid_degree")
    adaptive_terminal = {
        (int(row["seed"]), int(row["bo_iteration"]))
        for row in refinement_rows
        if row["stopping_reason"] in {"screening_tolerance", "all_factors_active"}
    }
    adaptive_heldout_rows = [
        row
        for row in acquisition_rows
        if row["method"] == "adaptive"
        and row["stage"] == "heldout_full_target"
    ]
    shared_hashes = True
    for seed in {int(row["seed"]) for row in trajectory_rows}:
        seed_rows = [row for row in trajectory_rows if int(row["seed"]) == seed]
        shared_hashes &= len(
            {row["preference_bank_sha256"] for row in seed_rows}
        ) == 1
        shared_hashes &= len({row["scalar_noise_sha256"] for row in seed_rows}) == 1
    checks = {
        "exactly_three_allowed_methods": methods
        == {"standard", "full", "adaptive"},
        "trajectory_row_count": len(trajectory_rows)
        == seed_count * horizon * 3,
        "preference_banks_generated_before_methods": len(bank_rows)
        == seed_count * int(config["preference_bank"]["factor_count"])
        and all(bool(row["generated_before_methods"]) for row in bank_rows),
        "complete_scalar_noise_maps": len(noise_rows)
        == seed_count * int(config["action_grid"]["count"]),
        "shared_seed_inputs_across_methods": bool(shared_hashes),
        "adaptive_refinement_terminated": len(adaptive_terminal)
        == seed_count * horizon,
        "heldout_full_validation_executed": len(adaptive_heldout_rows)
        == seed_count * horizon * int(config["action_grid"]["count"]),
        "heldout_validation_not_used_for_selection": all(
            not bool(row["used_for_selection"]) for row in adaptive_heldout_rows
        ),
        "no_prohibited_baseline": not methods.difference(
            {"standard", "full", "adaptive"}
        ),
        "scalar_observations_never_screened": all(
            row["screened_factor_source"]
            == "historical_preference_bank_only"
            for row in trajectory_rows
        ),
        "preference_edges_unique": len(undirected_edges) == pairs.shape[0],
        "preference_graph_degree_matches_config": expected_degrees is None
        or np.array_equal(graph_degrees, np.asarray(expected_degrees, dtype=int)),
        "preference_graph_declared_index_only": bool(
            config["preference_bank"].get(
                "graph_construction_depends_only_on_grid_indices", True
            )
        ),
    }
    return checks


def _inference_summary(
    trajectory_rows: list[dict[str, Any]],
    acquisition_rows: list[dict[str, Any]],
    numerical_caveats: list[str],
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "numerical_caveats": numerical_caveats,
        "unreliable_trajectory_rows": sum(
            not bool(row["numerical_accuracy_reliable"])
            for row in trajectory_rows
        ),
    }
    for method in ("full", "adaptive"):
        fractions = np.asarray(
            [
                float(row["inference_ess_fraction"])
                for row in trajectory_rows
                if row["method"] == method
                and row["inference_ess_fraction"] is not None
            ],
            dtype=float,
        )
        summary[method] = {
            "minimum_ess_fraction": float(fractions.min()),
            "median_ess_fraction": float(np.median(fractions)),
            "maximum_ess_fraction": float(fractions.max()),
            "total_wall_time_seconds": float(
                sum(
                    float(row["inference_wall_time_seconds"])
                    for row in trajectory_rows
                    if row["method"] == method
                )
            ),
            "total_raw_factor_likelihood_evaluations": int(
                sum(
                    int(row["raw_factor_likelihood_evaluation_count"])
                    for row in trajectory_rows
                    if row["method"] == method
                )
            ),
            "inference_sample_counts": sorted(
                {
                    int(row["inference_sample_count"])
                    for row in trajectory_rows
                    if row["method"] == method
                }
            ),
            "laplace_all_converged": all(
                bool(row["laplace_converged"])
                for row in trajectory_rows
                if row["method"] == method
            ),
            "maximum_laplace_gradient_infinity_norm": float(
                max(
                    float(row["laplace_gradient_infinity_norm"])
                    for row in trajectory_rows
                    if row["method"] == method
                )
            ),
            "maximum_split_half_ei_discrepancy": (
                None
                if method == "adaptive"
                else float(
                    max(
                        float(row["maximum_split_half_ei_discrepancy"])
                        for row in trajectory_rows
                        if row["method"] == method
                    )
                )
            ),
        }
    heldout_regrets = np.asarray(
        [
            float(row["adaptive_heldout_full_target_acquisition_regret"])
            for row in trajectory_rows
            if row["method"] == "adaptive"
        ]
    )
    heldout_fractions = np.asarray(
        [
            float(row["heldout_full_target_ess_fraction"])
            for row in trajectory_rows
            if row["method"] == "adaptive"
        ]
    )
    heldout_groups: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in acquisition_rows:
        if row["method"] == "adaptive" and row["stage"] == "heldout_full_target":
            key = (int(row["seed"]), int(row["bo_iteration"]))
            heldout_groups.setdefault(key, []).append(row)
    exact_agreement = 0
    for rows in heldout_groups.values():
        eligible = [row for row in rows if not row["was_observed_before_decision"]]
        best = max(eligible, key=lambda row: float(row["acquisition_estimate"]))
        selected = next(row for row in rows if row["is_selected_action"])
        exact_agreement += int(best["action_index"] == selected["action_index"])
    summary["adaptive_heldout_validation"] = {
        "maximum_acquisition_regret": float(heldout_regrets.max()),
        "median_acquisition_regret": float(np.median(heldout_regrets)),
        "minimum_ess_fraction": float(heldout_fractions.min()),
        "median_ess_fraction": float(np.median(heldout_fractions)),
        "maximum_ess_fraction": float(heldout_fractions.max()),
        "sample_counts": sorted(
            {
                int(row["heldout_full_target_sample_count"])
                for row in trajectory_rows
                if row["method"] == "adaptive"
            }
        ),
        "laplace_all_converged": all(
            bool(row["heldout_laplace_converged"])
            for row in trajectory_rows
            if row["method"] == "adaptive"
        ),
        "maximum_laplace_gradient_infinity_norm": float(
            max(
                float(row["heldout_laplace_gradient_infinity_norm"])
                for row in trajectory_rows
                if row["method"] == "adaptive"
            )
        ),
        "maximum_split_half_ei_discrepancy": float(
            max(
                float(row["heldout_maximum_split_half_ei_discrepancy"])
                for row in trajectory_rows
                if row["method"] == "adaptive"
            )
        ),
        "exact_action_agreement_count": exact_agreement,
        "decision_count": len(heldout_groups),
        "exact_action_agreement_fraction": exact_agreement / len(heldout_groups),
        "exact_action_agreement_is_pass_condition": False,
    }
    return summary


def _failure_diagnosis(
    verdict: str,
    gate_result: dict[str, Any],
    trajectory_rows: list[dict[str, Any]],
    refinement_rows: list[dict[str, Any]],
    numerical_caveats: list[str],
) -> dict[str, str] | None:
    if verdict == "PASS":
        return None
    if numerical_caveats:
        category = (
            "preference-model/inference numerical failure"
            if verdict == "FAIL-P1"
            else "inference error forcing excessive refinement"
        )
        return {
            "category": category,
            "evidence": "At least one frozen-cap ESS/split or working-gap diagnostic remained unreliable.",
        }
    if verdict == "FAIL-P1":
        if gate_result["median_t_0_10_standard"] <= 2.0:
            return {
                "category": "scalar BO already solving the ambiguity",
                "evidence": "The scalar-only median target time was already at most two post-initial evaluations.",
            }
        return {
            "category": "insufficient preference information",
            "evidence": "Full preference conditioning was numerically reliable but did not improve the preregistered median target time by one evaluation.",
        }
    adaptive_rows = [row for row in trajectory_rows if row["method"] == "adaptive"]
    activation_rows = [
        row for row in refinement_rows if row["stopping_reason"] == "activate_factor"
    ]
    structural_values = np.asarray(
        [float(row["B_struct"]) for row in activation_rows], dtype=float
    )
    inference_values = np.asarray(
        [float(row["B_infer"]) for row in activation_rows], dtype=float
    )
    heldout_max = max(
        float(row["adaptive_heldout_full_target_acquisition_regret"])
        for row in adaptive_rows
    )
    if (
        not gate_result["p2_sparsity_pass"]
        and activation_rows
        and float(np.mean(structural_values > inference_values)) >= 0.90
        and heldout_max <= 0.05
    ):
        return {
            "category": "conservative structural influence bounds",
            "evidence": (
                "Adaptive factor use failed the sparsity gate while structural "
                "terms dominated inference allowances in at least 90% of "
                "activation rounds and held-out full-target regret stayed within "
                "the screening tolerance."
            ),
        }
    return {
        "category": "genuinely broad decision dependence",
        "evidence": "Reliable adaptive decisions required broad factor activation without a dominant inference-error signal.",
    }


def _next_action(verdict: str, run_id: str) -> str:
    if run_id == "redundant_bank_pilot":
        if verdict == "PASS":
            return (
                "Treat the preliminary E3 phenomenon as sufficiently supported "
                "and next design one realistic larger preference-informed BO "
                "case; do not run more synthetic tuning."
            )
        if verdict == "FAIL-P1":
            return (
                "Preserve that the redundant bank did not retain the preference-"
                "value phenomenon and do not retune this synthetic bank."
            )
        return (
            "Preserve the failed redundant-bank gate and do not design another "
            "synthetic bank in this task; the PDE experiments remain the stronger "
            "setting for the sparsity/scaling claim."
        )
    if verdict == "PASS":
        return (
            "Proceed to the deferred full E3 baseline suite without changing this "
            "minimal-pilot result."
        )
    if verdict == "FAIL-P1":
        return (
            "Preserve the failed phenomenon gate and assess preference-bank "
            "informativeness in a separately preregistered future task; do not "
            "repair this pilot."
        )
    return (
        "Preserve the failed sparse-preservation gate and separately analyze "
        "structural-bound conservatism versus broad decision dependence; do not "
        "tune or rerun this pilot."
    )


def _make_diagnostic(
    path: Path,
    trajectory_rows: list[dict[str, Any]],
    horizon: int,
    n_factors: int,
    sparsity_threshold: float,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    iterations = np.arange(1, horizon + 1)
    colors = {"standard": "#4C78A8", "full": "#F58518", "adaptive": "#54A24B"}
    for method in ("standard", "full", "adaptive"):
        method_rows = [row for row in trajectory_rows if row["method"] == method]
        seeds = sorted({int(row["seed"]) for row in method_rows})
        curves = np.asarray(
            [
                [
                    float(
                        next(
                            row["simple_regret"]
                            for row in method_rows
                            if int(row["seed"]) == seed
                            and int(row["bo_iteration"]) == iteration
                        )
                    )
                    for iteration in iterations
                ]
                for seed in seeds
            ]
        )
        median = np.median(curves, axis=0)
        lower = np.quantile(curves, 0.25, axis=0)
        upper = np.quantile(curves, 0.75, axis=0)
        axes[0].plot(iterations, median, marker="o", label=method, color=colors[method])
        axes[0].fill_between(iterations, lower, upper, color=colors[method], alpha=0.16)
    axes[0].axhline(0.10, color="black", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Post-initial scalar BO iteration")
    axes[0].set_ylabel("Finite-grid simple regret")
    axes[0].set_title("Optimization trajectory (median, IQR)")
    axes[0].legend(frameon=False)

    adaptive = [row for row in trajectory_rows if row["method"] == "adaptive"]
    seeds = sorted({int(row["seed"]) for row in adaptive})
    fractions = np.asarray(
        [
            [
                int(
                    next(
                        row["M_t"]
                        for row in adaptive
                        if int(row["seed"]) == seed
                        and int(row["bo_iteration"]) == iteration
                    )
                )
                / n_factors
                for iteration in iterations
            ]
            for seed in seeds
        ]
    )
    axes[1].plot(
        iterations,
        np.median(fractions, axis=0),
        marker="o",
        color=colors["adaptive"],
    )
    axes[1].fill_between(
        iterations,
        np.quantile(fractions, 0.25, axis=0),
        np.quantile(fractions, 0.75, axis=0),
        color=colors["adaptive"],
        alpha=0.16,
    )
    axes[1].axhline(
        sparsity_threshold, color="black", linestyle="--", linewidth=1
    )
    axes[1].set_ylim(0.0, 1.03)
    axes[1].set_xlabel("Post-initial scalar BO iteration")
    axes[1].set_ylabel("Adaptive active fraction $M_t/N$")
    axes[1].set_title("Decision-specific factor use (median, IQR)")
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _results_markdown(summary: dict[str, Any]) -> str:
    if summary["scientific_gates_evaluated"]:
        gates = summary["gates"]
        gate_text = f"""
## Preregistered gates

- P1: median T_0.10 full = {gates['median_t_0_10_full']:.3g}; standard = {gates['median_t_0_10_standard']:.3g}; pass = {gates['p1_pass']}.
- P2 performance: median T_0.10 adaptive = {gates['median_t_0_10_adaptive']:.3g}; full = {gates['median_t_0_10_full']:.3g}; pass = {gates['p2_performance_pass']}.
- P2 sparsity: median R_factors = {gates['median_adaptive_factor_ratio']:.6g}; pass = {gates['p2_sparsity_pass']}.
- Overall verdict: **{summary['verdict']}**.
"""
    else:
        gate_text = "\n## Scientific gates\n\nNot evaluated: this was the reduced numerical smoke test.\n"
    caveat = (
        "None under the preregistered empirical ESS/split diagnostics."
        if not summary["inference_diagnostics"]["numerical_caveats"]
        else "; ".join(summary["inference_diagnostics"]["numerical_caveats"])
    )
    diagnosis = summary.get("failure_diagnosis")
    diagnosis_text = (
        "\n## Failure diagnosis\n\n"
        f"Main evidence: **{diagnosis['category']}**. {diagnosis['evidence']}\n"
        if diagnosis
        else ""
    )
    return f"""# {summary['result_title']}

Profile: `{summary['profile']}`

Configuration SHA-256: `{summary['config_sha256']}`

Starting commit: `{summary['starting_git_commit']}`
{gate_text}
## Mechanical status

All required mechanical checks passed: `{summary['all_mechanical_checks_passed']}`.

## Numerical caveat

{caveat}

## Interpretation

{summary['scientific_interpretation']}
{diagnosis_text}

## Next action

{summary['next_action']}
"""


def run(
    *,
    smoke: bool = False,
    output_directory: Path | None = None,
    config_path: Path = CONFIG_PATH,
    default_output_directory: Path = DEFAULT_OUTPUT_DIRECTORY,
    default_smoke_output_directory: Path = DEFAULT_SMOKE_OUTPUT_DIRECTORY,
    result_title: str = "E3 Preference-BO Minimal Pilot Results",
    entrypoint_path: Path | None = None,
) -> dict[str, Any]:
    """Execute one immutable smoke or full preference-pilot output directory."""

    config_path = config_path.resolve()
    entrypoint = (entrypoint_path or Path(__file__)).resolve()
    config = load_frozen_config(config_path)
    config_bytes = config_path.read_bytes()
    config_sha256 = hashlib.sha256(config_bytes).hexdigest()
    starting_commit = str(config["provenance"]["starting_git_commit"])
    current_main = _git_output("rev-parse", "main")
    current_head = _git_output("rev-parse", "HEAD")
    if current_main != starting_commit or current_head != starting_commit:
        raise RuntimeError(
            "current main/HEAD no longer matches the frozen starting commit "
            f"({current_main=}, {current_head=}, {starting_commit=})"
        )
    grid = action_grid(config)
    gp = config["gp_reference"]
    prior = gp_reference_posterior(
        grid,
        observed_indices=[],
        observed_values=[],
        kernel_amplitude=float(gp["kernel_amplitude"]),
        kernel_lengthscale=float(gp["kernel_lengthscale"]),
        observation_noise_standard_deviation=float(
            gp["observation_noise_standard_deviation"]
        ),
    )
    _, _, _, prior_influence = preference_influence_components(
        config, prior.precision
    )
    regression = config["influence"]["prior_regression"]
    regression_checks = {
        "minimum_eigenvalue": np.isclose(
            prior_influence.minimum_eigenvalue,
            float(regression["minimum_eigenvalue_approximately"]),
            rtol=0.0,
            atol=5e-8,
        ),
        "condition_number": np.isclose(
            prior_influence.condition_number,
            float(regression["condition_number_approximately"]),
            rtol=0.0,
            atol=0.01 if config["run_id"] == "minimal_pilot" else 5e-7,
        ),
        "spd": prior_influence.is_spd,
    }
    if "minimum_row_dominance_margin_approximately" in regression:
        regression_checks["minimum_row_dominance_margin"] = np.isclose(
            prior_influence.minimum_row_dominance_margin,
            float(regression["minimum_row_dominance_margin_approximately"]),
            rtol=0.0,
            atol=5e-8,
        )
    if not all(regression_checks.values()):
        raise RuntimeError(
            "frozen preference influence/SPD regression failed before execution: "
            f"{regression_checks}"
        )
    destination = output_directory or (
        default_smoke_output_directory if smoke else default_output_directory
    )
    if destination.exists():
        raise RuntimeError(
            f"output directory already exists and will not be overwritten: {destination}"
        )
    destination.mkdir(parents=True)
    shutil.copyfile(config_path, destination / "frozen_config.json")
    if hashlib.sha256((destination / "frozen_config.json").read_bytes()).hexdigest() != config_sha256:
        raise RuntimeError("frozen configuration copy changed bytes")

    run_started = _utc_now()
    wall_started = time.perf_counter()
    prepared_all = prepare_pilot_inputs(config)
    if smoke:
        requested_seed = int(config["smoke_test"]["seed"])
        prepared = tuple(item for item in prepared_all if item.seed == requested_seed)
        horizon = int(config["smoke_test"]["post_initial_horizon"])
    else:
        prepared = prepared_all
        horizon = int(config["scalar_observations"]["post_initial_horizon"])
    numerical_settings = numerical_settings_from_config(config, smoke=smoke)

    bank_rows = _preference_bank_rows(config, prepared, config_sha256)
    noise_rows = _scalar_noise_rows(config, prepared, config_sha256)
    _write_csv(destination / "preference_banks.csv", bank_rows)
    _write_csv(destination / "scalar_noise.csv", noise_rows)

    trajectory_rows: list[dict[str, Any]] = []
    refinement_rows: list[dict[str, Any]] = []
    acquisition_rows: list[dict[str, Any]] = []
    numerical_caveats: list[str] = []
    for seed_position, seed_input in enumerate(prepared, start=1):
        print(
            f"[{seed_position}/{len(prepared)}] seed={seed_input.seed}: "
            "running standard/full/adaptive",
            flush=True,
        )
        result = run_seed(
            config,
            seed_input,
            horizon=horizon,
            numerical_settings=numerical_settings,
            config_sha256=config_sha256,
        )
        trajectory_rows.extend(result.trajectory_rows)
        refinement_rows.extend(result.refinement_rows)
        acquisition_rows.extend(result.acquisition_rows)
        numerical_caveats.extend(result.numerical_caveats)

    validate_trajectory_rows(trajectory_rows)
    _write_csv(destination / "trajectory.csv", trajectory_rows)
    _write_csv(destination / "refinement_history.csv", refinement_rows)
    _write_csv(destination / "acquisition_validation.csv", acquisition_rows)
    checks = _mechanical_checks(
        config=config,
        trajectory_rows=trajectory_rows,
        refinement_rows=refinement_rows,
        acquisition_rows=acquisition_rows,
        bank_rows=bank_rows,
        noise_rows=noise_rows,
        seed_count=len(prepared),
        horizon=horizon,
    )
    if not all(checks.values()):
        raise RuntimeError(f"mechanical output validation failed: {checks}")

    n_factors = int(config["preference_bank"]["factor_count"])
    t_values, adaptive_ratios, per_seed = _metrics_from_trajectory(
        trajectory_rows, horizon, n_factors
    )
    if smoke:
        gate_result = None
        verdict = "NOT_EVALUATED"
        interpretation = (
            "Reduced-count smoke execution was mechanically correct; it is not "
            "scientific evidence for P1 or P2."
        )
        next_action = "Run the unchanged full 12-seed frozen pilot."
        failure_diagnosis = None
    else:
        gate_result = evaluate_gates(
            t_values,
            adaptive_ratios,
            p2_sparsity_threshold=float(
                config["gates"].get("p2_sparsity_threshold", 0.65)
            ),
        )
        verdict = str(gate_result["verdict"])
        failure_diagnosis = _failure_diagnosis(
            verdict,
            gate_result,
            trajectory_rows,
            refinement_rows,
            numerical_caveats,
        )
        if verdict == "PASS":
            interpretation = (
                "The fixed historical preference bank improved scalar-observation "
                "BO and adaptive decision-specific conditioning preserved that "
                "benefit at the preregistered sparsity threshold."
            )
        elif verdict == "FAIL-P1":
            interpretation = (
                "This minimal pilot did not establish that the fixed historical "
                "preference bank improves scalar-observation BO."
            )
        else:
            interpretation = (
                "The preference bank passed the value gate, but adaptive "
                "conditioning did not satisfy both preregistered preservation "
                "and sparsity conditions."
            )
        next_action = _next_action(verdict, str(config["run_id"]))

    adaptive_counts = [
        int(row["M_t"])
        for row in trajectory_rows
        if row["method"] == "adaptive"
    ]
    turnovers = [
        value
        for seed_summary in per_seed
        for value in seed_summary["consecutive_active_set_turnover"]
    ]
    inference_summary = _inference_summary(
        trajectory_rows, acquisition_rows, numerical_caveats
    )
    activation_rows = [
        row for row in refinement_rows if row["stopping_reason"] == "activate_factor"
    ]
    structural_values = np.asarray(
        [float(row["B_struct"]) for row in activation_rows], dtype=float
    )
    inference_values = np.asarray(
        [float(row["B_infer"]) for row in activation_rows], dtype=float
    )
    summary: dict[str, Any] = {
        "run_id": config["run_id"],
        "result_title": result_title,
        "runner_path": str(entrypoint.relative_to(REPOSITORY_ROOT)),
        "profile": "smoke" if smoke else "frozen_pilot",
        "starting_git_commit": starting_commit,
        "config_path": str(config_path.relative_to(REPOSITORY_ROOT)),
        "config_sha256": config_sha256,
        "handoff_path": config["provenance"]["handoff_path"],
        "seeds": [item.seed for item in prepared],
        "methods": config["methods"],
        "first_pilot_record": {
            "run_id": config["provenance"].get("first_pilot_run_id"),
            "verdict": config["provenance"].get("first_pilot_verdict"),
            "config_sha256": config["provenance"].get(
                "first_pilot_config_sha256"
            ),
            "unchanged_by_this_run": True,
        },
        "preference_graph": {
            "edge_count": int(config["preference_bank"]["factor_count"]),
            "endpoint_index_pairs": config["preference_bank"][
                "endpoint_index_pairs"
            ],
            "degrees": preference_graph_degrees(
                np.asarray(
                    config["preference_bank"]["endpoint_index_pairs"], dtype=int
                ),
                int(config["action_grid"]["count"]),
            ).tolist(),
            "construction_depends_only_on_grid_indices": bool(
                config["preference_bank"].get(
                    "graph_construction_depends_only_on_grid_indices", True
                )
            ),
        },
        "prior_influence_regression": {
            "minimum_eigenvalue": prior_influence.minimum_eigenvalue,
            "minimum_row_dominance_margin": prior_influence.minimum_row_dominance_margin,
            "condition_number": prior_influence.condition_number,
            "is_spd": prior_influence.is_spd,
            "checks": {key: bool(value) for key, value in regression_checks.items()},
        },
        "post_initial_horizon": horizon,
        "scientific_gates_evaluated": not smoke,
        "t_0_10_by_method": t_values,
        "adaptive_factor_ratios": adaptive_ratios,
        "gates": gate_result,
        "verdict": verdict,
        "per_seed": per_seed,
        "factor_use": {
            "minimum_M_t": min(adaptive_counts),
            "median_M_t": float(np.median(adaptive_counts)),
            "maximum_M_t": max(adaptive_counts),
            "mean_M_t": float(np.mean(adaptive_counts)),
            "median_scientific_factor_ratio": float(np.median(adaptive_ratios)),
            "total_adaptive_final_active_factor_use": int(sum(adaptive_counts)),
            "M_t_distribution": {
                str(value): int(adaptive_counts.count(value))
                for value in sorted(set(adaptive_counts))
            },
        },
        "factor_set_turnover_diagnostic": {
            "consecutive_turnover_values": turnovers,
            "median_turnover": float(np.median(turnovers)) if turnovers else None,
            "mean_turnover": float(np.mean(turnovers)) if turnovers else None,
            "is_pass_condition": False,
        },
        "inference_diagnostics": inference_summary,
        "structural_refinement_diagnostic": {
            "activation_rounds": len(activation_rows),
            "fraction_with_B_struct_greater_than_B_infer": float(
                np.mean(structural_values > inference_values)
            ),
            "median_B_struct_on_activation_rounds": float(
                np.median(structural_values)
            ),
            "median_B_infer_on_activation_rounds": float(
                np.median(inference_values)
            ),
        },
        "mechanical_checks": checks,
        "all_mechanical_checks_passed": all(checks.values()),
        "failure_diagnosis": failure_diagnosis,
        "scientific_interpretation": interpretation,
        "next_action": next_action,
        "wall_time_seconds": float(time.perf_counter() - wall_started),
        "run_started_utc": run_started,
        "run_completed_utc": _utc_now(),
    }
    (destination / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    (destination / "RESULTS.md").write_text(_results_markdown(summary))
    _make_diagnostic(
        destination / "diagnostic.png",
        trajectory_rows,
        horizon,
        n_factors,
        float(config["gates"].get("p2_sparsity_threshold", 0.65)),
    )

    provenance = {
        "run_id": config["run_id"],
        "profile": summary["profile"],
        "starting_git_commit": starting_commit,
        "main_commit_at_execution": current_main,
        "head_commit_at_execution": current_head,
        "branch": _git_output("branch", "--show-current"),
        "worktree_status_at_completion": _git_output("status", "--short"),
        "remote_url": _git_output("remote", "get-url", "origin"),
        "config_source_path": str(config_path.relative_to(REPOSITORY_ROOT)),
        "frozen_config_output_path": str(
            (destination / "frozen_config.json").relative_to(REPOSITORY_ROOT)
        ),
        "config_sha256": config_sha256,
        "handoff_path": config["provenance"]["handoff_path"],
        "handoff_sha256": hashlib.sha256(
            (REPOSITORY_ROOT / config["provenance"]["handoff_path"]).read_bytes()
        ).hexdigest(),
        "methods": config["methods"],
        "first_pilot_verdict_preserved": config["provenance"].get(
            "first_pilot_verdict"
        ),
        "prior_influence_regression": summary["prior_influence_regression"],
        "seeds": summary["seeds"],
        "post_initial_horizon": horizon,
        "scientific_configuration_unchanged": True,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "matplotlib_version": matplotlib.__version__,
        "runner_path": str(entrypoint.relative_to(REPOSITORY_ROOT)),
        "shared_runner_path": str(Path(__file__).relative_to(REPOSITORY_ROOT)),
        "preserved_attempts": [
            {
                "path": str(path.relative_to(REPOSITORY_ROOT)),
                "status": "preserved incomplete output; not used for final gates",
            }
            for path in sorted(destination.parent.iterdir())
            if path.is_dir()
            and path != destination
            and path.name.startswith(f"{config['run_id']}_")
            and "attempt" in path.name
        ],
        "run_started_utc": run_started,
        "run_completed_utc": summary["run_completed_utc"],
        "wall_time_seconds": summary["wall_time_seconds"],
    }
    (destination / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )
    print(
        f"completed profile={summary['profile']} verdict={verdict} "
        f"output={destination}",
        flush=True,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke", action="store_true", help="run the frozen reduced-count smoke profile"
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=None,
        help="optional new output directory; existing paths are never overwritten",
    )
    arguments = parser.parse_args()
    run(smoke=arguments.smoke, output_directory=arguments.output_directory)


if __name__ == "__main__":
    main()
