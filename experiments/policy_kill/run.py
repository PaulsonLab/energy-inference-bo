"""Run the prospectively frozen reference-only policy-kill gate."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from causal_policy_bo.model import TwoRegimeConfig, TwoRegimeModel
from causal_policy_bo.reference import (
    DESIGN_SCAN_LEVEL,
    REFERENCE_LEVELS,
    planning_opportunity,
    solve_reference_level,
)


OUTPUT_DIRECTORY = Path(__file__).resolve().parent / "outputs"
DELTAS = (0.15, 0.20, 0.25, 0.30, 0.35)
NOISE_LEVELS = (0.10, 0.15, 0.20, 0.25)
CLASSIFICATION = "POLICY_KILL_NEGATIVE_REVIEW_REQUIRED"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _scan_candidates() -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    candidate_index = 0
    for delta in DELTAS:
        for noise_std in NOISE_LEVELS:
            candidate_index += 1
            config = TwoRegimeConfig(delta=delta, noise_std=noise_std)
            start = time.perf_counter()
            solution = solve_reference_level(config, DESIGN_SCAN_LEVEL)
            elapsed = time.perf_counter() - start
            passes, diagnostics = planning_opportunity(
                solution, TwoRegimeModel(config)
            )
            candidate = {
                "candidate_index": candidate_index,
                "config": config,
                "solution": solution,
                "passes": passes,
                "diagnostics": diagnostics,
                "wall_clock_seconds": elapsed,
            }
            candidates.append(candidate)
            for reference_row in solution.rows:
                row = dict(reference_row)
                row.update(
                    {
                        "candidate_index": candidate_index,
                        "delta": delta,
                        "noise_std": noise_std,
                        "candidate_passes": passes,
                        "diagnostic_to_optimal_ei_ratio": diagnostics[
                            "diagnostic_to_optimal_ei_ratio"
                        ],
                        "wall_clock_seconds": elapsed,
                    }
                )
                rows.append(row)
            actions = ",".join(
                f"{float(row['first_action']):.3f}" for row in solution.rows
            )
            largest_loss = max(
                float(row["forced_myopic_relative_loss"])
                for row in solution.rows
            )
            print(
                f"[REFERENCE] candidate {candidate_index:02d}/20 "
                f"delta={delta:.2f} sigma={noise_std:.2f} "
                f"actions={actions} max_loss={100.0 * largest_loss:.4f}% "
                f"pass={passes} ... done",
                flush=True,
            )
    return pd.DataFrame(rows), candidates


def _audit_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    diagnostic_candidates = []
    for candidate in candidates:
        rows = candidate["solution"].rows
        has_diagnostic_action = any(
            0.45 <= float(row["first_action"]) <= 0.55
            for row in rows
            if int(row["horizon"]) in (2, 3)
        )
        if has_diagnostic_action:
            diagnostic_candidates.append(candidate)
    pool = diagnostic_candidates or candidates
    return max(
        pool,
        key=lambda candidate: max(
            float(row["forced_myopic_relative_loss"])
            for row in candidate["solution"].rows
        ),
    )


def _frozen_record(
    selected: dict[str, Any] | None, audit: dict[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment": "policy_kill",
        "selection_status": (
            "SELECTED" if selected is not None else "FAILED_NO_CANDIDATE"
        ),
        "selected_config": (
            selected["config"].to_dict() if selected is not None else None
        ),
        "audit_config": audit["config"].to_dict(),
        "audit_reason": (
            "closest candidate with a diagnostic first action and the largest "
            "forced-myopic value loss"
        ),
        "response_family": {
            "rbf": "exp(-0.5*((x-center)/width)^2)",
            "f0": "1.12*g_left + (1.10-delta)*g_right - 0.40*g_diagnostic",
            "f1": "(1.12-delta)*g_left + 1.10*g_right + 0.30*g_diagnostic",
        },
        "candidate_grid": {
            "delta": list(DELTAS),
            "noise_std": list(NOISE_LEVELS),
            "order": "delta-major lexicographic",
        },
        "design_scan_level": DESIGN_SCAN_LEVEL.to_dict(),
        "prospective_gates": {
            "one_step_action_in_exploit_region": True,
            "nonmyopic_action_in_diagnostic_region": [0.45, 0.55],
            "minimum_action_displacement": 0.10,
            "minimum_forced_myopic_relative_loss": 0.02,
            "maximum_diagnostic_to_optimal_one_step_ei": 0.25,
        },
        "terminal_utility": "max(0, max_t(Y_t) - incumbent)",
    }


def _reference_audit(
    audit: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    config = audit["config"]
    solutions = [audit["solution"]]
    for level in REFERENCE_LEVELS[1:]:
        start = time.perf_counter()
        solution = solve_reference_level(config, level)
        elapsed = time.perf_counter() - start
        for row in solution.rows:
            row["wall_clock_seconds"] = elapsed
        solutions.append(solution)
        print(f"[REFERENCE] {level.name} ... done ({elapsed:.2f}s)", flush=True)

    rows: list[dict[str, Any]] = []
    for solution_index, solution in enumerate(solutions):
        for row in solution.rows:
            record = dict(row)
            if solution_index == 0:
                record["level"] = "coarse"
            record.setdefault(
                "wall_clock_seconds", audit["wall_clock_seconds"]
            )
            record.update(config.to_dict())
            rows.append(record)
    table = pd.DataFrame(rows)

    medium = table[table["level"] == "medium"].set_index("horizon")
    fine = table[table["level"] == "fine"].set_index("horizon")
    action_drift = {
        str(horizon): abs(
            float(fine.loc[horizon, "first_action"])
            - float(medium.loc[horizon, "first_action"])
        )
        for horizon in (1, 2, 3)
    }
    relative_value_drift = {
        str(horizon): abs(
            float(fine.loc[horizon, "value"])
            - float(medium.loc[horizon, "value"])
        )
        / float(fine.loc[horizon, "value"])
        for horizon in (1, 2, 3)
    }
    convergence = {
        "action_drift_medium_to_fine": action_drift,
        "relative_value_drift_medium_to_fine": relative_value_drift,
        "maximum_action_drift": max(action_drift.values()),
        "maximum_relative_value_drift": max(relative_value_drift.values()),
        "action_tolerance": 0.01,
        "relative_value_tolerance": 0.0025,
        "passes": max(action_drift.values()) <= 0.01
        and max(relative_value_drift.values()) <= 0.0025,
        "refinement_required": False,
    }
    return table, convergence


def _save_figure(figure: plt.Figure, stem: str) -> None:
    figure.savefig(OUTPUT_DIRECTORY / f"{stem}.png", dpi=180, bbox_inches="tight")
    figure.savefig(OUTPUT_DIRECTORY / f"{stem}.pdf", bbox_inches="tight")
    plt.close(figure)


def _figure_planning(reference: pd.DataFrame, config: TwoRegimeConfig) -> None:
    model = TwoRegimeModel(config)
    x = np.linspace(0.0, 1.0, 1001)
    fine = reference[reference["level"] == "fine"].set_index("horizon")
    figure, axis = plt.subplots(figsize=(8.2, 4.8))
    axis.plot(x, model.response_numpy(x, 0), label=r"$f_0(x)$", linewidth=2.2)
    axis.plot(x, model.response_numpy(x, 1), label=r"$f_1(x)$", linewidth=2.2)
    axis.axvspan(0.15, 0.35, color="#4C78A8", alpha=0.08)
    axis.axvspan(0.65, 0.85, color="#4C78A8", alpha=0.08)
    axis.axvspan(0.45, 0.55, color="#F58518", alpha=0.12)
    colors = {1: "#2F4B7C", 2: "#59A14F", 3: "#E45756"}
    for horizon in (1, 2, 3):
        axis.axvline(
            float(fine.loc[horizon, "first_action"]),
            color=colors[horizon],
            linestyle=(0, (4, 3)),
            linewidth=1.8,
            label=f"DP first action H={horizon}",
        )
    axis.axhline(config.incumbent, color="black", linestyle=":", label="incumbent")
    axis.set(xlabel="Experiment x", ylabel="Response", xlim=(0.0, 1.0))
    axis.set_title("Closest construction: action moves, but the value gain is negligible")
    axis.legend(ncol=2, fontsize=8)
    figure.tight_layout()
    _save_figure(figure, "figure1_planning_opportunity")


def _figure_design_scan(scan: pd.DataFrame) -> None:
    horizon3 = scan[scan["horizon"] == 3]
    action = horizon3.pivot(index="delta", columns="noise_std", values="first_action")
    loss = 100.0 * horizon3.pivot(
        index="delta", columns="noise_std", values="forced_myopic_relative_loss"
    )
    figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), constrained_layout=True)
    image0 = axes[0].imshow(action.values, aspect="auto", vmin=0.0, vmax=1.0)
    image1 = axes[1].imshow(loss.values, aspect="auto", vmin=0.0, vmax=2.0)
    for axis, title in zip(
        axes,
        ("H=3 first action", "Forced-myopic value loss (%)"),
        strict=True,
    ):
        axis.set_xticks(range(len(action.columns)), [f"{v:.2f}" for v in action.columns])
        axis.set_yticks(range(len(action.index)), [f"{v:.2f}" for v in action.index])
        axis.set(xlabel="Observation noise σ", ylabel="Exploit separation Δ", title=title)
    for row in range(action.shape[0]):
        for column in range(action.shape[1]):
            axes[0].text(column, row, f"{action.iloc[row, column]:.2f}", ha="center", va="center", fontsize=8)
            axes[1].text(column, row, f"{loss.iloc[row, column]:.3f}", ha="center", va="center", fontsize=8)
    figure.colorbar(image0, ax=axes[0], shrink=0.82)
    figure.colorbar(image1, ax=axes[1], shrink=0.82)
    _save_figure(figure, "figure2_design_scan_failure")


def _figure_convergence(reference: pd.DataFrame) -> None:
    levels = ["coarse", "medium", "fine"]
    figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), constrained_layout=True)
    for horizon in (1, 2, 3):
        subset = reference[reference["horizon"] == horizon].set_index("level").loc[levels]
        axes[0].plot(levels, subset["first_action"], marker="o", label=f"H={horizon}")
        axes[1].plot(levels, subset["value"], marker="o", label=f"H={horizon}")
    axes[0].set(ylabel="First action", title="Reference action convergence")
    axes[1].set(ylabel="Expected terminal improvement", title="Reference value convergence")
    for axis in axes:
        axis.set_xlabel("Reference level")
        axis.grid(alpha=0.25)
        axis.legend()
    _save_figure(figure, "figure3_reference_convergence")


def _results_markdown(summary: dict[str, Any]) -> str:
    audit = summary["audit_reference"]
    return f"""# Policy Kill Results

## Decision

The prospectively frozen two-regime design scan did not produce a meaningful
nonmyopic planning opportunity. Learned-policy optimization was therefore not
run.

## Reference result

The closest candidate used `delta={audit['delta']:.2f}` and
`noise_std={audit['noise_std']:.2f}`. Its converged first actions were:

- H=1: `{audit['horizons']['1']['first_action']:.3f}`
- H=2: `{audit['horizons']['2']['first_action']:.3f}`
- H=3: `{audit['horizons']['3']['first_action']:.3f}`

The H=3 action moved to the diagnostic region, but forcing the H=1 action lost
only `{100.0 * audit['horizons']['3']['forced_myopic_relative_loss']:.4f}%` of
the optimal H=3 value, far below the frozen `2%` requirement. The medium-to-fine
value drift was at most
`{100.0 * summary['reference_convergence']['maximum_relative_value_drift']:.4f}%`.

## Scientific answers

1. **Nonmyopia:** No. The location changed in one construction, but the value
   difference was not meaningful.
2. **Policy representation:** Not tested because the prospective reference gate
   failed.
3. **Energy/transport value:** Not tested because the prospective reference gate
   failed.

## Scope respected

No rollout policy, critic, actor–critic baseline, long-horizon suite, or complex
posterior model was implemented.

{CLASSIFICATION}
"""


def main() -> None:
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    scan, candidates = _scan_candidates()
    scan.to_csv(OUTPUT_DIRECTORY / "design_scan.csv", index=False)
    selected = next((candidate for candidate in candidates if candidate["passes"]), None)
    audit = _audit_candidate(candidates)
    frozen = _frozen_record(selected, audit)
    _write_json(OUTPUT_DIRECTORY / "frozen_config.json", frozen)

    if selected is not None:
        raise RuntimeError(
            "The reference gate unexpectedly passed; the learned-policy phase "
            "must be reviewed before execution."
        )

    reference, convergence = _reference_audit(audit)
    reference.to_csv(OUTPUT_DIRECTORY / "exact_reference.csv", index=False)
    if not convergence["passes"]:
        raise RuntimeError("The failed-gate reference did not converge")

    fine = reference[reference["level"] == "fine"].set_index("horizon")
    audit_reference = {
        "delta": audit["config"].delta,
        "noise_std": audit["config"].noise_std,
        "horizons": {
            str(horizon): {
                "first_action": float(fine.loc[horizon, "first_action"]),
                "value": float(fine.loc[horizon, "value"]),
                "forced_myopic_value": float(
                    fine.loc[horizon, "forced_myopic_value"]
                ),
                "forced_myopic_relative_loss": float(
                    fine.loc[horizon, "forced_myopic_relative_loss"]
                ),
            }
            for horizon in (1, 2, 3)
        },
    }
    summary = {
        "classification": CLASSIFICATION,
        "candidate_count": len(candidates),
        "passing_candidate_count": 0,
        "planning_opportunity": False,
        "policy_representation": "NOT_RUN_REFERENCE_GATE_FAILED",
        "transport_advantage": "NOT_RUN_REFERENCE_GATE_FAILED",
        "audit_reference": audit_reference,
        "reference_convergence": convergence,
        "prospective_stop_honored": True,
    }
    _write_json(OUTPUT_DIRECTORY / "summary.json", summary)
    _figure_planning(reference, audit["config"])
    _figure_design_scan(scan)
    _figure_convergence(reference)
    (OUTPUT_DIRECTORY / "RESULTS.md").write_text(_results_markdown(summary))
    print("[RESULT] reference recovery: not run (reference gate failed)", flush=True)
    print("[RESULT] transport advantage: not run (reference gate failed)", flush=True)
    print(f"[RESULT] {CLASSIFICATION}", flush=True)


if __name__ == "__main__":
    main()
