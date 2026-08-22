"""Run the frozen prospective Gp2 E3 P1 gate profiles."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import platform
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from conditioned_bo.gp2_preference_bo import (  # noqa: E402
    GateRunResult,
    PreprocessingAmbiguity,
    PreprocessingInvalid,
    config_sha256,
    create_immutable_output_directory,
    load_gate_config,
    prepare_gp2_data,
    run_gate,
)


EXPERIMENT_DIRECTORY = Path(__file__).resolve().parent
DEFAULT_CONFIG = EXPERIMENT_DIRECTORY / "configs" / "p1_gate.json"
OUTPUTS = EXPERIMENT_DIRECTORY / "outputs"
OUTPUT_BY_MODE = {
    "preflight": OUTPUTS / "preflight",
    "smoke": OUTPUTS / "p1_gate_smoke",
    "scientific": OUTPUTS / "p1_gate",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value)!r}")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n"
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
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


def _provenance(
    *,
    config_path: Path,
    config_hash: str,
    prepared,
    mode: str,
    started_at: str,
    git_sha: str,
    git_status_before: str,
) -> dict[str, Any]:
    return {
        "run_id": "gp2_e3_p1_gate",
        "mode": mode,
        "started_at_utc": started_at,
        "completed_at_utc": _utc_now(),
        "git_sha": git_sha,
        "git_branch": _git("branch", "--show-current"),
        "git_status_before_run": git_status_before,
        "config_path": str(config_path.relative_to(REPOSITORY_ROOT)),
        "config_sha256": config_hash,
        "handoff_path": "project/E3_GP2_P1_GATE_HANDOFF.md",
        "external_repository": "HackelLab-UMN/DevRep",
        "external_commit": "e05023a8abe7be6c2e22f42d523b20bd76cd8da5",
        "external_files": [
            {
                "path": source.path,
                "sha256": source.sha256,
                "row_count": source.row_count,
                "cache_path": str(source.local_path.relative_to(REPOSITORY_ROOT)),
            }
            for source in prepared.source_provenance
        ],
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "target_values_used_in_preflight_reporting": False,
    }


def _factor_conflict_fraction(factors: pd.DataFrame) -> float:
    grouped = factors.groupby(["left_action_index", "right_action_index"])
    two_assay = [group for _, group in grouped if len(group) == 2]
    if not two_assay:
        return 0.0
    conflicts = sum(
        int(group["preference_sign"].nunique() == 2) for group in two_assay
    )
    return float(conflicts / len(two_assay))


def _deterministic_projection(result: GateRunResult) -> dict[str, Any]:
    inference = [
        {key: value for key, value in row.items() if key != "wall_time_seconds"}
        for row in result.inference_rows
    ]
    return {
        "trajectory": result.trajectory_rows,
        "inference": inference,
        "per_seed": result.per_seed_rows,
        "verdict": result.verdict,
        "numerically_reliable": result.numerically_reliable,
    }


def _make_diagnostic(path: Path, result: GateRunResult) -> None:
    rows = result.trajectory_rows
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    colors = {"scalar_only": "#4C78A8", "full_preference": "#F58518"}
    for method in ("scalar_only", "full_preference"):
        method_rows = [row for row in rows if row["method"] == method]
        seeds = sorted({int(row["seed"]) for row in method_rows})
        iterations = sorted({int(row["bo_iteration"]) for row in method_rows})
        if not seeds or not iterations:
            continue
        curves = np.asarray(
            [
                [
                    next(
                        float(row["normalized_simple_regret"])
                        for row in method_rows
                        if int(row["seed"]) == seed
                        and int(row["bo_iteration"]) == iteration
                    )
                    for iteration in iterations
                ]
                for seed in seeds
            ]
        )
        axes[0].plot(
            iterations,
            np.median(curves, axis=0),
            marker="o",
            color=colors[method],
            label=method,
        )
        axes[0].fill_between(
            iterations,
            np.quantile(curves, 0.25, axis=0),
            np.quantile(curves, 0.75, axis=0),
            color=colors[method],
            alpha=0.18,
        )
    axes[0].set_xlabel("Post-initial scalar evaluations")
    axes[0].set_ylabel("Normalized simple regret")
    axes[0].set_title("Median and IQR")
    axes[0].legend(frameon=False)

    paired: dict[int, dict[str, float]] = {}
    for row in result.per_seed_rows:
        paired.setdefault(int(row["seed"]), {})[str(row["method"])] = float(
            row["r_5"]
        )
    for seed, values in paired.items():
        if set(values) == {"scalar_only", "full_preference"}:
            axes[1].plot(
                [0, 1],
                [values["scalar_only"], values["full_preference"]],
                color="#999999",
                alpha=0.6,
            )
            axes[1].scatter(
                [0, 1],
                [values["scalar_only"], values["full_preference"]],
                color=[colors["scalar_only"], colors["full_preference"]],
                s=22,
            )
    axes[1].set_xticks([0, 1], ["scalar_only", "full_preference"])
    axes[1].set_ylabel("$r_5$")
    axes[1].set_title("Paired seed outcomes")
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _results_markdown(mode: str, summary: Mapping[str, Any]) -> str:
    if mode == "smoke":
        return f"""# Gp2 E3 P1 mechanical smoke

Verdict: **{summary['verdict']}**.

This reduced-count run checks mechanics only. The scientific P1 gate was not evaluated.
"""
    gates = summary["gate_summary"]
    return f"""# Gp2 E3 P1 prospective result

Verdict: **{summary['verdict']}**.

- Median scalar-only $r_5$: {gates['median_r_5_scalar_only']:.8g}
- Median full-preference $r_5$: {gates['median_r_5_full_preference']:.8g}
- Relative median-regret improvement: {gates['relative_median_regret_improvement']}

The numerical ESS and split-half checks are empirical reliability diagnostics, not rigorous certificates.
"""


def _write_run_outputs(
    destination: Path,
    *,
    config_path: Path,
    config_hash: str,
    prepared,
    provenance: dict[str, Any],
    result: GateRunResult,
    mode: str,
    deterministic_rerun_match: bool | None,
) -> dict[str, Any]:
    shutil.copyfile(config_path, destination / "frozen_config.json")
    if config_sha256(destination / "frozen_config.json") != config_hash:
        raise RuntimeError("frozen config copy changed bytes")
    _write_json(destination / "provenance.json", provenance)
    _write_json(destination / "preprocessing_summary.json", prepared.preprocessing_summary)
    prepared.factors.to_csv(destination / "preference_bank.csv", index=False)
    _write_csv(destination / "trajectory.csv", result.trajectory_rows)
    _write_csv(destination / "inference_diagnostics.csv", result.inference_rows)
    summary = {
        "profile": mode,
        "verdict": result.verdict,
        "scientific_gate_evaluated": mode == "scientific"
        and result.gate_summary is not None,
        "gate_summary": result.gate_summary,
        "numerically_reliable": result.numerically_reliable,
        "candidate_count": prepared.preprocessing_summary["final_candidate_count"],
        "graph_edge_count": prepared.preprocessing_summary["graph_edge_count"],
        "factor_count_by_assay": prepared.preprocessing_summary["factor_count_by_assay"],
        "total_factor_count": prepared.preprocessing_summary["total_factor_count"],
        "sort1_sort8_conflict_fraction": _factor_conflict_fraction(prepared.factors),
        "deterministic_rerun_match": deterministic_rerun_match,
        "config_sha256": config_hash,
        "git_sha": provenance["git_sha"],
        "per_seed": result.per_seed_rows,
    }
    _write_json(destination / "summary.json", summary)
    (destination / "RESULTS.md").write_text(_results_markdown(mode, summary))
    _make_diagnostic(destination / "diagnostic.png", result)
    return summary


def execute(config_path: Path, mode: str) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = load_gate_config(config_path)
    config_hash = config_sha256(config_path)
    destination = OUTPUT_BY_MODE[mode]
    if destination.exists():
        raise FileExistsError(
            f"output directory already exists and will not be overwritten: {destination}"
        )
    git_sha = _git("rev-parse", "HEAD")
    git_status_before = _git("status", "--short")
    if mode == "scientific" and git_status_before:
        raise RuntimeError(
            "scientific mode requires the exact clean preregistration commit"
        )
    started_at = _utc_now()
    try:
        prepared = prepare_gp2_data(config, REPOSITORY_ROOT)
    except PreprocessingAmbiguity:
        raise
    except PreprocessingInvalid:
        raise
    provenance = _provenance(
        config_path=config_path,
        config_hash=config_hash,
        prepared=prepared,
        mode=mode,
        started_at=started_at,
        git_sha=git_sha,
        git_status_before=git_status_before,
    )

    if mode == "preflight":
        output = create_immutable_output_directory(destination)
        shutil.copyfile(config_path, output / "frozen_config.json")
        _write_json(output / "provenance.json", provenance)
        _write_json(output / "preprocessing_summary.json", prepared.preprocessing_summary)
        verdict = str(prepared.preprocessing_summary["verdict"])
        _write_json(
            output / f"{verdict}.json",
            {
                "verdict": verdict,
                "criteria": prepared.preprocessing_summary["preprocessing_criteria"],
                "config_sha256": config_hash,
            },
        )
        return {
            "mode": mode,
            "verdict": verdict,
            "config_sha256": config_hash,
            **{
                key: prepared.preprocessing_summary[key]
                for key in (
                    "final_candidate_count",
                    "graph_edge_count",
                    "factor_count_by_assay",
                    "total_factor_count",
                )
            },
        }

    if prepared.preprocessing_summary["verdict"] != "PREPROCESSING_VALID":
        raise PreprocessingInvalid(
            "preprocessing is invalid; smoke/scientific execution is prohibited"
        )
    smoke = mode == "smoke"
    result = run_gate(
        prepared,
        config,
        config_hash=config_hash,
        smoke=smoke,
        progress=lambda message: print(message, flush=True),
    )
    deterministic_match: bool | None = None
    if smoke:
        rerun = run_gate(
            prepared,
            config,
            config_hash=config_hash,
            smoke=True,
        )
        deterministic_match = _deterministic_projection(result) == _deterministic_projection(rerun)
        if not deterministic_match:
            raise RuntimeError("the smoke rerun was not deterministic")
    output = create_immutable_output_directory(destination)
    summary = _write_run_outputs(
        output,
        config_path=config_path,
        config_hash=config_hash,
        prepared=prepared,
        provenance=provenance,
        result=result,
        mode=mode,
        deterministic_rerun_match=deterministic_match,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("preflight", "smoke", "scientific"), required=True
    )
    arguments = parser.parse_args()
    summary = execute(arguments.config, arguments.mode)
    print(json.dumps(summary, indent=2, sort_keys=True, default=_json_default))


if __name__ == "__main__":
    main()
