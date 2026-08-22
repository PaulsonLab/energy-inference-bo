"""Run exactly one frozen Gp2 E3 structural preflight."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
import platform
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from conditioned_bo.gp2_proxy_bo import (  # noqa: E402
    PreflightFailure,
    TARGET_COLUMN,
    compute_structural_sparsity,
    graph_edge_table,
    load_structural_preflight_config,
    prepare_structural_preflight,
    sha256_file,
)


EXPERIMENT_DIRECTORY = Path(__file__).resolve().parent
DEFAULT_CONFIG = EXPERIMENT_DIRECTORY / "configs" / "structural_preflight.json"
OUTPUT_DIRECTORY = EXPERIMENT_DIRECTORY / "outputs" / "structural_preflight"
HANDOFF_PATH = "project/archive/e3/E3_GP2_STRUCTURAL_PREFLIGHT_HANDOFF.md"
REQUIRED_OUTPUTS = {
    "frozen_config.json",
    "provenance.json",
    "preprocessing_summary.json",
    "calibration_summary.json",
    "graph_edges.csv",
    "proxy_factor_bank.csv",
    "structural_pairwise_sparsity.csv",
    "structural_sparsity_summary.json",
    "RESULTS.md",
    "structural_sparsity_ecdf.png",
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
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
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


def _provenance(
    *,
    config_path: Path,
    config_hash: str,
    git_sha: str,
    git_status_before: str,
    started_at: str,
    prepared,
) -> dict[str, Any]:
    return {
        "run_id": "gp2_e3_structural_preflight",
        "started_at_utc": started_at,
        "completed_at_utc": _utc_now(),
        "git_sha": git_sha,
        "git_branch": _git("branch", "--show-current"),
        "git_status_before_run": git_status_before,
        "config_path": str(config_path.relative_to(REPOSITORY_ROOT)),
        "config_sha256": config_hash,
        "handoff_path": HANDOFF_PATH,
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
        "source_role_audit": {
            "historical_calibration_source": "datasets/assay_to_yield_training_sequences.csv",
            "prospective_action_source": "datasets/test_sequences.csv",
            "sources_unioned_for_actions": False,
        },
        "target_leakage_audit": {
            "heldout_target_use": "finite_check_and_duplicate_consistency_only",
            "heldout_target_magnitude_available_to_graph": False,
            "heldout_target_magnitude_available_to_factors": False,
            "heldout_target_magnitude_available_to_theory": False,
            "heldout_target_magnitude_available_to_structural_calculation": False,
            "heldout_target_magnitude_saved_printed_ranked_plotted_or_correlated": False,
        },
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
    }


def _make_target_blind_ecdf(path: Path, ratios: np.ndarray) -> None:
    ordered = np.sort(np.asarray(ratios, dtype=float))
    cumulative = np.arange(1, ordered.size + 1, dtype=float) / ordered.size
    figure, axis = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    axis.step(ordered, cumulative, where="post", color="#3366A3", linewidth=2.0)
    axis.axvline(0.50, color="#C44E52", linestyle="--", linewidth=1.4)
    axis.axhline(0.90, color="#55A868", linestyle="--", linewidth=1.4)
    axis.scatter([0.50], [0.90], color="#222222", s=28, zorder=3)
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_xlabel(r"Active proxy-factor fraction $R_{0.05}$")
    axis.set_ylabel("Fraction of unordered action pairs")
    axis.set_title("Gp2 target-blind structural sparsity preflight")
    axis.grid(alpha=0.18)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _interpretation(verdict: str) -> str:
    if verdict == "PASS_STRUCTURAL_PREFLIGHT":
        return (
            "The source-faithful candidate-local Gp2 proxy formulation admits "
            "materially sparse theorem-backed decision influence under the existing "
            "covariance construction. Proceed to a separate numerical/full-proxy BO "
            "smoke gate; no BO claim has yet been established."
        )
    if verdict == "FAIL_STRUCTURAL_SPARSITY":
        return (
            "The source-faithful candidate-local Gp2 proxy formulation is "
            "mathematically compatible with the existing influence construction, but "
            "the real sequence graph does not exhibit the preregistered degree of "
            "structural decision sparsity. Abandon Gp2 as E3 rather than adding new "
            "theory or tuning the case study."
        )
    return "Gp2 is abandoned under the frozen structural-preflight outcome precedence."


def _results_markdown(prepared, result, provenance: Mapping[str, Any]) -> str:
    calibration = prepared.calibration.summary
    graph = prepared.graph_summary
    structural = result.summary
    return f"""# Gp2 E3 structural preflight result

Verdict: **{result.verdict}**.

- Preregistration commit: `{provenance['git_sha']}`
- Config SHA-256: `{provenance['config_sha256']}`
- Historical target-scale count: {calibration['historical_target_scale_count']}
- Calibration count: {calibration['calibration_count']}
- Historical target mean: {calibration['mu_hist']:.17g}
- Historical target sample SD: {calibration['s_hist']:.17g}
- OOF RMSE: {calibration['oof_rmse']:.17g}
- Proxy scale: {calibration['s_proxy']:.17g}
- Final action count: {graph['final_action_count']}
- Final graph edges/components: {graph['final_edge_count']} / {graph['final_component_count']}
- Proxy factors: {structural['factor_count']}
- Proxy-factor coverage: {prepared.preprocessing_summary['factor_coverage_fraction']:.17g}
- Fraction of pairs with R_0.05 <= 0.50: {structural['fraction_pairs_R_0_05_at_most_0_50']:.17g}
- Median R_0.05: {structural['median_R_0_05']:.17g}
- 90th-percentile R_0.05: {structural['percentile_90_R_0_05']:.17g}

{_interpretation(result.verdict)}

This target-blind gate did not run Bayesian optimization or non-Gaussian inference.
"""


def _audit_target_blind_outputs(prepared, pairwise: pd.DataFrame) -> dict[str, Any]:
    frames = {
        "actions": prepared.actions,
        "factors": prepared.factors,
        "pairwise": pairwise,
        "graph_edges": graph_edge_table(prepared.actions, prepared.edges),
    }
    leaking = {
        name: [column for column in frame.columns if column == TARGET_COLUMN]
        for name, frame in frames.items()
        if TARGET_COLUMN in frame.columns
    }
    if leaking:
        raise RuntimeError(f"held-out target column reached target-blind outputs: {leaking}")
    return {
        "passed": True,
        "heldout_target_column_absent_from_actions": True,
        "heldout_target_column_absent_from_graph_edges": True,
        "heldout_target_column_absent_from_proxy_factor_bank": True,
        "heldout_target_column_absent_from_structural_pairs": True,
        "audited_columns": {
            name: list(frame.columns) for name, frame in frames.items()
        },
    }


def _write_success_outputs(
    temporary: Path,
    *,
    config_path: Path,
    config_hash: str,
    prepared,
    result,
    provenance: dict[str, Any],
) -> None:
    shutil.copyfile(config_path, temporary / "frozen_config.json")
    if sha256_file(temporary / "frozen_config.json") != config_hash:
        raise RuntimeError("frozen config copy changed bytes")
    provenance["target_leakage_output_audit"] = _audit_target_blind_outputs(
        prepared, result.pairwise
    )
    _write_json(temporary / "provenance.json", provenance)
    preprocessing = {
        **prepared.preprocessing_summary,
        "graph": prepared.graph_summary,
    }
    _write_json(temporary / "preprocessing_summary.json", preprocessing)
    _write_json(temporary / "calibration_summary.json", prepared.calibration.summary)
    graph_edge_table(prepared.actions, prepared.edges).to_csv(
        temporary / "graph_edges.csv", index=False
    )
    prepared.factors.to_csv(temporary / "proxy_factor_bank.csv", index=False)
    result.pairwise.to_csv(
        temporary / "structural_pairwise_sparsity.csv", index=False
    )
    structural_summary = {
        **result.summary,
        "factor_coverage_fraction": prepared.preprocessing_summary[
            "factor_coverage_fraction"
        ],
        "graph_degree_summary": prepared.graph_summary["degree_summary"],
        "graph_diameter": prepared.graph_summary["diameter"],
        "theory_checks": prepared.theory.summary,
        "config_sha256": config_hash,
        "git_sha": provenance["git_sha"],
    }
    _write_json(
        temporary / "structural_sparsity_summary.json", structural_summary
    )
    (temporary / "RESULTS.md").write_text(
        _results_markdown(prepared, result, provenance)
    )
    _make_target_blind_ecdf(
        temporary / "structural_sparsity_ecdf.png",
        result.pairwise["R_0_05"].to_numpy(dtype=float),
    )
    actual_outputs = {path.name for path in temporary.iterdir() if path.is_file()}
    if actual_outputs != REQUIRED_OUTPUTS:
        raise RuntimeError(
            f"output set mismatch: expected {sorted(REQUIRED_OUTPUTS)}, "
            f"found {sorted(actual_outputs)}"
        )


def _write_failure_outputs(
    temporary: Path,
    *,
    config_path: Path,
    config_hash: str,
    git_sha: str,
    git_status_before: str,
    started_at: str,
    failure: PreflightFailure,
) -> None:
    shutil.copyfile(config_path, temporary / "frozen_config.json")
    payload = {
        "run_id": "gp2_e3_structural_preflight",
        "started_at_utc": started_at,
        "completed_at_utc": _utc_now(),
        "git_sha": git_sha,
        "git_branch": _git("branch", "--show-current"),
        "git_status_before_run": git_status_before,
        "config_sha256": config_hash,
        "verdict": failure.verdict,
        "reason": failure.reason,
        "details": failure.details,
        "target_leakage_audit": {
            "heldout_target_magnitudes_saved_printed_ranked_plotted_or_correlated": False
        },
    }
    _write_json(temporary / "provenance.json", payload)
    _write_json(
        temporary / "preprocessing_summary.json",
        {
            "verdict": failure.verdict,
            "reason": failure.reason,
            "details": failure.details,
        },
    )
    (temporary / "RESULTS.md").write_text(
        f"# Gp2 E3 structural preflight result\n\n"
        f"Verdict: **{failure.verdict}**.\n\n"
        f"Mechanical reason: {failure.reason}\n\n"
        f"{_interpretation(failure.verdict)}\n"
    )


def execute(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config_path = config_path.resolve()
    if OUTPUT_DIRECTORY.exists():
        raise FileExistsError(
            f"immutable output directory already exists: {OUTPUT_DIRECTORY}"
        )
    config = load_structural_preflight_config(config_path)
    config_hash = sha256_file(config_path)
    git_sha = _git("rev-parse", "HEAD")
    branch = _git("branch", "--show-current")
    git_status_before = _git("status", "--short")
    if branch != "main":
        raise RuntimeError("the real structural preflight must run from branch main")
    if git_status_before:
        raise RuntimeError(
            "the real structural preflight requires a clean preregistration commit"
        )
    started_at = _utc_now()
    OUTPUT_DIRECTORY.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=".structural_preflight_tmp_", dir=OUTPUT_DIRECTORY.parent
        )
    )
    try:
        try:
            print("Preparing frozen historical calibration and test-only actions...", flush=True)
            prepared = prepare_structural_preflight(config, REPOSITORY_ROOT)
            print("Computing all unordered action-pair structural tails...", flush=True)
            structural = config["structural_gate"]
            result = compute_structural_sparsity(
                prepared.theory.k0,
                prepared.factors["action_index"].to_numpy(dtype=np.int64),
                prepared.calibration.s_proxy,
                prepared.theory.graph_distances,
                epsilon_struct=float(structural["epsilon_struct"]),
                maximum_active_factor_fraction=float(
                    structural["maximum_active_factor_fraction"]
                ),
                minimum_passing_pair_fraction=float(
                    structural["minimum_passing_pair_fraction"]
                ),
            )
            provenance = _provenance(
                config_path=config_path,
                config_hash=config_hash,
                git_sha=git_sha,
                git_status_before=git_status_before,
                started_at=started_at,
                prepared=prepared,
            )
            _write_success_outputs(
                temporary,
                config_path=config_path,
                config_hash=config_hash,
                prepared=prepared,
                result=result,
                provenance=provenance,
            )
            summary = {
                "verdict": result.verdict,
                "git_sha": git_sha,
                "config_sha256": config_hash,
                "historical_target_scale_count": prepared.calibration.historical_target_scale_count,
                "calibration_count": prepared.calibration.calibration_count,
                "mu_hist": prepared.calibration.mu_hist,
                "s_hist": prepared.calibration.s_hist,
                "oof_rmse": prepared.calibration.oof_rmse,
                "s_proxy": prepared.calibration.s_proxy,
                "final_action_count": int(len(prepared.actions)),
                "graph_edge_count": int(len(prepared.edges)),
                "graph_component_count": prepared.graph_summary["final_component_count"],
                "proxy_factor_count": int(len(prepared.factors)),
                "factor_coverage_fraction": prepared.preprocessing_summary[
                    "factor_coverage_fraction"
                ],
                "fraction_pairs_R_0_05_at_most_0_50": result.summary[
                    "fraction_pairs_R_0_05_at_most_0_50"
                ],
                "median_R_0_05": result.summary["median_R_0_05"],
                "percentile_90_R_0_05": result.summary[
                    "percentile_90_R_0_05"
                ],
            }
        except PreflightFailure as failure:
            _write_failure_outputs(
                temporary,
                config_path=config_path,
                config_hash=config_hash,
                git_sha=git_sha,
                git_status_before=git_status_before,
                started_at=started_at,
                failure=failure,
            )
            summary = {
                "verdict": failure.verdict,
                "reason": failure.reason,
                "git_sha": git_sha,
                "config_sha256": config_hash,
            }
        os.replace(temporary, OUTPUT_DIRECTORY)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    arguments = parser.parse_args()
    summary = execute(arguments.config)
    print(json.dumps(summary, indent=2, sort_keys=True, default=_json_default))


if __name__ == "__main__":
    main()
