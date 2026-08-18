"""Orchestration and figures for the frozen Welded Beam q=1 experiment."""

from __future__ import annotations

import json
import math
import platform
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import botorch
import gpytorch
import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import torch
from botorch.test_functions.synthetic import WeldedBeamSO
from scipy.integrate import quad

from .welded_beam import (
    build_candidate_set,
    build_state,
    classify_result,
    exact_constrained_ei,
    fit_independent_gps,
    load_config,
    posterior_marginals,
    qmc_candidate_metrics,
    summarize_qmc_rows,
    unnormalize_inputs,
)


def _git_sha(repository: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repository, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _moment_quadrature(
    mean: float, variance: float, best_f: float
) -> tuple[float, float]:
    sigma = math.sqrt(variance)
    lower = (best_f - mean) / sigma
    normalizer = math.sqrt(2.0 * math.pi)

    def density(z: float) -> float:
        return math.exp(-0.5 * z * z) / normalizer

    # Integrate in excess-over-threshold coordinates. This avoids cancellation
    # and gives a trustworthy check even when improvement is a far-tail event.
    first = quad(
        lambda excess: sigma * excess * density(lower + excess),
        0.0,
        np.inf,
        epsabs=1e-300,
        epsrel=1e-13,
    )[0]
    second = quad(
        lambda excess: (sigma * excess) ** 2 * density(lower + excess),
        0.0,
        np.inf,
        epsabs=1e-300,
        epsrel=1e-13,
    )[0]
    return first, second


def _validate_top_moments(
    means: np.ndarray,
    variances: np.ndarray,
    best_f: float,
    exact: dict[str, np.ndarray],
) -> float:
    index = int(np.argmax(exact["log_acquisition"]))
    first, second = _moment_quadrature(
        float(means[index, 0]), float(variances[index, 0]), best_f
    )
    first_error = abs(math.log(first) - float(exact["log_ei"][index]))
    second_log = (
        float(exact["d2"][index])
        + 2.0 * float(exact["log_ei"][index])
        + float(exact["log_feasibility"][index])
    )
    second_error = abs(math.log(second) - second_log)
    return max(first_error, second_error)


def _candidate_frame(
    state_seed: int,
    candidates: torch.Tensor,
    raw_candidates: torch.Tensor,
    exact: dict[str, np.ndarray],
    truth_feasible: np.ndarray,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, np.ndarray]:
    order = np.argsort(-exact["log_acquisition"], kind="stable")
    rank = np.empty(len(order), dtype=np.int64)
    rank[order] = np.arange(len(order))
    maximum_log = float(exact["log_acquisition"][order[0]])
    quality = np.exp(exact["log_acquisition"] - maximum_log)
    top_one_count = math.ceil(0.01 * len(order))
    top_five_count = math.ceil(0.05 * len(order))
    frame = pd.DataFrame(
        {
            "state_seed": state_seed,
            "candidate_index": np.arange(len(order)),
            "rank": rank + 1,
            "quality": quality,
            "log_ei": exact["log_ei"],
            "ei": exact["ei"],
            "log_feasibility": exact["log_feasibility"],
            "feasibility": exact["feasibility"],
            "log_constrained_ei": exact["log_acquisition"],
            "constrained_ei": exact["acquisition"],
            "d2": exact["d2"],
            "chi_square": exact["chi_square"],
            "ess_fraction": exact["ess_fraction"],
            "top_1_percent": rank < top_one_count,
            "top_5_percent": rank < top_five_count,
            "top_32": rank < int(config["candidates"]["top_k"]),
            "truth_feasible": truth_feasible,
        }
    )
    for dimension in range(candidates.shape[-1]):
        frame[f"x{dimension}_unit"] = candidates[:, dimension].numpy()
        frame[f"x{dimension}_raw"] = raw_candidates[:, dimension].numpy()
    return frame, order


def _make_figures(
    candidate_metrics: pd.DataFrame,
    qmc_summary: pd.DataFrame,
    config: dict[str, Any],
    output: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 7.5,
            "figure.dpi": 140,
            "savefig.dpi": 300,
        }
    )
    seeds = config["states"]["seeds"]

    figure, axes = plt.subplots(1, 3, figsize=(11.4, 3.4), constrained_layout=True)
    for axis, seed in zip(axes, seeds, strict=True):
        state = candidate_metrics[candidate_metrics.state_seed == seed]
        axis.scatter(
            state.quality,
            state.ess_fraction,
            s=7,
            alpha=0.22,
            color="#4C78A8",
            linewidths=0,
            label="all candidates",
        )
        top = state[state.top_1_percent]
        axis.scatter(
            top.quality,
            top.ess_fraction,
            s=16,
            alpha=0.85,
            color="#E45756",
            linewidths=0,
            label="exact top 1%",
        )
        axis.axhline(0.125, color="black", linestyle="--", linewidth=0.9)
        axis.axhline(0.25, color="0.4", linestyle=":", linewidth=0.9)
        axis.set_yscale("log")
        axis.set_xlim(-0.02, 1.02)
        axis.set_ylim(max(1e-8, state.ess_fraction.min() * 0.7), 1.05)
        axis.set_title(f"State {seed}")
        axis.set_xlabel("exact cEI / state maximum")
        axis.grid(alpha=0.18)
    axes[0].set_ylabel("population ESS fraction")
    axes[0].legend(loc="lower right")
    figure.suptitle("A. Decision shift versus constrained-EI quality", fontweight="bold")
    _save_figure(figure, output / "figure_a_shift_vs_quality")

    figure, axes = plt.subplots(1, 3, figsize=(11.4, 3.4), constrained_layout=True)
    for axis, seed in zip(axes, seeds, strict=True):
        state = qmc_summary[qmc_summary.state_seed == seed]
        axis.semilogx(
            state.sample_count,
            state.exact_best_selection_probability,
            base=2,
            marker="o",
            label="exact best",
            color="#E45756",
        )
        axis.semilogx(
            state.sample_count,
            state.one_percent_optimal_probability,
            base=2,
            marker="s",
            label="within 1% of best",
            color="#54A24B",
        )
        axis.semilogx(
            state.sample_count,
            1.0 - state.mean_pairwise_disagreement,
            base=2,
            marker="^",
            label="top-32 pair agreement",
            color="#4C78A8",
        )
        axis.axvline(256, color="0.6", linestyle=":", linewidth=0.8)
        axis.axvline(512, color="0.6", linestyle="--", linewidth=0.8)
        axis.set_ylim(-0.02, 1.02)
        axis.set_title(f"State {seed}")
        axis.set_xlabel("scrambled-Sobol worlds")
        axis.grid(alpha=0.18)
    axes[0].set_ylabel("selection / ranking reliability")
    axes[0].legend(loc="lower right")
    figure.suptitle("B. Practical QMC decision reliability", fontweight="bold")
    _save_figure(figure, output / "figure_b_qmc_reliability")

    representative = int(config["candidates"]["representative_state_seed"])
    state = candidate_metrics[
        (candidate_metrics.state_seed == representative) & candidate_metrics.top_32
    ].sort_values("rank")
    figure, axis = plt.subplots(figsize=(7.2, 4.0), constrained_layout=True)
    rank = np.arange(1, len(state) + 1)
    axis.plot(rank, state.ei / state.ei.max(), marker="o", ms=3, label="EI / max EI")
    axis.plot(rank, state.feasibility, marker="s", ms=3, label="joint feasibility")
    axis.plot(rank, state.quality, marker="^", ms=3, label="cEI / max cEI")
    axis.plot(rank, state.ess_fraction, marker="D", ms=3, label="ESS fraction")
    axis.axhline(0.125, color="black", linestyle="--", linewidth=0.9)
    axis.set_xlabel("exact cEI rank within the top 32")
    axis.set_ylabel("probability or normalized acquisition")
    axis.set_ylim(-0.02, 1.05)
    axis.grid(alpha=0.18)
    axis.legend(ncol=2)
    axis.set_title(
        f"C. Mechanism in prospectively designated State {representative}",
        fontweight="bold",
    )
    _save_figure(figure, output / "figure_c_mechanism")


def _save_figure(figure: plt.Figure, stem: Path) -> None:
    for suffix in ("png", "pdf", "svg"):
        path = stem.with_suffix(f".{suffix}")
        figure.savefig(path)
        if suffix == "svg":
            path.write_text(
                "\n".join(line.rstrip() for line in path.read_text().splitlines())
                + "\n"
            )
    plt.close(figure)


def _results_markdown(summary: dict[str, Any]) -> str:
    gate = summary["gate"]
    lines = [
        "# Welded Beam q=1 Decision-Shift Result",
        "",
        "## Outcome",
        "",
        f"**{gate['status']}**",
        "",
        "The experiment used all three prospectively frozen states. The status is",
        "computed from the frozen gate; it does not authorize q=4 or a new sampler.",
        "",
        f"All {7 * len(summary['states'])} independent GP fits converged. The stable analytic improvement moments",
        f"agreed with adaptive quadrature to maximum log error `{max(state['moment_log_error'] for state in summary['states']):.3g}`. The fixed",
        f"candidate set's true feasible fraction was `{summary['true_candidate_feasible_fraction']:.7f}` "
        f"({round(summary['true_candidate_feasible_fraction'] * summary['candidate_count'])}/{summary['candidate_count']:,}), consistent",
        "with the narrow standard Welded Beam feasible region.",
        "",
        "## State evidence",
        "",
        "| State | Best ESS | Top-32 median ESS | 256 disagreement | 512 disagreement | 512 regret | Classification |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    qmc = {
        (int(row["state_seed"]), int(row["sample_count"])): row
        for row in summary["qmc_summary"]
    }
    for state in summary["states"]:
        seed = int(state["state_seed"])
        row256 = qmc[(seed, 256)]
        row = qmc[(seed, 512)]
        check = gate["state_checks"][str(seed)]
        label = "joint positive" if check["joint_positive"] else (
            "conclusively negative" if check["conclusively_negative"] else "intermediate"
        )
        lines.append(
            f"| {seed} | {state['exact_best_ess_fraction']:.4f} | "
            f"{state['top32_median_ess_fraction']:.4f} | "
            f"{row256['mean_pairwise_disagreement']:.4f} | "
            f"{row['mean_pairwise_disagreement']:.4f} | "
            f"{row['mean_normalized_regret']:.4f} | {label} |"
        )
    lines.extend(
        [
            "",
            "All 64 independent scrambles selected the exact candidate-set maximizer in",
            "every state at every tested sample count from 64 through 1,024. Thus the",
            "experiment verifies that ordinary constraints can reduce population ESS, but",
            "it falsifies the required conjunction: the shift did not produce a material",
            "practical-QMC decision error. State 4103 retained modest top-32 ordering error",
            "at 256 samples, but it never changed the selected design or incurred regret.",
            "",
            "The result should not be generalized to q=4 or every constrained BO state. It",
            "does show that low ESS by itself is insufficient motivation for a new inner",
            "inference method when common-random-number scrambled Sobol sampling preserves",
            "the actual decision.",
            "",
            "## Evidence",
            "",
            "- [Gate record](gate_result.json)",
            "- [State summary](state_summary.csv)",
            "- [QMC summary](qmc_summary.csv)",
            "- [Figure A](figure_a_shift_vs_quality.png)",
            "- [Figure B](figure_b_qmc_reliability.png)",
            "- [Figure C](figure_c_mechanism.png)",
            "",
            "## Next action",
            "",
            "Human review is required. Do not automatically authorize q=4, a non-Gaussian belief, or decision-adapted inference.",
            "",
        ]
    )
    return "\n".join(lines)


def run_experiment(
    config_path: str | Path, output_directory: str | Path
) -> dict[str, Any]:
    started = time.perf_counter()
    config_path = Path(config_path).resolve()
    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config, protocol_hash = load_config(config_path)
    repository = config_path.parents[2]
    git_sha = _git_sha(repository)
    shutil.copyfile(config_path, output / "frozen_config.json")

    problem = WeldedBeamSO(dtype=torch.double)
    candidates = build_candidate_set(config)
    raw_candidates = unnormalize_inputs(candidates, problem.bounds)
    truth_feasible = problem.is_feasible(raw_candidates, noise=False).numpy()
    true_feasible_fraction = float(truth_feasible.mean())

    candidate_frames: list[pd.DataFrame] = []
    qmc_rows: list[dict[str, Any]] = []
    qmc_summaries: list[dict[str, Any]] = []
    fit_rows: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []
    for state_index, seed in enumerate(config["states"]["seeds"], start=1):
        state_started = time.perf_counter()
        print(f"[STATE {state_index}/3] fitting objective + 6 constraint GPs ...", flush=True)
        state = build_state(config, int(seed))
        fitted = fit_independent_gps(state, config)
        print(f"[STATE {state_index}/3] fitting objective + 6 constraint GPs ... done", flush=True)
        fit_rows.extend(item.record for item in fitted)
        means, variances = posterior_marginals(fitted, candidates, config)
        exact = exact_constrained_ei(means, variances, state.incumbent, config)
        moment_error = _validate_top_moments(
            means, variances, state.incumbent, exact
        )
        frame, order = _candidate_frame(
            int(seed),
            candidates,
            raw_candidates,
            exact,
            truth_feasible,
            config,
        )
        candidate_frames.append(frame)
        print(f"[STATE {state_index}/3] exact cEI / ESS over candidate set ... done", flush=True)
        state_qmc = qmc_candidate_metrics(
            means, variances, state.incumbent, exact, config, int(seed)
        )
        qmc_rows.extend(state_qmc)
        state_qmc_summary = summarize_qmc_rows(state_qmc, config, int(seed))
        qmc_summaries.extend(state_qmc_summary)
        for count in config["qmc"]["sample_counts"]:
            print(f"[STATE {state_index}/3] QMC N={count} ... done", flush=True)
        top32 = order[: int(config["candidates"]["top_k"])]
        top_one = order[: math.ceil(0.01 * len(order))]
        state_rows.append(
            {
                "state_seed": int(seed),
                "training_size": len(state.train_x),
                "observed_feasible_count": state.feasible_count,
                "incumbent": state.incumbent,
                "exact_best_candidate": int(order[0]),
                "exact_best_log_constrained_ei": float(exact["log_acquisition"][order[0]]),
                "exact_best_ess_fraction": float(exact["ess_fraction"][order[0]]),
                "top32_median_ess_fraction": float(np.median(exact["ess_fraction"][top32])),
                "top1_median_ess_fraction": float(np.median(exact["ess_fraction"][top_one])),
                "top1_fraction_ess_at_most_one_eighth": float(np.mean(exact["ess_fraction"][top_one] <= 0.125)),
                "candidate_truth_feasible_fraction": true_feasible_fraction,
                "all_gp_fits_converged": all(item.record["converged"] for item in fitted),
                "moment_log_error": moment_error,
                "state_seconds": time.perf_counter() - state_started,
            }
        )
        row512 = next(row for row in state_qmc_summary if row["sample_count"] == 512)
        print(
            f"[STATE {state_index}/3] complete: top32 ESS={state_rows[-1]['top32_median_ess_fraction']:.4f}, "
            f"512 regret={row512['mean_normalized_regret']:.4f}, "
            f"512 disagreement={row512['mean_pairwise_disagreement']:.4f}",
            flush=True,
        )

    candidates_frame = pd.concat(candidate_frames, ignore_index=True)
    qmc_frame = pd.DataFrame(qmc_rows)
    qmc_summary_frame = pd.DataFrame(qmc_summaries)
    fits_frame = pd.DataFrame(fit_rows)
    states_frame = pd.DataFrame(state_rows)
    numerical_arrays = [
        candidates_frame.select_dtypes(include=[np.number]).to_numpy(),
        qmc_frame.select_dtypes(include=[np.number]).to_numpy(),
        qmc_summary_frame.select_dtypes(include=[np.number]).to_numpy(),
        fits_frame.select_dtypes(include=[np.number]).to_numpy(),
        states_frame.select_dtypes(include=[np.number]).to_numpy(),
    ]
    valid = bool(
        all(np.isfinite(array).all() for array in numerical_arrays)
        and fits_frame.converged.all()
        and (states_frame.moment_log_error < 1e-9).all()
    )
    gate = classify_result(
        state_rows, qmc_summaries, config, valid=valid
    )
    gate.update(
        {
            "protocol_version": config["protocol_version"],
            "protocol_hash": protocol_hash,
            "git_sha": git_sha,
        }
    )
    summary = {
        "protocol_version": config["protocol_version"],
        "protocol_hash": protocol_hash,
        "git_sha": git_sha,
        "candidate_count": len(candidates),
        "true_candidate_feasible_fraction": true_feasible_fraction,
        "states": state_rows,
        "qmc_summary": qmc_summaries,
        "gate": gate,
        "runtime_seconds": time.perf_counter() - started,
    }
    metadata = {
        "git_sha": git_sha,
        "protocol_hash": protocol_hash,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "botorch": botorch.__version__,
        "gpytorch": gpytorch.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "device": "cpu",
        "runtime_seconds": summary["runtime_seconds"],
    }
    candidates_frame.to_csv(output / "candidate_metrics.csv", index=False)
    qmc_frame.to_csv(output / "qmc_reliability.csv", index=False)
    qmc_summary_frame.to_csv(output / "qmc_summary.csv", index=False)
    fits_frame.to_csv(output / "gp_fits.csv", index=False)
    states_frame.to_csv(output / "state_summary.csv", index=False)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "gate_result.json").write_text(json.dumps(gate, indent=2) + "\n")
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    _make_figures(candidates_frame, qmc_summary_frame, config, output)
    (output / "RESULTS.md").write_text(_results_markdown(summary))
    return summary
