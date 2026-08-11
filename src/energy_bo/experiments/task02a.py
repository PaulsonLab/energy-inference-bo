"""Task 02A SAAS structural-posterior reuse falsification experiment."""

from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import pearsonr

from energy_bo.structural.acquisition import weighted_expected_improvement
from energy_bo.structural.diagnostics import (
    ei_comparison,
    lengthscale_summary,
    log_lengthscale_wasserstein,
    prediction_comparison,
    standardized_rbf_mmd,
    weighted_quantile,
)
from energy_bo.structural.exact_gp import ExactGPBatchState, full_log_marginal_likelihood
from energy_bo.structural.particles import ParticleWeights, SaasParticles
from energy_bo.structural.preprocessing import (
    FrozenOutputTransform,
    deterministic_benchmark_inputs,
    negative_branin,
)
from energy_bo.structural.saas_reference import NutsConfig, fit_saas_reference


@dataclass(frozen=True)
class Task02AConfig:
    profile: str
    seeds: tuple[int, ...]
    dimension: int
    initial_count: int
    final_count: int
    test_count: int
    candidate_count: int
    prediction_chunk_size: int
    reference_checkpoints: tuple[int, ...]
    nuts_warmup: int
    nuts_samples: int
    nuts_thinning: int
    nuts_tree_depth: int
    noise_variance: float = 1e-4

    @classmethod
    def smoke(cls, seeds: tuple[int, ...] = (0,)) -> "Task02AConfig":
        return cls(
            profile="smoke",
            seeds=seeds,
            dimension=4,
            initial_count=8,
            final_count=11,
            test_count=64,
            candidate_count=256,
            prediction_chunk_size=256,
            reference_checkpoints=(8, 11),
            nuts_warmup=32,
            nuts_samples=32,
            nuts_thinning=1,
            nuts_tree_depth=4,
        )

    @classmethod
    def full(cls, seeds: tuple[int, ...] = (0, 1, 2)) -> "Task02AConfig":
        return cls(
            profile="full",
            seeds=seeds,
            dimension=10,
            initial_count=16,
            final_count=40,
            test_count=512,
            candidate_count=2048,
            prediction_chunk_size=256,
            reference_checkpoints=(16, 20, 24, 32, 40),
            nuts_warmup=512,
            nuts_samples=512,
            nuts_thinning=2,
            nuts_tree_depth=6,
        )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def _top_two_active_probability(
    particles: SaasParticles, weights: ParticleWeights
) -> float:
    top_two = torch.topk(particles.lengthscales.reciprocal(), k=2, dim=1).indices
    correct = (torch.sort(top_two, dim=1).values == torch.tensor([0, 1])).all(dim=1)
    return float(weights.probabilities[correct].sum())


def _fresh_reference(
    x: torch.Tensor,
    y: torch.Tensor,
    config: Task02AConfig,
    seed: int,
    count: int,
) -> tuple[SaasParticles, float, dict[str, Any]]:
    nuts = NutsConfig(
        warmup_steps=config.nuts_warmup,
        num_samples=config.nuts_samples,
        thinning=config.nuts_thinning,
        max_tree_depth=config.nuts_tree_depth,
        seed=10_000 * seed + count,
    )
    fit = fit_saas_reference(x[:count], y[:count], config.noise_variance, nuts)
    return fit.particles, fit.elapsed_seconds, fit.environment


def _checkpoint_metrics(
    *,
    seed: int,
    count: int,
    reused_state: ExactGPBatchState,
    reused_weights: ParticleWeights,
    fresh_particles: SaasParticles,
    fresh_nuts_seconds: float,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    candidates: torch.Tensor,
    chunk_size: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    fresh_weights = ParticleWeights.uniform(fresh_particles.num_particles)
    fresh_state = ExactGPBatchState.build(
        fresh_particles,
        train_x[:count],
        train_y[:count],
        reused_state.noise_variance,
    )
    wasserstein = log_lengthscale_wasserstein(
        reused_state.particles, reused_weights, fresh_particles
    )
    prediction = prediction_comparison(
        reused_state,
        reused_weights,
        fresh_state,
        fresh_weights,
        test_x,
        test_y,
        chunk_size=chunk_size,
    )
    best_f = float(train_y[:count].max())
    reused_ei = weighted_expected_improvement(
        reused_state, reused_weights, candidates, best_f, chunk_size=chunk_size
    )
    fresh_ei = weighted_expected_improvement(
        fresh_state, fresh_weights, candidates, best_f, chunk_size=chunk_size
    )
    ei_metrics = ei_comparison(reused_ei, fresh_ei, candidates)
    row: dict[str, Any] = {
        "seed": seed,
        "count": count,
        "ess_fraction": reused_weights.ess_fraction,
        "negative_log_ess_fraction": -math.log(reused_weights.ess_fraction),
        "mmd": standardized_rbf_mmd(
            reused_state.particles, reused_weights, fresh_particles
        ),
        "mean_log_lengthscale_w1": float(np.mean(wasserstein)),
        "max_log_lengthscale_w1": float(np.max(wasserstein)),
        "active_log_lengthscale_w1": float(np.mean(wasserstein[:2])),
        "inactive_log_lengthscale_w1": float(np.mean(wasserstein[2:])),
        "reused_top2_active_probability": _top_two_active_probability(
            reused_state.particles, reused_weights
        ),
        "fresh_top2_active_probability": _top_two_active_probability(
            fresh_particles, fresh_weights
        ),
        "fresh_nuts_seconds": fresh_nuts_seconds,
        **prediction,
        **ei_metrics,
    }
    lengthscale_rows: list[dict[str, Any]] = []
    for label, particles, weights in (
        ("reused", reused_state.particles, reused_weights),
        ("fresh", fresh_particles, fresh_weights),
    ):
        for summary in lengthscale_summary(particles, weights):
            lengthscale_rows.append(
                {"seed": seed, "count": count, "posterior": label, **summary}
            )
    ei_rows = [
        {
            "seed": seed,
            "count": count,
            "candidate_index": index,
            "reused_ei": float(reused_ei[index]),
            "fresh_ei": float(fresh_ei[index]),
        }
        for index in range(candidates.shape[0])
    ]
    return row, lengthscale_rows, ei_rows


def _coordinate_drift_rows(
    seed: int,
    fresh_references: dict[int, SaasParticles],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    checkpoints = sorted(fresh_references)
    for previous, current in zip(checkpoints[:-1], checkpoints[1:], strict=True):
        before = fresh_references[previous]
        after = fresh_references[current]
        before_weights = ParticleWeights.uniform(before.num_particles)
        after_weights = ParticleWeights.uniform(after.num_particles)
        distances = log_lengthscale_wasserstein(before, before_weights, after)
        for dimension, distance in enumerate(distances):
            before_median = weighted_quantile(
                -before.lengthscales[:, dimension].log(), before_weights.probabilities, 0.5
            )
            after_median = weighted_quantile(
                -after.lengthscales[:, dimension].log(), after_weights.probabilities, 0.5
            )
            rows.append(
                {
                    "seed": seed,
                    "previous_count": previous,
                    "current_count": current,
                    "dimension": dimension,
                    "active": dimension < 2,
                    "absolute_median_log_inverse_lengthscale_change": abs(
                        after_median - before_median
                    ),
                    "log_lengthscale_w1": distance,
                }
            )
    return rows


def _run_seed(config: Task02AConfig, seed: int) -> dict[str, Any]:
    train_x = deterministic_benchmark_inputs(config.final_count, config.dimension, seed)
    raw_y = negative_branin(train_x)
    transform = FrozenOutputTransform.fit(raw_y[: config.initial_count])
    train_y = transform.transform(raw_y)
    test_x = deterministic_benchmark_inputs(config.test_count, config.dimension, seed + 1_000)
    test_y = transform.transform(negative_branin(test_x))
    candidates = deterministic_benchmark_inputs(
        config.candidate_count, config.dimension, seed + 2_000
    )

    initial_particles, initial_nuts_seconds, environment = _fresh_reference(
        train_x, train_y, config, seed, config.initial_count
    )
    weights = ParticleWeights.uniform(initial_particles.num_particles)
    state = ExactGPBatchState.build(
        initial_particles,
        train_x[: config.initial_count],
        train_y[: config.initial_count],
        config.noise_variance,
    )
    thresholds = (0.75, 0.50, 0.25, 0.10)
    crossings: dict[str, int | None] = {str(value): None for value in thresholds}
    fresh_references = {config.initial_count: initial_particles}
    nuts_times = {config.initial_count: initial_nuts_seconds}
    round_rows: list[dict[str, Any]] = [
        {
            "seed": seed,
            "count": config.initial_count,
            "added_observations": 0,
            "ess": weights.ess,
            "ess_fraction": weights.ess_fraction,
            "negative_log_ess_fraction": 0.0,
            "conditional_ess_fraction": 1.0,
            "incremental_log_likelihood_variance": 0.0,
            "sequential_update_seconds": 0.0,
            "marginal_increment_max_abs_error": 0.0,
            "cache_chol_max_abs_error": 0.0,
            "cache_mean_max_abs_error": 0.0,
            "cache_variance_max_abs_error": 0.0,
        }
    ]
    crossing_checkpoint: int | None = None
    for count in range(config.initial_count + 1, config.final_count + 1):
        old_lml = full_log_marginal_likelihood(
            initial_particles, state.train_x, state.train_y, config.noise_variance
        )
        update_start = time.perf_counter()
        increment = state.predictive_log_likelihood(train_x[count - 1], train_y[count - 1])
        conditional_ess = weights.conditional_ess_fraction(increment)
        increment_variance = weights.weighted_variance(increment)
        weights = weights.update(increment)
        state.append(train_x[count - 1], train_y[count - 1])
        sequential_seconds = time.perf_counter() - update_start
        new_lml = full_log_marginal_likelihood(
            initial_particles, state.train_x, state.train_y, config.noise_variance
        )
        state.counters.validation_factorizations += 2 * initial_particles.num_particles
        increment_error = float((new_lml - old_lml - increment).abs().max())
        cache_check = state.validate_against_full(test_x[: min(8, config.test_count)])
        for threshold in thresholds:
            key = str(threshold)
            if crossings[key] is None and weights.ess_fraction < threshold:
                crossings[key] = count - config.initial_count
        if (
            config.profile == "full"
            and crossing_checkpoint is None
            and weights.ess_fraction < 0.5
            and count not in config.reference_checkpoints
        ):
            crossing_checkpoint = count
        round_rows.append(
            {
                "seed": seed,
                "count": count,
                "added_observations": count - config.initial_count,
                "ess": weights.ess,
                "ess_fraction": weights.ess_fraction,
                "negative_log_ess_fraction": -math.log(weights.ess_fraction),
                "conditional_ess_fraction": conditional_ess,
                "incremental_log_likelihood_variance": increment_variance,
                "sequential_update_seconds": sequential_seconds,
                "marginal_increment_max_abs_error": increment_error,
                "cache_chol_max_abs_error": cache_check["chol_max_abs"],
                "cache_mean_max_abs_error": cache_check["mean_max_abs"],
                "cache_variance_max_abs_error": cache_check["variance_max_abs"],
                "validation_seconds": cache_check["validation_seconds"],
                **state.counters.to_dict(),
            }
        )
        requested = count in config.reference_checkpoints or count == crossing_checkpoint
        if requested and count not in fresh_references:
            particles, elapsed, _ = _fresh_reference(train_x, train_y, config, seed, count)
            fresh_references[count] = particles
            nuts_times[count] = elapsed

    checkpoint_rows: list[dict[str, Any]] = []
    lengthscale_rows: list[dict[str, Any]] = []
    ei_rows: list[dict[str, Any]] = []
    for count in sorted(fresh_references):
        if count == config.final_count:
            checkpoint_state = state
            checkpoint_weights = weights
        elif count == config.initial_count:
            checkpoint_state = ExactGPBatchState.build(
                initial_particles,
                train_x[:count],
                train_y[:count],
                config.noise_variance,
            )
            checkpoint_weights = ParticleWeights.uniform(initial_particles.num_particles)
        else:
            checkpoint_state = ExactGPBatchState.build(
                initial_particles,
                train_x[: config.initial_count],
                train_y[: config.initial_count],
                config.noise_variance,
            )
            checkpoint_weights = ParticleWeights.uniform(initial_particles.num_particles)
            for index in range(config.initial_count, count):
                update = checkpoint_state.predictive_log_likelihood(train_x[index], train_y[index])
                checkpoint_weights = checkpoint_weights.update(update)
                checkpoint_state.append(train_x[index], train_y[index])
        row, lengthscales, eis = _checkpoint_metrics(
            seed=seed,
            count=count,
            reused_state=checkpoint_state,
            reused_weights=checkpoint_weights,
            fresh_particles=fresh_references[count],
            fresh_nuts_seconds=nuts_times[count],
            train_x=train_x,
            train_y=train_y,
            test_x=test_x,
            test_y=test_y,
            candidates=candidates,
            chunk_size=config.prediction_chunk_size,
        )
        row["added_observations"] = count - config.initial_count
        checkpoint_rows.append(row)
        lengthscale_rows.extend(lengthscales)
        ei_rows.extend(eis)

    return {
        "seed": seed,
        "transform": asdict(transform),
        "environment": environment,
        "crossings": crossings,
        "round_rows": round_rows,
        "checkpoint_rows": checkpoint_rows,
        "lengthscale_rows": lengthscale_rows,
        "ei_rows": ei_rows,
        "drift_rows": _coordinate_drift_rows(seed, fresh_references),
        "cache_counters": state.counters.to_dict(),
        "nuts_times": nuts_times,
    }


def _plots(
    output_dir: Path,
    rounds: list[dict[str, Any]],
    checkpoints: list[dict[str, Any]],
    lengthscales: list[dict[str, Any]],
    drift: list[dict[str, Any]],
    ei_rows: list[dict[str, Any]],
) -> None:
    figure, axis = plt.subplots(figsize=(6.4, 4.0))
    for seed in sorted({int(row["seed"]) for row in rounds}):
        selected = [row for row in rounds if int(row["seed"]) == seed]
        axis.plot(
            [row["added_observations"] for row in selected],
            [row["ess_fraction"] for row in selected],
            marker="o",
            label=f"seed {seed}",
        )
    for threshold in (0.75, 0.5, 0.25, 0.1):
        axis.axhline(threshold, color="0.75", linewidth=0.8, linestyle="--")
    axis.set(xlabel="new observations", ylabel="ESS / P", ylim=(0, 1.03))
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "ess_reuse_horizon.png", dpi=160)
    plt.close(figure)

    final_count = max(int(row["count"]) for row in lengthscales)
    selected = [row for row in lengthscales if int(row["count"]) == final_count]
    figure, axis = plt.subplots(figsize=(7.0, 4.0))
    for posterior, offset, color in (("reused", -0.08, "C0"), ("fresh", 0.08, "C1")):
        rows = [row for row in selected if row["posterior"] == posterior]
        dimensions = np.array([int(row["dimension"]) for row in rows]) + offset
        medians = np.array([float(row["median"]) for row in rows])
        lower = medians - np.array([float(row["q25"]) for row in rows])
        upper = np.array([float(row["q75"]) for row in rows]) - medians
        axis.errorbar(dimensions, medians, yerr=[lower, upper], fmt="o", color=color, label=posterior)
    axis.set(xlabel="dimension", ylabel="lengthscale median and IQR", yscale="log")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "lengthscale_posteriors.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(5.5, 4.0))
    groups = {
        "active": [float(row["log_lengthscale_w1"]) for row in drift if bool(row["active"])],
        "inactive": [float(row["log_lengthscale_w1"]) for row in drift if not bool(row["active"])],
    }
    axis.bar(groups.keys(), [np.mean(values) if values else 0.0 for values in groups.values()])
    axis.set(ylabel="fresh-posterior log-lengthscale W1")
    figure.tight_layout()
    figure.savefig(output_dir / "coordinate_drift.png", dpi=160)
    plt.close(figure)

    first_seed = min(int(row["seed"]) for row in ei_rows)
    final_ei = [
        row
        for row in ei_rows
        if int(row["count"]) == final_count and int(row["seed"]) == first_seed
    ]
    figure, axis = plt.subplots(figsize=(6.4, 4.0))
    axis.plot([row["candidate_index"] for row in final_ei], [row["fresh_ei"] for row in final_ei], label="fresh NUTS")
    axis.plot([row["candidate_index"] for row in final_ei], [row["reused_ei"] for row in final_ei], label="reweighted")
    axis.set(xlabel="fixed Sobol candidate index", ylabel="q=1 EI")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "ei_comparison.png", dpi=160)
    plt.close(figure)

    sequential = [float(row["sequential_update_seconds"]) for row in rounds if int(row["added_observations"]) > 0]
    nuts = [float(row["fresh_nuts_seconds"]) for row in checkpoints]
    figure, axis = plt.subplots(figsize=(5.5, 4.0))
    axis.bar(("sequential update", "fresh NUTS"), (np.median(sequential), np.median(nuts)))
    axis.set(ylabel="wall time (seconds)", yscale="log")
    figure.tight_layout()
    figure.savefig(output_dir / "timing_cost.png", dpi=160)
    plt.close(figure)


def _safe_correlation(x: list[float], y: list[float]) -> float | None:
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(pearsonr(x, y).statistic)


def _write_summary(
    path: Path,
    config: Task02AConfig,
    results: list[dict[str, Any]],
    rounds: list[dict[str, Any]],
    checkpoints: list[dict[str, Any]],
    drift: list[dict[str, Any]],
) -> dict[str, Any]:
    noninitial_checkpoints = [row for row in checkpoints if int(row["added_observations"]) > 0]
    high_ess = [row for row in noninitial_checkpoints if float(row["ess_fraction"]) >= 0.5]
    informative = high_ess or noninitial_checkpoints
    d2 = [float(row["negative_log_ess_fraction"]) for row in noninitial_checkpoints]
    discrepancy = [float(row["mmd"]) for row in noninitial_checkpoints]
    correlation = _safe_correlation(d2, discrepancy)
    sequential_times = [
        float(row["sequential_update_seconds"])
        for row in rounds
        if int(row["added_observations"]) > 0
    ]
    nuts_times = [float(row["fresh_nuts_seconds"]) for row in checkpoints]
    speedup = float(np.median(nuts_times) / np.median(sequential_times))
    active_drift = [float(row["log_lengthscale_w1"]) for row in drift if bool(row["active"])]
    inactive_drift = [float(row["log_lengthscale_w1"]) for row in drift if not bool(row["active"])]
    active_mean = float(np.mean(active_drift)) if active_drift else float("nan")
    inactive_mean = float(np.mean(inactive_drift)) if inactive_drift else float("nan")
    final_count = max(int(row["count"]) for row in noninitial_checkpoints)
    final_rows = [row for row in noninitial_checkpoints if int(row["count"]) == final_count]
    final_mean = lambda key: float(np.mean([float(row[key]) for row in final_rows]))
    final_ess = final_mean("ess_fraction")
    decision_agreement = float(
        np.mean(
            [
                int(row["reused_ei_index"]) == int(row["fresh_ei_index"])
                for row in final_rows
            ]
        )
    )
    if final_ess < 0.1:
        recommendation = "02B-B"
        reason = "fixed-support importance weights collapsed severely"
    elif final_ess < 0.5:
        recommendation = "02B-A"
        reason = "reuse is finite but needs adaptive annealed resample-move updates"
    elif inactive_mean < 0.7 * active_mean and decision_agreement == 1.0:
        recommendation = "02C"
        reason = "reuse remained effective and irrelevant-coordinate drift was distinctly smaller"
    elif decision_agreement == 1.0:
        recommendation = "02B-A"
        reason = "reuse preserved the smoke decision but coordinate selectivity was not established"
    else:
        recommendation = "STOP/PIVOT"
        reason = "the fixed-support reuse diagnostic did not preserve the q=1 decision"

    crossing_lines = []
    for threshold in (0.75, 0.5, 0.25, 0.1):
        values = [result["crossings"][str(threshold)] for result in results]
        formatted = [
            str(value) if value is not None else f">{config.final_count - config.initial_count}"
            for value in values
        ]
        crossing_lines.append(f"ESS/P < {threshold:.2f}: `{', '.join(formatted)}` new observations by seed")

    mean_metric = lambda key: float(np.mean([float(row[key]) for row in informative]))
    cache_max = max(float(row["marginal_increment_max_abs_error"]) for row in rounds)
    chol_max = max(float(row["cache_chol_max_abs_error"]) for row in rounds)
    smoke_caveat = (
        "This is one deliberately reduced NUTS smoke run; its posterior is not scientifically "
        "interpretable. The D=10, three-seed Colab study is required before advancing a stage."
        if config.profile == "smoke"
        else "These values are from the planned three-seed full Colab configuration."
    )
    high_ess_basis = (
        f"Across {len(high_ess)} noninitial checkpoint(s) with ESS/P >= 0.5"
        if high_ess
        else "There was no noninitial fresh checkpoint with ESS/P >= 0.5; using the final checkpoint only as low-ESS context"
    )
    reused_indices = [int(row["reused_ei_index"]) for row in final_rows]
    fresh_indices = [int(row["fresh_ei_index"]) for row in final_rows]
    text = f"""# Task 02A — SAAS structural-posterior reuse diagnostic

## Scope

This is only the Task 02A falsification experiment: trusted BoTorch SAAS NUTS particles are held fixed while exact one-step predictive likelihoods update their weights and rank-one formulas update their GP caches. It does **not** implement rejuvenation, particle movement, selective-coordinate updates, Vecchia, an output EBM, q>1 acquisition, or a BO loop.

Profile: `{config.profile}`; seeds `{list(config.seeds)}`; D={config.dimension}; n0={config.initial_count}; nfinal={config.final_count}; retained particles={config.nuts_samples // config.nuts_thinning}; NUTS warmup/samples/thinning/tree depth={config.nuts_warmup}/{config.nuts_samples}/{config.nuts_thinning}/{config.nuts_tree_depth}. {smoke_caveat}

## Eight completion questions

1. **Reuse horizon.** {'; '.join(crossing_lines)}.
2. **Does -log(ESS/P) track fresh-posterior discrepancy?** The Pearson correlation with pooled-standardized RBF MMD is `{correlation if correlation is not None else 'not estimable with fewer than three informative fresh checkpoints'}`. Final mean `-log(ESS/P)={final_mean('negative_log_ess_fraction'):.4g}` and MMD=`{final_mean('mmd'):.4g}`.
3. **Structural marginals while ESS is high.** {high_ess_basis}: mean log-lengthscale W1=`{mean_metric('mean_log_lengthscale_w1'):.4g}`, maximum dimensionwise W1=`{mean_metric('max_log_lengthscale_w1'):.4g}`, and pooled MMD=`{mean_metric('mmd'):.4g}`. Final active-top-2 probability is `{final_mean('reused_top2_active_probability'):.3f}` reused versus `{final_mean('fresh_top2_active_probability'):.3f}` fresh.
4. **Predictive mixtures while ESS is high.** On the same checkpoint basis, mean predictive-mean RMSE=`{mean_metric('predictive_mean_rmse'):.4g}`, total-variance RMSE=`{mean_metric('predictive_variance_rmse'):.4g}`, and reused-minus-fresh held-out mixture log score=`{mean_metric('reused_log_score') - mean_metric('fresh_log_score'):.4g}`. Total variance includes between-particle variance; log score also includes fixed observation noise.
5. **q=1 EI decision.** At the final checkpoint, mean Spearman=`{final_mean('ei_spearman'):.4f}`, mean top-5% overlap=`{final_mean('ei_top5_overlap'):.3f}`, mean maximum-EI relative error=`{final_mean('ei_max_relative_error'):.4g}`, selected indices reused/fresh by seed=`{reused_indices}/{fresh_indices}`, decision agreement=`{decision_agreement:.3f}`, and mean candidate distance=`{final_mean('ei_candidate_distance'):.4g}`.
6. **Measured cost.** Median sequential likelihood/weight/cache update=`{np.median(sequential_times):.4g}` s versus median fresh NUTS fit=`{np.median(nuts_times):.4g}` s, a `{speedup:.3g}x` wall-time ratio. Exact increment error was at most `{cache_max:.3e}` and rank-one/full Cholesky error at most `{chol_max:.3e}`; validation-only full recomputations are separately counted in JSON.
7. **Coordinate stability.** Mean fresh-posterior log-lengthscale W1 drift was `{active_mean:.4g}` for active dimensions 0–1 and `{inactive_mean:.4g}` for irrelevant dimensions. On this evidence, irrelevant dimensions were `{'more stable' if inactive_mean < active_mean else 'not more stable'}`.
8. **Stage recommendation.** **{recommendation}** provisionally, because {reason}. For the smoke profile this recommendation is diagnostic only: do not implement Task 02B until the full Colab evidence is reviewed.

## Reproduction and files

- `artifacts/task02a/{config.profile}/task02a_config.json` records the frozen affine transform, seeds, numerical settings, package versions, JAX backend/devices, and counters.
- The same directory contains round, checkpoint, lengthscale, coordinate-drift, and EI CSVs plus ESS, lengthscale, drift, EI, and timing PNGs.
- Unit tests include exact marginal-increment, rank-one cache, GPyTorch-kernel, stable weighting, weighted-EI, and frozen-preprocessing identities.
- `COLAB.md` gives the exact CPU full-run command and optional NVIDIA JAX setup.

No Task 02B code is included.
"""
    path.write_text(text)
    return {
        "recommendation": recommendation,
        "median_sequential_seconds": float(np.median(sequential_times)),
        "median_nuts_seconds": float(np.median(nuts_times)),
        "speedup_ratio": speedup,
        "d2_mmd_correlation": correlation,
        "active_mean_w1_drift": active_mean,
        "inactive_mean_w1_drift": inactive_mean,
    }


def run_task02a(config: Task02AConfig, output_dir: Path, summary_path: Path) -> None:
    torch.set_default_dtype(torch.double)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = [_run_seed(config, seed) for seed in config.seeds]
    rounds = [row for result in results for row in result["round_rows"]]
    checkpoints = [row for result in results for row in result["checkpoint_rows"]]
    lengthscales = [row for result in results for row in result["lengthscale_rows"]]
    ei_rows = [row for result in results for row in result["ei_rows"]]
    drift = [row for result in results for row in result["drift_rows"]]
    _write_csv(output_dir / "task02a_rounds.csv", rounds)
    _write_csv(output_dir / "task02a_checkpoints.csv", checkpoints)
    _write_csv(output_dir / "task02a_lengthscales.csv", lengthscales)
    _write_csv(output_dir / "task02a_coordinate_drift.csv", drift)
    _write_csv(output_dir / "task02a_ei.csv", ei_rows)
    _plots(output_dir, rounds, checkpoints, lengthscales, drift, ei_rows)
    summary = _write_summary(summary_path, config, results, rounds, checkpoints, drift)
    payload = {
        "configuration": asdict(config),
        "runs": [
            {
                "seed": result["seed"],
                "transform": result["transform"],
                "environment": result["environment"],
                "crossings": result["crossings"],
                "cache_counters": result["cache_counters"],
                "nuts_times": result["nuts_times"],
            }
            for result in results
        ],
        "summary": summary,
    }
    (output_dir / "task02a_config.json").write_text(json.dumps(payload, indent=2) + "\n")
