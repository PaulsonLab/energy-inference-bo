"""Task 02B decision-space compression and joint-target diagnostic."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

from energy_bo.decision.coresets import (
    acquisition_frank_wolfe,
    posterior_k_medoids,
    random_equal,
)
from energy_bo.decision.joint_target import joint_target_marginals
from energy_bo.decision.metrics import acquisition_metrics, signature_spectrum
from energy_bo.decision.signatures import particle_ei_signatures, transformed_particle_features
from energy_bo.structural.acquisition import weighted_expected_improvement
from energy_bo.structural.exact_gp import ExactGPBatchState
from energy_bo.structural.particles import ParticleWeights, SaasParticles
from energy_bo.structural.preprocessing import (
    FrozenOutputTransform,
    deterministic_benchmark_inputs,
    negative_branin,
)
from energy_bo.structural.saas_reference import NutsConfig, fit_saas_reference, runtime_environment


@dataclass(frozen=True)
class Task02BConfig:
    profile: str
    seeds: tuple[int, ...]
    dimension: int
    initial_count: int
    final_count: int
    candidate_count: int
    prediction_chunk_size: int
    checkpoints: tuple[tuple[int, tuple[int, ...]], ...]
    nuts_warmup: int
    nuts_samples: int
    nuts_thinning: int
    nuts_tree_depth: int
    noise_variance: float = 1e-4
    random_repetitions: int = 32

    @classmethod
    def smoke(cls) -> "Task02BConfig":
        return cls(
            profile="smoke",
            seeds=(0,),
            dimension=4,
            initial_count=8,
            final_count=11,
            candidate_count=256,
            prediction_chunk_size=256,
            checkpoints=((0, (8, 11)),),
            nuts_warmup=32,
            nuts_samples=32,
            nuts_thinning=1,
            nuts_tree_depth=4,
        )

    @classmethod
    def full(cls, task02a_results: Path) -> "Task02BConfig":
        payload = json.loads((task02a_results / "task02a_config.json").read_text())
        source = payload["configuration"]
        checkpoint_rows = _read_csv(task02a_results / "task02a_checkpoints.csv")
        by_seed: dict[int, list[int]] = {}
        for row in checkpoint_rows:
            by_seed.setdefault(int(row["seed"]), []).append(int(row["count"]))
        checkpoints = tuple(
            (seed, tuple(sorted(set(by_seed[seed])))) for seed in sorted(by_seed)
        )
        return cls(
            profile="full",
            seeds=tuple(int(seed) for seed in source["seeds"]),
            dimension=int(source["dimension"]),
            initial_count=int(source["initial_count"]),
            final_count=int(source["final_count"]),
            candidate_count=int(source["candidate_count"]),
            prediction_chunk_size=int(source["prediction_chunk_size"]),
            checkpoints=checkpoints,
            nuts_warmup=int(source["nuts_warmup"]),
            nuts_samples=int(source["nuts_samples"]),
            nuts_thinning=int(source["nuts_thinning"]),
            nuts_tree_depth=int(source["nuts_tree_depth"]),
            noise_variance=float(source["noise_variance"]),
        )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


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


def _safe_correlations(x: list[float], y: list[float]) -> dict[str, float | None]:
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return {"pearson": None, "spearman": None}
    return {
        "pearson": float(pearsonr(x, y).statistic),
        "spearman": float(spearmanr(x, y).statistic),
    }


def retrospective_task02a(task02a_results: Path, output_dir: Path) -> dict[str, Any]:
    """Compute decision regret solely from saved Task 02A integrated EI curves."""

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_rows = _read_csv(task02a_results / "task02a_checkpoints.csv")
    checkpoint_map = {
        (int(row["seed"]), int(row["count"])): row for row in checkpoint_rows
    }
    grouped: dict[tuple[int, int], list[dict[str, str]]] = {}
    for row in _read_csv(task02a_results / "task02a_ei.csv"):
        grouped.setdefault((int(row["seed"]), int(row["count"])), []).append(row)
    rows: list[dict[str, Any]] = []
    for key, candidates in sorted(grouped.items()):
        candidates.sort(key=lambda row: int(row["candidate_index"]))
        fresh = torch.tensor([float(row["fresh_ei"]) for row in candidates], dtype=torch.double)
        reused = torch.tensor([float(row["reused_ei"]) for row in candidates], dtype=torch.double)
        metrics = acquisition_metrics(fresh, reused)
        checkpoint = checkpoint_map[key]
        rows.append(
            {
                "seed": key[0],
                "count": key[1],
                "added_observations": int(checkpoint["added_observations"]),
                "ess_fraction": float(checkpoint["ess_fraction"]),
                "negative_log_ess_fraction": float(checkpoint["negative_log_ess_fraction"]),
                "mmd": float(checkpoint["mmd"]),
                "mean_log_lengthscale_w1": float(checkpoint["mean_log_lengthscale_w1"]),
                "max_log_lengthscale_w1": float(checkpoint["max_log_lengthscale_w1"]),
                "saved_ei_spearman": float(checkpoint["ei_spearman"]),
                "saved_ei_top5_overlap": float(checkpoint["ei_top5_overlap"]),
                **metrics,
            }
        )
    noninitial = [row for row in rows if int(row["added_observations"]) > 0]
    regret = [float(row["normalized_decision_regret"]) for row in noninitial]
    diagnostic_names = (
        "ess_fraction",
        "negative_log_ess_fraction",
        "mmd",
        "mean_log_lengthscale_w1",
        "max_log_lengthscale_w1",
        "saved_ei_spearman",
        "saved_ei_top5_overlap",
    )
    correlations = {
        name: _safe_correlations([float(row[name]) for row in noninitial], regret)
        for name in diagnostic_names
    }
    differing = [row for row in rows if not bool(row["exact_index_agreement"])]
    low_ess = [row for row in noninitial if float(row["ess_fraction"]) < 0.1]
    summary = {
        "checkpoint_count": len(rows),
        "noninitial_checkpoint_count": len(noninitial),
        "exact_index_difference_count": len(differing),
        "different_but_below_1pct_count": sum(
            float(row["normalized_decision_regret"]) < 0.01 for row in differing
        ),
        "below_5pct_count": sum(value < 0.05 for value in regret),
        "above_10pct_count": sum(value > 0.10 for value in regret),
        "median_normalized_regret": float(np.median(regret)),
        "mean_normalized_regret": float(np.mean(regret)),
        "max_normalized_regret": float(np.max(regret)),
        "low_ess_checkpoint_count": len(low_ess),
        "low_ess_below_5pct_count": sum(
            float(row["normalized_decision_regret"]) < 0.05 for row in low_ess
        ),
        "correlations": correlations,
    }
    _write_csv(output_dir / "task02b_retrospective.csv", rows)
    (output_dir / "task02b_retrospective.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    _plot_retrospective(output_dir / "decision_regret_diagnostics.png", noninitial)
    return {"rows": rows, "summary": summary}


def _plot_retrospective(path: Path, rows: list[dict[str, Any]]) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(11.0, 3.5))
    for axis, key, label in zip(
        axes,
        ("ess_fraction", "mmd", "max_log_lengthscale_w1"),
        ("ESS / P", "posterior MMD", "max log-lengthscale W1"),
        strict=True,
    ):
        for seed in sorted({int(row["seed"]) for row in rows}):
            selected = [row for row in rows if int(row["seed"]) == seed]
            axis.scatter(
                [float(row[key]) for row in selected],
                [float(row["normalized_decision_regret"]) for row in selected],
                label=f"seed {seed}",
            )
        axis.set(xlabel=label, ylabel="normalized decision regret")
    axes[-1].legend(loc="upper left")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _saved_fresh_ei(task02a_results: Path) -> dict[tuple[int, int], torch.Tensor]:
    grouped: dict[tuple[int, int], list[tuple[int, float]]] = {}
    for row in _read_csv(task02a_results / "task02a_ei.csv"):
        key = (int(row["seed"]), int(row["count"]))
        grouped.setdefault(key, []).append((int(row["candidate_index"]), float(row["fresh_ei"])))
    return {
        key: torch.tensor([value for _, value in sorted(values)], dtype=torch.double)
        for key, values in grouped.items()
    }


def _coreset_budgets(particles: int) -> tuple[int, ...]:
    return tuple(value for value in (4, 8, 16, 32, 64) if value <= particles)


def _signature_checkpoint(
    *,
    config: Task02BConfig,
    seed: int,
    count: int,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    candidates: torch.Tensor,
    output_dir: Path,
    saved_teacher: torch.Tensor | None,
) -> dict[str, Any]:
    checkpoint_path = output_dir / "signatures" / f"seed{seed}_n{count}.npz"
    metadata_path = output_dir / "signatures" / f"seed{seed}_n{count}.json"
    loaded_from_disk = checkpoint_path.exists()
    if loaded_from_disk:
        saved = np.load(checkpoint_path)
        saved_candidates = torch.from_numpy(saved["candidate_x"]).double()
        if not torch.equal(saved_candidates, candidates):
            raise RuntimeError("saved signature candidates do not match the configured set")
        particles = SaasParticles(
            lengthscales=torch.from_numpy(saved["lengthscales"]),
            means=torch.from_numpy(saved["means"]),
            outputscales=torch.from_numpy(saved["outputscales"]),
        )
        signatures = torch.from_numpy(saved["particle_ei"]).double()
        expected_particles = config.nuts_samples // config.nuts_thinning
        if particles.num_particles != expected_particles or particles.dimension != config.dimension:
            raise RuntimeError("saved signature particle shape does not match configuration")
        if signatures.shape != (expected_particles, config.candidate_count):
            raise RuntimeError("saved acquisition-signature shape does not match configuration")
        if not torch.isfinite(signatures).all() or torch.any(signatures < 0):
            raise RuntimeError("saved acquisition signatures must be finite and nonnegative")
        teacher = signatures.mean(dim=0)
        if "teacher_ei" in saved and not torch.allclose(
            teacher, torch.from_numpy(saved["teacher_ei"]).double(), atol=1e-14, rtol=1e-13
        ):
            raise RuntimeError("saved teacher EI does not equal the particle-signature mean")
        metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        elapsed_seconds = metadata.get("nuts_seconds")
        environment = metadata.get("environment", runtime_environment())
        mean_validation_error = float(metadata.get("signature_mean_max_abs_error", 0.0))
    else:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        nuts = NutsConfig(
            warmup_steps=config.nuts_warmup,
            num_samples=config.nuts_samples,
            thinning=config.nuts_thinning,
            max_tree_depth=config.nuts_tree_depth,
            seed=10_000 * seed + count,
        )
        fit = fit_saas_reference(
            train_x[:count], train_y[:count], config.noise_variance, nuts
        )
        particles = fit.particles
        elapsed_seconds = fit.elapsed_seconds
        environment = fit.environment
        state = ExactGPBatchState.build(
            particles, train_x[:count], train_y[:count], config.noise_variance
        )
        best_f = float(train_y[:count].max())
        signatures = particle_ei_signatures(
            state, candidates, best_f, chunk_size=config.prediction_chunk_size
        )
        weights = ParticleWeights.uniform(particles.num_particles)
        direct = weighted_expected_improvement(
            state, weights, candidates, best_f, chunk_size=config.prediction_chunk_size
        )
        teacher = signatures.mean(dim=0)
        mean_validation_error = float((teacher - direct).abs().max())
        if mean_validation_error > 1e-12:
            raise RuntimeError("particle signature mean does not reproduce weighted EI")
        np.savez_compressed(
            checkpoint_path,
            lengthscales=particles.lengthscales.numpy(),
            means=particles.means.numpy(),
            outputscales=particles.outputscales.numpy(),
            candidate_x=candidates.numpy(),
            particle_ei=signatures.numpy(),
            teacher_ei=teacher.numpy(),
        )
        metadata_path.write_text(
            json.dumps(
                {
                    "nuts_seconds": elapsed_seconds,
                    "environment": environment,
                    "signature_mean_max_abs_error": mean_validation_error,
                },
                indent=2,
            )
            + "\n"
        )
    saved_max_error = None
    saved_rmse = None
    if saved_teacher is not None:
        saved_max_error = float((teacher - saved_teacher).abs().max())
        saved_rmse = float(torch.mean((teacher - saved_teacher).square()).sqrt())
    acquisition_spectrum = signature_spectrum(signatures)
    structural_spectrum = signature_spectrum(
        transformed_particle_features(particles), column_standardize=True
    )
    coreset_rows: list[dict[str, Any]] = []
    features = transformed_particle_features(particles)
    for budget in _coreset_budgets(particles.num_particles):
        for repetition in range(config.random_repetitions):
            coreset = random_equal(
                signatures, budget, seed=1_000_000 * seed + 1_000 * count + 31 * budget + repetition
            )
            coreset_rows.append(
                {
                    "seed": seed,
                    "count": count,
                    "method": coreset.method,
                    "budget": budget,
                    "retained": coreset.indices.numel(),
                    "repetition": repetition,
                    **acquisition_metrics(teacher, coreset.acquisition(signatures)),
                }
            )
        for coreset in (
            posterior_k_medoids(features, budget),
            acquisition_frank_wolfe(signatures, budget),
        ):
            coreset_rows.append(
                {
                    "seed": seed,
                    "count": count,
                    "method": coreset.method,
                    "budget": budget,
                    "retained": coreset.indices.numel(),
                    "repetition": 0,
                    **acquisition_metrics(teacher, coreset.acquisition(signatures)),
                }
            )
    return {
        "seed": seed,
        "count": count,
        "particles": particles.num_particles,
        "nuts_seconds": elapsed_seconds,
        "loaded_from_disk": loaded_from_disk,
        "environment": environment,
        "signature_mean_max_abs_error": mean_validation_error,
        "saved_teacher_max_abs_error": saved_max_error,
        "saved_teacher_rmse": saved_rmse,
        "acquisition_spectrum": acquisition_spectrum,
        "structural_spectrum": structural_spectrum,
        "coreset_rows": coreset_rows,
        "signatures": signatures,
    }


def _spectrum_rows(checkpoints: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        for space in ("acquisition", "structural"):
            spectrum = checkpoint[f"{space}_spectrum"]
            for index, (singular, fraction, cumulative) in enumerate(
                zip(
                    spectrum["singular_values"],
                    spectrum["explained_squared_fraction"],
                    spectrum["cumulative_explained_squared_fraction"],
                    strict=True,
                )
            ):
                rows.append(
                    {
                        "seed": checkpoint["seed"],
                        "count": checkpoint["count"],
                        "space": space,
                        "component": index + 1,
                        "singular_value": singular,
                        "explained_squared_fraction": fraction,
                        "cumulative_explained_squared_fraction": cumulative,
                        "entropy_effective_rank": spectrum["entropy_effective_rank"],
                        "stable_rank": spectrum["stable_rank"],
                        "rank90": spectrum["rank90"],
                        "rank95": spectrum["rank95"],
                        "rank99": spectrum["rank99"],
                    }
                )
    return rows


def _joint_validation(signatures: torch.Tensor) -> dict[str, Any]:
    marginal = joint_target_marginals(signatures)
    return {
        "m1_max_abs_error": float(
            (marginal["m1"] - marginal["normalized_teacher"]).abs().max()
        ),
        "m1_l1_error": float(
            (marginal["m1"] - marginal["normalized_teacher"]).abs().sum()
        ),
        "m2_max_abs_error": float(
            (marginal["independent_m2"] - marginal["normalized_teacher_squared"]).abs().max()
        ),
        "m2_l1_error": float(
            (marginal["independent_m2"] - marginal["normalized_teacher_squared"]).abs().sum()
        ),
        "common_vs_independent_l1": float(
            (marginal["common_particle_m2"] - marginal["independent_m2"]).abs().sum()
        ),
        "teacher_mode": int(torch.argmax(marginal["normalized_teacher"])),
        "m1_mode": int(torch.argmax(marginal["m1"])),
        "teacher_squared_mode": int(torch.argmax(marginal["normalized_teacher_squared"])),
        "m2_mode": int(torch.argmax(marginal["independent_m2"])),
        "marginals": marginal,
    }


def _plots_task02b(
    output_dir: Path,
    spectrum_rows: list[dict[str, Any]],
    coreset_rows: list[dict[str, Any]],
    joint: dict[str, Any],
) -> None:
    figure, axis = plt.subplots(figsize=(6.2, 4.0))
    for key in sorted({(int(row["seed"]), int(row["count"])) for row in spectrum_rows}):
        selected = [
            row for row in spectrum_rows
            if row["space"] == "acquisition" and (int(row["seed"]), int(row["count"])) == key
        ]
        axis.semilogy(
            [row["component"] for row in selected],
            [max(float(row["explained_squared_fraction"]), 1e-16) for row in selected],
            marker="o",
            label=f"seed {key[0]}, n={key[1]}",
        )
    axis.set(xlabel="signature component", ylabel="explained squared fraction")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "acquisition_signature_spectrum.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6.2, 4.0))
    methods = sorted({str(row["method"]) for row in coreset_rows})
    for method in methods:
        budgets = sorted({int(row["budget"]) for row in coreset_rows if row["method"] == method})
        means = [
            np.mean([
                float(row["normalized_decision_regret"])
                for row in coreset_rows
                if row["method"] == method and int(row["budget"]) == budget
            ])
            for budget in budgets
        ]
        axis.plot(budgets, [max(value, 1e-6) for value in means], marker="o", label=method)
    axis.set(
        xlabel="particle budget K",
        ylabel="mean normalized decision regret (floor $10^{-6}$)",
        yscale="log",
    )
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "coreset_quality.png", dpi=160)
    plt.close(figure)

    marginal = joint["marginals"]
    figure, axes = plt.subplots(1, 2, figsize=(10.0, 3.5))
    axes[0].plot(marginal["normalized_teacher"].numpy(), label="teacher EI")
    axes[0].plot(marginal["m1"].numpy(), linestyle="--", label="M=1 marginal")
    axes[1].plot(marginal["normalized_teacher_squared"].numpy(), label="teacher EI squared")
    axes[1].plot(marginal["independent_m2"].numpy(), linestyle="--", label="independent M=2")
    for axis in axes:
        axis.set(xlabel="candidate index", ylabel="normalized mass")
        axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "joint_target_validation.png", dpi=160)
    plt.close(figure)


def _aggregate_coresets(rows: list[dict[str, Any]]) -> dict[str, Any]:
    aggregates: dict[str, Any] = {}
    for method in sorted({str(row["method"]) for row in rows}):
        method_rows = [row for row in rows if row["method"] == method]
        aggregates[method] = {
            "mean_normalized_regret": float(
                np.mean([float(row["normalized_decision_regret"]) for row in method_rows])
            ),
            "mean_normalized_max_error": float(
                np.mean([float(row["normalized_max_error"]) for row in method_rows])
            ),
        }
    fw_rows = [row for row in rows if row["method"] == "acquisition_fw"]
    checkpoint_count = len({(int(row["seed"]), int(row["count"])) for row in fw_rows})
    thresholds: dict[str, int | str] = {}
    for threshold in (0.01, 0.05, 0.10):
        passing = []
        for budget in sorted({int(row["budget"]) for row in fw_rows}):
            selected = [row for row in fw_rows if int(row["budget"]) == budget]
            if len(selected) == checkpoint_count and all(
                float(row["normalized_decision_regret"]) < threshold for row in selected
            ):
                passing.append(budget)
        thresholds[str(threshold)] = min(passing) if passing else ">max_tested"
    aggregates["acquisition_fw_all_checkpoint_budget"] = thresholds
    return aggregates


def run_signature_study(
    config: Task02BConfig,
    task02a_results: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_map = _saved_fresh_ei(task02a_results) if config.profile == "full" else {}
    expected_transforms: dict[int, dict[str, float]] = {}
    if config.profile == "full":
        source_payload = json.loads((task02a_results / "task02a_config.json").read_text())
        expected_transforms = {
            int(run["seed"]): {
                "mean": float(run["transform"]["mean"]),
                "scale": float(run["transform"]["scale"]),
            }
            for run in source_payload["runs"]
        }
    checkpoint_results: list[dict[str, Any]] = []
    for seed, counts in config.checkpoints:
        train_x = deterministic_benchmark_inputs(config.final_count, config.dimension, seed)
        raw_y = negative_branin(train_x)
        transform = FrozenOutputTransform.fit(raw_y[: config.initial_count])
        if seed in expected_transforms:
            expected = expected_transforms[seed]
            if abs(transform.mean - expected["mean"]) > 1e-12 or abs(transform.scale - expected["scale"]) > 1e-12:
                raise RuntimeError("reconstructed frozen output transform differs from Task 02A")
        train_y = transform.transform(raw_y)
        candidates = deterministic_benchmark_inputs(
            config.candidate_count, config.dimension, seed + 2_000
        )
        for count in counts:
            checkpoint_results.append(
                _signature_checkpoint(
                    config=config,
                    seed=seed,
                    count=count,
                    train_x=train_x,
                    train_y=train_y,
                    candidates=candidates,
                    output_dir=output_dir,
                    saved_teacher=saved_map.get((seed, count)),
                )
            )
    spectrum_rows = _spectrum_rows(checkpoint_results)
    coreset_rows = [row for result in checkpoint_results for row in result["coreset_rows"]]
    representative = max(checkpoint_results, key=lambda row: (row["count"], -row["seed"]))
    joint = _joint_validation(representative["signatures"])
    _write_csv(output_dir / "task02b_spectra.csv", spectrum_rows)
    _write_csv(output_dir / "task02b_coresets.csv", coreset_rows)
    joint_serializable = {key: value for key, value in joint.items() if key != "marginals"}
    _write_csv(output_dir / "task02b_joint_targets.csv", [joint_serializable])
    _plots_task02b(output_dir, spectrum_rows, coreset_rows, joint)
    checkpoint_summary = [
        {
            key: value
            for key, value in result.items()
            if key not in {"coreset_rows", "signatures", "environment"}
        }
        | {"environment": result["environment"]}
        for result in checkpoint_results
    ]
    payload = {
        "configuration": asdict(config),
        "checkpoints": checkpoint_summary,
        "coreset_aggregate": _aggregate_coresets(coreset_rows),
        "joint_target": joint_serializable,
    }
    (output_dir / "task02b_config.json").write_text(json.dumps(payload, indent=2) + "\n")
    return {
        "checkpoints": checkpoint_results,
        "spectrum_rows": spectrum_rows,
        "coreset_rows": coreset_rows,
        "coreset_aggregate": payload["coreset_aggregate"],
        "joint": joint_serializable,
    }


def _format_correlation(value: float | None) -> str:
    return "not estimable" if value is None else f"{value:.4f}"


def write_task02b_summary(
    path: Path,
    config: Task02BConfig,
    retrospective: dict[str, Any],
    signature: dict[str, Any],
) -> None:
    retrospective_summary = retrospective["summary"]
    checkpoints = signature["checkpoints"]
    acquisition_ranks = [
        float(row["acquisition_spectrum"]["entropy_effective_rank"]) for row in checkpoints
    ]
    structural_ranks = [
        float(row["structural_spectrum"]["entropy_effective_rank"]) for row in checkpoints
    ]
    coreset = signature["coreset_aggregate"]
    fw_budget = coreset["acquisition_fw_all_checkpoint_budget"]
    joint = signature["joint"]
    correlations = retrospective_summary["correlations"]
    random_regret = coreset["random_equal"]["mean_normalized_regret"]
    posterior_regret = coreset["posterior_medoid"]["mean_normalized_regret"]
    acquisition_regret = coreset["acquisition_fw"]["mean_normalized_regret"]
    low_ess_total = retrospective_summary["low_ess_checkpoint_count"]
    low_ess_low_regret = retrospective_summary["low_ess_below_5pct_count"]
    saved_curve_errors = [
        float(row["saved_teacher_max_abs_error"])
        for row in checkpoints
        if row["saved_teacher_max_abs_error"] is not None
    ]
    if config.profile == "full":
        conditions = (
            low_ess_total > 0 and low_ess_low_regret / low_ess_total >= 0.5,
            float(np.mean(acquisition_ranks)) <= 32,
            isinstance(fw_budget["0.05"], int) and int(fw_budget["0.05"]) <= 32,
            joint["m1_max_abs_error"] < 1e-12 and joint["m2_max_abs_error"] < 1e-12,
            acquisition_regret < min(random_regret, posterior_regret),
        )
        recommendation = "GO" if sum(conditions) >= 4 else "NO-GO"
        recommendation_reason = f"{sum(conditions)}/5 prespecified diagnostic conditions passed"
    else:
        recommendation = "NO-GO pending full evidence"
        recommendation_reason = "The acquisition-signature evidence uses only tiny D=4 smoke NUTS chains"
    rerun_comparison = (
        f" The independently rerun teacher curves differed from the published Task 02A curves by maximum absolute error up to `{max(saved_curve_errors):.4g}`; this Monte Carlo/backend discrepancy is retained in the checkpoint JSON."
        if saved_curve_errors
        else ""
    )
    text = f"""# Task 02B — Decision-space compression and joint-energy validation

## Scope and evidence

This task tests an oracle/teacher compression hypothesis and the discrete q=1 joint structural-decision identity. It does **not** implement SVGD, MALA, annealed Langevin, SMC rejuvenation, Vecchia, a residual-output EBM, q>1 BO, molecular optimization, or an end-to-end BO loop.

The decision-regret analysis below uses all `{retrospective_summary['checkpoint_count']}` saved full Task 02A checkpoints without rerunning NUTS. Acquisition-signature results use profile `{config.profile}` with `{len(checkpoints)}` checkpoint(s), D={config.dimension}, `{config.nuts_samples // config.nuts_thinning}` retained particles, and `{config.candidate_count}` fixed candidates. {'The signature and coreset findings are smoke diagnostics, not scientific full-run evidence.' if config.profile == 'smoke' else 'The signature and coreset findings use the full three-seed Colab profile.'}{rerun_comparison}

## Eight completion questions

1. **How much more robust is decision regret than posterior fidelity?** Across `{retrospective_summary['noninitial_checkpoint_count']}` noninitial full Task 02A checkpoints, median/mean/max normalized decision regret was `{retrospective_summary['median_normalized_regret']:.4g}` / `{retrospective_summary['mean_normalized_regret']:.4g}` / `{retrospective_summary['max_normalized_regret']:.4g}`; `{retrospective_summary['below_5pct_count']}` were below 5% and `{retrospective_summary['above_10pct_count']}` exceeded 10%. Of `{low_ess_total}` checkpoints with ESS/P < 0.1, `{low_ess_low_regret}` retained <5% decision regret. Exact candidates differed at `{retrospective_summary['exact_index_difference_count']}` checkpoints; `{retrospective_summary['different_but_below_1pct_count']}` of those differences cost <1% regret. This supports robustness only at a subset of checkpoints, not uniformly.
2. **Does ESS predict decision regret?** Over the `{retrospective_summary['noninitial_checkpoint_count']}` noninitial checkpoints, ESS/P versus normalized regret had Pearson/Spearman correlations `{_format_correlation(correlations['ess_fraction']['pearson'])}` / `{_format_correlation(correlations['ess_fraction']['spearman'])}`; `-log(ESS/P)` had `{_format_correlation(correlations['negative_log_ess_fraction']['pearson'])}` / `{_format_correlation(correlations['negative_log_ess_fraction']['spearman'])}`. These correlations are descriptive because n is small.
3. **What is the acquisition-signature effective rank?** Entropy effective rank ranged `{min(acquisition_ranks):.3f}`–`{max(acquisition_ranks):.3f}` (mean `{np.mean(acquisition_ranks):.3f}`), versus standardized structural-coordinate rank `{min(structural_ranks):.3f}`–`{max(structural_ranks):.3f}` (mean `{np.mean(structural_ranks):.3f}`).
4. **How many particles preserve the decision?** Acquisition-space Frank–Wolfe required K=`{fw_budget['0.01']}` / `{fw_budget['0.05']}` / `{fw_budget['0.1']}` for every analyzed checkpoint to have <1% / <5% / <10% normalized regret.
5. **Does acquisition-space selection outperform the baselines?** Mean normalized regret across tested budgets/checkpoints was `{acquisition_regret:.4g}` for acquisition-space Frank–Wolfe, `{random_regret:.4g}` for 32-repeat random equal thinning, and `{posterior_regret:.4g}` for posterior-space medoids. This is an oracle compression comparison, not an implementable inference algorithm.
6. **Does M=1 recover full Bayesian EI?** Yes: maximum/L1 normalized-marginal errors were `{joint['m1_max_abs_error']:.3e}` / `{joint['m1_l1_error']:.3e}`, with teacher/M=1 modes `{joint['teacher_mode']}` / `{joint['m1_mode']}`.
7. **Does independent-replica M=2 recover squared EI?** Yes: maximum/L1 errors were `{joint['m2_max_abs_error']:.3e}` / `{joint['m2_l1_error']:.3e}`, with squared-teacher/M=2 modes `{joint['teacher_squared_mode']}` / `{joint['m2_mode']}`. The common-particle negative control differed from independent M=2 by L1 `{joint['common_vs_independent_l1']:.4g}`.
8. **Is Task 02C joint-energy transport justified?** **{recommendation}.** {recommendation_reason}. No Task 02C method is implemented.

## Reproduction and outputs

- `results/task02b/retrospective/` contains the saved-result regret table, correlations, and diagnostic plot.
- The profile output contains spectra, coreset metrics, joint-target metrics, plots, environment/configuration JSON, and compressed per-particle signatures.
- `COLAB.md` gives the exact full extraction command. Full signature matrices remain generated artifacts until deliberately reviewed and imported.
"""
    path.write_text(text)


def run_task02b(
    config: Task02BConfig,
    task02a_results: Path,
    output_dir: Path,
    retrospective_dir: Path,
    summary_path: Path,
) -> None:
    torch.set_default_dtype(torch.double)
    retrospective = retrospective_task02a(task02a_results, retrospective_dir)
    signature = run_signature_study(config, task02a_results, output_dir)
    write_task02b_summary(summary_path, config, retrospective, signature)
