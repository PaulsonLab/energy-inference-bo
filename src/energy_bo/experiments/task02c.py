"""Task 02C teacher preflight and matched decision-tilted SVGD experiment."""

from __future__ import annotations

import csv
import json
import math
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial.distance import cdist, pdist
from scipy.stats import wasserstein_distance
from torch.quasirandom import SobolEngine

from energy_bo.decision.metrics import signature_spectrum
from energy_bo.structural.exact_gp import ExactGPBatchState
from energy_bo.structural.particles import SaasParticles
from energy_bo.structural.preprocessing import (
    FrozenOutputTransform,
    deterministic_benchmark_inputs,
    negative_branin,
)
from energy_bo.transport.logei import OperationalMixture, cached_particle_log_ei
from energy_bo.transport.potential import SaasUnconstrainedPotential
from energy_bo.transport.preflight import run_point_preflight
from energy_bo.transport.svgd import (
    AdamState,
    Whitening,
    adam_ascent,
    choose_tempering_increment,
    design_retilt_cess,
    maximin_subset,
    svgd_direction,
)
from energy_bo.transport.teacher import (
    TeacherSamples,
    continuous_teacher_optimum,
    fit_fresh_teacher,
    map_saas_initial_design,
    operational_mixture_from_samples,
)


@dataclass(frozen=True)
class Task02CConfig:
    profile: str
    seeds: tuple[int, ...]
    counts: tuple[int, ...]
    dimension: int
    initial_count: int
    final_count: int
    candidate_count: int
    particle_counts: tuple[int, ...]
    structural_budgets: tuple[int, ...]
    repeats: int
    initialization_steps: int
    initialization_reach_steps: int
    structural_block: int
    design_steps_per_block: int
    prior_whitening_draws: int
    nuts_warmup: int
    nuts_samples: int
    nuts_thinning: int
    nuts_tree_depth: int
    noise_variance: float = 1e-4

    @classmethod
    def smoke(cls) -> "Task02CConfig":
        return cls(
            profile="smoke",
            seeds=(0,),
            counts=(16,),
            dimension=10,
            initial_count=16,
            final_count=40,
            candidate_count=2048,
            particle_counts=(8,),
            structural_budgets=(8,),
            repeats=1,
            initialization_steps=4,
            initialization_reach_steps=4,
            structural_block=4,
            design_steps_per_block=2,
            prior_whitening_draws=512,
            nuts_warmup=0,
            nuts_samples=0,
            nuts_thinning=1,
            nuts_tree_depth=4,
        )

    @classmethod
    def full(cls) -> "Task02CConfig":
        return cls(
            profile="full",
            seeds=(0, 1, 2),
            counts=(16, 40),
            dimension=10,
            initial_count=16,
            final_count=40,
            candidate_count=2048,
            particle_counts=(8, 16, 32),
            structural_budgets=(32, 64),
            repeats=3,
            initialization_steps=16,
            initialization_reach_steps=12,
            structural_block=4,
            design_steps_per_block=4,
            prior_whitening_draws=4096,
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
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def matched_factorization_budget(
    particle_count: int,
    initialization_steps: int,
    structural_budget: int,
    structural_block: int,
) -> dict[str, int]:
    """Budget charged identically to each branch of a paired comparison."""

    if min(particle_count, initialization_steps, structural_budget, structural_block) <= 0:
        raise ValueError("all budget terms must be positive")
    if structural_budget % structural_block:
        raise ValueError("structural budget must be divisible by its block size")
    cache_builds = structural_budget // structural_block
    post = structural_budget + cache_builds
    return {
        "design_cache_builds": cache_builds,
        "post_initialization_per_particle": post,
        "total_per_particle": initialization_steps + post,
        "total": particle_count * (initialization_steps + post),
    }


def _case_data(seed: int, count: int, candidate_path: Path | None = None) -> dict[str, Any]:
    train_x = deterministic_benchmark_inputs(40, 10, seed)
    raw_y = negative_branin(train_x)
    transform = FrozenOutputTransform.fit(raw_y[:16])
    train_y = transform.transform(raw_y)
    if candidate_path is not None and candidate_path.exists():
        candidates = torch.from_numpy(np.load(candidate_path)["candidate_x"]).double()
    else:
        candidates = deterministic_benchmark_inputs(2048, 10, seed + 2_000)
    return {
        "train_x": train_x[:count],
        "train_y": train_y[:count],
        "candidates": candidates,
        "transform": transform,
    }


def _load_saved_particles(path: Path) -> tuple[SaasParticles, torch.Tensor]:
    values = np.load(path)
    return (
        SaasParticles(
            torch.from_numpy(values["lengthscales"]),
            torch.from_numpy(values["means"]),
            torch.from_numpy(values["outputscales"]),
        ),
        torch.from_numpy(values["candidate_x"]).double(),
    )


def _preflight_pass(rows: list[dict[str, Any]], potential: dict[str, float]) -> bool:
    return (
        max(float(row["autodiff_envelope_max_abs"]) for row in rows) <= 1e-9
        and max(float(row["finite_difference_max_abs"]) for row in rows) <= 1e-5
        and max(float(row["torch_value_abs"]) for row in rows) <= 1e-9
        and max(float(row["torch_gradient_max_abs"]) for row in rows) <= 1e-8
        and potential["centered_value_max_abs"] <= 1e-8
        and potential["gradient_max_abs"] <= 1e-7
    )


def run_saved_teacher_preflight(
    signature_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Run all six early/late tilt checks without fitting NUTS."""

    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    potential_validation: dict[str, float] | None = None
    point_metadata: list[dict[str, Any]] = []
    for seed in (0, 1, 2):
        for count in (16, 40):
            signature_path = signature_dir / f"seed{seed}_n{count}.npz"
            if not signature_path.exists():
                raise FileNotFoundError(
                    f"missing Task 02B teacher signatures: {signature_path}"
                )
            particles, archived_candidates = _load_saved_particles(signature_path)
            data = _case_data(seed, count, signature_path)
            train_x, train_y = data["train_x"], data["train_y"]
            mixture = operational_mixture_from_samples(train_x, train_y, 1e-4, particles)
            state = ExactGPBatchState.build(particles, train_x, train_y, 1e-4)
            map_x, map_info = map_saas_initial_design(
                train_x,
                train_y,
                1e-4,
                archived_candidates,
                seed=10_000 * seed + count,
            )
            extra = SobolEngine(10, scramble=True, seed=seed + 30_000).draw(4096).double().numpy()
            teacher_x, teacher_ei, teacher_info = continuous_teacher_optimum(
                mixture,
                float(train_y.max()),
                archived_candidates.numpy(),
                extra_candidates=extra,
                map_design=map_x,
            )
            midpoint = 0.5 * (map_x + teacher_x)
            for label, point in (
                ("map", map_x),
                ("teacher", teacher_x),
                ("midpoint", midpoint),
            ):
                for row in run_point_preflight(
                    mixture, particles, state, point, float(train_y.max()), label
                ):
                    rows.append({"seed": seed, "count": count, **row})
            point_metadata.append(
                {
                    "seed": seed,
                    "count": count,
                    "map": map_info,
                    "teacher": teacher_info,
                    "teacher_ei": teacher_ei,
                    "map_x": map_x.tolist(),
                    "teacher_x": teacher_x.tolist(),
                }
            )
            if potential_validation is None:
                potential = SaasUnconstrainedPotential.build(train_x, train_y, 1e-4, seed=71)
                validation_vectors = potential.initialization_vectors(3, seed=91)
                potential_validation = potential.validate_fused(validation_vectors, map_x)
    assert potential_validation is not None
    passed = _preflight_pass(rows, potential_validation)
    _write_csv(output_dir / "task02c_tilt_preflight.csv", rows)
    payload = {
        "passed": passed,
        "thresholds": {
            "autodiff_envelope_max_abs": 1e-9,
            "finite_difference_max_abs": 1e-5,
            "torch_value_max_abs": 1e-9,
            "torch_gradient_max_abs": 1e-8,
            "fused_centered_value_max_abs": 1e-8,
            "fused_gradient_max_abs": 1e-7,
        },
        "potential_validation": potential_validation,
        "points": point_metadata,
        "maxima": {
            key: max(float(row[key]) for row in rows)
            for key in (
                "autodiff_envelope_max_abs",
                "finite_difference_max_abs",
                "torch_value_abs",
                "torch_gradient_max_abs",
            )
        },
        "beta1_ess_fraction_min": min(
            float(row["ess_fraction"]) for row in rows if float(row["beta"]) == 1.0
        ),
        "beta1_ess_fraction_max": max(
            float(row["ess_fraction"]) for row in rows if float(row["beta"]) == 1.0
        ),
    }
    (output_dir / "task02c_preflight.json").write_text(json.dumps(payload, indent=2) + "\n")
    _plot_preflight(output_dir / "tilt_ess_path.png", rows)
    return payload


def _plot_preflight(path: Path, rows: list[dict[str, Any]]) -> None:
    figure, axis = plt.subplots(figsize=(6.4, 4.0))
    for key in sorted({(r["seed"], r["count"], r["point"]) for r in rows}):
        selected = [
            row
            for row in rows
            if (row["seed"], row["count"], row["point"]) == key
        ]
        axis.plot(
            [row["beta"] for row in selected],
            [row["ess_fraction"] for row in selected],
            alpha=0.65,
            label=f"s{key[0]} n{key[1]} {key[2]}",
        )
    axis.axhline(0.1, color="black", linestyle="--", linewidth=0.8)
    axis.set(xlabel=r"decision tilt $\beta$", ylabel="teacher tilt ESS / P", ylim=(0, 1.02))
    axis.legend(fontsize=6, ncol=2)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _teacher_from_eta(potential: SaasUnconstrainedPotential, eta: jax.Array) -> OperationalMixture:
    values = potential.operational(eta)
    return OperationalMixture.build(
        potential.train_x,
        potential.train_y,
        potential.noise_variance,
        values["lengthscales"],
        values["means"],
        values["outputscales"],
    )


def _final_particle_diagnostics(
    eta: jax.Array,
    potential: SaasUnconstrainedPotential,
    teacher: OperationalMixture,
    candidates: torch.Tensor,
) -> dict[str, float]:
    """Posterior and acquisition diversity diagnostics, excluded from method budget."""

    operational = potential.operational(eta)
    method_features = np.concatenate(
        (
            np.log(np.asarray(operational["lengthscales"])),
            np.asarray(operational["means"])[:, None],
            np.log(np.asarray(operational["outputscales"]))[:, None],
        ),
        axis=1,
    )
    teacher_features = np.concatenate(
        (
            np.log(np.asarray(teacher.lengthscales)),
            np.asarray(teacher.means)[:, None],
            np.log(np.asarray(teacher.outputscales))[:, None],
        ),
        axis=1,
    )
    pooled = np.concatenate((method_features, teacher_features), axis=0)
    scale = np.maximum(np.std(pooled, axis=0, ddof=0), 1e-12)
    method_standard = method_features / scale
    teacher_standard = teacher_features / scale
    pooled_standard = np.concatenate((method_standard, teacher_standard), axis=0)
    distances = pdist(pooled_standard)
    bandwidth = max(float(np.median(distances**2)), 1e-12)
    method_kernel = np.exp(-cdist(method_standard, method_standard, "sqeuclidean") / bandwidth)
    teacher_kernel = np.exp(-cdist(teacher_standard, teacher_standard, "sqeuclidean") / bandwidth)
    cross_kernel = np.exp(-cdist(method_standard, teacher_standard, "sqeuclidean") / bandwidth)
    mmd_squared = max(
        0.0,
        float(method_kernel.mean() + teacher_kernel.mean() - 2.0 * cross_kernel.mean()),
    )
    lengthscale_w1 = [
        wasserstein_distance(
            method_features[:, dimension], teacher_features[:, dimension]
        )
        for dimension in range(potential.dimension)
    ]
    method_mixture = _teacher_from_eta(potential, eta)
    diagnostic_candidates = jnp.asarray(candidates[: min(256, len(candidates))].numpy())
    signatures = jax.vmap(
        lambda point: jnp.exp(
            method_mixture.particle_log_ei(point, potential.best_f)
        )
    )(diagnostic_candidates).T
    spectrum = signature_spectrum(torch.from_numpy(np.asarray(signatures).copy()))
    return {
        "posterior_mmd": math.sqrt(mmd_squared),
        "mean_log_lengthscale_w1": float(np.mean(lengthscale_w1)),
        "max_log_lengthscale_w1": float(np.max(lengthscale_w1)),
        "active_log_lengthscale_w1": float(np.mean(lengthscale_w1[:2])),
        "inactive_log_lengthscale_w1": float(np.mean(lengthscale_w1[2:])),
        "acquisition_signature_entropy_rank": float(
            spectrum["entropy_effective_rank"]
        ),
        "diagnostic_factorizations": int(eta.shape[0]),
    }


def _cached_design_objective(
    train_x: jax.Array,
    lengthscales: jax.Array,
    means: jax.Array,
    outputscales: jax.Array,
    chol: jax.Array,
    alpha: jax.Array,
    x: jax.Array,
    best_f: float,
    beta: float,
    posterior_focused: bool,
) -> jax.Array:
    log_ei = cached_particle_log_ei(
        train_x, lengthscales, means, outputscales, chol, alpha, x, best_f
    )
    if posterior_focused:
        return jax.scipy.special.logsumexp(log_ei) - jnp.log(log_ei.size)
    # DT particles already approximate q_{x,beta}; applying a second beta
    # reweighting here would target the wrong distribution.  The envelope update
    # is the unweighted particle average of LogEI gradients under that tilt.
    del beta
    return jnp.mean(log_ei)


_cached_design_value_grad = jax.jit(
    jax.value_and_grad(_cached_design_objective, argnums=6),
    static_argnums=9,
)


def _svgd_update(
    eta: jax.Array,
    whitening: Whitening,
    score_function,
    adam: AdamState,
    *,
    learning_rate: float = 0.01,
) -> tuple[jax.Array, AdamState, dict[str, float]]:
    white = whitening.whiten(eta)
    scores_eta = score_function(eta)
    scores_white = scores_eta * whitening.scale
    direction, raw = svgd_direction(white, scores_white)
    for retry in range(4):
        proposed_white, proposed_adam, adam_info = adam_ascent(
            white, direction, adam, learning_rate=learning_rate / (2**retry)
        )
        proposed = whitening.unwhiten(proposed_white)
        if bool(jnp.all(jnp.isfinite(proposed))):
            pairwise = jnp.sqrt(
                jnp.maximum(
                    jnp.sum(
                        (proposed_white[:, None] - proposed_white[None, :]) ** 2,
                        axis=-1,
                    ),
                    0.0,
                )
            )
            mask = ~jnp.eye(eta.shape[0], dtype=bool)
            centered = proposed_white - jnp.mean(proposed_white, axis=0, keepdims=True)
            singular = jnp.linalg.svd(centered, compute_uv=False)
            squared = singular**2
            proportions = squared / jnp.maximum(jnp.sum(squared), 1e-30)
            positive = jnp.maximum(proportions, 1e-30)
            effective_rank = jnp.exp(-jnp.sum(proportions * jnp.log(positive)))
            off_diagonal = pairwise[mask]
            return proposed, proposed_adam, {
                **{key: float(value) for key, value in raw.items()},
                **adam_info,
                "backtracks": retry,
                "score_rms": float(jnp.sqrt(jnp.mean(scores_white**2))),
                "median_pairwise_distance": float(jnp.median(off_diagonal)),
                "minimum_pairwise_distance": float(jnp.min(off_diagonal)),
                "duplicate_pair_fraction": float(jnp.mean(off_diagonal < 1e-8)),
                "particle_covariance_effective_rank": float(effective_rank),
                "bandwidth_floor_hit": bool(float(raw["bandwidth"]) <= 1e-3 * (1.0 + 1e-12)),
                "repulsion_attraction_ratio": float(
                    raw["repulsion_norm"] / jnp.maximum(raw["attraction_norm"], 1e-30)
                ),
            }
    raise FloatingPointError("SVGD proposal remained nonfinite after three halvings")


def _common_initialization(
    potential: SaasUnconstrainedPotential,
    config: Task02CConfig,
    particle_count: int,
    seed: int,
) -> tuple[jax.Array, Whitening, AdamState, list[dict[str, Any]]]:
    prior = potential.prior_vectors(config.prior_whitening_draws, seed + 1)
    whitening = Whitening.fit(prior)
    candidates = potential.initialization_vectors(4 * particle_count, seed + 10_000)
    indices = maximin_subset(whitening.whiten(candidates), particle_count)
    eta = candidates[indices]
    adam = AdamState.zeros(eta)
    lam = 0.0
    rows: list[dict[str, Any]] = []

    def nll_one(value: jax.Array) -> jax.Array:
        nll, _ = potential.nll_and_log_ei(value, jnp.zeros(potential.dimension))
        return nll

    nll_batch = jax.jit(jax.vmap(nll_one))
    score = jax.jit(
        jax.vmap(jax.grad(lambda value, power: -potential.prior_potential(value) - power * nll_one(value)), in_axes=(0, None))
    )
    for step in range(config.initialization_steps):
        remaining = max(config.initialization_reach_steps - step, 1)
        required = max(0.0, (1.0 - lam) / remaining) if step < config.initialization_reach_steps else 0.0
        nll = nll_batch(eta)
        increment, cess, forced = choose_tempering_increment(
            -nll,
            lam,
            target_cess=0.8,
            maximum_increment=0.25,
            required_minimum=required,
        )
        lam = min(1.0, lam + increment)
        eta, adam, diagnostics = _svgd_update(
            eta,
            whitening,
            lambda values, power=lam: score(values, power),
            adam,
        )
        rows.append(
            {
                "phase": "initialization",
                "step": step + 1,
                "likelihood_power": lam,
                "conditional_ess_fraction": cess,
                "forced_progress": forced,
                **diagnostics,
            }
        )
    if abs(lam - 1.0) > 1e-12:
        raise RuntimeError("common initialization did not reach the posterior")
    return eta, whitening, adam, rows


def _design_updates(
    eta: jax.Array,
    potential: SaasUnconstrainedPotential,
    x: jax.Array,
    state: AdamState,
    *,
    method: str,
    beta: float,
    steps: int,
) -> tuple[jax.Array, AdamState, list[dict[str, Any]]]:
    mixture = _teacher_from_eta(potential, eta)
    arguments = (
        mixture.train_x,
        mixture.lengthscales,
        mixture.means,
        mixture.outputscales,
        mixture.chol,
        mixture.alpha,
    )
    posterior_focused = method == "posterior"
    rows: list[dict[str, Any]] = []
    for index in range(steps):
        old_value, gradient = _cached_design_value_grad(
            *arguments, x, potential.best_f, beta, posterior_focused
        )
        proposed_batch, proposed_state, info = adam_ascent(
            x[None, :], gradient[None, :], state, coordinate_clip=0.02
        )
        raw_proposal = jnp.clip(proposed_batch[0], 0.0, 1.0)
        accepted = False
        cess = 1.0
        proposal = x
        for retry in range(4):
            candidate = x + (raw_proposal - x) / (2**retry)
            if method == "decision" and beta > 0:
                old_logei = mixture.particle_log_ei(x, potential.best_f)
                new_logei = mixture.particle_log_ei(candidate, potential.best_f)
                cess = float(design_retilt_cess(old_logei, new_logei, beta))
                if cess < 0.9:
                    continue
            proposal = candidate
            accepted = True
            info["backtracks"] = retry
            break
        if accepted:
            x, state = proposal, proposed_state
        else:
            info["backtracks"] = 4
        new_value = float(
            _cached_design_objective(
                *arguments, x, potential.best_f, beta, posterior_focused
            )
        )
        rows.append(
            {
                "design_step": index + 1,
                "accepted": accepted,
                "conditional_ess_fraction": cess,
                "objective_before": float(old_value),
                "objective_after": new_value,
                **info,
            }
        )
    return x, state, rows


def _run_method(
    *,
    method: str,
    initial_eta: jax.Array,
    initial_adam: AdamState,
    whitening: Whitening,
    initial_x: np.ndarray,
    potential: SaasUnconstrainedPotential,
    teacher: OperationalMixture,
    teacher_x: np.ndarray,
    teacher_ei: float,
    config: Task02CConfig,
    structural_budget: int,
    diagnostic_candidates: torch.Tensor,
) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray]:
    eta = jnp.array(initial_eta)
    adam = initial_adam.copy()
    x_adam = AdamState.zeros(jnp.asarray(initial_x)[None, :])
    x = jnp.asarray(initial_x, dtype=jnp.float64)
    beta = 0.0
    rows: list[dict[str, Any]] = []
    structural_step = 0

    score_at = jax.jit(
        jax.vmap(
            jax.grad(
                lambda z, point, power: -potential.fused_potential(z, point, power),
                argnums=0,
            ),
            in_axes=(0, None, None),
        )
    )

    for block in range(structural_budget // config.structural_block):
        x, x_adam, design_rows = _design_updates(
            eta,
            potential,
            x,
            x_adam,
            method=method,
            beta=beta,
            steps=config.design_steps_per_block,
        )
        for row in design_rows:
            rows.append(
                {
                    "method": method,
                    "block": block + 1,
                    "event": "design",
                    "beta": beta,
                    **row,
                }
            )
        for _ in range(config.structural_block):
            structural_step += 1
            forced = False
            beta_cess = 1.0
            if method == "decision" and structural_step <= structural_budget // 2:
                log_ei = jax.vmap(lambda z: potential.nll_and_log_ei(z, x)[1])(eta)
                remaining = structural_budget // 2 - structural_step + 1
                required = max(0.0, (1.0 - beta) / remaining)
                increment, beta_cess, forced = choose_tempering_increment(
                    log_ei,
                    beta,
                    maximum_increment=0.25,
                    target_cess=0.8,
                    required_minimum=required,
                )
                beta = min(1.0, beta + increment)
            target_beta = beta if method == "decision" else 0.0
            eta, adam, diagnostics = _svgd_update(
                eta,
                whitening,
                lambda values, point=x, power=target_beta: score_at(
                    values, point, power
                ),
                adam,
            )
            rows.append(
                {
                    "method": method,
                    "block": block + 1,
                    "event": "structural",
                    "structural_step": structural_step,
                    "beta": beta,
                    "beta_conditional_ess_fraction": beta_cess,
                    "forced_beta_progress": forced,
                    **diagnostics,
                }
            )
    if method == "decision" and abs(beta - 1.0) > 1e-12:
        raise RuntimeError("decision transport did not reach beta=1")
    method_log_ei = float(teacher.log_integrated_ei(x, potential.best_f))
    method_ei = math.exp(method_log_ei)
    absolute_regret = max(0.0, teacher_ei - method_ei)
    budget = matched_factorization_budget(
        int(eta.shape[0]),
        config.initialization_steps,
        structural_budget,
        config.structural_block,
    )
    cache_builds = budget["design_cache_builds"]
    particle_diagnostics = _final_particle_diagnostics(
        eta, potential, teacher, diagnostic_candidates
    )
    return {
        "method": method,
        "final_x": np.asarray(x).tolist(),
        "teacher_ei_at_method": method_ei,
        "teacher_optimum_ei": teacher_ei,
        "absolute_regret": absolute_regret,
        "normalized_regret": absolute_regret / max(abs(teacher_ei), 1e-15),
        "distance_to_teacher": float(np.linalg.norm(np.asarray(x) - teacher_x)),
        "final_beta": beta,
        "structural_steps": structural_budget,
        "design_attempts": cache_builds * config.design_steps_per_block,
        "design_cache_builds": cache_builds,
        "post_initialization_factorizations_per_particle": budget[
            "post_initialization_per_particle"
        ],
        "factorization_equivalents_per_particle": budget["total_per_particle"],
        "factorization_equivalents": budget["total"],
        **particle_diagnostics,
    }, rows, np.asarray(eta)


def run_smoke_transport(
    signature_dir: Path,
    preflight_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """One D=10 wiring smoke, using saved NUTS particles only for evaluation."""

    preflight = json.loads(preflight_path.read_text())
    if not preflight.get("passed", False):
        raise RuntimeError("Task 02C transport is blocked by a failed preflight")
    output_dir.mkdir(parents=True, exist_ok=True)
    config = Task02CConfig.smoke()
    signature_path = signature_dir / "seed0_n16.npz"
    teacher_particles, candidates = _load_saved_particles(signature_path)
    data = _case_data(0, 16, signature_path)
    train_x, train_y = data["train_x"], data["train_y"]
    teacher = operational_mixture_from_samples(
        train_x, train_y, config.noise_variance, teacher_particles
    )
    map_x, map_info = map_saas_initial_design(
        train_x, train_y, config.noise_variance, candidates, seed=16
    )
    extra = SobolEngine(10, scramble=True, seed=30_000).draw(4096).double().numpy()
    teacher_x, teacher_ei, teacher_info = continuous_teacher_optimum(
        teacher,
        float(train_y.max()),
        candidates.numpy(),
        extra_candidates=extra,
        map_design=map_x,
    )
    potential = SaasUnconstrainedPotential.build(
        train_x, train_y, config.noise_variance, seed=501
    )
    start = time.perf_counter()
    initial_eta, whitening, initial_adam, trace = _common_initialization(
        potential, config, 8, 700
    )
    results: list[dict[str, Any]] = []
    for method in ("posterior", "decision"):
        result, method_trace, _ = _run_method(
            method=method,
            initial_eta=initial_eta,
            initial_adam=initial_adam,
            whitening=whitening,
            initial_x=map_x,
            potential=potential,
            teacher=teacher,
            teacher_x=teacher_x,
            teacher_ei=teacher_ei,
            config=config,
            structural_budget=8,
            diagnostic_candidates=candidates,
        )
        results.append(result)
        trace.extend(method_trace)
    elapsed = time.perf_counter() - start
    payload = {
        "configuration": asdict(config),
        "environment": {
            "python": platform.python_version(),
            "jax": jax.__version__,
            "jax_x64": bool(jax.config.jax_enable_x64),
            "jax_backend": jax.default_backend(),
            "jax_devices": [str(device) for device in jax.devices()],
        },
        "map": map_info,
        "teacher": teacher_info | {"x": teacher_x.tolist(), "ei": teacher_ei},
        "results": results,
        "elapsed_seconds": elapsed,
        "scientific_interpretation": "wiring smoke only; no Task 02C GO/NO-GO inference",
    }
    (output_dir / "task02c_smoke.json").write_text(json.dumps(payload, indent=2) + "\n")
    _write_csv(output_dir / "task02c_trace.csv", trace)
    _write_csv(output_dir / "task02c_methods.csv", results)
    _plot_smoke(output_dir / "matched_regret_smoke.png", results)
    return payload


def _plot_smoke(path: Path, rows: list[dict[str, Any]]) -> None:
    figure, axis = plt.subplots(figsize=(5.0, 3.7))
    axis.bar(
        [row["method"] for row in rows],
        [row["normalized_regret"] for row in rows],
    )
    axis.set(ylabel="normalized teacher decision regret", title="Task 02C wiring smoke")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_full(
    output_dir: Path,
    results: list[dict[str, Any]],
    traces: list[dict[str, Any]],
    preflight_rows: list[dict[str, Any]],
) -> None:
    transport = [row for row in results if row["method"] in {"posterior", "decision"}]
    figure, axes = plt.subplots(1, 2, figsize=(10.0, 3.8))
    for method in ("posterior", "decision"):
        for budget, linestyle in ((32, "-"), (64, "--")):
            rows = [
                row
                for row in transport
                if row["method"] == method and int(row["budget"]) == budget
            ]
            particle_counts = sorted({int(row["particles"]) for row in rows})
            axes[0].plot(
                particle_counts,
                [
                    max(1e-8, np.mean(
                        [
                            float(row["normalized_regret"])
                            for row in rows
                            if int(row["particles"]) == count
                        ]
                    ))
                    for count in particle_counts
                ],
                marker="o",
                linestyle=linestyle,
                label=f"{method}, B={budget}",
            )
    axes[0].set(
        xlabel="structural particles K",
        ylabel="mean normalized teacher regret",
        yscale="log",
    )
    axes[0].legend(fontsize=7)
    timing_methods = ("posterior", "decision")
    axes[1].bar(
        timing_methods,
        [
            np.median(
                [float(row["elapsed_seconds"]) for row in transport if row["method"] == method]
            )
            for method in timing_methods
        ],
    )
    axes[1].set(ylabel="median method wall time (seconds)")
    figure.tight_layout()
    figure.savefig(output_dir / "matched_regret_and_timing.png", dpi=160)
    plt.close(figure)

    structural = [row for row in traces if row.get("event") == "structural"]
    figure, axes = plt.subplots(1, 2, figsize=(10.0, 3.8))
    for method in ("posterior", "decision"):
        selected = [row for row in structural if row["method"] == method]
        grouped: dict[int, list[float]] = {}
        for row in selected:
            grouped.setdefault(int(row["structural_step"]), []).append(
                float(row["median_pairwise_distance"])
            )
        axes[0].plot(
            sorted(grouped),
            [np.median(grouped[step]) for step in sorted(grouped)],
            label=method,
        )
        grouped_score: dict[int, list[float]] = {}
        for row in selected:
            grouped_score.setdefault(int(row["structural_step"]), []).append(
                float(row["score_rms"])
            )
        axes[1].plot(
            sorted(grouped_score),
            [np.median(grouped_score[step]) for step in sorted(grouped_score)],
            label=method,
        )
    axes[0].set(xlabel="structural step", ylabel="median whitened pair distance")
    axes[1].set(xlabel="structural step", ylabel="median whitened score RMS", yscale="log")
    for axis in axes:
        axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "particle_geometry.png", dpi=160)
    plt.close(figure)
    _plot_preflight(output_dir / "fresh_teacher_tilt_ess.png", preflight_rows)


def _write_full_summary(
    path: Path,
    results: list[dict[str, Any]],
    preflight_rows: list[dict[str, Any]],
    potential_validations: list[dict[str, float]],
    traces: list[dict[str, Any]],
    case_metadata: list[dict[str, Any]],
) -> None:
    transport = [row for row in results if row["method"] in {"posterior", "decision"}]
    posterior = [row for row in transport if row["method"] == "posterior"]
    decision = [row for row in transport if row["method"] == "decision"]
    pairs = {
        (row["seed"], row["count"], row["particles"], row["budget"], row["repeat"]): row
        for row in posterior
    }
    improvements = [
        float(pairs[(row["seed"], row["count"], row["particles"], row["budget"], row["repeat"])]["normalized_regret"])
        - float(row["normalized_regret"])
        for row in decision
    ]
    small = [row for row in decision if int(row["particles"]) <= 16]
    dt_better = float(np.mean(np.asarray(improvements) > 0.0))
    median_p = float(np.median([row["normalized_regret"] for row in posterior]))
    median_dt = float(np.median([row["normalized_regret"] for row in decision]))
    below_five = float(np.mean([row["normalized_regret"] < 0.05 for row in small]))
    forced = [
        row
        for row in preflight_rows
        if float(row["beta"]) == 1.0 and float(row["ess_fraction"]) < 0.1
    ]
    structural = [row for row in traces if row.get("event") == "structural"]
    final_keys: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in structural:
        key = (
            row["seed"],
            row["count"],
            row["particles"],
            row["budget"],
            row["repeat"],
            row["method"],
        )
        if key not in final_keys or int(row["structural_step"]) > int(final_keys[key]["structural_step"]):
            final_keys[key] = row
    final_structural = list(final_keys.values())
    collapse_fraction = float(
        np.mean(
            [
                float(row["median_pairwise_distance"]) < 0.1
                or float(row["particle_covariance_effective_rank"]) < 2.0
                for row in final_structural
            ]
        )
    )
    beta_forced_fraction = float(
        np.mean(
            [
                bool(row["forced_beta_progress"])
                for row in structural
                if row["method"] == "decision"
            ]
        )
    )
    bandwidth_floor_fraction = float(
        np.mean([bool(row["bandwidth_floor_hit"]) for row in structural])
    )
    median_dt_seconds = float(np.median([row["elapsed_seconds"] for row in decision]))
    median_nuts_seconds = float(
        np.median([row["teacher_nuts_seconds"] for row in case_metadata])
    )
    if (
        dt_better >= 2 / 3
        and median_dt < median_p
        and below_five >= 0.5
        and collapse_fraction < 0.25
        and median_dt_seconds <= median_nuts_seconds
    ):
        recommendation = "STRONG GO"
        reason = "decision tilting won the matched comparison with healthy particles and competitive runtime"
    elif dt_better >= 2 / 3 and median_dt < median_p and below_five >= 0.5:
        recommendation = "QUALIFIED GO"
        reason = "decision tilting improved the matched comparison and often met the small-K decision target"
    elif collapse_fraction >= 0.25:
        recommendation = "TRANSPORT PIVOT / OPTIMIZATION"
        reason = "particle geometry failed often enough that the bounded SVGD result does not isolate the energy target"
    else:
        recommendation = "NO-GO"
        reason = "decision tilting did not consistently outperform posterior-focused SVGD at matched compute"
    text = f"""# Task 02C — full decision-tilted SVGD result

This is a q=1, D=10 embedded-Branin falsification study. It is not an end-to-end BO
benchmark and contains no Vecchia, residual-output EBM, q>1, molecular optimization,
or post-Task-02C method.

## Eight completion questions

1. **Did the teacher preflight pass?** Yes. The largest envelope autodiff and finite-difference errors were `{max(float(r['autodiff_envelope_max_abs']) for r in preflight_rows):.3e}` and `{max(float(r['finite_difference_max_abs']) for r in preflight_rows):.3e}`.
2. **Was the exact SAAS energy reproduced?** Yes. The largest centered-value and unconstrained-gradient errors were `{max(r['centered_value_max_abs'] for r in potential_validations):.3e}` and `{max(r['gradient_max_abs'] for r in potential_validations):.3e}`.
3. **Was the decision tilt intrinsically degenerate?** `{len(forced)}` beta-one teacher checks had ESS/P below 0.1; the observed range was `{min(float(r['ess_fraction']) for r in preflight_rows if float(r['beta']) == 1):.3f}`–`{max(float(r['ess_fraction']) for r in preflight_rows if float(r['beta']) == 1):.3f}`.
4. **Was expensive compute matched?** Yes. Every posterior/decision pair used identical initial particles, K, initialization steps, structural steps, cache builds, and design attempts; recorded factorization-equivalent counts match pairwise.
5. **Did DT-SVGD improve decision regret?** It had lower regret in `{dt_better:.1%}` of paired runs. Median normalized regret was `{median_p:.4g}` for P-SVGD and `{median_dt:.4g}` for DT-SVGD.
6. **Were K=8–16 particles sufficient?** `{below_five:.1%}` of DT-SVGD runs with K<=16 achieved below 5% normalized regret. K=8 remains a stress test.
7. **What is the compute and geometry interpretation?** Median charged DT-SVGD time was `{median_dt_seconds:.3g}` s versus `{median_nuts_seconds:.3g}` s for fresh NUTS. Final-particle collapse occurred in `{collapse_fraction:.1%}` of runs, forced beta progress in `{beta_forced_fraction:.1%}` of DT structural steps, and the bandwidth floor in `{bandwidth_floor_fraction:.1%}` of all structural steps. Factorization-equivalents are recorded in `task02c_methods.csv`; shared initialization is charged equally.
8. **Should the structural decision-energy program continue?** **{recommendation}.** {reason}.

Task 02B Frank–Wolfe K=8/K=16 results remain an unattainable oracle compression
ceiling and were never used for initialization.
"""
    path.write_text(text)


def run_full_transport(output_dir: Path, signature_dir: Path) -> None:
    """GPU-preferred six-case study with fresh x64 NUTS teachers."""

    config = Task02CConfig.full()
    output_dir.mkdir(parents=True, exist_ok=True)
    all_results: list[dict[str, Any]] = []
    all_traces: list[dict[str, Any]] = []
    full_preflight_rows: list[dict[str, Any]] = []
    potential_validations: list[dict[str, float]] = []
    case_metadata: list[dict[str, Any]] = []
    for seed in config.seeds:
        for count in config.counts:
            archived_path = signature_dir / f"seed{seed}_n{count}.npz"
            data = _case_data(seed, count, archived_path if archived_path.exists() else None)
            train_x, train_y, candidates = (
                data["train_x"],
                data["train_y"],
                data["candidates"],
            )
            potential = SaasUnconstrainedPotential.build(
                train_x, train_y, config.noise_variance, seed=10_000 * seed + count
            )
            teacher_path = output_dir / "teachers" / f"seed{seed}_n{count}.npz"
            if teacher_path.exists():
                saved = np.load(teacher_path)
                metadata = json.loads(teacher_path.with_suffix(".json").read_text())
                teacher_samples = TeacherSamples(
                    unconstrained=saved["unconstrained"],
                    lengthscales=saved["lengthscales"],
                    means=saved["means"],
                    outputscales=saved["outputscales"],
                    elapsed_seconds=float(metadata["elapsed_seconds"]),
                    metadata=metadata,
                )
            else:
                teacher_samples = fit_fresh_teacher(
                    potential,
                    warmup_steps=config.nuts_warmup,
                    num_samples=config.nuts_samples,
                    thinning=config.nuts_thinning,
                    max_tree_depth=config.nuts_tree_depth,
                    seed=10_000 * seed + count,
                    progress_bar=True,
                )
                teacher_samples.save(teacher_path)
            teacher = operational_mixture_from_samples(
                train_x, train_y, config.noise_variance, teacher_samples
            )
            map_x, map_info = map_saas_initial_design(
                train_x,
                train_y,
                config.noise_variance,
                candidates,
                seed=10_000 * seed + count,
            )
            extra = SobolEngine(10, scramble=True, seed=seed + 30_000).draw(4096).double().numpy()
            continuous_map_x = np.asarray(map_info["continuous_x"], dtype=np.float64)
            teacher_x, teacher_ei, _ = continuous_teacher_optimum(
                teacher,
                float(train_y.max()),
                candidates.numpy(),
                extra_candidates=extra,
                map_design=continuous_map_x,
            )
            validation = potential.validate_fused(
                jnp.asarray(teacher_samples.unconstrained[:8]), map_x
            )
            potential_validations.append(validation)
            teacher_particles = SaasParticles(
                torch.from_numpy(teacher_samples.lengthscales),
                torch.from_numpy(teacher_samples.means),
                torch.from_numpy(teacher_samples.outputscales),
            )
            teacher_state = ExactGPBatchState.build(
                teacher_particles, train_x, train_y, config.noise_variance
            )
            for point_label, point in (
                ("map", map_x),
                ("teacher", teacher_x),
                ("midpoint", 0.5 * (map_x + teacher_x)),
            ):
                for row in run_point_preflight(
                    teacher,
                    teacher_particles,
                    teacher_state,
                    point,
                    float(train_y.max()),
                    point_label,
                ):
                    full_preflight_rows.append(
                        {"seed": seed, "count": count, **row}
                    )
            case_rows = [
                row
                for row in full_preflight_rows
                if row["seed"] == seed and row["count"] == count
            ]
            if not _preflight_pass(case_rows, validation):
                raise RuntimeError(
                    f"fresh-teacher Task 02C preflight failed for seed={seed}, n={count}"
                )
            archived_max = archived_rmse = None
            if archived_path.exists():
                archived = np.load(archived_path)
                fresh_curve = np.exp(
                    np.asarray(
                        jax.vmap(
                            lambda point: teacher.log_integrated_ei(
                                point, float(train_y.max())
                            )
                        )(jnp.asarray(candidates.numpy()))
                    )
                )
                archived_curve = archived["teacher_ei"]
                archived_max = float(np.max(np.abs(fresh_curve - archived_curve)))
                archived_rmse = float(np.sqrt(np.mean((fresh_curve - archived_curve) ** 2)))
            map_teacher_ei = math.exp(
                float(
                    teacher.log_integrated_ei(
                        jnp.asarray(continuous_map_x), float(train_y.max())
                    )
                )
            )
            map_regret = max(0.0, teacher_ei - map_teacher_ei)
            all_results.append(
                {
                    "seed": seed,
                    "count": count,
                    "method": "map",
                    "particles": 1,
                    "budget": 0,
                    "repeat": 0,
                    "teacher_ei_at_method": map_teacher_ei,
                    "teacher_optimum_ei": teacher_ei,
                    "absolute_regret": map_regret,
                    "normalized_regret": map_regret / max(teacher_ei, 1e-15),
                    "distance_to_teacher": float(
                        np.linalg.norm(continuous_map_x - teacher_x)
                    ),
                    "final_x": continuous_map_x.tolist(),
                }
            )
            case_metadata.append(
                {
                    "seed": seed,
                    "count": count,
                    "teacher_nuts_seconds": teacher_samples.elapsed_seconds,
                    "archived_teacher_max_abs_error": archived_max,
                    "archived_teacher_rmse": archived_rmse,
                    "potential_validation": validation,
                }
            )
            _write_csv(output_dir / "task02c_teacher_preflight.csv", full_preflight_rows)
            initialization_cache: dict[
                tuple[int, int], tuple[jax.Array, Whitening, AdamState, float]
            ] = {}
            for particle_count in config.particle_counts:
                for budget in config.structural_budgets:
                    for repeat in range(config.repeats):
                        run_seed = 1_000_000 * seed + 10_000 * count + 100 * particle_count + repeat
                        initialization_key = (particle_count, repeat)
                        if initialization_key not in initialization_cache:
                            initialization_started = time.perf_counter()
                            initial_eta, whitening, initial_adam, init_trace = _common_initialization(
                                potential, config, particle_count, run_seed
                            )
                            initialization_seconds = time.perf_counter() - initialization_started
                            initialization_cache[initialization_key] = (
                                initial_eta,
                                whitening,
                                initial_adam,
                                initialization_seconds,
                            )
                            for row in init_trace:
                                all_traces.append(
                                    {
                                        "seed": seed,
                                        "count": count,
                                        "particles": particle_count,
                                        "budget": "shared",
                                        "repeat": repeat,
                                        **row,
                                    }
                                )
                        initial_eta, whitening, initial_adam, initialization_seconds = initialization_cache[
                            initialization_key
                        ]
                        for method in ("posterior", "decision"):
                            started = time.perf_counter()
                            result, trace, final_eta = _run_method(
                                method=method,
                                initial_eta=initial_eta,
                                initial_adam=initial_adam,
                                whitening=whitening,
                                initial_x=map_x,
                                potential=potential,
                                teacher=teacher,
                                teacher_x=teacher_x,
                                teacher_ei=teacher_ei,
                                config=config,
                                structural_budget=budget,
                                diagnostic_candidates=candidates,
                            )
                            if result["teacher_ei_at_method"] > teacher_ei + 1e-10:
                                teacher_x, teacher_ei, strengthened = continuous_teacher_optimum(
                                    teacher,
                                    float(train_y.max()),
                                    candidates.numpy(),
                                    extra_candidates=extra,
                                    map_design=np.asarray(result["final_x"]),
                                )
                                if teacher_ei + 1e-10 < result["teacher_ei_at_method"]:
                                    raise RuntimeError(
                                        "method exceeded the strengthened continuous teacher reference"
                                    )
                                for prior_result in all_results:
                                    if prior_result.get("seed") == seed and prior_result.get("count") == count:
                                        value = float(prior_result["teacher_ei_at_method"])
                                        regret = max(0.0, teacher_ei - value)
                                        prior_result["teacher_optimum_ei"] = teacher_ei
                                        prior_result["absolute_regret"] = regret
                                        prior_result["normalized_regret"] = regret / max(teacher_ei, 1e-15)
                                        prior_result["distance_to_teacher"] = float(
                                            np.linalg.norm(np.asarray(prior_result["final_x"]) - teacher_x)
                                        )
                                result["teacher_reference_strengthened"] = True
                                result["teacher_strengthened_starts"] = strengthened["starts"]
                                value = float(result["teacher_ei_at_method"])
                                regret = max(0.0, teacher_ei - value)
                                result["teacher_optimum_ei"] = teacher_ei
                                result["absolute_regret"] = regret
                                result["normalized_regret"] = regret / max(teacher_ei, 1e-15)
                                result["distance_to_teacher"] = float(
                                    np.linalg.norm(np.asarray(result["final_x"]) - teacher_x)
                                )
                            else:
                                result["teacher_reference_strengthened"] = False
                            method_seconds = time.perf_counter() - started
                            result.update(
                                seed=seed,
                                count=count,
                                particles=particle_count,
                                budget=budget,
                                repeat=repeat,
                                post_initialization_elapsed_seconds=method_seconds,
                                shared_initialization_elapsed_seconds=initialization_seconds,
                                elapsed_seconds=initialization_seconds + method_seconds,
                            )
                            all_results.append(result)
                            particle_path = (
                                output_dir
                                / "particles"
                                / f"seed{seed}_n{count}_k{particle_count}_b{budget}_r{repeat}_{method}.npz"
                            )
                            particle_path.parent.mkdir(parents=True, exist_ok=True)
                            np.savez_compressed(
                                particle_path,
                                unconstrained=final_eta,
                                final_x=np.asarray(result["final_x"]),
                            )
                            for row in trace:
                                all_traces.append(
                                    {
                                        "seed": seed,
                                        "count": count,
                                        "particles": particle_count,
                                        "budget": budget,
                                        "repeat": repeat,
                                        **row,
                                    }
                                )
                        _write_csv(output_dir / "task02c_methods.csv", all_results)
                        _write_csv(output_dir / "task02c_trace.csv", all_traces)
    (output_dir / "task02c_config.json").write_text(
        json.dumps(
            {
                "configuration": asdict(config),
                "environment": {
                    "python": platform.python_version(),
                    "jax": jax.__version__,
                    "jax_x64": bool(jax.config.jax_enable_x64),
                    "jax_backend": jax.default_backend(),
                    "jax_devices": [str(device) for device in jax.devices()],
                },
                "cases": case_metadata,
            },
            indent=2,
        )
        + "\n"
    )
    _write_full_summary(
        output_dir / "SUMMARY.md",
        all_results,
        full_preflight_rows,
        potential_validations,
        all_traces,
        case_metadata,
    )
    _plot_full(output_dir, all_results, all_traces, full_preflight_rows)
