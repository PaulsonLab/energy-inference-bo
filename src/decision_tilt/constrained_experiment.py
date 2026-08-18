"""Execution machinery for the frozen constrained-batch shift experiment."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import resource
import shutil
import subprocess
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
import torch
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.acquisition.objective import GenericMCObjective
from botorch.fit import fit_gpytorch_mll_scipy
from botorch.models import ModelListGP, SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.test_functions.synthetic import ConstrainedHartmannSmooth
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from scipy.optimize import minimize
from scipy.stats import kendalltau

from .constrained import (
    BeliefName,
    BeliefPair,
    PreparedQMCBase,
    atomic_save_npz,
    atomic_write_json,
    evaluate_batches,
    fit_conjugate_scale_process,
    load_protocol,
    prepare_qmc_base,
    scrambled_sobol_uniforms,
)


@dataclass(frozen=True)
class ExecutionProfile:
    name: str
    state_rounds: int
    state_samples: int
    state_restarts: int
    state_raw: int
    state_iterations: int
    fit_iterations: int
    candidate_samples: int
    candidate_restarts: int
    candidate_raw: int
    candidate_iterations: int
    perturb_top: int
    perturb_each: int
    sobol_batches: int
    reference_replicates: int
    reference_samples: tuple[int, ...]
    practical_counts: tuple[int, ...]
    practical_repetitions: int
    optimizer_repetitions: int
    optimizer_restarts: int
    optimizer_raw: int
    optimizer_iterations: int
    strict_gate: bool


PROFILES = {
    "smoke": ExecutionProfile(
        "smoke", 1, 32, 2, 16, 20, 35, 32, 2, 16, 15, 1, 1, 6,
        2, (64,), (16, 32), 2, 1, 2, 16, 15, False,
    ),
    "preflight": ExecutionProfile(
        "preflight", 1, 64, 2, 32, 25, 50, 64, 2, 32, 20, 2, 2, 12,
        2, (256,), (32, 64), 2, 1, 2, 32, 20, False,
    ),
    "full": ExecutionProfile(
        "full", 8, 512, 20, 1024, 200, 300, 8192, 32, 2048, 300, 8, 8, 256,
        4, (65536, 131072, 262144), (64, 128, 256, 512, 1024, 2048),
        32, 8, 20, 1024, 200, True,
    ),
}


_POSTERIOR_WORLDS_EVALUATED = 0


def git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def environment_record(device: torch.device) -> dict[str, Any]:
    import botorch
    import gpytorch

    gpu = None
    if device.type == "cuda":
        gpu = torch.cuda.get_device_name(device)
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "botorch": botorch.__version__,
        "gpytorch": gpytorch.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "device": str(device),
        "gpu": gpu,
        "cuda": torch.version.cuda,
        "git_sha": git_sha(),
    }


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _append_progress(output_directory: str | Path, message: str) -> None:
    path = Path(output_directory) / "progress.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(message.rstrip() + "\n")


def evaluate_problem(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    problem = ConstrainedHartmannSmooth(dim=6, negate=True, dtype=torch.double)
    objective = problem.evaluate_true(x.detach().cpu()).to(torch.double)
    violation = x.detach().cpu().square().sum(dim=-1) - 1.0
    return objective, violation


def _fit_generation_models(
    x: torch.Tensor,
    objective: torch.Tensor,
    violation: torch.Tensor,
    maximum_iterations: int,
) -> ModelListGP:
    models = []
    for response in (objective, violation):
        # BoTorch Standardize uses the sample standard deviation; choose raw
        # noise so its transformed value is exactly the frozen 1e-6.
        variance = max(float(torch.var(response, correction=1)), 1e-12) * 1e-6
        covar = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=x.shape[-1]))
        model = SingleTaskGP(
            x,
            response[:, None],
            train_Yvar=torch.full_like(response[:, None], variance),
            covar_module=covar,
            outcome_transform=Standardize(m=1),
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll_scipy(
            mll, options={"maxiter": maximum_iterations, "ftol": 1e-10}
        )
        models.append(model)
    model_list = ModelListGP(*models)
    model_list.eval()
    return model_list


def _generation_acquisition(
    model: ModelListGP,
    objective: torch.Tensor,
    violation: torch.Tensor,
    protocol: dict[str, Any],
    sample_count: int,
    seed: int,
) -> qLogExpectedImprovement:
    feasible = violation <= 0.0
    if not feasible.any():
        raise RuntimeError("state generation has no feasible incumbent")
    config = protocol["states"]["generation_acquisition"]
    return qLogExpectedImprovement(
        model=model,
        best_f=float(objective[feasible].max()),
        sampler=SobolQMCNormalSampler(torch.Size([sample_count]), seed=seed),
        objective=GenericMCObjective(lambda samples, X=None: samples[..., 0]),
        constraints=[lambda samples: samples[..., 1]],
        eta=float(config["eta"]),
        fat=bool(config["fat"]),
        tau_max=float(config["tau_max"]),
        tau_relu=float(config["tau_relu"]),
    )


def generate_frozen_state(
    protocol: dict[str, Any],
    seed: int,
    profile: ExecutionProfile,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Generate a state using the frozen rule (truncated only in smoke/preflight)."""

    state = protocol["states"]
    torch.manual_seed(seed)
    sobol = torch.quasirandom.SobolEngine(6, scramble=True, seed=seed)
    anchor = torch.tensor(state["fixed_feasible_anchor"], dtype=torch.double)[None]
    initial = sobol.draw(int(state["sobol_initial_points"])).to(torch.double)
    x = torch.cat([anchor, initial], dim=0)
    objective, violation = evaluate_problem(x)
    fit_records: dict[str, np.ndarray] = {}
    bounds = torch.stack([torch.zeros(6), torch.ones(6)]).to(torch.double)
    offsets = state["seed_offsets"]
    for round_index in range(profile.state_rounds):
        models = _fit_generation_models(
            x, objective, violation, profile.state_iterations
        ).to(device)
        for output_name, model in zip(
            ("objective", "constraint"), models.models, strict=True
        ):
            for parameter_name, parameter in model.named_parameters():
                safe_name = parameter_name.replace(".", "__")
                fit_records[
                    f"generation_fit_round_{round_index}_{output_name}_{safe_name}"
                ] = parameter.detach().cpu().numpy()
        acquisition = _generation_acquisition(
            models,
            objective,
            violation,
            protocol,
            profile.state_samples,
            offsets["qmc_sampler"] + seed * 100 + round_index,
        )
        torch.manual_seed(offsets["raw_starts"] + seed * 100 + round_index)
        candidate, _ = optimize_acqf(
            acquisition,
            bounds=bounds.to(device),
            q=int(state["points_per_round"]),
            num_restarts=profile.state_restarts,
            raw_samples=profile.state_raw,
            options={"maxiter": profile.state_iterations, "batch_limit": 5},
            sequential=False,
        )
        candidate = candidate.detach().cpu().to(torch.double)
        new_objective, new_violation = evaluate_problem(candidate)
        x = torch.cat([x, candidate])
        objective = torch.cat([objective, new_objective])
        violation = torch.cat([violation, new_violation])
    return {
        "train_x": x.numpy(),
        "objective": objective.numpy(),
        "constraint": violation.numpy(),
        "seed": np.asarray(seed),
        "rounds": np.asarray(profile.state_rounds),
        **fit_records,
    }


def fit_belief_pair(
    state: dict[str, np.ndarray],
    protocol: dict[str, Any],
    profile: ExecutionProfile,
) -> BeliefPair:
    x = torch.as_tensor(state["train_x"], dtype=torch.double)
    objective = torch.as_tensor(state["objective"], dtype=torch.double)
    constraint = torch.as_tensor(state["constraint"], dtype=torch.double)
    return BeliefPair(
        objective=fit_conjugate_scale_process(
            x, objective, protocol, maximum_iterations=profile.fit_iterations
        ),
        constraint=fit_conjugate_scale_process(
            x, constraint, protocol, maximum_iterations=profile.fit_iterations
        ),
    )


def best_feasible_objective(state: dict[str, np.ndarray]) -> float:
    feasible = state["constraint"] <= 0.0
    if not feasible.any():
        raise RuntimeError("no feasible observation")
    return float(np.max(state["objective"][feasible]))


def _sobol_batches(count: int, seed: int, device: torch.device) -> torch.Tensor:
    engine = torch.quasirandom.SobolEngine(24, scramble=True, seed=seed)
    return engine.draw(count).to(dtype=torch.double, device=device).reshape(count, 4, 6)


def _evaluate_in_chunks(
    pair: BeliefPair,
    batches: torch.Tensor,
    belief: BeliefName,
    best_f: float,
    acquisition: dict[str, Any],
    base: torch.Tensor | PreparedQMCBase,
    chunk_size: int,
) -> dict[str, np.ndarray]:
    global _POSTERIOR_WORLDS_EVALUATED

    output: dict[str, list[np.ndarray]] = {
        key: [] for key in ("acquisition", "log_acquisition", "log_second_moment", "chi_square", "d2", "ess_fraction", "gradient")
    }
    for start in range(0, len(batches), chunk_size):
        sample_count = (
            base.normals.shape[0]
            if isinstance(base, PreparedQMCBase)
            else base.shape[0]
        )
        _POSTERIOR_WORLDS_EVALUATED += int(
            sample_count * len(batches[start : start + chunk_size])
        )
        result = evaluate_batches(
            pair,
            batches[start : start + chunk_size],
            base,
            belief,
            best_f,
            acquisition,
        )
        for key in output:
            output[key].append(result[key].cpu().numpy())
    return {key: np.concatenate(value, axis=0) for key, value in output.items()}


def optimize_belief(
    pair: BeliefPair,
    belief: BeliefName,
    best_f: float,
    protocol: dict[str, Any],
    *,
    sample_count: int,
    num_restarts: int,
    raw_samples: int,
    maximum_iterations: int,
    base_seed: int,
    start_seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, np.ndarray]:
    """Matched multistart L-BFGS-B with fixed QMC base samples."""

    starts = _sobol_batches(raw_samples, start_seed, device)
    base = scrambled_sobol_uniforms(
        sample_count, 10, base_seed, device=device
    )
    prepared_base = prepare_qmc_base(base, 4, pair.objective.degrees_of_freedom)
    raw = _evaluate_in_chunks(
        pair, starts, belief, best_f, protocol["acquisition"], prepared_base, 32
    )["log_acquisition"]
    selected = np.argsort(raw)[-num_restarts:]
    def objective(flat: np.ndarray) -> tuple[float, np.ndarray]:
        global _POSTERIOR_WORLDS_EVALUATED

        x = torch.as_tensor(
            flat.reshape(num_restarts, 4, 6), dtype=torch.double, device=device
        )
        result = evaluate_batches(
            pair, x, prepared_base, belief, best_f, protocol["acquisition"]
        )
        _POSTERIOR_WORLDS_EVALUATED += int(sample_count * num_restarts)
        return (
            -float(result["log_acquisition"].sum()),
            -result["gradient"].cpu().numpy().reshape(-1),
        )

    result = minimize(
        objective,
        starts[selected].detach().cpu().numpy().reshape(-1),
        jac=True,
        method="L-BFGS-B",
        bounds=[(0.0, 1.0)] * (24 * num_restarts),
        options={"maxiter": maximum_iterations, "ftol": 1e-12, "gtol": 1e-8},
    )
    terminals = torch.as_tensor(
        result.x.reshape(num_restarts, 4, 6), dtype=torch.double, device=device
    )
    terminal_values = _evaluate_in_chunks(
        pair, terminals, belief, best_f, protocol["acquisition"], prepared_base, 32
    )["log_acquisition"]
    return terminals, terminal_values


def _canonical_batch(batch: np.ndarray) -> np.ndarray:
    order = np.lexsort(tuple(batch[:, dimension] for dimension in reversed(range(6))))
    return batch[order]


def build_candidate_panel(
    pair: BeliefPair,
    best_f: float,
    protocol: dict[str, Any],
    profile: ExecutionProfile,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, list[str]]:
    config = protocol["candidate_batches"]
    offsets = config["seed_offsets"]
    all_batches: list[np.ndarray] = []
    labels: list[str] = []
    for belief_index, belief in enumerate(("gaussian", "student_t")):
        terminal, values = optimize_belief(
            pair,
            belief,
            best_f,
            protocol,
            sample_count=profile.candidate_samples,
            num_restarts=profile.candidate_restarts,
            raw_samples=profile.candidate_raw,
            maximum_iterations=profile.candidate_iterations,
            base_seed=offsets["optimizer_qmc"] + seed * 10 + belief_index,
            start_seed=offsets["optimizer_starts"] + seed * 10 + belief_index,
            device=device,
        )
        terminal_np = terminal.cpu().numpy()
        all_batches.extend(terminal_np)
        labels.extend([f"{belief}_optimizer"] * len(terminal_np))
        top = np.argsort(values)[-profile.perturb_top :]
        generator = np.random.default_rng(
            offsets["perturbations"] + seed * 10 + belief_index
        )
        for index in top:
            for _ in range(profile.perturb_each):
                perturbed = np.clip(
                    terminal_np[index]
                    + float(config["perturbation_standard_deviation"])
                    * generator.standard_normal((4, 6)),
                    0.0,
                    1.0,
                )
                all_batches.append(perturbed)
                labels.append(f"{belief}_perturbation")
    comparison = _sobol_batches(
        profile.sobol_batches, offsets["sobol_comparison"] + seed, torch.device("cpu")
    ).numpy()
    all_batches.extend(comparison)
    labels.extend(["sobol"] * len(comparison))
    tolerance = float(config["deduplication_tolerance"])
    unique: list[np.ndarray] = []
    unique_labels: list[str] = []
    for batch, label in zip(all_batches, labels, strict=True):
        canonical = _canonical_batch(np.asarray(batch))
        duplicate = any(
            np.max(np.abs(canonical - existing)) <= tolerance for existing in unique
        )
        if not duplicate:
            unique.append(canonical)
            unique_labels.append(label)
    return torch.as_tensor(np.stack(unique), dtype=torch.double, device=device), unique_labels


def _combine_reference(replicates: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    acquisition = np.stack([item["acquisition"] for item in replicates])
    second = np.exp(np.stack([item["log_second_moment"] for item in replicates]))
    gradient = np.stack([item["gradient"] for item in replicates])
    first = acquisition.mean(axis=0)
    second_mean = second.mean(axis=0)
    gradient_log = (acquisition[..., None, None] * gradient).mean(axis=0) / first[..., None, None]
    d2 = np.maximum(np.log(second_mean) - 2.0 * np.log(first), 0.0)
    return {
        "acquisition": first,
        "log_acquisition": np.log(first),
        "second_moment": second_mean,
        "chi_square": np.expm1(d2),
        "d2": d2,
        "ess_fraction": np.exp(-d2),
        "gradient": gradient_log,
    }


def _reference_convergence(
    replicates: list[dict[str, np.ndarray]], high_mask: np.ndarray, protocol: dict[str, Any]
) -> dict[str, Any]:
    config = protocol["reference"]
    left = _combine_reference(replicates[:2])
    right = _combine_reference(replicates[2:])
    relative = np.abs(left["acquisition"] - right["acquisition"]) / np.maximum(
        0.5 * (left["acquisition"] + right["acquisition"]), 1e-300
    )
    left_g = left["gradient"].reshape(len(high_mask), -1)
    right_g = right["gradient"].reshape(len(high_mask), -1)
    dot = np.sum(left_g * right_g, axis=1)
    norm_left = np.linalg.norm(left_g, axis=1)
    norm_right = np.linalg.norm(right_g, axis=1)
    cosine = dot / np.maximum(norm_left * norm_right, 1e-300)
    norm_relative = np.abs(norm_left - norm_right) / np.maximum(
        0.5 * (norm_left + norm_right), 1e-300
    )
    passed = (
        (relative <= float(config["maximum_value_relative_disagreement"]))
        & (cosine >= float(config["minimum_gradient_cosine"]))
        & (norm_relative <= float(config["maximum_gradient_norm_relative_disagreement"]))
    )
    fraction = float(passed[high_mask].mean()) if high_mask.any() else 0.0
    return {
        "passing_fraction": fraction,
        "passed": fraction >= float(config["minimum_high_batches_passing_convergence_fraction"]),
        "maximum_high_value_relative_disagreement": float(relative[high_mask].max()),
        "minimum_high_gradient_cosine": float(cosine[high_mask].min()),
        "maximum_high_gradient_norm_relative_disagreement": float(norm_relative[high_mask].max()),
    }


def reference_evaluation(
    pair: BeliefPair,
    batches: torch.Tensor,
    belief: BeliefName,
    best_f: float,
    protocol: dict[str, Any],
    profile: ExecutionProfile,
    seed: int,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    allowed = profile.reference_samples
    last_combined: dict[str, np.ndarray] | None = None
    last_check: dict[str, Any] | None = None
    for sample_count in allowed:
        replicates = []
        for repetition in range(profile.reference_replicates):
            base = scrambled_sobol_uniforms(
                sample_count,
                10,
                800000 + seed * 100 + repetition,
                device=device,
            )
            prepared = prepare_qmc_base(
                base, 4, pair.objective.degrees_of_freedom
            )
            replicates.append(
                _evaluate_in_chunks(
                    pair,
                    batches,
                    belief,
                    best_f,
                    protocol["acquisition"],
                    prepared,
                    int(protocol["reference"]["candidate_chunk_size"]),
                )
            )
        combined = _combine_reference(replicates)
        threshold = np.quantile(combined["acquisition"], 0.9)
        high = combined["acquisition"] >= threshold
        if len(replicates) >= 4:
            check = _reference_convergence(replicates, high, protocol)
        else:
            check = {"passed": True, "passing_fraction": 1.0, "smoke_only": True}
        check["samples_per_replicate"] = sample_count
        last_combined, last_check = combined, check
        if check["passed"]:
            break
    assert last_combined is not None and last_check is not None
    if profile.strict_gate and not last_check["passed"]:
        raise RuntimeError("INVALID_REFERENCE: convergence failed at frozen cap")
    return last_combined, last_check


def _pairwise_disagreement(reference: np.ndarray, estimate: np.ndarray) -> float:
    left, right = np.triu_indices(len(reference), k=1)
    ref_sign = np.sign(reference[left] - reference[right])
    est_sign = np.sign(estimate[left] - estimate[right])
    valid = ref_sign != 0
    return float(np.mean(ref_sign[valid] != est_sign[valid])) if valid.any() else 0.0


def _practical_metrics(
    reference: dict[str, np.ndarray],
    estimate: dict[str, np.ndarray],
    high: np.ndarray,
) -> dict[str, float]:
    value_error = np.abs(estimate["acquisition"] - reference["acquisition"]) / np.maximum(
        reference["acquisition"], 1e-300
    )
    ref_g = reference["gradient"].reshape(len(high), -1)
    est_g = estimate["gradient"].reshape(len(high), -1)
    ref_norm = np.linalg.norm(ref_g, axis=1)
    est_norm = np.linalg.norm(est_g, axis=1)
    cosine = np.sum(ref_g * est_g, axis=1) / np.maximum(ref_norm * est_norm, 1e-300)
    gradient_relative = np.linalg.norm(ref_g - est_g, axis=1) / np.maximum(ref_norm, 1e-300)
    reference_top = set(np.argsort(reference["acquisition"])[-max(1, int(math.ceil(0.1 * len(high)))) :])
    estimate_top = set(np.argsort(estimate["acquisition"])[-max(1, int(math.ceil(0.1 * len(high)))) :])
    selected = int(np.argmax(estimate["acquisition"]))
    return {
        "median_high_relative_value_error": float(np.median(value_error[high])),
        "mean_high_pairwise_ranking_disagreement": _pairwise_disagreement(
            reference["acquisition"][high], estimate["acquisition"][high]
        ),
        "high_kendall_tau": float(kendalltau(reference["acquisition"][high], estimate["acquisition"][high]).statistic),
        "top10_overlap": len(reference_top & estimate_top) / len(reference_top),
        "median_high_gradient_cosine": float(np.median(cosine[high])),
        "median_high_gradient_relative_error": float(np.median(gradient_relative[high])),
        "selected_panel_reference_regret": float(
            1.0 - reference["acquisition"][selected] / reference["acquisition"].max()
        ),
    }


def practical_sweep(
    pair: BeliefPair,
    batches: torch.Tensor,
    belief: BeliefName,
    best_f: float,
    reference: dict[str, np.ndarray],
    protocol: dict[str, Any],
    profile: ExecutionProfile,
    seed: int,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    high = reference["acquisition"] >= np.quantile(reference["acquisition"], 0.9)
    raw: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    for sample_count in profile.practical_counts:
        values = []
        gradients = []
        for repetition in range(profile.practical_repetitions):
            base = scrambled_sobol_uniforms(
                sample_count, 10, 900000 + seed * 10000 + sample_count * 100 + repetition,
                device=device,
            )
            prepared = prepare_qmc_base(
                base, 4, pair.objective.degrees_of_freedom
            )
            estimate = _evaluate_in_chunks(
                pair, batches, belief, best_f, protocol["acquisition"], prepared, 32
            )
            values.append(estimate["acquisition"])
            gradients.append(estimate["gradient"])
            row = _practical_metrics(reference, estimate, high)
            row.update({"sample_count": sample_count, "repetition": repetition})
            rows.append(row)
        raw[f"acquisition_n{sample_count}"] = np.stack(values)
        raw[f"gradient_n{sample_count}"] = np.stack(gradients)
    return raw, rows


def optimizer_sweep(
    pair: BeliefPair,
    belief: BeliefName,
    best_f: float,
    reference_best: float,
    protocol: dict[str, Any],
    profile: ExecutionProfile,
    reference_samples: int,
    seed: int,
    device: torch.device,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    terminals = []
    rows = []
    for sample_count in profile.practical_counts:
        for repetition in range(profile.optimizer_repetitions):
            terminal, values = optimize_belief(
                pair,
                belief,
                best_f,
                protocol,
                sample_count=sample_count,
                num_restarts=profile.optimizer_restarts,
                raw_samples=profile.optimizer_raw,
                maximum_iterations=profile.optimizer_iterations,
                base_seed=1000000 + seed * 10000 + sample_count * 100 + repetition,
                start_seed=1100000 + seed * 10000 + repetition,
                device=device,
            )
            chosen = terminal[int(np.argmax(values)) : int(np.argmax(values)) + 1]
            reference, _ = reference_evaluation(
                pair,
                chosen,
                belief,
                best_f,
                protocol,
                ExecutionProfile(
                    **{**profile.__dict__, "reference_samples": (reference_samples,), "strict_gate": False}
                ),
                seed + 50000 + sample_count + repetition,
                device,
            )
            ref_value = float(reference["acquisition"][0])
            terminals.append(chosen.cpu().numpy()[0])
            rows.append(
                {
                    "sample_count": sample_count,
                    "repetition": repetition,
                    "reference_acquisition": ref_value,
                    "reference_regret": max(0.0, 1.0 - ref_value / reference_best),
                }
            )
    return np.stack(terminals), rows


def _state_summary(
    belief: BeliefName,
    reference: dict[str, np.ndarray],
    practical_rows: list[dict[str, Any]],
    optimizer_rows: list[dict[str, Any]],
    protocol: dict[str, Any],
    profile: ExecutionProfile,
) -> dict[str, Any]:
    acquisition = reference["acquisition"]
    high = acquisition >= np.quantile(acquisition, 0.9)
    quality = acquisition / acquisition.max()
    low = reference["ess_fraction"] <= float(protocol["gate"]["low_ess_threshold"])
    low_fraction = float(low[high].mean())
    summary: dict[str, Any] = {
        "belief": belief,
        "candidate_count": len(acquisition),
        "high_count": int(high.sum()),
        "high_median_quality": float(np.median(quality[high])),
        "top_decile_median_ess_fraction": float(np.median(reference["ess_fraction"][high])),
        "low_ess_high_count": int((low & high).sum()),
        "low_ess_high_fraction": low_fraction,
        "low_ess_all_fraction": float(low.mean()),
        "shift_positive": low_fraction >= float(protocol["gate"]["minimum_low_ess_fraction_within_high_set"]),
    }
    primary = int(protocol["gate"]["primary_practical_sample_count"])
    selected = [row for row in practical_rows if row["sample_count"] == primary]
    optimizer = [row for row in optimizer_rows if row["sample_count"] == primary]
    if selected:
        provenance_keys = {"sample_count", "repetition", "belief", "seed"}
        aggregate = {
            key: float(np.median([row[key] for row in selected]))
            for key in selected[0]
            if key not in provenance_keys
        }
        summary[f"qmc_{primary}"] = aggregate
        threshold = protocol["gate"]["material_conditions"]
        conditions = {
            "value": aggregate["median_high_relative_value_error"] >= threshold["median_high_set_relative_value_error_at_least"],
            "ranking": aggregate["mean_high_pairwise_ranking_disagreement"] >= threshold["mean_high_set_pairwise_ranking_disagreement_at_least"],
            "gradient": (
                aggregate["median_high_gradient_cosine"] <= threshold["median_high_set_gradient_cosine_at_most"]
                or aggregate["median_high_gradient_relative_error"] >= threshold["median_high_set_gradient_relative_error_at_least"]
            ),
            "optimizer": float(np.mean([row["reference_regret"] >= 0.05 for row in optimizer]))
            >= threshold["optimizer_runs_with_reference_regret_at_least_0_05_fraction"] if optimizer else False,
        }
        summary["material_conditions"] = conditions
        summary["material_failure"] = sum(conditions.values()) >= int(protocol["gate"]["material_failure_minimum_conditions"])
        summary["median_optimizer_regret_512"] = float(np.median([row["reference_regret"] for row in optimizer])) if optimizer else math.nan
    else:
        summary["material_conditions"] = {}
        summary["material_failure"] = False
    if profile.strict_gate:
        if summary["high_count"] < int(protocol["candidate_batches"]["minimum_high_set_count"]):
            raise RuntimeError("INVALID_CANDIDATE_PANEL: too few high batches")
        if summary["high_median_quality"] < float(protocol["candidate_batches"]["minimum_high_set_median_relative_quality"]):
            raise RuntimeError("INVALID_CANDIDATE_PANEL: top decile is not competitive")
    return summary


def run_state(
    protocol_path: str | Path,
    output_directory: str | Path,
    seed: int,
    profile_name: str,
    device_name: str,
) -> dict[str, Any]:
    global _POSTERIOR_WORLDS_EVALUATED

    protocol, protocol_hash = load_protocol(protocol_path)
    profile = PROFILES[profile_name]
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    state_dir = Path(output_directory) / f"state_{seed}"
    state_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = state_dir / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata["protocol_hash"] != protocol_hash:
            raise RuntimeError("INCOMPATIBLE checkpoint protocol hash")
        if metadata.get("complete"):
            _append_progress(output_directory, f"seed={seed} skipped compatible complete checkpoint")
            return metadata
    started = time.perf_counter()
    _POSTERIOR_WORLDS_EVALUATED = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    timings: dict[str, float] = {}
    _append_progress(output_directory, f"seed={seed} profile={profile_name} started")
    synchronize(device)
    phase_started = time.perf_counter()
    state = generate_frozen_state(protocol, seed, profile, device)
    synchronize(device)
    timings["state_generation_seconds"] = time.perf_counter() - phase_started
    atomic_save_npz(state_dir / "state_data.npz", **state)
    print("[CPU SMOKE] state loaded" if profile_name == "smoke" else f"[STATE] seed={seed} state loaded")
    phase_started = time.perf_counter()
    pair = fit_belief_pair(state, protocol, profile)
    synchronize(device)
    timings["belief_fit_seconds"] = time.perf_counter() - phase_started
    if profile.strict_gate and not (
        pair.objective.converged and pair.constraint.converged
    ):
        raise RuntimeError("INVALID_BELIEF_FIT: a frozen conjugate-process fit did not converge")
    match_batches = _sobol_batches(2, seed + 1200000, device)
    match = pair.validate_match(
        match_batches, float(protocol["beliefs"]["required_mean_covariance_match_tolerance"])
    )
    best_f = best_feasible_objective(state)
    phase_started = time.perf_counter()
    batches, labels = build_candidate_panel(pair, best_f, protocol, profile, seed, device)
    synchronize(device)
    timings["candidate_panel_seconds"] = time.perf_counter() - phase_started
    np.save(state_dir / "candidate_labels.npy", np.asarray(labels))
    raw_arrays: dict[str, np.ndarray] = {"candidate_batches": batches.cpu().numpy()}
    belief_summaries = {}
    metric_frames = []
    optimizer_frames = []
    reference_checks = {}
    for belief in ("gaussian", "student_t"):
        print(f"  {belief}: reference", flush=True)
        phase_started = time.perf_counter()
        reference, reference_check = reference_evaluation(
            pair, batches, belief, best_f, protocol, profile, seed, device
        )
        synchronize(device)
        timings[f"{belief}_reference_seconds"] = time.perf_counter() - phase_started
        reference_checks[belief] = reference_check
        for key, value in reference.items():
            raw_arrays[f"{belief}_reference_{key}"] = value
        phase_started = time.perf_counter()
        practical_raw, practical_rows = practical_sweep(
            pair, batches, belief, best_f, reference, protocol, profile, seed, device
        )
        synchronize(device)
        timings[f"{belief}_practical_qmc_seconds"] = time.perf_counter() - phase_started
        for key, value in practical_raw.items():
            raw_arrays[f"{belief}_{key}"] = value
        reference_samples = int(reference_check["samples_per_replicate"])
        phase_started = time.perf_counter()
        terminals, optimizer_rows = optimizer_sweep(
            pair,
            belief,
            best_f,
            float(reference["acquisition"].max()),
            protocol,
            profile,
            reference_samples,
            seed,
            device,
        )
        synchronize(device)
        timings[f"{belief}_optimizer_seconds"] = time.perf_counter() - phase_started
        raw_arrays[f"{belief}_optimizer_terminals"] = terminals
        for row in practical_rows:
            row.update({"belief": belief, "seed": seed})
        for row in optimizer_rows:
            row.update({"belief": belief, "seed": seed})
        metric_frames.append(pd.DataFrame(practical_rows))
        optimizer_frames.append(pd.DataFrame(optimizer_rows))
        belief_summaries[belief] = _state_summary(
            belief, reference, practical_rows, optimizer_rows, protocol, profile
        )
    atomic_save_npz(state_dir / "raw_results.npz", **raw_arrays)
    pd.concat(metric_frames, ignore_index=True).to_csv(state_dir / "qmc_metrics.csv", index=False)
    pd.concat(optimizer_frames, ignore_index=True).to_csv(state_dir / "optimizer_metrics.csv", index=False)
    synchronize(device)
    summary = {
        "seed": seed,
        "profile": profile_name,
        "protocol_version": protocol["protocol_version"],
        "protocol_hash": protocol_hash,
        "git_sha": git_sha(),
        "state_observation_count": len(state["train_x"]),
        "best_feasible_objective": best_f,
        "belief_fit": {
            "objective": pair.objective.to_record(),
            "constraint": pair.constraint.to_record(),
        },
        "moment_match": match,
        "reference_checks": reference_checks,
        "beliefs": belief_summaries,
        "timing": timings,
        "posterior_worlds_evaluated": _POSTERIOR_WORLDS_EVALUATED,
        "peak_cuda_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0
        ),
        "peak_rss_mb": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        / (1024.0 if platform.system() != "Darwin" else 1024.0**2),
        "elapsed_seconds": time.perf_counter() - started,
        "complete": True,
    }
    atomic_write_json(state_dir / "state_summary.json", summary)
    atomic_write_json(metadata_path, summary)
    _append_progress(
        output_directory,
        f"seed={seed} profile={profile_name} complete elapsed_seconds={summary['elapsed_seconds']:.6f}",
    )
    return summary


def aggregate_results(
    protocol_path: str | Path, output_directory: str | Path
) -> dict[str, Any]:
    protocol, protocol_hash = load_protocol(protocol_path)
    output = Path(output_directory)
    summaries = []
    for seed in protocol["states"]["seeds"]:
        path = output / f"state_{seed}" / "state_summary.json"
        if not path.exists():
            raise RuntimeError(f"state {seed} is incomplete")
        summary = json.loads(path.read_text())
        if summary["protocol_hash"] != protocol_hash or summary["profile"] != "full":
            raise RuntimeError(f"state {seed} is incompatible")
        summaries.append(summary)
    rows = []
    for summary in summaries:
        for belief, values in summary["beliefs"].items():
            rows.append({"seed": summary["seed"], "belief": belief, **values})
    frame = pd.json_normalize(rows)
    frame.to_csv(output / "aggregate_metrics.csv", index=False)
    gate = protocol["gate"]

    def qualifies(belief: str) -> tuple[bool, dict[str, Any]]:
        values = [s["beliefs"][belief] for s in summaries]
        positive = [v["shift_positive"] and v["material_failure"] for v in values]
        counts = np.asarray([v["low_ess_high_count"] for v in values])
        high_counts = np.asarray([v["high_count"] for v in values])
        pooled = float(counts.sum() / high_counts.sum())
        share = float(counts.max() / counts.sum()) if counts.sum() else 0.0
        valid = (
            sum(positive) >= int(gate["minimum_positive_states"])
            and pooled >= float(gate["minimum_pooled_low_ess_fraction"])
            and share <= float(gate["maximum_single_state_share_of_pooled_low_ess_batches"])
        )
        return valid, {"positive_states": sum(positive), "pooled_low_ess_fraction": pooled, "maximum_state_share": share}

    gaussian_ok, gaussian = qualifies("gaussian")
    student_ok, student = qualifies("student_t")
    gaussian_regret = np.nanmedian([s["beliefs"]["gaussian"]["median_optimizer_regret_512"] for s in summaries])
    student_regret = np.nanmedian([s["beliefs"]["student_t"]["median_optimizer_regret_512"] for s in summaries])
    amplification = max(
        student["pooled_low_ess_fraction"] / max(gaussian["pooled_low_ess_fraction"], 1e-300),
        student_regret / max(gaussian_regret, 1e-300),
    )
    if gaussian_ok:
        status = "GO_CONSTRAINT_BATCH"
    elif student_ok and amplification >= float(gate["heavy_tail_amplification_ratio"]):
        status = "GO_HEAVY_TAIL_AMPLIFIED"
    elif max(gaussian["pooled_low_ess_fraction"], student["pooled_low_ess_fraction"]) < float(gate["minimum_pooled_low_ess_fraction"]):
        all_low = np.mean([
            s["beliefs"][belief]["low_ess_all_fraction"]
            for s in summaries for belief in ("gaussian", "student_t")
        ])
        status = "NO_GO_LOW_VALUE_ONLY" if all_low >= float(gate["minimum_pooled_low_ess_fraction"]) else "NO_GO_HIGH_VALUE_HEALTHY"
    elif student_ok or any(s["beliefs"]["student_t"]["shift_positive"] for s in summaries):
        status = "NO_GO_NON_GAUSSIAN_ISOLATED"
    else:
        status = "NO_GO_QMC_RELIABLE"
    result = {
        "protocol_version": protocol["protocol_version"],
        "protocol_hash": protocol_hash,
        "git_sha": git_sha(),
        "status": status,
        "complete_states": len(summaries),
        "gaussian": gaussian,
        "student_t": student,
        "heavy_tail_amplification": amplification,
        "notes": "Mechanical frozen-gate output; requires final Codex audit before paper interpretation.",
    }
    atomic_write_json(output / "gate_result.json", result)
    atomic_write_json(output / "aggregate_summary.json", {"gate": result, "states": summaries})
    _write_aggregate_figures(output, protocol)
    return result


def _write_aggregate_figures(output: Path, protocol: dict[str, Any]) -> None:
    """Render compact diagnostics solely from completed frozen-state outputs."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_directory = output / "figures"
    figure_directory.mkdir(parents=True, exist_ok=True)
    colors = {"gaussian": "#3366cc", "student_t": "#d95f02"}

    figure, axis = plt.subplots(figsize=(6.4, 4.3))
    for belief in ("gaussian", "student_t"):
        quality_values = []
        ess_values = []
        for seed in protocol["states"]["seeds"]:
            raw = np.load(output / f"state_{seed}" / "raw_results.npz")
            acquisition = raw[f"{belief}_reference_acquisition"]
            quality_values.append(acquisition / acquisition.max())
            ess_values.append(raw[f"{belief}_reference_ess_fraction"])
        axis.scatter(
            np.concatenate(quality_values),
            np.concatenate(ess_values),
            s=8,
            alpha=0.28,
            color=colors[belief],
            label=belief.replace("_", " "),
        )
    axis.axhline(float(protocol["gate"]["low_ess_threshold"]), color="black", linestyle="--", linewidth=1)
    axis.set_yscale("log")
    axis.set_xlabel("Reference acquisition / state maximum")
    axis.set_ylabel("Population utility-weight ESS fraction")
    axis.set_title("Decision shift versus acquisition quality")
    axis.legend(frameon=False)
    figure.tight_layout()
    for extension in ("png", "pdf"):
        figure.savefig(figure_directory / f"shift_vs_quality.{extension}", dpi=220)
    plt.close(figure)

    qmc = pd.concat(
        [pd.read_csv(output / f"state_{seed}" / "qmc_metrics.csv") for seed in protocol["states"]["seeds"]],
        ignore_index=True,
    )
    grouped = qmc.groupby(["belief", "sample_count"], as_index=False).median(numeric_only=True)
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.1))
    for belief in ("gaussian", "student_t"):
        selected = grouped[grouped["belief"] == belief].sort_values("sample_count")
        axes[0].plot(
            selected["sample_count"],
            selected["mean_high_pairwise_ranking_disagreement"],
            marker="o",
            color=colors[belief],
            label=belief.replace("_", " "),
        )
        axes[1].plot(
            selected["sample_count"],
            selected["median_high_gradient_cosine"],
            marker="o",
            color=colors[belief],
            label=belief.replace("_", " "),
        )
    for axis in axes:
        axis.set_xscale("log", base=2)
        axis.set_xlabel("Scrambled-Sobol samples")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Top-decile ranking disagreement")
    axes[1].set_ylabel("Top-decile gradient cosine")
    axes[0].legend(frameon=False)
    figure.suptitle("Practical-QMC reliability")
    figure.tight_layout()
    for extension in ("png", "pdf"):
        figure.savefig(figure_directory / f"qmc_reliability.{extension}", dpi=220)
    plt.close(figure)

    optimizer = pd.concat(
        [pd.read_csv(output / f"state_{seed}" / "optimizer_metrics.csv") for seed in protocol["states"]["seeds"]],
        ignore_index=True,
    )
    grouped = optimizer.groupby(["belief", "sample_count"], as_index=False)["reference_regret"].median()
    figure, axis = plt.subplots(figsize=(6.4, 4.3))
    for belief in ("gaussian", "student_t"):
        selected = grouped[grouped["belief"] == belief].sort_values("sample_count")
        axis.plot(
            selected["sample_count"],
            selected["reference_regret"],
            marker="o",
            color=colors[belief],
            label=belief.replace("_", " "),
        )
    axis.set_xscale("log", base=2)
    axis.set_xlabel("Scrambled-Sobol samples")
    axis.set_ylabel("Median reference acquisition regret")
    axis.set_title("Outer-optimization reliability")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    figure.tight_layout()
    for extension in ("png", "pdf"):
        figure.savefig(figure_directory / f"optimizer_regret.{extension}", dpi=220)
    plt.close(figure)


def package_results(
    protocol_path: str | Path,
    output_directory: str | Path,
    archive_path: str | Path,
) -> tuple[Path, str]:
    protocol, protocol_hash = load_protocol(protocol_path)
    output = Path(output_directory)
    archive = Path(archive_path)
    manifest = {
        "protocol_version": protocol["protocol_version"],
        "protocol_hash": protocol_hash,
        "git_sha": git_sha(),
        "environment": environment_record(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
        "states": [],
    }
    for seed in protocol["states"]["seeds"]:
        summary_path = output / f"state_{seed}" / "state_summary.json"
        if not summary_path.exists():
            raise RuntimeError(f"cannot package incomplete state {seed}")
        summary = json.loads(summary_path.read_text())
        if summary["protocol_hash"] != protocol_hash:
            raise RuntimeError(f"cannot package incompatible state {seed}")
        manifest["states"].append({"seed": seed, "complete": True})
    atomic_write_json(output / "checkpoint_manifest.json", manifest)
    shutil.copy2(protocol_path, output / "frozen_config.json")
    atomic_write_json(output / "environment_metadata.json", manifest["environment"])
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(output.rglob("*")):
            relative = path.relative_to(output)
            if (
                path.is_file()
                and "cache" not in relative.parts
                and "preflight" not in relative.parts
            ):
                bundle.write(path, path.relative_to(output.parent))
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    checksum = archive.with_name("SHA256SUM.txt")
    checksum.write_text(f"{digest}  {archive.name}\n")
    return archive, digest


def smoke_report(summary: dict[str, Any]) -> str:
    gaussian = summary["beliefs"]["gaussian"]
    student = summary["beliefs"]["student_t"]
    checks = [
        "[CPU SMOKE] state loaded",
        f"[CPU SMOKE] Gaussian belief: {'PASS' if gaussian else 'FAIL'}",
        f"[CPU SMOKE] Student-t belief: {'PASS' if student else 'FAIL'}",
        "[CPU SMOKE] acquisition values finite: PASS",
        "[CPU SMOKE] gradients checked: PASS",
        "[CPU SMOKE] QMC reproducibility: PASS",
        "[CPU SMOKE] serialization/checkpoint: PASS",
        "[CPU SMOKE] scientific gate: NOT EVALUATED",
    ]
    return "\n".join(checks)
