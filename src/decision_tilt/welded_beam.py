"""Exact q=1 constrained-EI diagnostics for the Welded Beam experiment."""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from botorch.fit import fit_gpytorch_mll_scipy
from botorch.models import SingleTaskGP
from botorch.test_functions.synthetic import WeldedBeamSO
from gpytorch.mlls import ExactMarginalLogLikelihood
from scipy.special import log_ndtr
from scipy.stats import kendalltau


def canonical_hash(value: dict[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def load_config(path: str | Path) -> tuple[dict[str, Any], str]:
    value = json.loads(Path(path).read_text())
    return value, canonical_hash(value)


@dataclass(frozen=True)
class FrozenState:
    seed: int
    train_x: torch.Tensor
    train_x_raw: torch.Tensor
    train_outputs: torch.Tensor
    incumbent: float
    feasible_count: int


@dataclass(frozen=True)
class OutputTransform:
    center: float
    scale: float

    def standardize(self, value: torch.Tensor) -> torch.Tensor:
        return (value - self.center) / self.scale

    def untransform_moments(
        self, mean: torch.Tensor, variance: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.center + self.scale * mean, self.scale**2 * variance


@dataclass
class FittedOutputGP:
    model: SingleTaskGP
    transform: OutputTransform
    record: dict[str, Any]


def normalize_inputs(raw_x: torch.Tensor, bounds: torch.Tensor) -> torch.Tensor:
    return (raw_x - bounds[0]) / (bounds[1] - bounds[0])


def unnormalize_inputs(unit_x: torch.Tensor, bounds: torch.Tensor) -> torch.Tensor:
    return bounds[0] + (bounds[1] - bounds[0]) * unit_x


def build_state(config: dict[str, Any], seed: int) -> FrozenState:
    """Build one prospectively frozen state without inspecting acquisition results."""

    problem = WeldedBeamSO(dtype=torch.double)
    state_config = config["states"]
    anchor_raw = torch.tensor(
        [state_config["fixed_feasible_anchor_raw"]], dtype=torch.double
    )
    anchor_slack = problem.evaluate_slack(anchor_raw, noise=False)
    if not bool(torch.all(anchor_slack > 0.0)):
        raise RuntimeError("the frozen anchor must be strictly feasible")
    anchor = normalize_inputs(anchor_raw, problem.bounds)
    sobol = torch.quasirandom.SobolEngine(
        problem.dim, scramble=True, seed=seed
    ).draw(int(state_config["sobol_points_per_state"]))
    unit_x = torch.cat([anchor, sobol.to(torch.double)], dim=0)
    if len(unit_x) != int(state_config["training_size"]):
        raise RuntimeError("frozen training-size contract was violated")
    raw_x = unnormalize_inputs(unit_x, problem.bounds)
    objective = -problem.evaluate_true(raw_x)
    slacks = problem.evaluate_slack(raw_x, noise=False)
    outputs = torch.cat([objective[:, None], slacks], dim=-1)
    feasible = torch.all(slacks >= 0.0, dim=-1)
    if not bool(feasible.any()):
        raise RuntimeError("frozen feasible anchor did not define an incumbent")
    return FrozenState(
        seed=seed,
        train_x=unit_x,
        train_x_raw=raw_x,
        train_outputs=outputs,
        incumbent=float(objective[feasible].max()),
        feasible_count=int(feasible.sum()),
    )


def build_candidate_set(config: dict[str, Any]) -> torch.Tensor:
    candidate_config = config["candidates"]
    return torch.quasirandom.SobolEngine(
        int(config["problem"]["dimension"]),
        scramble=True,
        seed=int(candidate_config["seed"]),
    ).draw(int(candidate_config["count"])).to(torch.double)


def fit_independent_gps(
    state: FrozenState, config: dict[str, Any]
) -> list[FittedOutputGP]:
    gp_config = config["gp"]
    fitted: list[FittedOutputGP] = []
    for output_index in range(state.train_outputs.shape[-1]):
        raw_y = state.train_outputs[:, output_index]
        center = float(raw_y.mean())
        scale = max(
            float(torch.sqrt(torch.mean((raw_y - center).square()))),
            float(gp_config["standardization_scale_floor"]),
        )
        transform = OutputTransform(center=center, scale=scale)
        standardized = transform.standardize(raw_y)[:, None]
        noise = torch.full_like(
            standardized, float(gp_config["standardized_fixed_noise_variance"])
        )
        model = SingleTaskGP(
            state.train_x,
            standardized,
            train_Yvar=noise,
            outcome_transform=None,
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        started = time.perf_counter()
        result = fit_gpytorch_mll_scipy(
            mll, options={"maxiter": int(gp_config["maximum_iterations"])}
        )
        elapsed = time.perf_counter() - started
        model.eval()
        fitted.append(
            FittedOutputGP(
                model=model,
                transform=transform,
                record={
                    "seed": state.seed,
                    "output_index": output_index,
                    "role": "objective" if output_index == 0 else f"constraint_{output_index}",
                    "center": center,
                    "scale": scale,
                    "optimizer_status": result.status.name,
                    "converged": result.status.name == "SUCCESS",
                    "iterations": int(result.step),
                    "negative_mll": float(result.fval),
                    "message": result.message or "",
                    "fit_seconds": elapsed,
                },
            )
        )
    return fitted


def posterior_marginals(
    fitted: list[FittedOutputGP],
    candidate_x: torch.Tensor,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    chunk_size = int(config["candidates"]["prediction_chunk_size"])
    floor = float(config["gp"]["posterior_variance_floor"])
    means: list[np.ndarray] = []
    variances: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(candidate_x), chunk_size):
            chunk = candidate_x[start : start + chunk_size]
            chunk_means = []
            chunk_variances = []
            for fitted_output in fitted:
                posterior = fitted_output.model.posterior(
                    chunk, observation_noise=False
                )
                mean, variance = fitted_output.transform.untransform_moments(
                    posterior.mean.squeeze(-1), posterior.variance.squeeze(-1)
                )
                chunk_means.append(mean)
                chunk_variances.append(variance.clamp_min(floor))
            means.append(torch.stack(chunk_means, dim=-1).cpu().numpy())
            variances.append(torch.stack(chunk_variances, dim=-1).cpu().numpy())
    return np.concatenate(means), np.concatenate(variances)


def _asymptotic_positive_tail_factors(t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return h1/phi and h2/phi for z=-t using asymptotic series."""

    first = np.zeros_like(t)
    second = np.zeros_like(t)
    double_factorial = 1.0
    sign = 1.0
    for order in range(1, 13):
        if order > 1:
            double_factorial *= 2 * order - 1
            sign *= -1.0
        first += sign * double_factorial / np.power(t, 2 * order)
        second += (
            sign
            * (2 * order)
            * double_factorial
            / np.power(t, 2 * order + 1)
        )
    if np.any(first <= 0.0) or np.any(second <= 0.0):
        raise FloatingPointError("negative-tail moment expansion lost positivity")
    return first, second


def gaussian_improvement_log_moments(
    mean: np.ndarray | float,
    variance: np.ndarray | float,
    best_f: float,
    *,
    negative_tail_switch: float = -8.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Stable logs of the first two raw-improvement moments."""

    mean_array, variance_array = np.broadcast_arrays(
        np.asarray(mean, dtype=np.float64), np.asarray(variance, dtype=np.float64)
    )
    sigma = np.sqrt(np.maximum(variance_array, np.finfo(np.float64).tiny))
    z = (mean_array - best_f) / sigma
    log_phi = -0.5 * z**2 - 0.5 * math.log(2.0 * math.pi)
    log_first = np.empty_like(z)
    log_second = np.empty_like(z)
    tail = z < negative_tail_switch
    regular = ~tail
    if np.any(regular):
        zr = z[regular]
        cdf = np.exp(log_ndtr(zr))
        pdf = np.exp(-0.5 * zr**2) / math.sqrt(2.0 * math.pi)
        first_factor = np.maximum(pdf + zr * cdf, np.finfo(np.float64).tiny)
        second_factor = np.maximum(
            (zr**2 + 1.0) * cdf + zr * pdf, np.finfo(np.float64).tiny
        )
        log_first[regular] = np.log(sigma[regular]) + np.log(first_factor)
        log_second[regular] = 2.0 * np.log(sigma[regular]) + np.log(second_factor)
    if np.any(tail):
        first_factor, second_factor = _asymptotic_positive_tail_factors(-z[tail])
        log_first[tail] = np.log(sigma[tail]) + log_phi[tail] + np.log(first_factor)
        log_second[tail] = (
            2.0 * np.log(sigma[tail]) + log_phi[tail] + np.log(second_factor)
        )
    return log_first, log_second


def exact_constrained_ei(
    means: np.ndarray,
    variances: np.ndarray,
    best_f: float,
    config: dict[str, Any],
) -> dict[str, np.ndarray]:
    if means.shape != variances.shape or means.shape[-1] != 7:
        raise ValueError("means and variances must have shape candidate x 7")
    log_first, log_second = gaussian_improvement_log_moments(
        means[:, 0],
        variances[:, 0],
        best_f,
        negative_tail_switch=float(config["numerics"]["negative_tail_switch"]),
    )
    constraint_z = means[:, 1:] / np.sqrt(variances[:, 1:])
    log_feasibility = log_ndtr(constraint_z).sum(axis=-1)
    log_acquisition = log_first + log_feasibility
    d2 = log_second - 2.0 * log_first - log_feasibility
    d2 = np.maximum(d2, 0.0)
    return {
        "log_ei": log_first,
        "ei": np.exp(log_first),
        "log_feasibility": log_feasibility,
        "feasibility": np.exp(log_feasibility),
        "log_acquisition": log_acquisition,
        "acquisition": np.exp(log_acquisition),
        "d2": d2,
        "chi_square": np.expm1(np.minimum(d2, 700.0)),
        "ess_fraction": np.exp(-d2),
    }


def qmc_standard_normals(sample_count: int, dimension: int, seed: int) -> np.ndarray:
    if sample_count <= 0 or sample_count & (sample_count - 1):
        raise ValueError("Sobol sample_count must be a positive power of two")
    uniforms = torch.quasirandom.SobolEngine(
        dimension, scramble=True, seed=seed
    ).draw(sample_count).to(torch.double)
    epsilon = torch.finfo(torch.double).eps
    return torch.special.ndtri(uniforms.clamp(epsilon, 1.0 - epsilon)).numpy()


def pairwise_ranking_disagreement(
    exact_values: np.ndarray,
    estimated_values: np.ndarray,
    *,
    tie_credit: float = 0.5,
) -> float:
    exact = np.asarray(exact_values)
    estimate = np.asarray(estimated_values)
    left, right = np.triu_indices(len(exact), k=1)
    exact_sign = np.sign(exact[left] - exact[right])
    estimate_sign = np.sign(estimate[left] - estimate[right])
    informative = exact_sign != 0
    if not np.any(informative):
        return 0.0
    product = exact_sign[informative] * estimate_sign[informative]
    return float(np.mean(np.where(product < 0, 1.0, np.where(product == 0, tie_credit, 0.0))))


def qmc_candidate_metrics(
    means: np.ndarray,
    variances: np.ndarray,
    best_f: float,
    exact: dict[str, np.ndarray],
    config: dict[str, Any],
    state_seed: int,
) -> list[dict[str, Any]]:
    candidate_config = config["candidates"]
    qmc_config = config["qmc"]
    sample_counts = [int(value) for value in qmc_config["sample_counts"]]
    maximum = max(sample_counts)
    chunk_size = int(candidate_config["qmc_chunk_size"])
    top_k = min(int(candidate_config["top_k"]), len(means))
    exact_log = exact["log_acquisition"]
    exact_best = int(np.argmax(exact_log))
    maximum_log = float(exact_log[exact_best])
    exact_top = np.argsort(-exact_log, kind="stable")[:top_k]
    exact_top_values = exact_log[exact_top]
    state_offset = config["states"]["seeds"].index(state_seed) * 1000
    rows: list[dict[str, Any]] = []
    for repetition in range(int(qmc_config["scramble_repetitions"])):
        qmc_seed = int(qmc_config["scramble_seed_base"]) + state_offset + repetition
        normals = qmc_standard_normals(maximum, 7, qmc_seed)
        estimates = np.empty((len(sample_counts), len(means)), dtype=np.float64)
        for start in range(0, len(means), chunk_size):
            stop = min(start + chunk_size, len(means))
            samples = means[start:stop, None, :] + np.sqrt(
                variances[start:stop, None, :]
            ) * normals[None, :, :]
            utility = np.maximum(samples[..., 0] - best_f, 0.0)
            utility *= np.all(samples[..., 1:] >= 0.0, axis=-1)
            cumulative = np.cumsum(utility, axis=1)
            for count_index, count in enumerate(sample_counts):
                estimates[count_index, start:stop] = cumulative[:, count - 1] / count
        for count_index, count in enumerate(sample_counts):
            estimate = estimates[count_index]
            selected = int(np.argmax(estimate))
            estimated_top = np.argsort(-estimate, kind="stable")[:top_k]
            tau = kendalltau(exact_top_values, estimate[exact_top], variant="b").statistic
            rows.append(
                {
                    "state_seed": state_seed,
                    "sample_count": count,
                    "repetition": repetition,
                    "qmc_seed": qmc_seed,
                    "selected_candidate": selected,
                    "exact_best_selected": selected == exact_best,
                    "one_percent_optimal_selected": bool(
                        exact_log[selected] >= maximum_log + math.log(0.99)
                    ),
                    "normalized_regret": float(
                        -math.expm1(min(0.0, exact_log[selected] - maximum_log))
                    ),
                    "pairwise_disagreement_top32": pairwise_ranking_disagreement(
                        exact_top_values,
                        estimate[exact_top],
                        tie_credit=float(config["numerics"]["ranking_tie_credit"]),
                    ),
                    "kendall_tau_top32": 0.0 if np.isnan(tau) else float(tau),
                    "top32_overlap": float(
                        len(set(exact_top.tolist()) & set(estimated_top.tolist())) / top_k
                    ),
                }
            )
    return rows


def bootstrap_mean_interval(
    values: np.ndarray,
    *,
    repetitions: int,
    seed: int,
    confidence_level: float,
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(values), size=(repetitions, len(values)))
    means = values[indices].mean(axis=1)
    alpha = 1.0 - confidence_level
    return (
        float(values.mean()),
        float(np.quantile(means, alpha)),
        float(np.quantile(means, 1.0 - alpha)),
    )


def summarize_qmc_rows(
    rows: list[dict[str, Any]], config: dict[str, Any], state_seed: int
) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    numeric = config["numerics"]
    for sample_count in config["qmc"]["sample_counts"]:
        selected = [
            row
            for row in rows
            if row["state_seed"] == state_seed and row["sample_count"] == sample_count
        ]
        regrets = np.array([row["normalized_regret"] for row in selected])
        disagreements = np.array(
            [row["pairwise_disagreement_top32"] for row in selected]
        )
        regret = bootstrap_mean_interval(
            regrets,
            repetitions=int(numeric["bootstrap_repetitions"]),
            seed=int(numeric["bootstrap_seed"]) + state_seed + int(sample_count),
            confidence_level=float(numeric["confidence_level"]),
        )
        disagreement = bootstrap_mean_interval(
            disagreements,
            repetitions=int(numeric["bootstrap_repetitions"]),
            seed=int(numeric["bootstrap_seed"]) + 100000 + state_seed + int(sample_count),
            confidence_level=float(numeric["confidence_level"]),
        )
        summary.append(
            {
                "state_seed": state_seed,
                "sample_count": sample_count,
                "mean_normalized_regret": regret[0],
                "regret_lower_95": regret[1],
                "regret_upper_95": regret[2],
                "mean_pairwise_disagreement": disagreement[0],
                "pairwise_lower_95": disagreement[1],
                "pairwise_upper_95": disagreement[2],
                "exact_best_selection_probability": float(
                    np.mean([row["exact_best_selected"] for row in selected])
                ),
                "one_percent_optimal_probability": float(
                    np.mean([row["one_percent_optimal_selected"] for row in selected])
                ),
                "mean_kendall_tau": float(
                    np.mean([row["kendall_tau_top32"] for row in selected])
                ),
                "mean_top32_overlap": float(
                    np.mean([row["top32_overlap"] for row in selected])
                ),
            }
        )
    return summary


def classify_result(
    state_rows: list[dict[str, Any]],
    qmc_summary: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    valid: bool,
) -> dict[str, Any]:
    gate = config["prospective_gate"]
    state_checks: dict[str, Any] = {}
    for state in state_rows:
        seed = int(state["state_seed"])
        by_count = {
            int(row["sample_count"]): row
            for row in qmc_summary
            if int(row["state_seed"]) == seed
        }
        failure = by_count[int(gate["qmc_failure_sample_count"])]
        shift_positive = (
            state["top32_median_ess_fraction"]
            <= gate["shift_positive_top32_median_ess_maximum"]
        )
        qmc_failure = (
            failure["regret_lower_95"]
            > gate["qmc_failure_regret_lower_bound_minimum"]
            and failure["pairwise_lower_95"]
            > gate["qmc_failure_pairwise_disagreement_lower_bound_minimum"]
        )
        shift_negative = (
            state["top32_median_ess_fraction"]
            >= gate["shift_negative_top32_median_ess_minimum"]
        )
        reliable = all(
            by_count[int(count)]["regret_upper_95"]
            < gate["qmc_reliable_regret_upper_bound_maximum"]
            and by_count[int(count)]["pairwise_upper_95"]
            < gate["qmc_reliable_pairwise_disagreement_upper_bound_maximum"]
            for count in gate["qmc_reliable_sample_counts"]
        )
        state_checks[str(seed)] = {
            "shift_positive": bool(shift_positive),
            "qmc_failure_positive": bool(qmc_failure),
            "joint_positive": bool(shift_positive and qmc_failure),
            "shift_negative": bool(shift_negative),
            "qmc_reliable": bool(reliable),
            "conclusively_negative": bool(shift_negative or reliable),
        }
    positive = sum(value["joint_positive"] for value in state_checks.values())
    negative = sum(
        value["conclusively_negative"] for value in state_checks.values()
    )
    if not valid:
        status = "WELDED_BEAM_SHIFT_INCONCLUSIVE_REVIEW_REQUIRED"
    elif positive >= int(gate["minimum_positive_states"]):
        status = "WELDED_BEAM_SHIFT_POSITIVE_REVIEW_REQUIRED"
    elif negative >= int(gate["minimum_negative_states"]):
        status = "WELDED_BEAM_SHIFT_NEGATIVE_REVIEW_REQUIRED"
    else:
        status = "WELDED_BEAM_SHIFT_INCONCLUSIVE_REVIEW_REQUIRED"
    return {
        "status": status,
        "valid": valid,
        "positive_state_count": positive,
        "negative_state_count": negative,
        "state_checks": state_checks,
    }
