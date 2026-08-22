"""Finite-grid preference-informed Bayesian optimization components.

This module implements the frozen E3 phenomenon pilot with an exact scalar-data
GP reference, logistic historical preferences, Laplace-preconditioned
self-normalized importance sampling, and separate adaptive factor screening.
The empirical inference diagnostics here are not rigorous certificates.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import expit, logsumexp, ndtr

from conditioned_bo.preference_influence import (
    comparison_diagnostics,
    comparison_matrix_from_precision,
    ei_decision_footprint,
    factor_block_metadata,
    factor_sensitivity_matrix,
    omitted_factor_load,
    preference_blocks,
    ranked_omitted_contributions,
    structural_bound,
)


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


REQUIRED_TRAJECTORY_FIELDS = {
    "method",
    "seed",
    "bo_iteration",
    "selected_action_index",
    "selected_action_x",
    "observed_scalar_outcome",
    "incumbent",
    "noise_free_best_value_observed",
    "simple_regret",
    "t_0_10_status",
    "t_0_10",
    "active_factor_indices",
    "M_t",
    "N",
    "cumulative_final_active_factor_use",
    "raw_factor_likelihood_evaluation_count",
    "decision_factor_likelihood_evaluation_count",
    "heldout_factor_likelihood_evaluation_count",
    "inference_sample_count",
    "inference_ess",
    "inference_ess_fraction",
    "inference_wall_time_seconds",
    "heldout_full_target_sample_count",
    "heldout_full_target_ess",
    "heldout_full_target_ess_fraction",
    "heldout_full_target_wall_time_seconds",
    "selected_action_acquisition_estimate",
    "structural_bound_at_worst_challenger",
    "empirical_inference_allowance",
    "adaptive_heldout_full_target_acquisition_regret",
    "config_sha256",
    "starting_git_commit",
    "preference_bank_sha256",
    "scalar_noise_sha256",
    "screened_factor_source",
    "scalar_observation_count",
}


@dataclass(frozen=True)
class GPPosterior:
    """Exact finite-grid scalar-data GP posterior."""

    mean: FloatArray
    covariance: FloatArray
    precision: FloatArray


@dataclass(frozen=True)
class PreparedSeedInputs:
    """Preference bank and location-indexed scalar noise generated pre-method."""

    seed: int
    preference_signs: IntArray
    preference_probabilities: FloatArray
    scalar_noise: FloatArray
    preference_bank_sha256: str
    scalar_noise_sha256: str
    generated_before_methods: bool = True


@dataclass(frozen=True)
class LaplaceResult:
    """Mode, local precision, and deterministic optimization diagnostics."""

    mode: FloatArray
    hessian: FloatArray
    gradient_infinity_norm: float
    iterations: int
    factor_likelihood_evaluations: int
    converged: bool


@dataclass(frozen=True)
class ImportanceSampleResult:
    """One self-normalized importance calculation and retained diagnostics."""

    acquisition: FloatArray
    split_half_acquisition_1: FloatArray
    split_half_acquisition_2: FloatArray
    maximum_split_half_discrepancy: float
    normalized_weights: FloatArray
    utilities: FloatArray
    log_weights: FloatArray
    ess: float
    ess_fraction: float
    draws: int
    active_factor_count: int
    factor_likelihood_evaluations: int
    wall_time_seconds: float
    laplace: LaplaceResult
    accuracy_reliable: bool = True


@dataclass(frozen=True)
class NumericalSettings:
    """Only numerical sample schedules; scientific settings remain in JSON."""

    working_draws_per_batch_schedule: tuple[int, ...]
    full_draws_schedule: tuple[int, ...]

    def __post_init__(self) -> None:
        for schedule in (
            self.working_draws_per_batch_schedule,
            self.full_draws_schedule,
        ):
            if not schedule or any(value < 2 for value in schedule):
                raise ValueError("sample schedules must contain values of at least two")
            if any(later <= earlier for earlier, later in zip(schedule, schedule[1:])):
                raise ValueError("sample schedules must be strictly increasing")


@dataclass(frozen=True)
class MethodTrajectoryResult:
    """Machine-readable rows produced by one method/seed trajectory."""

    trajectory_rows: list[dict[str, Any]]
    refinement_rows: list[dict[str, Any]]
    acquisition_rows: list[dict[str, Any]]
    numerical_caveats: list[str]


@dataclass(frozen=True)
class SeedRunResult:
    """Combined result rows for the three frozen methods for one seed."""

    trajectory_rows: list[dict[str, Any]]
    refinement_rows: list[dict[str, Any]]
    acquisition_rows: list[dict[str, Any]]
    numerical_caveats: list[str]


@dataclass(frozen=True)
class _RawImportanceBatch:
    utilities: FloatArray
    log_weights: FloatArray
    factor_likelihood_evaluations: int


def load_frozen_config(path: str | Path) -> dict[str, Any]:
    """Load and minimally validate the preregistered JSON configuration."""

    config = json.loads(Path(path).read_text())
    if config.get("status") != "preregistered_frozen":
        raise ValueError("configuration is not marked preregistered_frozen")
    if config.get("methods") != ["standard", "full", "adaptive"]:
        raise ValueError("the frozen pilot must contain standard/full/adaptive only")
    if config["scalar_observations"].get("screened") is not False:
        raise ValueError("scalar observations must never be screened")
    return config


def numerical_settings_from_config(
    config: Mapping[str, Any], *, smoke: bool = False
) -> NumericalSettings:
    """Select the preregistered full or smoke numerical profile."""

    if smoke:
        smoke_config = config["smoke_test"]
        return NumericalSettings(
            working_draws_per_batch_schedule=tuple(
                int(value)
                for value in smoke_config["working_draws_per_batch_schedule"]
            ),
            full_draws_schedule=(
                int(smoke_config["full_initial_draws"]),
                int(smoke_config["full_accuracy_refinement_draws"]),
            ),
        )
    importance = config["importance_sampling"]
    return NumericalSettings(
        working_draws_per_batch_schedule=tuple(
            int(value)
            for value in importance["adaptive_working"][
                "draws_per_batch_schedule"
            ]
        ),
        full_draws_schedule=(
            int(importance["full_target"]["initial_draws"]),
            int(importance["full_target"]["accuracy_refinement_draws"]),
        ),
    )


def action_grid(config: Mapping[str, Any]) -> FloatArray:
    """Return the frozen equally spaced finite action grid."""

    settings = config["action_grid"]
    return np.linspace(
        float(settings["minimum"]),
        float(settings["maximum"]),
        int(settings["count"]),
        dtype=float,
    )


def true_objective(x: ArrayLike) -> FloatArray:
    """Evaluate the frozen two-mode synthetic objective."""

    values = np.asarray(x, dtype=float)
    return np.asarray(
        0.55 * np.exp(-0.5 * ((values - 0.22) / 0.07) ** 2)
        + 1.00 * np.exp(-0.5 * ((values - 0.76) / 0.065) ** 2)
        + 0.04 * np.sin(4.0 * np.pi * values),
        dtype=float,
    )


def ou_covariance(
    points: ArrayLike, kernel_amplitude: float, kernel_lengthscale: float
) -> FloatArray:
    """Return the frozen OU covariance matrix on arbitrary finite points."""

    values = np.asarray(points, dtype=float)
    if values.ndim != 1:
        raise ValueError("points must be one-dimensional")
    if kernel_amplitude <= 0.0 or kernel_lengthscale <= 0.0:
        raise ValueError("kernel parameters must be positive")
    return np.asarray(
        kernel_amplitude
        * np.exp(-np.abs(values[:, None] - values[None, :]) / kernel_lengthscale),
        dtype=float,
    )


def gp_reference_posterior(
    grid: ArrayLike,
    observed_indices: Sequence[int],
    observed_values: ArrayLike,
    kernel_amplitude: float,
    kernel_lengthscale: float,
    observation_noise_standard_deviation: float,
) -> GPPosterior:
    """Construct the exact finite-grid GP posterior from scalar data only."""

    points = np.asarray(grid, dtype=float)
    indices = np.asarray(tuple(observed_indices), dtype=np.int64)
    observations = np.asarray(observed_values, dtype=float)
    if indices.shape != observations.shape:
        raise ValueError("observed indices and values must match")
    if indices.size != np.unique(indices).size:
        raise ValueError("repeated scalar action indices are prohibited")
    if indices.size and (indices.min() < 0 or indices.max() >= points.size):
        raise ValueError("observed action index is outside the grid")
    if observation_noise_standard_deviation <= 0.0:
        raise ValueError("observation noise must be positive")

    covariance_prior = ou_covariance(
        points, kernel_amplitude, kernel_lengthscale
    )
    precision_prior = np.linalg.solve(
        covariance_prior, np.eye(points.size, dtype=float)
    )
    precision = np.asarray(precision_prior, dtype=float)
    natural_parameter = np.zeros(points.size, dtype=float)
    noise_variance = observation_noise_standard_deviation**2
    if indices.size:
        precision[indices, indices] += 1.0 / noise_variance
        natural_parameter[indices] += observations / noise_variance
    precision = 0.5 * (precision + precision.T)
    mean = np.linalg.solve(precision, natural_parameter)
    covariance = np.linalg.solve(precision, np.eye(points.size, dtype=float))
    covariance = 0.5 * (covariance + covariance.T)
    return GPPosterior(mean=mean, covariance=covariance, precision=precision)


def analytic_expected_improvement(
    mean: ArrayLike, variance: ArrayLike, incumbent: float
) -> FloatArray:
    """Evaluate ordinary Gaussian expected improvement elementwise."""

    means = np.asarray(mean, dtype=float)
    variances = np.asarray(variance, dtype=float)
    if means.shape != variances.shape or np.any(variances < -1e-12):
        raise ValueError("mean and nonnegative variance arrays must match")
    variances = np.maximum(variances, 0.0)
    sigma = np.sqrt(variances)
    centered = means - float(incumbent)
    result = np.maximum(centered, 0.0)
    positive = sigma > 0.0
    if np.any(positive):
        z = centered[positive] / sigma[positive]
        density = np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)
        result[positive] = centered[positive] * ndtr(z) + sigma[positive] * density
    return np.asarray(result, dtype=float)


def select_unobserved_argmax(
    acquisition: ArrayLike, observed_indices: Sequence[int]
) -> int:
    """Exhaustively select the first maximizing unobserved finite-grid action."""

    values = np.asarray(acquisition, dtype=float)
    if values.ndim != 1:
        raise ValueError("acquisition must be one-dimensional")
    observed = np.asarray(tuple(observed_indices), dtype=np.int64)
    if observed.size and (observed.min() < 0 or observed.max() >= values.size):
        raise ValueError("observed action index is outside the grid")
    eligible = np.ones(values.size, dtype=bool)
    eligible[observed] = False
    if not np.any(eligible):
        raise ValueError("no unobserved actions remain")
    masked = np.where(eligible, values, -np.inf)
    return int(np.argmax(masked))


def _active_array(active_factors: Sequence[int], n_factors: int) -> IntArray:
    active = np.asarray(tuple(active_factors), dtype=np.int64)
    if active.size and (active.min() < 0 or active.max() >= n_factors):
        raise ValueError("active preference index is out of range")
    if np.unique(active).size != active.size:
        raise ValueError("active preference indices must be unique")
    return np.sort(active)


def preference_energy_batch(
    samples: ArrayLike,
    endpoint_pairs: ArrayLike,
    signs: ArrayLike,
    temperature: float,
    active_factors: Sequence[int],
) -> FloatArray:
    """Evaluate the summed active logistic energy for a sample matrix."""

    values = np.asarray(samples, dtype=float)
    pairs = np.asarray(endpoint_pairs, dtype=np.int64)
    sign_values = np.asarray(signs, dtype=np.int64)
    if values.ndim != 2 or pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("samples or endpoint pair shape is invalid")
    if sign_values.shape != (pairs.shape[0],) or not np.all(
        np.isin(sign_values, (-1, 1))
    ):
        raise ValueError("one +/-1 sign is required per preference factor")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    active = _active_array(active_factors, pairs.shape[0])
    if active.size == 0:
        return np.zeros(values.shape[0], dtype=float)
    selected = pairs[active]
    z = (
        sign_values[active]
        * (values[:, selected[:, 0]] - values[:, selected[:, 1]])
        / temperature
    )
    return np.asarray(np.logaddexp(0.0, -z).sum(axis=1), dtype=float)


def _preference_potential(
    latent: FloatArray,
    posterior: GPPosterior,
    endpoint_pairs: IntArray,
    signs: IntArray,
    temperature: float,
    active: IntArray,
    *,
    derivatives: bool,
) -> tuple[float, FloatArray | None, FloatArray | None, int]:
    centered = latent - posterior.mean
    energy = 0.5 * float(centered @ posterior.precision @ centered)
    gradient = posterior.precision @ centered if derivatives else None
    hessian = posterior.precision.copy() if derivatives else None
    for factor_index in active:
        left, right = endpoint_pairs[factor_index]
        sign = int(signs[factor_index])
        z = sign * (latent[left] - latent[right]) / temperature
        energy += float(np.logaddexp(0.0, -z))
        if derivatives:
            q = float(expit(-z))
            coefficient = -sign * q / temperature
            gradient[left] += coefficient
            gradient[right] -= coefficient
            curvature = q * (1.0 - q) / temperature**2
            hessian[left, left] += curvature
            hessian[right, right] += curvature
            hessian[left, right] -= curvature
            hessian[right, left] -= curvature
    return float(energy), gradient, hessian, int(active.size)


def laplace_preference_mode(
    posterior: GPPosterior,
    endpoint_pairs: ArrayLike,
    signs: ArrayLike,
    temperature: float,
    active_factors: Sequence[int],
    laplace_settings: Mapping[str, Any],
) -> LaplaceResult:
    """Find the unique active-target mode by damped exact Newton steps."""

    pairs = np.asarray(endpoint_pairs, dtype=np.int64)
    sign_values = np.asarray(signs, dtype=np.int64)
    active = _active_array(active_factors, pairs.shape[0])
    tolerance = float(laplace_settings["gradient_infinity_norm_tolerance"])
    maximum_iterations = int(laplace_settings["maximum_newton_iterations"])
    backtracking = bool(laplace_settings["backtracking_line_search"])
    latent = posterior.mean.copy()
    factor_evaluations = 0
    final_hessian = posterior.precision.copy()
    final_gradient_norm = np.inf

    for iteration in range(maximum_iterations + 1):
        potential, gradient, hessian, evaluated = _preference_potential(
            latent,
            posterior,
            pairs,
            sign_values,
            temperature,
            active,
            derivatives=True,
        )
        factor_evaluations += evaluated
        assert gradient is not None and hessian is not None
        final_hessian = hessian
        final_gradient_norm = float(np.linalg.norm(gradient, ord=np.inf))
        if final_gradient_norm <= tolerance:
            return LaplaceResult(
                mode=latent,
                hessian=final_hessian,
                gradient_infinity_norm=final_gradient_norm,
                iterations=iteration,
                factor_likelihood_evaluations=factor_evaluations,
                converged=True,
            )
        if iteration == maximum_iterations:
            break
        step = np.linalg.solve(hessian, gradient)
        step_decrement = float(gradient @ step)
        scale = 1.0
        roundoff_descent_floor = (
            8.0 * np.finfo(float).eps * max(1.0, abs(potential))
        )
        while True:
            candidate = latent - scale * step
            candidate_potential, _, _, evaluated = _preference_potential(
                candidate,
                posterior,
                pairs,
                sign_values,
                temperature,
                active,
                derivatives=False,
            )
            factor_evaluations += evaluated
            if (
                not backtracking
                or candidate_potential
                <= potential - 1e-4 * scale * step_decrement
                or (scale == 1.0 and step_decrement <= roundoff_descent_floor)
            ):
                latent = candidate
                break
            scale *= 0.5
            if scale < 2.0**-50:
                raise RuntimeError("Laplace Newton line search failed")

    return LaplaceResult(
        mode=latent,
        hessian=final_hessian,
        gradient_infinity_norm=final_gradient_norm,
        iterations=maximum_iterations,
        factor_likelihood_evaluations=factor_evaluations,
        converged=False,
    )


def _draw_importance_batch(
    *,
    posterior: GPPosterior,
    endpoint_pairs: IntArray,
    signs: IntArray,
    temperature: float,
    active: IntArray,
    incumbent: float,
    draws: int,
    rng: np.random.Generator,
    laplace: LaplaceResult,
) -> _RawImportanceBatch:
    if draws < 1:
        raise ValueError("draws must be positive")
    cholesky = np.linalg.cholesky(laplace.hessian)
    standard = rng.standard_normal((draws, posterior.mean.size))
    deviations = np.linalg.solve(cholesky.T, standard.T).T
    samples = laplace.mode + deviations
    centered_reference = samples - posterior.mean
    gaussian_energy = 0.5 * np.einsum(
        "ni,ij,nj->n", centered_reference, posterior.precision, centered_reference
    )
    preference_energy = preference_energy_batch(
        samples, endpoint_pairs, signs, temperature, active
    )
    proposal_quadratic = 0.5 * np.einsum(
        "ni,ij,nj->n", deviations, laplace.hessian, deviations
    )
    log_weights = -gaussian_energy - preference_energy + proposal_quadratic
    utilities = np.maximum(samples - float(incumbent), 0.0)
    return _RawImportanceBatch(
        utilities=np.asarray(utilities, dtype=float),
        log_weights=np.asarray(log_weights, dtype=float),
        factor_likelihood_evaluations=int(draws * active.size),
    )


def _concatenate_raw(chunks: Sequence[_RawImportanceBatch]) -> _RawImportanceBatch:
    if not chunks:
        raise ValueError("at least one importance chunk is required")
    return _RawImportanceBatch(
        utilities=np.concatenate([chunk.utilities for chunk in chunks], axis=0),
        log_weights=np.concatenate([chunk.log_weights for chunk in chunks]),
        factor_likelihood_evaluations=sum(
            chunk.factor_likelihood_evaluations for chunk in chunks
        ),
    )


def _normalized_weights(log_weights: FloatArray) -> FloatArray:
    return np.asarray(np.exp(log_weights - logsumexp(log_weights)), dtype=float)


def _weighted_acquisition(utilities: FloatArray, log_weights: FloatArray) -> FloatArray:
    return np.asarray(_normalized_weights(log_weights) @ utilities, dtype=float)


def _summarize_importance(
    raw: _RawImportanceBatch,
    laplace: LaplaceResult,
    active_count: int,
    elapsed: float,
    *,
    accuracy_reliable: bool = True,
) -> ImportanceSampleResult:
    weights = _normalized_weights(raw.log_weights)
    acquisition = np.asarray(weights @ raw.utilities, dtype=float)
    midpoint = raw.utilities.shape[0] // 2
    first = _weighted_acquisition(raw.utilities[:midpoint], raw.log_weights[:midpoint])
    second = _weighted_acquisition(raw.utilities[midpoint:], raw.log_weights[midpoint:])
    ess = min(float(weights.size), float(1.0 / np.square(weights).sum()))
    return ImportanceSampleResult(
        acquisition=acquisition,
        split_half_acquisition_1=first,
        split_half_acquisition_2=second,
        maximum_split_half_discrepancy=float(np.max(np.abs(first - second))),
        normalized_weights=weights,
        utilities=raw.utilities,
        log_weights=raw.log_weights,
        ess=ess,
        ess_fraction=ess / weights.size,
        draws=weights.size,
        active_factor_count=active_count,
        factor_likelihood_evaluations=(
            laplace.factor_likelihood_evaluations
            + raw.factor_likelihood_evaluations
        ),
        wall_time_seconds=float(elapsed),
        laplace=laplace,
        accuracy_reliable=accuracy_reliable,
    )


def importance_sample_preference_target(
    *,
    posterior: GPPosterior,
    endpoint_pairs: ArrayLike,
    signs: ArrayLike,
    temperature: float,
    active_factors: Sequence[int],
    incumbent: float,
    draws: int,
    rng: np.random.Generator,
    laplace_settings: Mapping[str, Any],
) -> ImportanceSampleResult:
    """Run one Laplace-preconditioned self-normalized IS calculation."""

    started = time.perf_counter()
    pairs = np.asarray(endpoint_pairs, dtype=np.int64)
    sign_values = np.asarray(signs, dtype=np.int64)
    active = _active_array(active_factors, pairs.shape[0])
    laplace = laplace_preference_mode(
        posterior, pairs, sign_values, temperature, active, laplace_settings
    )
    if not laplace.converged:
        raise RuntimeError("Laplace mode failed the frozen gradient tolerance")
    raw = _draw_importance_batch(
        posterior=posterior,
        endpoint_pairs=pairs,
        signs=sign_values,
        temperature=temperature,
        active=active,
        incumbent=incumbent,
        draws=draws,
        rng=rng,
        laplace=laplace,
    )
    return _summarize_importance(
        raw, laplace, active.size, time.perf_counter() - started
    )


def full_target_inference(
    *,
    posterior: GPPosterior,
    endpoint_pairs: ArrayLike,
    signs: ArrayLike,
    temperature: float,
    incumbent: float,
    draws: int,
    rng: np.random.Generator,
    laplace_settings: Mapping[str, Any],
) -> ImportanceSampleResult:
    """Run the identical target implementation with all factors active."""

    pairs = np.asarray(endpoint_pairs, dtype=np.int64)
    return importance_sample_preference_target(
        posterior=posterior,
        endpoint_pairs=pairs,
        signs=signs,
        temperature=temperature,
        active_factors=range(pairs.shape[0]),
        incumbent=incumbent,
        draws=draws,
        rng=rng,
        laplace_settings=laplace_settings,
    )


def _rng_for(
    config: Mapping[str, Any],
    seed: int,
    stream_name: str,
    bo_iteration: int = 0,
    refinement_index: int = 0,
    accuracy_level: int = 0,
) -> np.random.Generator:
    code = int(config["rng"]["stream_codes"][stream_name])
    sequence = np.random.SeedSequence(
        [int(seed), code, int(bo_iteration), int(refinement_index), int(accuracy_level)]
    )
    return np.random.default_rng(sequence)


def generate_preference_bank(
    grid: ArrayLike,
    endpoint_pairs: ArrayLike,
    temperature: float,
    rng: np.random.Generator,
) -> tuple[IntArray, FloatArray]:
    """Generate one complete noisy historical preference bank."""

    points = np.asarray(grid, dtype=float)
    pairs = np.asarray(endpoint_pairs, dtype=np.int64)
    objective = true_objective(points)
    logits = (objective[pairs[:, 0]] - objective[pairs[:, 1]]) / temperature
    probabilities = np.asarray(expit(logits), dtype=float)
    signs = np.where(rng.random(pairs.shape[0]) < probabilities, 1, -1)
    return np.asarray(signs, dtype=np.int64), probabilities


def generate_scalar_noise(
    grid_size: int, standard_deviation: float, rng: np.random.Generator
) -> FloatArray:
    """Generate one location-indexed scalar-noise lookup table."""

    if grid_size < 1 or standard_deviation <= 0.0:
        raise ValueError("grid size and noise standard deviation must be positive")
    return np.asarray(rng.normal(0.0, standard_deviation, size=grid_size), dtype=float)


def prepare_pilot_inputs(config: Mapping[str, Any]) -> tuple[PreparedSeedInputs, ...]:
    """Generate every bank/noise table before any method is executed."""

    grid = action_grid(config)
    preference = config["preference_bank"]
    pairs = np.asarray(preference["endpoint_index_pairs"], dtype=np.int64)
    noise_sd = float(config["scalar_observations"]["noise_standard_deviation"])
    prepared: list[PreparedSeedInputs] = []
    for seed in preference["pilot_seeds"]:
        signs, probabilities = generate_preference_bank(
            grid,
            pairs,
            float(preference["temperature"]),
            _rng_for(config, int(seed), "preference_bank"),
        )
        noise = generate_scalar_noise(
            grid.size, noise_sd, _rng_for(config, int(seed), "scalar_noise")
        )
        prepared.append(
            PreparedSeedInputs(
                seed=int(seed),
                preference_signs=signs,
                preference_probabilities=probabilities,
                scalar_noise=noise,
                preference_bank_sha256=hashlib.sha256(signs.tobytes()).hexdigest(),
                scalar_noise_sha256=hashlib.sha256(noise.tobytes()).hexdigest(),
            )
        )
    return tuple(prepared)


def _posterior_from_history(
    config: Mapping[str, Any],
    grid: FloatArray,
    observed_indices: Sequence[int],
    observed_values: Sequence[float],
) -> GPPosterior:
    gp = config["gp_reference"]
    return gp_reference_posterior(
        grid,
        observed_indices,
        observed_values,
        float(gp["kernel_amplitude"]),
        float(gp["kernel_lengthscale"]),
        float(gp["observation_noise_standard_deviation"]),
    )


def _initial_history(
    config: Mapping[str, Any], seed_input: PreparedSeedInputs, objective: FloatArray
) -> tuple[list[int], list[float]]:
    indices = [
        int(index)
        for index in config["scalar_observations"]["initial_action_indices"]
    ]
    values = [
        float(objective[index] + seed_input.scalar_noise[index]) for index in indices
    ]
    return indices, values


def _complete_t_metric(rows: list[dict[str, Any]], threshold: float, unreached: int) -> int:
    reached = next(
        (
            int(row["bo_iteration"])
            for row in rows
            if float(row["simple_regret"]) <= threshold
        ),
        unreached,
    )
    for row in rows:
        row["t_0_10"] = reached
        row["t_0_10_status"] = (
            "reached" if int(row["bo_iteration"]) >= reached and reached != unreached else "not_reached"
        )
    return reached


def _trajectory_row(
    *,
    config: Mapping[str, Any],
    seed_input: PreparedSeedInputs,
    config_sha256: str,
    method: str,
    bo_iteration: int,
    action_index: int,
    grid: FloatArray,
    observation: float,
    incumbent_before: float,
    incumbent_after: float,
    observed_indices: Sequence[int],
    objective: FloatArray,
    active_factors: Sequence[int],
    raw_factor_evaluations: int,
    decision_factor_evaluations: int,
    heldout_factor_evaluations: int,
    inference_sample_count: int,
    inference_ess: float | None,
    inference_ess_fraction: float | None,
    inference_wall_time: float,
    heldout_sample_count: int | None,
    heldout_ess: float | None,
    heldout_ess_fraction: float | None,
    heldout_wall_time: float | None,
    selected_acquisition: float,
    structural_bound_value: float | None,
    inference_allowance: float | None,
    heldout_regret: float | None,
    numerical_accuracy_reliable: bool,
) -> dict[str, Any]:
    best_true = float(objective[np.asarray(observed_indices, dtype=int)].max())
    simple_regret = float(objective.max() - best_true)
    return {
        "method": method,
        "seed": seed_input.seed,
        "bo_iteration": int(bo_iteration),
        "selected_action_index": int(action_index),
        "selected_action_x": float(grid[action_index]),
        "observed_scalar_outcome": float(observation),
        "incumbent_before_observation": float(incumbent_before),
        "incumbent": float(incumbent_after),
        "noise_free_best_value_observed": best_true,
        "simple_regret": simple_regret,
        "t_0_10_status": "not_reached",
        "t_0_10": None,
        "active_factor_indices": json.dumps(sorted(int(i) for i in active_factors)),
        "M_t": len(tuple(active_factors)),
        "N": int(config["preference_bank"]["factor_count"]),
        "cumulative_factor_use_through_iteration": None,
        "cumulative_final_active_factor_use": None,
        "raw_factor_likelihood_evaluation_count": int(raw_factor_evaluations),
        "decision_factor_likelihood_evaluation_count": int(
            decision_factor_evaluations
        ),
        "heldout_factor_likelihood_evaluation_count": int(
            heldout_factor_evaluations
        ),
        "inference_sample_count": int(inference_sample_count),
        "inference_ess": inference_ess,
        "inference_ess_fraction": inference_ess_fraction,
        "inference_wall_time_seconds": float(inference_wall_time),
        "heldout_full_target_sample_count": heldout_sample_count,
        "heldout_full_target_ess": heldout_ess,
        "heldout_full_target_ess_fraction": heldout_ess_fraction,
        "heldout_full_target_wall_time_seconds": heldout_wall_time,
        "selected_action_acquisition_estimate": float(selected_acquisition),
        "structural_bound_at_worst_challenger": structural_bound_value,
        "empirical_inference_allowance": inference_allowance,
        "adaptive_heldout_full_target_acquisition_regret": heldout_regret,
        "numerical_accuracy_reliable": bool(numerical_accuracy_reliable),
        "config_sha256": config_sha256,
        "starting_git_commit": config["provenance"]["starting_git_commit"],
        "preference_bank_sha256": seed_input.preference_bank_sha256,
        "scalar_noise_sha256": seed_input.scalar_noise_sha256,
        "screened_factor_source": "historical_preference_bank_only",
        "scalar_observation_count": len(observed_indices),
    }


def _finalize_method_rows(
    rows: list[dict[str, Any]], config: Mapping[str, Any]
) -> None:
    cumulative = 0
    for row in rows:
        cumulative += int(row["M_t"])
        row["cumulative_factor_use_through_iteration"] = cumulative
    for row in rows:
        row["cumulative_final_active_factor_use"] = cumulative
    metric = config["primary_metric"]
    _complete_t_metric(
        rows,
        float(metric["simple_regret_threshold"]),
        int(metric["unreached_value"]),
    )


def _acquisition_curve_rows(
    *,
    method: str,
    seed: int,
    bo_iteration: int,
    refinement_index: int | None,
    stage: str,
    acquisition: FloatArray,
    observed_indices: Sequence[int],
    selected_action_index: int | None,
    used_for_selection: bool,
    config_sha256: str,
) -> list[dict[str, Any]]:
    observed = set(int(index) for index in observed_indices)
    return [
        {
            "method": method,
            "seed": int(seed),
            "bo_iteration": int(bo_iteration),
            "refinement_index": refinement_index,
            "stage": stage,
            "action_index": int(index),
            "acquisition_estimate": float(value),
            "was_observed_before_decision": index in observed,
            "is_selected_action": selected_action_index == index,
            "used_for_selection": bool(used_for_selection),
            "config_sha256": config_sha256,
        }
        for index, value in enumerate(acquisition)
    ]


def run_standard_trajectory(
    config: Mapping[str, Any],
    seed_input: PreparedSeedInputs,
    *,
    config_sha256: str,
    horizon: int | None = None,
) -> list[dict[str, Any]]:
    """Run the complete analytic scalar-only GP-EI trajectory."""

    grid = action_grid(config)
    objective = true_objective(grid)
    observed_indices, observed_values = _initial_history(config, seed_input, objective)
    total_horizon = int(
        config["scalar_observations"]["post_initial_horizon"]
        if horizon is None
        else horizon
    )
    rows: list[dict[str, Any]] = []
    for iteration in range(1, total_horizon + 1):
        posterior = _posterior_from_history(
            config, grid, observed_indices, observed_values
        )
        incumbent_before = float(max(observed_values))
        acquisition = analytic_expected_improvement(
            posterior.mean, np.diag(posterior.covariance), incumbent_before
        )
        selected = select_unobserved_argmax(acquisition, observed_indices)
        observation = float(objective[selected] + seed_input.scalar_noise[selected])
        observed_indices.append(selected)
        observed_values.append(observation)
        rows.append(
            _trajectory_row(
                config=config,
                seed_input=seed_input,
                config_sha256=config_sha256,
                method="standard",
                bo_iteration=iteration,
                action_index=selected,
                grid=grid,
                observation=observation,
                incumbent_before=incumbent_before,
                incumbent_after=max(observed_values),
                observed_indices=observed_indices,
                objective=objective,
                active_factors=[],
                raw_factor_evaluations=0,
                decision_factor_evaluations=0,
                heldout_factor_evaluations=0,
                inference_sample_count=0,
                inference_ess=None,
                inference_ess_fraction=None,
                inference_wall_time=0.0,
                heldout_sample_count=None,
                heldout_ess=None,
                heldout_ess_fraction=None,
                heldout_wall_time=None,
                selected_acquisition=float(acquisition[selected]),
                structural_bound_value=None,
                inference_allowance=None,
                heldout_regret=None,
                numerical_accuracy_reliable=True,
            )
        )
    _finalize_method_rows(rows, config)
    return rows


def _full_inference_with_accuracy(
    *,
    config: Mapping[str, Any],
    posterior: GPPosterior,
    signs: IntArray,
    incumbent: float,
    sample_schedule: Sequence[int],
    rng: np.random.Generator,
) -> ImportanceSampleResult:
    started = time.perf_counter()
    pairs = np.asarray(
        config["preference_bank"]["endpoint_index_pairs"], dtype=np.int64
    )
    active = np.arange(pairs.shape[0], dtype=np.int64)
    temperature = float(config["preference_bank"]["temperature"])
    laplace = laplace_preference_mode(
        posterior, pairs, signs, temperature, active, config["laplace"]
    )
    if not laplace.converged:
        raise RuntimeError("full-target Laplace mode failed")
    chunks: list[_RawImportanceBatch] = []
    current = 0
    minimum_ess = float(
        config["importance_sampling"]["full_target"]["minimum_ess_fraction"]
    )
    maximum_discrepancy = float(
        config["importance_sampling"]["full_target"][
            "maximum_split_half_ei_discrepancy"
        ]
    )
    result: ImportanceSampleResult | None = None
    for desired in sample_schedule:
        chunks.append(
            _draw_importance_batch(
                posterior=posterior,
                endpoint_pairs=pairs,
                signs=signs,
                temperature=temperature,
                active=active,
                incumbent=incumbent,
                draws=int(desired) - current,
                rng=rng,
                laplace=laplace,
            )
        )
        current = int(desired)
        raw = _concatenate_raw(chunks)
        reliable = False
        result = _summarize_importance(
            raw, laplace, active.size, time.perf_counter() - started
        )
        reliable = (
            result.ess_fraction >= minimum_ess
            and result.maximum_split_half_discrepancy <= maximum_discrepancy
        )
        if reliable:
            return result
    assert result is not None
    return ImportanceSampleResult(
        **{**result.__dict__, "accuracy_reliable": False}
    )


def _pooled_gap_diagnostics(
    first_raw: _RawImportanceBatch,
    second_raw: _RawImportanceBatch,
    leader_index: int,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, float, float]:
    pooled = _concatenate_raw((first_raw, second_raw))
    weights = _normalized_weights(pooled.log_weights)
    acquisition = np.asarray(weights @ pooled.utilities, dtype=float)
    gap_values = pooled.utilities - pooled.utilities[:, [leader_index]]
    gaps = np.asarray(weights @ gap_values, dtype=float)
    centered = gap_values - gaps
    standard_errors = np.sqrt(np.sum(np.square(weights[:, None] * centered), axis=0))
    first_acquisition = _weighted_acquisition(
        first_raw.utilities, first_raw.log_weights
    )
    second_acquisition = _weighted_acquisition(
        second_raw.utilities, second_raw.log_weights
    )
    batch_gap_difference = np.abs(
        (first_acquisition - first_acquisition[leader_index])
        - (second_acquisition - second_acquisition[leader_index])
    )
    inference_allowance = np.maximum(2.0 * standard_errors, batch_gap_difference)
    gaps[leader_index] = 0.0
    inference_allowance[leader_index] = 0.0
    ess = min(float(weights.size), float(1.0 / np.square(weights).sum()))
    return (
        acquisition,
        gaps,
        inference_allowance,
        weights,
        ess,
        ess / weights.size,
    )


def _run_full_method(
    config: Mapping[str, Any],
    seed_input: PreparedSeedInputs,
    *,
    horizon: int,
    numerical_settings: NumericalSettings,
    config_sha256: str,
) -> MethodTrajectoryResult:
    grid = action_grid(config)
    objective = true_objective(grid)
    observed_indices, observed_values = _initial_history(config, seed_input, objective)
    rows: list[dict[str, Any]] = []
    acquisition_rows: list[dict[str, Any]] = []
    caveats: list[str] = []
    all_factors = list(range(int(config["preference_bank"]["factor_count"])))
    for iteration in range(1, horizon + 1):
        posterior = _posterior_from_history(
            config, grid, observed_indices, observed_values
        )
        incumbent_before = float(max(observed_values))
        inference = _full_inference_with_accuracy(
            config=config,
            posterior=posterior,
            signs=seed_input.preference_signs,
            incumbent=incumbent_before,
            sample_schedule=numerical_settings.full_draws_schedule,
            rng=_rng_for(config, seed_input.seed, "full_method", iteration),
        )
        selected = select_unobserved_argmax(inference.acquisition, observed_indices)
        acquisition_rows.extend(
            _acquisition_curve_rows(
                method="full",
                seed=seed_input.seed,
                bo_iteration=iteration,
                refinement_index=None,
                stage="full_target",
                acquisition=inference.acquisition,
                observed_indices=observed_indices,
                selected_action_index=selected,
                used_for_selection=True,
                config_sha256=config_sha256,
            )
        )
        if not inference.accuracy_reliable:
            caveats.append(
                f"seed={seed_input.seed} full iteration={iteration}: "
                "full-target ESS/split diagnostic remained unreliable at the frozen cap"
            )
        observation = float(objective[selected] + seed_input.scalar_noise[selected])
        observed_indices.append(selected)
        observed_values.append(observation)
        rows.append(
            _trajectory_row(
                config=config,
                seed_input=seed_input,
                config_sha256=config_sha256,
                method="full",
                bo_iteration=iteration,
                action_index=selected,
                grid=grid,
                observation=observation,
                incumbent_before=incumbent_before,
                incumbent_after=max(observed_values),
                observed_indices=observed_indices,
                objective=objective,
                active_factors=all_factors,
                raw_factor_evaluations=inference.factor_likelihood_evaluations,
                decision_factor_evaluations=inference.factor_likelihood_evaluations,
                heldout_factor_evaluations=0,
                inference_sample_count=inference.draws,
                inference_ess=inference.ess,
                inference_ess_fraction=inference.ess_fraction,
                inference_wall_time=inference.wall_time_seconds,
                heldout_sample_count=None,
                heldout_ess=None,
                heldout_ess_fraction=None,
                heldout_wall_time=None,
                selected_acquisition=float(inference.acquisition[selected]),
                structural_bound_value=0.0,
                inference_allowance=None,
                heldout_regret=0.0,
                numerical_accuracy_reliable=inference.accuracy_reliable,
            )
        )
    _finalize_method_rows(rows, config)
    return MethodTrajectoryResult(rows, [], acquisition_rows, caveats)


def _run_adaptive_method(
    config: Mapping[str, Any],
    seed_input: PreparedSeedInputs,
    *,
    horizon: int,
    numerical_settings: NumericalSettings,
    config_sha256: str,
) -> MethodTrajectoryResult:
    grid = action_grid(config)
    objective = true_objective(grid)
    observed_indices, observed_values = _initial_history(config, seed_input, objective)
    pairs = np.asarray(
        config["preference_bank"]["endpoint_index_pairs"], dtype=np.int64
    )
    temperature = float(config["preference_bank"]["temperature"])
    blocks = preference_blocks(grid.size)
    metadata = factor_block_metadata(pairs, blocks)
    factor_sensitivities = factor_sensitivity_matrix(
        metadata, len(blocks), temperature
    )
    tolerance = float(config["method_contract"]["adaptive"]["screening_tolerance"])
    working_config = config["importance_sampling"]["adaptive_working"]
    minimum_ess = float(working_config["minimum_ess_fraction"])
    maximum_allowance = float(
        working_config["maximum_worst_challenger_inference_allowance"]
    )

    rows: list[dict[str, Any]] = []
    refinement_rows: list[dict[str, Any]] = []
    acquisition_rows: list[dict[str, Any]] = []
    caveats: list[str] = []
    for iteration in range(1, horizon + 1):
        posterior = _posterior_from_history(
            config, grid, observed_indices, observed_values
        )
        comparison, _, _ = comparison_matrix_from_precision(
            posterior.precision, blocks
        )
        diagnostics = comparison_diagnostics(comparison)
        if not diagnostics.is_spd:
            raise RuntimeError("preference comparison matrix is not SPD")
        incumbent_before = float(max(observed_values))
        active: list[int] = []
        refinement = 0
        decision_factor_evaluations = 0
        decision_wall_time = 0.0
        selected = -1
        final_acquisition: FloatArray | None = None
        final_ess = np.nan
        final_ess_fraction = np.nan
        final_bound = np.nan
        final_allowance = np.nan
        final_reliable = True
        final_sample_count = 0

        while True:
            refinement_started = time.perf_counter()
            active_array = np.asarray(active, dtype=np.int64)
            laplace = laplace_preference_mode(
                posterior,
                pairs,
                seed_input.preference_signs,
                temperature,
                active,
                config["laplace"],
            )
            if not laplace.converged:
                raise RuntimeError("adaptive Laplace mode failed")
            decision_factor_evaluations += laplace.factor_likelihood_evaluations
            first_rng = _rng_for(
                config,
                seed_input.seed,
                "adaptive_working_batch_1",
                iteration,
                refinement,
            )
            second_rng = _rng_for(
                config,
                seed_input.seed,
                "adaptive_working_batch_2",
                iteration,
                refinement,
            )
            first_chunks: list[_RawImportanceBatch] = []
            second_chunks: list[_RawImportanceBatch] = []
            current = 0
            worst = -1
            structural = np.zeros(grid.size, dtype=float)
            contributions = np.full(pairs.shape[0], -np.inf)
            inference_allowance = np.zeros(grid.size, dtype=float)
            gaps = np.zeros(grid.size, dtype=float)
            acquisition = np.zeros(grid.size, dtype=float)
            ess = np.nan
            ess_fraction = np.nan
            envelope = np.zeros(grid.size, dtype=float)
            reliable = False

            for accuracy_level, desired in enumerate(
                numerical_settings.working_draws_per_batch_schedule
            ):
                additional = int(desired) - current
                first_chunk = _draw_importance_batch(
                    posterior=posterior,
                    endpoint_pairs=pairs,
                    signs=seed_input.preference_signs,
                    temperature=temperature,
                    active=active_array,
                    incumbent=incumbent_before,
                    draws=additional,
                    rng=first_rng,
                    laplace=laplace,
                )
                second_chunk = _draw_importance_batch(
                    posterior=posterior,
                    endpoint_pairs=pairs,
                    signs=seed_input.preference_signs,
                    temperature=temperature,
                    active=active_array,
                    incumbent=incumbent_before,
                    draws=additional,
                    rng=second_rng,
                    laplace=laplace,
                )
                first_chunks.append(first_chunk)
                second_chunks.append(second_chunk)
                decision_factor_evaluations += (
                    first_chunk.factor_likelihood_evaluations
                    + second_chunk.factor_likelihood_evaluations
                )
                current = int(desired)
                first_raw = _concatenate_raw(first_chunks)
                second_raw = _concatenate_raw(second_chunks)
                pooled_raw = _concatenate_raw((first_raw, second_raw))
                provisional_acquisition = _weighted_acquisition(
                    pooled_raw.utilities, pooled_raw.log_weights
                )
                selected = select_unobserved_argmax(
                    provisional_acquisition, observed_indices
                )
                (
                    acquisition,
                    gaps,
                    inference_allowance,
                    _,
                    ess,
                    ess_fraction,
                ) = _pooled_gap_diagnostics(first_raw, second_raw, selected)
                omitted = np.ones(pairs.shape[0], dtype=bool)
                omitted[active_array] = False
                load = omitted_factor_load(factor_sensitivities, omitted)
                for action_index in range(grid.size):
                    if action_index == selected:
                        structural[action_index] = 0.0
                    else:
                        footprint = ei_decision_footprint(
                            action_index, selected, blocks
                        )
                        structural[action_index] = structural_bound(
                            comparison, footprint, load
                        )
                envelope = gaps + inference_allowance + structural
                envelope[np.asarray(observed_indices, dtype=int)] = -np.inf
                envelope[selected] = 0.0
                worst = int(np.argmax(envelope))
                reliable = (
                    ess_fraction >= minimum_ess
                    and inference_allowance[worst] <= maximum_allowance
                )
                if reliable:
                    break

            if not reliable:
                final_reliable = False
                caveats.append(
                    f"seed={seed_input.seed} adaptive iteration={iteration} "
                    f"refinement={refinement}: working inference diagnostic "
                    "remained unreliable at the frozen cap"
                )
            footprint_worst = ei_decision_footprint(worst, selected, blocks)
            omitted = np.ones(pairs.shape[0], dtype=bool)
            omitted[np.asarray(active, dtype=int)] = False
            contributions = ranked_omitted_contributions(
                comparison, footprint_worst, factor_sensitivities, omitted
            )
            stop_tolerance = float(envelope[worst]) <= tolerance
            all_active = len(active) == pairs.shape[0]
            if stop_tolerance:
                stop_reason = "screening_tolerance"
                newly_activated: int | None = None
            elif all_active:
                stop_reason = "all_factors_active"
                newly_activated = None
            else:
                stop_reason = "activate_factor"
                newly_activated = int(np.argmax(contributions))

            refinement_wall = time.perf_counter() - refinement_started
            decision_wall_time += refinement_wall
            contribution_payload = [
                None if not np.isfinite(value) else float(value)
                for value in contributions
            ]
            refinement_rows.append(
                {
                    "method": "adaptive",
                    "seed": seed_input.seed,
                    "bo_iteration": iteration,
                    "refinement_index": refinement,
                    "leader_index": selected,
                    "worst_challenger_index": worst,
                    "active_set_before_refinement": json.dumps(active),
                    "newly_activated_factor": newly_activated,
                    "per_factor_contribution_scores": json.dumps(
                        contribution_payload
                    ),
                    "active_acquisition_gap": float(gaps[worst]),
                    "B_struct": float(structural[worst]),
                    "B_infer": float(inference_allowance[worst]),
                    "stopping_envelope": float(envelope[worst]),
                    "screening_tolerance": tolerance,
                    "stopping_reason": stop_reason,
                    "inference_draws_per_batch": current,
                    "inference_total_draws": 2 * current,
                    "inference_ess": float(ess),
                    "inference_ess_fraction": float(ess_fraction),
                    "laplace_mode_iterations": laplace.iterations,
                    "laplace_gradient_infinity_norm": laplace.gradient_infinity_norm,
                    "comparison_minimum_eigenvalue": diagnostics.minimum_eigenvalue,
                    "comparison_condition_number": diagnostics.condition_number,
                    "raw_factor_likelihood_evaluation_count_cumulative": decision_factor_evaluations,
                    "refinement_wall_time_seconds": refinement_wall,
                    "numerical_accuracy_reliable": reliable,
                    "config_sha256": config_sha256,
                }
            )
            acquisition_rows.extend(
                _acquisition_curve_rows(
                    method="adaptive",
                    seed=seed_input.seed,
                    bo_iteration=iteration,
                    refinement_index=refinement,
                    stage="active_target",
                    acquisition=acquisition,
                    observed_indices=observed_indices,
                    selected_action_index=selected,
                    used_for_selection=True,
                    config_sha256=config_sha256,
                )
            )
            final_acquisition = acquisition
            final_ess = ess
            final_ess_fraction = ess_fraction
            final_bound = float(structural[worst])
            final_allowance = float(inference_allowance[worst])
            final_sample_count = 2 * current
            if stop_tolerance or all_active:
                break
            assert newly_activated is not None
            active.append(newly_activated)
            active.sort()
            refinement += 1

        assert final_acquisition is not None and selected >= 0
        heldout = _full_inference_with_accuracy(
            config=config,
            posterior=posterior,
            signs=seed_input.preference_signs,
            incumbent=incumbent_before,
            sample_schedule=numerical_settings.full_draws_schedule,
            rng=_rng_for(
                config,
                seed_input.seed,
                "adaptive_heldout_full",
                iteration,
            ),
        )
        if not heldout.accuracy_reliable:
            final_reliable = False
            caveats.append(
                f"seed={seed_input.seed} adaptive iteration={iteration}: "
                "held-out full-target diagnostic remained unreliable at the frozen cap"
            )
        eligible = np.ones(grid.size, dtype=bool)
        eligible[np.asarray(observed_indices, dtype=int)] = False
        heldout_best = float(np.max(heldout.acquisition[eligible]))
        heldout_regret = max(
            0.0, heldout_best - float(heldout.acquisition[selected])
        )
        acquisition_rows.extend(
            _acquisition_curve_rows(
                method="adaptive",
                seed=seed_input.seed,
                bo_iteration=iteration,
                refinement_index=None,
                stage="heldout_full_target",
                acquisition=heldout.acquisition,
                observed_indices=observed_indices,
                selected_action_index=selected,
                used_for_selection=False,
                config_sha256=config_sha256,
            )
        )
        observation = float(objective[selected] + seed_input.scalar_noise[selected])
        observed_indices.append(selected)
        observed_values.append(observation)
        rows.append(
            _trajectory_row(
                config=config,
                seed_input=seed_input,
                config_sha256=config_sha256,
                method="adaptive",
                bo_iteration=iteration,
                action_index=selected,
                grid=grid,
                observation=observation,
                incumbent_before=incumbent_before,
                incumbent_after=max(observed_values),
                observed_indices=observed_indices,
                objective=objective,
                active_factors=active,
                raw_factor_evaluations=(
                    decision_factor_evaluations
                    + heldout.factor_likelihood_evaluations
                ),
                decision_factor_evaluations=decision_factor_evaluations,
                heldout_factor_evaluations=heldout.factor_likelihood_evaluations,
                inference_sample_count=final_sample_count,
                inference_ess=float(final_ess),
                inference_ess_fraction=float(final_ess_fraction),
                inference_wall_time=(
                    decision_wall_time + heldout.wall_time_seconds
                ),
                heldout_sample_count=heldout.draws,
                heldout_ess=heldout.ess,
                heldout_ess_fraction=heldout.ess_fraction,
                heldout_wall_time=heldout.wall_time_seconds,
                selected_acquisition=float(final_acquisition[selected]),
                structural_bound_value=final_bound,
                inference_allowance=final_allowance,
                heldout_regret=heldout_regret,
                numerical_accuracy_reliable=final_reliable,
            )
        )
    _finalize_method_rows(rows, config)
    return MethodTrajectoryResult(rows, refinement_rows, acquisition_rows, caveats)


def _standard_method_result(
    config: Mapping[str, Any],
    seed_input: PreparedSeedInputs,
    *,
    horizon: int,
    config_sha256: str,
) -> MethodTrajectoryResult:
    rows = run_standard_trajectory(
        config,
        seed_input,
        config_sha256=config_sha256,
        horizon=horizon,
    )
    grid = action_grid(config)
    objective = true_objective(grid)
    observed_indices, observed_values = _initial_history(config, seed_input, objective)
    acquisition_rows: list[dict[str, Any]] = []
    for row in rows:
        iteration = int(row["bo_iteration"])
        posterior = _posterior_from_history(
            config, grid, observed_indices, observed_values
        )
        acquisition = analytic_expected_improvement(
            posterior.mean,
            np.diag(posterior.covariance),
            max(observed_values),
        )
        selected = int(row["selected_action_index"])
        acquisition_rows.extend(
            _acquisition_curve_rows(
                method="standard",
                seed=seed_input.seed,
                bo_iteration=iteration,
                refinement_index=None,
                stage="scalar_gp",
                acquisition=acquisition,
                observed_indices=observed_indices,
                selected_action_index=selected,
                used_for_selection=True,
                config_sha256=config_sha256,
            )
        )
        observed_indices.append(selected)
        observed_values.append(float(row["observed_scalar_outcome"]))
    return MethodTrajectoryResult(rows, [], acquisition_rows, [])


def run_seed(
    config: Mapping[str, Any],
    seed_input: PreparedSeedInputs,
    *,
    horizon: int,
    numerical_settings: NumericalSettings,
    config_sha256: str,
) -> SeedRunResult:
    """Run standard, full, and adaptive with one immutable shared seed input."""

    results = (
        _standard_method_result(
            config,
            seed_input,
            horizon=horizon,
            config_sha256=config_sha256,
        ),
        _run_full_method(
            config,
            seed_input,
            horizon=horizon,
            numerical_settings=numerical_settings,
            config_sha256=config_sha256,
        ),
        _run_adaptive_method(
            config,
            seed_input,
            horizon=horizon,
            numerical_settings=numerical_settings,
            config_sha256=config_sha256,
        ),
    )
    return SeedRunResult(
        trajectory_rows=[row for result in results for row in result.trajectory_rows],
        refinement_rows=[row for result in results for row in result.refinement_rows],
        acquisition_rows=[row for result in results for row in result.acquisition_rows],
        numerical_caveats=[
            caveat for result in results for caveat in result.numerical_caveats
        ],
    )


def evaluate_gates(
    t_values_by_method: Mapping[str, Sequence[int]],
    adaptive_factor_ratios: Sequence[float],
) -> dict[str, Any]:
    """Evaluate P1/P2 with the preregistered precedence and thresholds."""

    required = {"standard", "full", "adaptive"}
    if set(t_values_by_method) != required:
        raise ValueError("gate input must contain exactly standard/full/adaptive")
    medians = {
        method: float(np.median(np.asarray(values, dtype=float)))
        for method, values in t_values_by_method.items()
    }
    median_ratio = float(np.median(np.asarray(adaptive_factor_ratios, dtype=float)))
    p1_pass = medians["full"] <= medians["standard"] - 1.0
    p2_performance_pass = medians["adaptive"] <= medians["full"] + 1.0
    p2_sparsity_pass = median_ratio <= 0.65
    p2_pass = p2_performance_pass and p2_sparsity_pass
    if not p1_pass:
        verdict = "FAIL-P1"
    elif not p2_pass:
        verdict = "FAIL-P2"
    else:
        verdict = "PASS"
    return {
        "median_t_0_10_standard": medians["standard"],
        "median_t_0_10_full": medians["full"],
        "median_t_0_10_adaptive": medians["adaptive"],
        "median_adaptive_factor_ratio": median_ratio,
        "p1_pass": bool(p1_pass),
        "p2_performance_pass": bool(p2_performance_pass),
        "p2_sparsity_pass": bool(p2_sparsity_pass),
        "p2_pass": bool(p2_pass),
        "verdict": verdict,
    }


def active_set_turnover(active_sets: Sequence[Sequence[int]]) -> list[float]:
    """Return consecutive Jaccard turnover (one minus overlap)."""

    turnover: list[float] = []
    for previous, current in zip(active_sets, active_sets[1:]):
        left, right = set(previous), set(current)
        union = left | right
        overlap = 1.0 if not union else len(left & right) / len(union)
        turnover.append(float(1.0 - overlap))
    return turnover


def validate_trajectory_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    """Validate required schema and the no-repeat/scalar-screening invariants."""

    if not rows:
        raise ValueError("trajectory rows must be nonempty")
    seen: dict[tuple[int, str], set[int]] = {}
    for row in rows:
        missing = REQUIRED_TRAJECTORY_FIELDS.difference(row)
        if missing:
            raise ValueError(f"trajectory row is missing fields: {sorted(missing)}")
        if row["screened_factor_source"] != "historical_preference_bank_only":
            raise ValueError("scalar observations entered the screened factor set")
        key = (int(row["seed"]), str(row["method"]))
        selected = int(row["selected_action_index"])
        selected_set = seen.setdefault(key, set())
        if selected in selected_set:
            raise ValueError("a BO trajectory repeated an action")
        selected_set.add(selected)
        active = json.loads(str(row["active_factor_indices"]))
        if any(not 0 <= int(index) < int(row["N"]) for index in active):
            raise ValueError("active factor index is outside the preference bank")


__all__ = [
    "GPPosterior",
    "ImportanceSampleResult",
    "LaplaceResult",
    "MethodTrajectoryResult",
    "NumericalSettings",
    "PreparedSeedInputs",
    "REQUIRED_TRAJECTORY_FIELDS",
    "SeedRunResult",
    "action_grid",
    "active_set_turnover",
    "analytic_expected_improvement",
    "evaluate_gates",
    "full_target_inference",
    "generate_preference_bank",
    "generate_scalar_noise",
    "gp_reference_posterior",
    "importance_sample_preference_target",
    "laplace_preference_mode",
    "load_frozen_config",
    "numerical_settings_from_config",
    "ou_covariance",
    "preference_energy_batch",
    "prepare_pilot_inputs",
    "run_seed",
    "run_standard_trajectory",
    "select_unobserved_argmax",
    "true_objective",
    "validate_trajectory_rows",
]
