"""Frozen Gaussian/Laplace BO routines for the Sun-oxide value pilot.

The FULL_PBE routines operate only on the exact Gaussian marginal over the
frozen support.  The returned Gaussian is a Laplace approximation to the
non-Gaussian conditioned target; it is never represented as the exact target.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import linalg, sparse
from scipy.optimize import minimize
from scipy.sparse.linalg import SuperLU, splu
from scipy.special import expit, ndtr


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
SIGMA_OBS = 0.05
SCALE_FLOOR_EV = 0.25
WEIGHT = 1.0 / 499.0


class NumericalFailure(RuntimeError):
    """Raised when a frozen numerical acceptance criterion is not met."""


@dataclass(frozen=True)
class FrozenTargetScale:
    mean_ev: float
    scale_ev: float

    def standardize(self, values_ev: ArrayLike) -> FloatArray:
        return (np.asarray(values_ev, dtype=np.float64) - self.mean_ev) / self.scale_ev


@dataclass(frozen=True)
class GaussianReferenceState:
    precision: sparse.csc_matrix
    factorization: SuperLU
    information: FloatArray
    mean: FloatArray
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class ExactSupportMarginal:
    mean: FloatArray
    covariance: FloatArray
    precision: FloatArray
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class LaplaceState:
    map: FloatArray
    precision: FloatArray
    cholesky: FloatArray
    covariance: FloatArray
    diagnostics: dict[str, Any]


def fixed_initial_positions(action_count: int, initial_count: int, seed: int) -> IntArray:
    """Draw the prospective initial action set with the required NumPy RNG."""

    if not 0 < initial_count <= action_count:
        raise ValueError("initial_count must be between one and action_count")
    rng = np.random.default_rng(seed)
    return np.asarray(
        rng.choice(action_count, size=initial_count, replace=False), dtype=np.int64
    )


def freeze_target_scale(
    initial_values_ev: ArrayLike, *, scale_floor_ev: float = SCALE_FLOOR_EV
) -> FrozenTargetScale:
    values = np.asarray(initial_values_ev, dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or not np.all(np.isfinite(values)):
        raise ValueError("Initial targets must be a finite vector with at least two values")
    if scale_floor_ev <= 0.0:
        raise ValueError("scale_floor_ev must be positive")
    return FrozenTargetScale(
        mean_ev=float(np.mean(values)),
        scale_ev=max(float(np.std(values, ddof=1)), float(scale_floor_ev)),
    )


def _relative_residual(matrix: ArrayLike, solution: ArrayLike, rhs: ArrayLike) -> float:
    values = np.asarray(matrix @ solution, dtype=np.float64) - np.asarray(rhs, dtype=np.float64)
    denominator = max(float(np.linalg.norm(rhs)), np.finfo(np.float64).tiny)
    return float(np.linalg.norm(values) / denominator)


def construct_gaussian_reference(
    q0: sparse.spmatrix,
    observed_nodes: ArrayLike,
    standardized_observations: ArrayLike,
    *,
    sigma_obs: float = SIGMA_OBS,
    residual_tolerance: float = 1e-9,
) -> GaussianReferenceState:
    """Construct and factor the exact frozen Gaussian reference posterior."""

    base = sparse.csc_matrix(q0, dtype=np.float64)
    nodes = np.asarray(observed_nodes, dtype=np.int64)
    values = np.asarray(standardized_observations, dtype=np.float64)
    if base.shape[0] != base.shape[1]:
        raise ValueError("Q0 must be square")
    if nodes.ndim != 1 or values.shape != nodes.shape:
        raise ValueError("Observed nodes and values must be aligned vectors")
    if len(set(nodes.tolist())) != nodes.size:
        raise ValueError("Observed nodes must be unique")
    if nodes.size and (nodes.min() < 0 or nodes.max() >= base.shape[0]):
        raise ValueError("Observed node lies outside Q0")
    if sigma_obs <= 0.0 or not np.all(np.isfinite(values)):
        raise ValueError("Invalid observation noise or standardized observations")

    observation_precision = sigma_obs**-2
    started = time.perf_counter()
    diagonal = sparse.csc_matrix(
        (
            np.full(nodes.size, observation_precision, dtype=np.float64),
            (nodes, nodes),
        ),
        shape=base.shape,
    )
    precision = sparse.csc_matrix(base + diagonal)
    precision.sort_indices()
    construction_seconds = time.perf_counter() - started

    information = np.zeros(base.shape[0], dtype=np.float64)
    information[nodes] = observation_precision * values
    started = time.perf_counter()
    factorization = splu(precision)
    factorization_seconds = time.perf_counter() - started
    started = time.perf_counter()
    mean = np.asarray(factorization.solve(information), dtype=np.float64)
    mean_solve_seconds = time.perf_counter() - started
    residual = _relative_residual(precision, mean, information)
    if residual > residual_tolerance:
        raise NumericalFailure(
            f"Gaussian posterior mean solve residual {residual} exceeds {residual_tolerance}"
        )
    return GaussianReferenceState(
        precision=precision,
        factorization=factorization,
        information=information,
        mean=mean,
        diagnostics={
            "qt_construction_seconds": construction_seconds,
            "sparse_factorization_seconds": factorization_seconds,
            "posterior_mean_solve_seconds": mean_solve_seconds,
            "posterior_mean_solve_relative_residual": residual,
            "observation_count": int(nodes.size),
            "sigma_obs_standardized": float(sigma_obs),
        },
    )


def selected_gaussian_marginals(
    state: GaussianReferenceState,
    selected_nodes: ArrayLike,
    *,
    residual_tolerance: float = 1e-9,
) -> tuple[FloatArray, FloatArray, dict[str, Any]]:
    nodes = np.asarray(selected_nodes, dtype=np.int64)
    if nodes.ndim != 1 or len(set(nodes.tolist())) != nodes.size:
        raise ValueError("Selected nodes must be a unique vector")
    rhs = np.zeros((state.precision.shape[0], nodes.size), dtype=np.float64)
    rhs[nodes, np.arange(nodes.size)] = 1.0
    started = time.perf_counter()
    columns = np.asarray(state.factorization.solve(rhs), dtype=np.float64)
    seconds = time.perf_counter() - started
    residual = _relative_residual(state.precision, columns, rhs)
    if residual > residual_tolerance:
        raise NumericalFailure(
            f"Selected variance solve residual {residual} exceeds {residual_tolerance}"
        )
    variances = np.maximum(columns[nodes, np.arange(nodes.size)], 0.0)
    return state.mean[nodes], variances, {
        "selected_variance_solves_seconds": seconds,
        "selected_variance_solve_relative_residual": residual,
        "selected_rhs_count": int(nodes.size),
    }


def exact_support_marginal(
    state: GaussianReferenceState,
    support_nodes: ArrayLike,
    *,
    residual_tolerance: float = 1e-9,
) -> ExactSupportMarginal:
    """Return (Q^-1)[H,H] and its inverse, never Q[H,H]."""

    support = np.asarray(support_nodes, dtype=np.int64)
    if support.ndim != 1 or len(set(support.tolist())) != support.size:
        raise ValueError("Support nodes must be a unique vector")
    rhs = np.zeros((state.precision.shape[0], support.size), dtype=np.float64)
    rhs[support, np.arange(support.size)] = 1.0
    started = time.perf_counter()
    columns = np.asarray(state.factorization.solve(rhs), dtype=np.float64)
    marginalization_seconds = time.perf_counter() - started
    solve_residual = _relative_residual(state.precision, columns, rhs)
    if solve_residual > residual_tolerance:
        raise NumericalFailure(
            f"Exact marginal solve residual {solve_residual} exceeds {residual_tolerance}"
        )
    covariance = np.asarray(columns[support, :], dtype=np.float64)
    covariance = 0.5 * (covariance + covariance.T)

    started = time.perf_counter()
    covariance_cholesky = linalg.cho_factor(
        covariance, lower=True, overwrite_a=False, check_finite=True
    )
    precision = linalg.cho_solve(
        covariance_cholesky, np.eye(support.size, dtype=np.float64), check_finite=True
    )
    precision = 0.5 * (precision + precision.T)
    precision_seconds = time.perf_counter() - started
    inverse_residual = _relative_residual(covariance, precision, np.eye(support.size))
    if inverse_residual > residual_tolerance:
        raise NumericalFailure(
            f"Marginal precision residual {inverse_residual} exceeds {residual_tolerance}"
        )
    return ExactSupportMarginal(
        mean=np.asarray(state.mean[support], dtype=np.float64),
        covariance=covariance,
        precision=precision,
        diagnostics={
            "marginalization_500_rhs_seconds": marginalization_seconds,
            "marginalization_solve_relative_residual": solve_residual,
            "dense_marginal_precision_seconds": precision_seconds,
            "dense_marginal_precision_relative_residual": inverse_residual,
            "support_rhs_count": int(support.size),
            "principal_precision_submatrix_used": False,
        },
    )


def schur_complement_precision(precision: ArrayLike, support: ArrayLike) -> FloatArray:
    """Dense fixture helper for testing the exact marginal identity."""

    matrix = np.asarray(precision, dtype=np.float64)
    selected = np.asarray(support, dtype=np.int64)
    remainder = np.setdiff1d(np.arange(matrix.shape[0]), selected, assume_unique=False)
    q_hh = matrix[np.ix_(selected, selected)]
    if remainder.size == 0:
        return q_hh.copy()
    q_hr = matrix[np.ix_(selected, remainder)]
    q_rr = matrix[np.ix_(remainder, remainder)]
    q_rh = matrix[np.ix_(remainder, selected)]
    return q_hh - q_hr @ np.linalg.solve(q_rr, q_rh)


def weighted_logistic_energy_gradient(
    values: ArrayLike,
    endpoint_pairs: ArrayLike,
    signs: ArrayLike,
    *,
    weight: float = WEIGHT,
    chunk_size: int = 16384,
) -> tuple[float, FloatArray]:
    latent = np.asarray(values, dtype=np.float64)
    pairs = np.asarray(endpoint_pairs, dtype=np.int64)
    directions = np.asarray(signs, dtype=np.int8)
    if latent.ndim != 1:
        raise ValueError("Latent values must be a vector")
    if pairs.ndim != 2 or pairs.shape[1] != 2 or directions.shape != (pairs.shape[0],):
        raise ValueError("Factor endpoints and signs do not align")
    if weight <= 0.0 or chunk_size < 1:
        raise ValueError("Invalid factor weight or chunk size")
    energy = 0.0
    gradient = np.zeros(latent.size, dtype=np.float64)
    for start in range(0, pairs.shape[0], chunk_size):
        stop = min(start + chunk_size, pairs.shape[0])
        left = pairs[start:stop, 0]
        right = pairs[start:stop, 1]
        sign = directions[start:stop].astype(np.float64)
        margin = sign * (latent[left] - latent[right])
        energy += weight * float(np.sum(np.logaddexp(0.0, -margin)))
        coefficient = -weight * sign * expit(-margin)
        gradient += np.bincount(left, weights=coefficient, minlength=latent.size)
        gradient += np.bincount(right, weights=-coefficient, minlength=latent.size)
    return energy, gradient


def weighted_logistic_hessian_dense(
    values: ArrayLike,
    endpoint_pairs: ArrayLike,
    signs: ArrayLike,
    *,
    weight: float = WEIGHT,
    chunk_size: int = 16384,
) -> FloatArray:
    latent = np.asarray(values, dtype=np.float64)
    pairs = np.asarray(endpoint_pairs, dtype=np.int64)
    directions = np.asarray(signs, dtype=np.int8)
    if pairs.ndim != 2 or pairs.shape[1] != 2 or directions.shape != (pairs.shape[0],):
        raise ValueError("Factor endpoints and signs do not align")
    hessian = np.zeros((latent.size, latent.size), dtype=np.float64)
    diagonal = np.zeros(latent.size, dtype=np.float64)
    for start in range(0, pairs.shape[0], chunk_size):
        stop = min(start + chunk_size, pairs.shape[0])
        left = pairs[start:stop, 0]
        right = pairs[start:stop, 1]
        sign = directions[start:stop].astype(np.float64)
        margin = sign * (latent[left] - latent[right])
        curvature = weight * expit(margin) * expit(-margin)
        diagonal += np.bincount(left, weights=curvature, minlength=latent.size)
        diagonal += np.bincount(right, weights=curvature, minlength=latent.size)
        hessian[left, right] -= curvature
        hessian[right, left] -= curvature
    hessian[np.diag_indices(latent.size)] += diagonal
    return hessian


def weighted_logistic_energy_samples(
    samples: ArrayLike,
    endpoint_pairs: ArrayLike,
    signs: ArrayLike,
    *,
    weight: float = WEIGHT,
    sample_chunk_size: int = 64,
    factor_chunk_size: int = 8192,
) -> FloatArray:
    """Chunked energy evaluation that never forms samples-by-all-factors."""

    values = np.asarray(samples, dtype=np.float64)
    pairs = np.asarray(endpoint_pairs, dtype=np.int64)
    directions = np.asarray(signs, dtype=np.int8)
    if values.ndim != 2:
        raise ValueError("Samples must have shape (n_samples, dimension)")
    if sample_chunk_size < 1 or factor_chunk_size < 1:
        raise ValueError("Chunk sizes must be positive")
    result = np.zeros(values.shape[0], dtype=np.float64)
    for sample_start in range(0, values.shape[0], sample_chunk_size):
        sample_stop = min(sample_start + sample_chunk_size, values.shape[0])
        block = values[sample_start:sample_stop]
        block_energy = np.zeros(block.shape[0], dtype=np.float64)
        for factor_start in range(0, pairs.shape[0], factor_chunk_size):
            factor_stop = min(factor_start + factor_chunk_size, pairs.shape[0])
            left = pairs[factor_start:factor_stop, 0]
            right = pairs[factor_start:factor_stop, 1]
            sign = directions[factor_start:factor_stop].astype(np.float64)
            margins = (block[:, left] - block[:, right]) * sign[None, :]
            block_energy += weight * np.sum(np.logaddexp(0.0, -margins), axis=1)
        result[sample_start:sample_stop] = block_energy
    return result


class TimedFactorBank:
    """Frozen factor arrays with timing/call diagnostics for one MAP decision."""

    def __init__(
        self,
        endpoint_pairs: ArrayLike,
        signs: ArrayLike,
        *,
        dimension: int,
        weight: float = WEIGHT,
        chunk_size: int = 16384,
    ) -> None:
        self.endpoint_pairs = np.asarray(endpoint_pairs, dtype=np.int64)
        self.signs = np.asarray(signs, dtype=np.int8)
        self.dimension = int(dimension)
        self.weight = float(weight)
        self.chunk_size = int(chunk_size)
        self.energy_gradient_calls = 0
        self.energy_gradient_seconds = 0.0
        self.hessian_calls = 0
        self.hessian_seconds = 0.0

    def energy_gradient(self, values: ArrayLike) -> tuple[float, FloatArray]:
        started = time.perf_counter()
        result = weighted_logistic_energy_gradient(
            values,
            self.endpoint_pairs,
            self.signs,
            weight=self.weight,
            chunk_size=self.chunk_size,
        )
        self.energy_gradient_seconds += time.perf_counter() - started
        self.energy_gradient_calls += 1
        return result

    def hessian(self, values: ArrayLike) -> FloatArray:
        started = time.perf_counter()
        result = weighted_logistic_hessian_dense(
            values,
            self.endpoint_pairs,
            self.signs,
            weight=self.weight,
            chunk_size=self.chunk_size,
        )
        self.hessian_seconds += time.perf_counter() - started
        self.hessian_calls += 1
        return result


def fit_laplace_approximation(
    marginal_mean: ArrayLike,
    marginal_precision: ArrayLike,
    factor_bank: TimedFactorBank,
    initial_map: ArrayLike,
    *,
    retry_map: ArrayLike | None = None,
    gradient_tolerance: float = 1e-5,
    optimizer_gradient_tolerance: float = 1e-8,
    function_tolerance: float = 1e-15,
    maximum_iterations: int = 2000,
    residual_tolerance: float = 1e-9,
) -> LaplaceState:
    """Find the unique MAP and construct the 500-D Laplace Gaussian."""

    mean = np.asarray(marginal_mean, dtype=np.float64)
    precision = np.asarray(marginal_precision, dtype=np.float64)
    first = np.asarray(initial_map, dtype=np.float64)
    if precision.shape != (mean.size, mean.size) or first.shape != mean.shape:
        raise ValueError("Laplace dimensions do not align")
    starts = [first]
    if retry_map is not None:
        retry = np.asarray(retry_map, dtype=np.float64)
        if retry.shape != mean.shape:
            raise ValueError("Retry MAP dimension does not align")
        if not np.array_equal(retry, first):
            starts.append(retry)

    def objective(values: FloatArray) -> tuple[float, FloatArray]:
        delta = values - mean
        factor_energy, factor_gradient = factor_bank.energy_gradient(values)
        gaussian_gradient = precision @ delta
        return (
            0.5 * float(delta @ gaussian_gradient) + factor_energy,
            np.asarray(gaussian_gradient + factor_gradient, dtype=np.float64),
        )

    attempts: list[dict[str, Any]] = []
    accepted_result = None
    map_started = time.perf_counter()
    for attempt_index, start in enumerate(starts):
        result = minimize(
            objective,
            start,
            method="L-BFGS-B",
            jac=True,
            options={
                "gtol": optimizer_gradient_tolerance,
                "ftol": function_tolerance,
                "maxiter": maximum_iterations,
                "maxls": 50,
            },
        )
        objective_value, gradient = objective(np.asarray(result.x, dtype=np.float64))
        gradient_inf = float(np.linalg.norm(gradient, ord=np.inf))
        attempt = {
            "attempt": attempt_index + 1,
            "success": bool(result.success),
            "status": int(result.status),
            "message": str(result.message),
            "iterations": int(result.nit),
            "function_evaluations": int(result.nfev),
            "objective": float(objective_value),
            "gradient_infinity_norm": gradient_inf,
        }
        attempts.append(attempt)
        if result.success and gradient_inf <= gradient_tolerance:
            accepted_result = result
            break
    map_seconds = time.perf_counter() - map_started
    if accepted_result is None:
        raise NumericalFailure(f"FULL_PBE MAP failed frozen acceptance checks: {attempts}")

    map_value = np.asarray(accepted_result.x, dtype=np.float64)
    started = time.perf_counter()
    factor_hessian = factor_bank.hessian(map_value)
    hessian_seconds = time.perf_counter() - started
    laplace_precision = np.asarray(precision + factor_hessian, dtype=np.float64)
    laplace_precision = 0.5 * (laplace_precision + laplace_precision.T)
    started = time.perf_counter()
    try:
        cholesky = linalg.cholesky(
            laplace_precision, lower=True, overwrite_a=False, check_finite=True
        )
    except linalg.LinAlgError as exc:
        raise NumericalFailure("FULL_PBE Laplace Hessian is not SPD") from exc
    cholesky_seconds = time.perf_counter() - started
    covariance = linalg.cho_solve(
        (cholesky, True), np.eye(mean.size, dtype=np.float64), check_finite=True
    )
    covariance = 0.5 * (covariance + covariance.T)
    solve_residual = _relative_residual(
        laplace_precision, covariance, np.eye(mean.size, dtype=np.float64)
    )
    if solve_residual > residual_tolerance:
        raise NumericalFailure(
            f"Laplace solve residual {solve_residual} exceeds {residual_tolerance}"
        )
    accepted_attempt = attempts[-1]
    return LaplaceState(
        map=map_value,
        precision=laplace_precision,
        cholesky=cholesky,
        covariance=covariance,
        diagnostics={
            "approximation": "Laplace Gaussian; not the exact conditioned posterior",
            "optimizer": "scipy.optimize.minimize/L-BFGS-B",
            "optimizer_success": bool(accepted_attempt["success"]),
            "optimizer_iterations": int(accepted_attempt["iterations"]),
            "optimizer_function_evaluations": int(
                accepted_attempt["function_evaluations"]
            ),
            "gradient_infinity_norm": float(
                accepted_attempt["gradient_infinity_norm"]
            ),
            "map_optimization_seconds": map_seconds,
            "optimizer_attempts": attempts,
            "factor_energy_gradient_calls": factor_bank.energy_gradient_calls,
            "factor_energy_gradient_seconds": factor_bank.energy_gradient_seconds,
            "hessian_construction_seconds": hessian_seconds,
            "factor_hessian_calls": factor_bank.hessian_calls,
            "factor_hessian_seconds": factor_bank.hessian_seconds,
            "dense_cholesky_seconds": cholesky_seconds,
            "laplace_solve_relative_residual": solve_residual,
            "laplace_spd_cholesky_success": True,
        },
    )


def gaussian_expected_improvement(
    means: ArrayLike, variances: ArrayLike, incumbent: float
) -> FloatArray:
    means_array = np.asarray(means, dtype=np.float64)
    variances_array = np.asarray(variances, dtype=np.float64)
    if means_array.shape != variances_array.shape or np.any(variances_array < 0.0):
        raise ValueError("Gaussian means and nonnegative variances must align")
    standard_deviation = np.sqrt(variances_array)
    difference = means_array - float(incumbent)
    result = np.maximum(difference, 0.0)
    positive = standard_deviation > np.sqrt(np.finfo(np.float64).tiny)
    gamma = np.zeros_like(difference)
    gamma[positive] = difference[positive] / standard_deviation[positive]
    result[positive] = (
        difference[positive] * ndtr(gamma[positive])
        + standard_deviation[positive]
        * np.exp(-0.5 * gamma[positive] ** 2)
        / math.sqrt(2.0 * math.pi)
    )
    return result


def select_unobserved_action(
    acquisition: ArrayLike,
    observed_positions: ArrayLike,
    action_keys: Sequence[str],
) -> int:
    values = np.asarray(acquisition, dtype=np.float64)
    observed = np.asarray(observed_positions, dtype=np.int64)
    if values.shape != (len(action_keys),) or not np.all(np.isfinite(values)):
        raise ValueError("Acquisition values and action keys do not align")
    available = np.ones(values.size, dtype=bool)
    available[observed] = False
    if not np.any(available):
        raise ValueError("No unobserved action remains")
    maximum = float(np.max(values[available]))
    tied = np.flatnonzero(available & (values == maximum))
    return min((int(index) for index in tied), key=lambda index: action_keys[index])


def draw_laplace_samples(
    state: LaplaceState, sample_count: int, rng: np.random.Generator
) -> FloatArray:
    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    noise = rng.standard_normal((state.map.size, sample_count))
    deviations = linalg.solve_triangular(
        state.cholesky.T, noise, lower=False, check_finite=False
    )
    return np.asarray(state.map[None, :] + deviations.T, dtype=np.float64)


def laplace_log_importance_weights(
    samples: ArrayLike,
    marginal_mean: ArrayLike,
    marginal_precision: ArrayLike,
    laplace_state: LaplaceState,
    endpoint_pairs: ArrayLike,
    signs: ArrayLike,
    *,
    weight: float = WEIGHT,
    sample_chunk_size: int = 64,
    factor_chunk_size: int = 8192,
) -> tuple[FloatArray, dict[str, Any]]:
    """Compute -V + 0.5*(y-MAP)'H*(y-MAP), omitting constants."""

    values = np.asarray(samples, dtype=np.float64)
    mean = np.asarray(marginal_mean, dtype=np.float64)
    precision = np.asarray(marginal_precision, dtype=np.float64)
    started = time.perf_counter()
    factor_energy = weighted_logistic_energy_samples(
        values,
        endpoint_pairs,
        signs,
        weight=weight,
        sample_chunk_size=sample_chunk_size,
        factor_chunk_size=factor_chunk_size,
    )
    factor_seconds = time.perf_counter() - started
    target_delta = values - mean[None, :]
    proposal_delta = values - laplace_state.map[None, :]
    target_quadratic = np.sum((target_delta @ precision) * target_delta, axis=1)
    proposal_quadratic = np.sum(
        (proposal_delta @ laplace_state.precision) * proposal_delta, axis=1
    )
    log_weights = -0.5 * target_quadratic - factor_energy + 0.5 * proposal_quadratic
    return np.asarray(log_weights, dtype=np.float64), {
        "factor_sample_evaluation_seconds": factor_seconds,
        "maximum_materialized_factor_sample_shape": [
            min(sample_chunk_size, values.shape[0]),
            min(factor_chunk_size, np.asarray(endpoint_pairs).shape[0]),
        ],
        "full_samples_by_factors_materialized": False,
    }


def stable_self_normalized_weights(log_weights: ArrayLike) -> tuple[FloatArray, float]:
    values = np.asarray(log_weights, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("Log weights must be a nonempty finite vector")
    shifted = np.exp(values - np.max(values))
    normalized = shifted / np.sum(shifted)
    ess = float(1.0 / np.sum(normalized**2))
    return np.asarray(normalized, dtype=np.float64), ess


def snis_expected_improvement(
    action_samples: ArrayLike, normalized_weights: ArrayLike, incumbent: float
) -> FloatArray:
    samples = np.asarray(action_samples, dtype=np.float64)
    weights = np.asarray(normalized_weights, dtype=np.float64)
    if samples.ndim != 2 or weights.shape != (samples.shape[0],):
        raise ValueError("Action samples and normalized weights do not align")
    improvement = np.maximum(samples - float(incumbent), 0.0)
    return np.asarray(weights @ improvement, dtype=np.float64)


def snis_pairwise_gap_standard_error(
    gap_samples: ArrayLike, normalized_weights: ArrayLike
) -> tuple[float, float]:
    """Return the SNIS gap and its direct asymptotic Monte Carlo SE estimate."""

    gaps = np.asarray(gap_samples, dtype=np.float64)
    weights = np.asarray(normalized_weights, dtype=np.float64)
    if gaps.ndim != 1 or weights.shape != gaps.shape:
        raise ValueError("Gap samples and normalized weights do not align")
    estimate = float(weights @ gaps)
    standard_error = float(np.sqrt(np.sum(weights**2 * (gaps - estimate) ** 2)))
    return estimate, standard_error


def simple_regret_trajectory(
    sequential_values_ev: ArrayLike,
    initial_values_ev: ArrayLike,
    oracle_maximum_ev: float,
) -> FloatArray:
    sequential = np.asarray(sequential_values_ev, dtype=np.float64)
    initial = np.asarray(initial_values_ev, dtype=np.float64)
    if sequential.ndim != 1 or initial.ndim != 1 or initial.size == 0:
        raise ValueError("Regret inputs must be vectors with initial observations")
    running = np.maximum.accumulate(
        np.concatenate(([float(np.max(initial))], sequential))
    )[1:]
    return np.asarray(float(oracle_maximum_ev) - running, dtype=np.float64)


def aurc(regrets: ArrayLike) -> float:
    values = np.asarray(regrets, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("Regrets must be a finite vector")
    return float(np.sum(values))


class RetrospectiveOracle:
    """Audit-friendly oracle that separates query access from post-run evaluation."""

    def __init__(self, action_keys: Sequence[str], values_ev: ArrayLike) -> None:
        self._keys = tuple(str(key) for key in action_keys)
        self._values = np.asarray(values_ev, dtype=np.float64)
        if self._values.shape != (len(self._keys),) or not np.all(np.isfinite(self._values)):
            raise ValueError("Oracle keys and finite values do not align")
        if len(set(self._keys)) != len(self._keys):
            raise ValueError("Oracle action keys must be unique")
        self._evaluation_unlocked = False
        self.access_log: list[dict[str, Any]] = []

    def query(self, action_position: int, *, seed: int, method: str, stage: str) -> float:
        position = int(action_position)
        if not 0 <= position < len(self._keys):
            raise IndexError("Oracle action position is outside the action set")
        self.access_log.append(
            {
                "access": "queried_action",
                "action_position": position,
                "action_key": self._keys[position],
                "seed": int(seed),
                "method": str(method),
                "stage": str(stage),
            }
        )
        return float(self._values[position])

    def unlock_post_run_evaluation(self) -> None:
        self._evaluation_unlocked = True
        self.access_log.append({"access": "post_run_evaluation_unlocked"})

    def evaluation_values(self) -> FloatArray:
        if not self._evaluation_unlocked:
            raise PermissionError("Full oracle evaluation is locked until all BO rollouts finish")
        self.access_log.append({"access": "full_oracle_evaluation"})
        return self._values.copy()
