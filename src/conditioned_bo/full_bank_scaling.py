"""Memory-bounded development tools for Sun-oxide full-bank scaling.

This module contains no oracle access.  It generalizes the frozen normalized
all-pairs PBE calculations to development support sizes while preserving the
same logistic energy, Laplace approximation, EI, Menz comparison matrix, and
adaptive activation rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import math
import resource
import sys
import time
from typing import Any, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import linalg, sparse
from scipy.optimize import minimize
from scipy.sparse.linalg import LinearOperator, cg, eigsh, splu
from scipy.special import expit
from scipy.stats import spearmanr

from conditioned_bo.bo_value import (
    NumericalFailure,
    gaussian_expected_improvement,
    select_unobserved_action,
)


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
Int32Array = NDArray[np.int32]
Int8Array = NDArray[np.int8]
BoolArray = NDArray[np.bool_]


class LocalResourceBlocked(RuntimeError):
    """Raised when the development probe reaches its local resource guard."""


def peak_rss_bytes() -> int:
    """Return process peak RSS in bytes on macOS and Linux."""

    raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return raw if sys.platform == "darwin" else raw * 1024


@dataclass
class ResourceGuard:
    rss_limit_bytes: int
    phase_time_limits_seconds: dict[str, float]
    records: list[dict[str, Any]]

    @classmethod
    def create(
        cls, rss_limit_gb: float, phase_time_limits_seconds: dict[str, float]
    ) -> ResourceGuard:
        return cls(
            rss_limit_bytes=int(rss_limit_gb * 1_000_000_000),
            phase_time_limits_seconds=dict(phase_time_limits_seconds),
            records=[],
        )

    def check(
        self,
        phase: str,
        *,
        phase_started: float | None = None,
        projected_additional_bytes: int = 0,
    ) -> None:
        peak = peak_rss_bytes()
        if peak + int(projected_additional_bytes) > self.rss_limit_bytes:
            raise LocalResourceBlocked(
                f"LOCAL_RESOURCE_BLOCKED: {phase} projected/observed RSS "
                f"{peak + int(projected_additional_bytes)} exceeds "
                f"{self.rss_limit_bytes}"
            )
        if phase_started is not None:
            limit = self.phase_time_limits_seconds.get(phase)
            if limit is not None and time.perf_counter() - phase_started > limit:
                raise LocalResourceBlocked(
                    f"LOCAL_RESOURCE_BLOCKED: {phase} exceeded {limit} seconds"
                )

    def record(self, phase: str, started: float, **metadata: Any) -> None:
        self.check(phase, phase_started=started)
        self.records.append(
            {
                "phase": phase,
                "elapsed_seconds": time.perf_counter() - started,
                "peak_rss_bytes": peak_rss_bytes(),
                "peak_rss_gb": peak_rss_bytes() / 1_000_000_000.0,
                **metadata,
            }
        )


@dataclass(frozen=True)
class CompactPairBank:
    model_name: str
    support_nodes: Int32Array
    endpoint_pairs: Int32Array
    signs: Int8Array
    tie_group_ids: Int32Array
    tie_group_sizes: Int32Array
    weight: float
    omitted_tie_pair_count: int

    @property
    def support_count(self) -> int:
        return int(self.support_nodes.size)

    @property
    def factor_count(self) -> int:
        return int(self.endpoint_pairs.shape[0])

    @property
    def maximum_weighted_incident_degree(self) -> float:
        degree = self.weight * (
            self.support_count - self.tie_group_sizes[self.tie_group_ids]
        )
        return float(np.max(degree))


def build_compact_pair_bank(
    support_nodes: ArrayLike,
    pbe_values: Sequence[Decimal],
    *,
    model_name: str,
) -> CompactPairBank:
    """Build every strict pair using compact endpoints and exact Decimal ties."""

    support = np.asarray(support_nodes, dtype=np.int32)
    if support.ndim != 1 or support.size < 2:
        raise ValueError("Support must be a vector with at least two nodes")
    if np.any(np.diff(support.astype(np.int64)) <= 0):
        raise ValueError("Support nodes must be unique and increasing")
    if int(support[0]) < 0 or int(support[-1]) >= len(pbe_values):
        raise ValueError("Support node lies outside PBE values")

    selected_values = [pbe_values[int(index)] for index in support]
    unique_values = sorted(set(selected_values))
    rank_by_value = {value: rank for rank, value in enumerate(unique_values)}
    ranks = np.asarray(
        [rank_by_value[value] for value in selected_values], dtype=np.int32
    )
    tie_group_ids = ranks.copy()
    tie_group_sizes = np.bincount(
        tie_group_ids, minlength=len(unique_values)
    ).astype(np.int32)
    omitted_ties = int(
        np.sum(
            tie_group_sizes.astype(np.int64)
            * (tie_group_sizes.astype(np.int64) - 1)
            // 2
        )
    )

    left, right = np.triu_indices(support.size, k=1)
    rank_difference = ranks[left] - ranks[right]
    keep = rank_difference != 0
    endpoints = np.empty((int(np.count_nonzero(keep)), 2), dtype=np.int32)
    endpoints[:, 0] = left[keep].astype(np.int32, copy=False)
    endpoints[:, 1] = right[keep].astype(np.int32, copy=False)
    signs = np.where(rank_difference[keep] > 0, 1, -1).astype(np.int8)
    weight = 1.0 / float(support.size - 1)
    return CompactPairBank(
        model_name=model_name,
        support_nodes=support,
        endpoint_pairs=endpoints,
        signs=signs,
        tie_group_ids=tie_group_ids,
        tie_group_sizes=tie_group_sizes,
        weight=weight,
        omitted_tie_pair_count=omitted_ties,
    )


class CompleteMinusTieAdjacency:
    """Implicit normalized all-pairs adjacency on the full latent domain."""

    def __init__(self, bank: CompactPairBank, full_dimension: int) -> None:
        self.bank = bank
        self.full_dimension = int(full_dimension)
        if int(bank.support_nodes[-1]) >= self.full_dimension:
            raise ValueError("Support lies outside the full latent domain")

    def matvec(self, values: ArrayLike) -> FloatArray:
        vector = np.asarray(values, dtype=np.float64)
        if vector.shape != (self.full_dimension,):
            raise ValueError("Adjacency vector has the wrong dimension")
        support_values = vector[self.bank.support_nodes]
        group_sums = np.bincount(
            self.bank.tie_group_ids,
            weights=support_values,
            minlength=self.bank.tie_group_sizes.size,
        )
        support_result = self.bank.weight * (
            float(np.sum(support_values))
            - group_sums[self.bank.tie_group_ids]
        )
        result = np.zeros(self.full_dimension, dtype=np.float64)
        result[self.bank.support_nodes] = support_result
        return result

    def linear_operator(self) -> LinearOperator:
        return LinearOperator(
            (self.full_dimension, self.full_dimension),
            matvec=self.matvec,
            rmatvec=self.matvec,
            dtype=np.float64,
        )

    def explicit_sparse(self) -> sparse.csr_matrix:
        nodes = self.bank.support_nodes[self.bank.endpoint_pairs]
        row = np.concatenate((nodes[:, 0], nodes[:, 1]))
        col = np.concatenate((nodes[:, 1], nodes[:, 0]))
        data = np.full(row.size, self.bank.weight, dtype=np.float64)
        result = sparse.coo_matrix(
            (data, (row, col)),
            shape=(self.full_dimension, self.full_dimension),
        ).tocsr()
        result.sum_duplicates()
        result.sort_indices()
        return result


def implicit_theory_diagnostics(
    q0: sparse.spmatrix,
    bank: CompactPairBank,
    *,
    eigensolver_tolerance: float = 1e-10,
) -> dict[str, Any]:
    """Check W and A0 spectrally without a dense inverse."""

    precision = sparse.csr_matrix(q0, dtype=np.float64)
    adjacency = CompleteMinusTieAdjacency(bank, precision.shape[0])
    w_operator = adjacency.linear_operator()
    w_started = time.perf_counter()
    w_value, w_vector = eigsh(
        w_operator,
        k=1,
        which="LM",
        tol=eigensolver_tolerance,
        return_eigenvectors=True,
    )
    w_value_scalar = float(w_value[0])
    w_vec = np.asarray(w_vector[:, 0], dtype=np.float64)
    w_residual = float(
        np.linalg.norm(adjacency.matvec(w_vec) - w_value_scalar * w_vec)
        / np.linalg.norm(w_vec)
    )
    w_seconds = time.perf_counter() - w_started

    def a0_matvec(values: FloatArray) -> FloatArray:
        return np.asarray(precision @ values, dtype=np.float64) - 0.25 * adjacency.matvec(values)

    a0_operator = LinearOperator(
        precision.shape,
        matvec=a0_matvec,
        rmatvec=a0_matvec,
        dtype=np.float64,
    )
    a_started = time.perf_counter()
    a_value, a_vector = eigsh(
        a0_operator,
        k=1,
        which="SA",
        tol=eigensolver_tolerance,
        return_eigenvectors=True,
    )
    a_value_scalar = float(a_value[0])
    a_vec = np.asarray(a_vector[:, 0], dtype=np.float64)
    a_residual = float(
        np.linalg.norm(a0_matvec(a_vec) - a_value_scalar * a_vec)
        / np.linalg.norm(a_vec)
    )
    return {
        "support_count": bank.support_count,
        "factor_count": bank.factor_count,
        "omitted_exact_tie_pair_count": bank.omitted_tie_pair_count,
        "omega": bank.weight,
        "maximum_weighted_incident_degree": bank.maximum_weighted_incident_degree,
        "weighted_adjacency_spectral_norm": abs(w_value_scalar),
        "weighted_adjacency_eigenvalue": w_value_scalar,
        "weighted_adjacency_eigen_residual": w_residual,
        "weighted_adjacency_eigensolver_seconds": w_seconds,
        "analytic_a0_lambda_minimum_lower_bound": 0.75,
        "a0_smallest_eigenvalue": a_value_scalar,
        "a0_eigen_residual": a_residual,
        "a0_eigensolver_seconds": time.perf_counter() - a_started,
        "a0_spd": a_value_scalar > 0.0,
        "dense_inverse_formed": False,
        "new_theorem_introduced": False,
    }


@dataclass(frozen=True)
class SupportPrecisionReference:
    support_nodes: Int32Array
    precision: FloatArray | sparse.csr_matrix
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class FixedReferenceState:
    mean: FloatArray
    precision: FloatArray | sparse.csr_matrix
    action_variances: FloatArray
    diagnostics: dict[str, Any]


def construct_support_precision_reference(
    q0: sparse.spmatrix,
    support_nodes: ArrayLike,
    *,
    residual_tolerance: float,
    guard: ResourceGuard,
) -> SupportPrecisionReference:
    """Construct exact marginal precision, using Q0 directly for full support."""

    base = sparse.csc_matrix(q0, dtype=np.float64)
    support = np.asarray(support_nodes, dtype=np.int32)
    started = time.perf_counter()
    if support.size == base.shape[0] and np.array_equal(
        support, np.arange(base.shape[0], dtype=np.int32)
    ):
        return SupportPrecisionReference(
            support_nodes=support,
            precision=base.tocsr(),
            diagnostics={
                "construction_kind": "full_support_J0_equals_Q0",
                "support_count": int(support.size),
                "rhs_count": 0,
                "total_seconds": time.perf_counter() - started,
                "dense_inverse_formed": False,
            },
        )

    guard.check(
        "support_reference",
        phase_started=started,
        projected_additional_bytes=int(8 * base.shape[0] * support.size * 2),
    )
    factor_started = time.perf_counter()
    factorization = splu(base)
    factor_seconds = time.perf_counter() - factor_started
    rhs = np.zeros((base.shape[0], support.size), dtype=np.float64)
    rhs[support, np.arange(support.size)] = 1.0
    solve_started = time.perf_counter()
    columns = np.asarray(factorization.solve(rhs), dtype=np.float64)
    solve_seconds = time.perf_counter() - solve_started
    solve_residual = float(
        np.linalg.norm(base @ columns - rhs) / np.linalg.norm(rhs)
    )
    if solve_residual > residual_tolerance:
        raise NumericalFailure("Exact support covariance solve residual failed")
    covariance = np.asarray(columns[support], dtype=np.float64)
    covariance = 0.5 * (covariance + covariance.T)
    del columns, rhs
    inverse_started = time.perf_counter()
    chol = linalg.cho_factor(covariance, lower=True, check_finite=True)
    precision = linalg.cho_solve(
        chol, np.eye(support.size, dtype=np.float64), check_finite=True
    )
    precision = np.asarray(0.5 * (precision + precision.T), dtype=np.float64)
    inverse_seconds = time.perf_counter() - inverse_started
    inverse_residual = float(
        np.linalg.norm(covariance @ precision - np.eye(support.size))
        / math.sqrt(support.size)
    )
    if inverse_residual > residual_tolerance:
        raise NumericalFailure("Exact support marginal precision residual failed")
    return SupportPrecisionReference(
        support_nodes=support,
        precision=precision,
        diagnostics={
            "construction_kind": "J0_inverse_of_Q0_inverse_support_block",
            "support_count": int(support.size),
            "rhs_count": int(support.size),
            "sparse_factorization_seconds": factor_seconds,
            "rhs_solve_seconds": solve_seconds,
            "support_solve_relative_residual": solve_residual,
            "dense_precision_seconds": inverse_seconds,
            "dense_precision_relative_residual": inverse_residual,
            "total_seconds": time.perf_counter() - started,
            "dense_full_inverse_formed": False,
        },
    )


def update_fixed_reference_state(
    reference: SupportPrecisionReference,
    observed_support_positions: ArrayLike,
    standardized_observations: ArrayLike,
    action_support_positions: ArrayLike,
    *,
    observation_precision: float = 400.0,
    residual_tolerance: float = 1e-9,
) -> FixedReferenceState:
    """Apply exact observation information and obtain reference action variances."""

    observed = np.asarray(observed_support_positions, dtype=np.int64)
    values = np.asarray(standardized_observations, dtype=np.float64)
    actions = np.asarray(action_support_positions, dtype=np.int64)
    dimension = reference.support_nodes.size
    information = np.zeros(dimension, dtype=np.float64)
    information[observed] = observation_precision * values
    rhs = np.zeros((dimension, actions.size), dtype=np.float64)
    rhs[actions, np.arange(actions.size)] = 1.0
    started = time.perf_counter()
    if sparse.issparse(reference.precision):
        diagonal = np.zeros(dimension, dtype=np.float64)
        diagonal[observed] = observation_precision
        precision = sparse.csr_matrix(
            reference.precision + sparse.diags(diagonal)
        )
        factor_started = time.perf_counter()
        factorization = splu(precision.tocsc())
        factor_seconds = time.perf_counter() - factor_started
        mean = np.asarray(factorization.solve(information), dtype=np.float64)
        columns = np.asarray(factorization.solve(rhs), dtype=np.float64)
    else:
        precision = np.asarray(reference.precision, dtype=np.float64).copy()
        precision[observed, observed] += observation_precision
        factor_started = time.perf_counter()
        factorization = linalg.cho_factor(
            precision, lower=True, overwrite_a=False, check_finite=True
        )
        factor_seconds = time.perf_counter() - factor_started
        mean = np.asarray(
            linalg.cho_solve(factorization, information, check_finite=True),
            dtype=np.float64,
        )
        columns = np.asarray(
            linalg.cho_solve(factorization, rhs, check_finite=True),
            dtype=np.float64,
        )
    mean_denominator = max(float(np.linalg.norm(information)), np.finfo(float).tiny)
    mean_residual = float(
        np.linalg.norm(precision @ mean - information) / mean_denominator
    )
    column_residual = float(
        np.linalg.norm(precision @ columns - rhs) / np.linalg.norm(rhs)
    )
    if max(mean_residual, column_residual) > residual_tolerance:
        raise NumericalFailure("Fixed Gaussian reference residual failed")
    action_variances = np.maximum(
        columns[actions, np.arange(actions.size)], 0.0
    )
    return FixedReferenceState(
        mean=mean,
        precision=precision,
        action_variances=action_variances,
        diagnostics={
            "observation_count": int(observed.size),
            "observation_precision": observation_precision,
            "factorization_seconds": factor_seconds,
            "mean_relative_residual": mean_residual,
            "action_variance_relative_residual": column_residual,
            "action_variance_rhs_count": int(actions.size),
            "total_reference_seconds": time.perf_counter() - started,
        },
    )


class TimedCompactFactorBank:
    """Contiguous compact factor arrays with hardware-independent work counts."""

    def __init__(
        self,
        bank: CompactPairBank,
        factor_indices: ArrayLike | None,
        *,
        chunk_size: int,
    ) -> None:
        if factor_indices is None:
            self.endpoint_pairs = bank.endpoint_pairs
            self.signs = bank.signs
        else:
            indices = np.asarray(factor_indices, dtype=np.int64)
            self.endpoint_pairs = np.ascontiguousarray(bank.endpoint_pairs[indices])
            self.signs = np.ascontiguousarray(bank.signs[indices])
        self.dimension = bank.support_count
        self.weight = bank.weight
        self.chunk_size = int(chunk_size)
        self.energy_gradient_calls = 0
        self.energy_gradient_seconds = 0.0
        self.hessian_calls = 0
        self.hessian_seconds = 0.0

    @property
    def factor_count(self) -> int:
        return int(self.endpoint_pairs.shape[0])

    @property
    def energy_gradient_element_work(self) -> int:
        return self.factor_count * self.energy_gradient_calls

    @property
    def hessian_element_work(self) -> int:
        return self.factor_count * self.hessian_calls

    def energy_gradient(self, values: ArrayLike) -> tuple[float, FloatArray]:
        started = time.perf_counter()
        latent = np.asarray(values, dtype=np.float64)
        energy = 0.0
        gradient = np.zeros(self.dimension, dtype=np.float64)
        for start in range(0, self.factor_count, self.chunk_size):
            stop = min(start + self.chunk_size, self.factor_count)
            left = self.endpoint_pairs[start:stop, 0]
            right = self.endpoint_pairs[start:stop, 1]
            signs = self.signs[start:stop].astype(np.float64)
            margin = signs * (latent[left] - latent[right])
            energy += self.weight * float(np.sum(np.logaddexp(0.0, -margin)))
            coefficient = -self.weight * signs * expit(-margin)
            gradient += np.bincount(
                left, weights=coefficient, minlength=self.dimension
            )
            gradient += np.bincount(
                right, weights=-coefficient, minlength=self.dimension
            )
        self.energy_gradient_calls += 1
        self.energy_gradient_seconds += time.perf_counter() - started
        return energy, gradient

    def hessian(self, values: ArrayLike) -> FloatArray:
        started = time.perf_counter()
        latent = np.asarray(values, dtype=np.float64)
        hessian = np.zeros((self.dimension, self.dimension), dtype=np.float64)
        diagonal = np.zeros(self.dimension, dtype=np.float64)
        for start in range(0, self.factor_count, self.chunk_size):
            stop = min(start + self.chunk_size, self.factor_count)
            left = self.endpoint_pairs[start:stop, 0]
            right = self.endpoint_pairs[start:stop, 1]
            signs = self.signs[start:stop].astype(np.float64)
            margin = signs * (latent[left] - latent[right])
            curvature = self.weight * expit(margin) * expit(-margin)
            diagonal += np.bincount(
                left, weights=curvature, minlength=self.dimension
            )
            diagonal += np.bincount(
                right, weights=curvature, minlength=self.dimension
            )
            hessian[left, right] = -curvature
            hessian[right, left] = -curvature
        hessian[np.diag_indices(self.dimension)] = diagonal
        self.hessian_calls += 1
        self.hessian_seconds += time.perf_counter() - started
        return hessian


@dataclass(frozen=True)
class CompactLaplaceState:
    map: FloatArray
    action_variances: FloatArray
    diagnostics: dict[str, Any]


def _precision_matvec(
    precision: FloatArray | sparse.spmatrix, values: FloatArray
) -> FloatArray:
    return np.asarray(precision @ values, dtype=np.float64)


def fit_compact_laplace(
    reference: FixedReferenceState,
    factor_bank: TimedCompactFactorBank,
    initial_map: ArrayLike,
    action_support_positions: ArrayLike,
    *,
    retry_map: ArrayLike | None,
    gradient_tolerance: float,
    optimizer_gradient_tolerance: float,
    function_tolerance: float,
    maximum_iterations: int,
    residual_tolerance: float,
    guard: ResourceGuard,
    phase_name: str,
) -> CompactLaplaceState:
    """Fit the exact active/full Laplace target without a dense covariance."""

    mean = reference.mean
    precision = reference.precision
    first = np.asarray(initial_map, dtype=np.float64)
    actions = np.asarray(action_support_positions, dtype=np.int64)
    if first.shape != mean.shape:
        raise ValueError("Initial MAP has the wrong dimension")
    retry = None if retry_map is None else np.asarray(retry_map, dtype=np.float64)
    if retry is not None and np.array_equal(first, retry):
        retry = None
    phase_started = time.perf_counter()

    def objective(values: FloatArray) -> tuple[float, FloatArray]:
        guard.check(phase_name, phase_started=phase_started)
        delta = values - mean
        energy, gradient = factor_bank.energy_gradient(values)
        gaussian_gradient = _precision_matvec(precision, delta)
        return (
            0.5 * float(delta @ gaussian_gradient) + energy,
            gaussian_gradient + gradient,
        )

    attempts: list[dict[str, Any]] = []
    accepted = None
    start = first
    start_source = "initial_map"
    map_started = time.perf_counter()
    for attempt_index in range(2):
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
        attempts.append(
            {
                "attempt": attempt_index + 1,
                "start_source": start_source,
                "success": bool(result.success),
                "status": int(result.status),
                "message": str(result.message),
                "iterations": int(result.nit),
                "function_evaluations": int(result.nfev),
                "objective": float(objective_value),
                "gradient_infinity_norm": gradient_inf,
            }
        )
        if result.success and gradient_inf <= gradient_tolerance:
            accepted = result
            break
        if attempt_index == 0:
            candidate = np.asarray(result.x, dtype=np.float64)
            if np.all(np.isfinite(candidate)):
                # Continue the same converged finite candidate when ftol stops
                # just above the independent gradient acceptance threshold.
                # This is the deterministic retry used by the frozen first-
                # state FULL implementation and changes no objective or
                # optimizer setting.
                start = candidate
                start_source = "first_attempt_candidate"
            elif retry is not None:
                start = retry
                start_source = "provided_retry_map_nonfinite_candidate"
            else:
                start = np.asarray(result.x, dtype=np.float64)
                start_source = "first_attempt_candidate"
    map_seconds = time.perf_counter() - map_started
    if accepted is None:
        raise NumericalFailure(f"Compact Laplace MAP failed: {attempts}")
    map_value = np.asarray(accepted.x, dtype=np.float64)

    guard.check(
        phase_name,
        phase_started=phase_started,
        projected_additional_bytes=int(8 * mean.size * mean.size * 3),
    )
    hessian_started = time.perf_counter()
    factor_hessian = factor_bank.hessian(map_value)
    hessian_seconds = time.perf_counter() - hessian_started
    if sparse.issparse(precision):
        laplace_precision = np.asarray(precision.toarray(), dtype=np.float64)
        laplace_precision += factor_hessian
    else:
        laplace_precision = np.asarray(precision + factor_hessian, dtype=np.float64)
    laplace_precision = 0.5 * (laplace_precision + laplace_precision.T)
    del factor_hessian
    cholesky_started = time.perf_counter()
    try:
        cholesky = linalg.cholesky(
            laplace_precision, lower=True, overwrite_a=False, check_finite=True
        )
    except linalg.LinAlgError as exc:
        raise NumericalFailure("Compact Laplace precision is not SPD") from exc
    cholesky_seconds = time.perf_counter() - cholesky_started
    rhs = np.zeros((mean.size, actions.size), dtype=np.float64)
    rhs[actions, np.arange(actions.size)] = 1.0
    variance_started = time.perf_counter()
    columns = linalg.cho_solve((cholesky, True), rhs, check_finite=True)
    variance_seconds = time.perf_counter() - variance_started
    solve_residual = float(
        np.linalg.norm(laplace_precision @ columns - rhs) / np.linalg.norm(rhs)
    )
    if solve_residual > residual_tolerance:
        raise NumericalFailure("Compact Laplace action-variance residual failed")
    action_variances = np.maximum(
        columns[actions, np.arange(actions.size)], 0.0
    )
    accepted_attempt = attempts[-1]
    return CompactLaplaceState(
        map=map_value,
        action_variances=action_variances,
        diagnostics={
            "optimizer": "scipy.optimize.minimize/L-BFGS-B",
            "optimizer_success": bool(accepted_attempt["success"]),
            "optimizer_iterations": int(accepted_attempt["iterations"]),
            "optimizer_function_evaluations": int(
                accepted_attempt["function_evaluations"]
            ),
            "gradient_infinity_norm": float(
                accepted_attempt["gradient_infinity_norm"]
            ),
            "optimizer_attempts": attempts,
            "map_optimization_seconds": map_seconds,
            "factor_count": factor_bank.factor_count,
            "factor_energy_gradient_calls": factor_bank.energy_gradient_calls,
            "factor_energy_gradient_element_work": (
                factor_bank.energy_gradient_element_work
            ),
            "factor_energy_gradient_seconds": factor_bank.energy_gradient_seconds,
            "hessian_construction_seconds": hessian_seconds,
            "factor_hessian_calls": factor_bank.hessian_calls,
            "factor_hessian_element_work": factor_bank.hessian_element_work,
            "factor_hessian_seconds": factor_bank.hessian_seconds,
            "dense_cholesky_seconds": cholesky_seconds,
            "selected_variance_solve_seconds": variance_seconds,
            "selected_variance_rhs_count": int(actions.size),
            "laplace_solve_relative_residual": solve_residual,
            "total_conditioning_seconds": time.perf_counter() - phase_started,
            "peak_rss_bytes": peak_rss_bytes(),
        },
    )


def fit_compact_map_only(
    precision: FloatArray | sparse.spmatrix,
    factor_bank: TimedCompactFactorBank,
    *,
    gradient_tolerance: float,
    optimizer_gradient_tolerance: float,
    function_tolerance: float,
    maximum_iterations: int,
    maximum_retries: int,
    guard: ResourceGuard,
) -> tuple[FloatArray, dict[str, Any]]:
    """Target-blind full-factor MAP for the exact support reference."""

    dimension = factor_bank.dimension
    started = time.perf_counter()

    def objective(values: FloatArray) -> tuple[float, FloatArray]:
        guard.check("pbe_signal", phase_started=started)
        energy, gradient = factor_bank.energy_gradient(values)
        gaussian_gradient = _precision_matvec(precision, values)
        return 0.5 * float(values @ gaussian_gradient) + energy, gaussian_gradient + gradient

    if maximum_retries != 1:
        raise ValueError("The development PBE-only MAP permits exactly one retry")
    attempts: list[dict[str, Any]] = []
    start = np.zeros(dimension, dtype=np.float64)
    accepted = None
    objective_value = math.inf
    gradient_inf = math.inf
    for attempt_index in range(maximum_retries + 1):
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
        attempts.append(
            {
                "attempt": attempt_index + 1,
                "success": bool(result.success),
                "status": int(result.status),
                "message": str(result.message),
                "iterations": int(result.nit),
                "function_evaluations": int(result.nfev),
                "objective": float(objective_value),
                "gradient_infinity_norm": gradient_inf,
            }
        )
        if result.success and gradient_inf <= gradient_tolerance:
            accepted = result
            break
        start = np.asarray(result.x, dtype=np.float64)
    if accepted is None:
        raise NumericalFailure(
            f"PBE-only MAP failed frozen checks: {attempts}"
        )
    result = accepted
    return np.asarray(result.x, dtype=np.float64), {
        "optimizer": "scipy.optimize.minimize/L-BFGS-B",
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "iterations": int(result.nit),
        "function_evaluations": int(result.nfev),
        "objective": float(objective_value),
        "gradient_infinity_norm": gradient_inf,
        "optimizer_attempts": attempts,
        "factor_energy_gradient_calls": factor_bank.energy_gradient_calls,
        "factor_energy_gradient_element_work": factor_bank.energy_gradient_element_work,
        "factor_energy_gradient_seconds": factor_bank.energy_gradient_seconds,
        "wall_seconds": time.perf_counter() - started,
        "peak_rss_bytes": peak_rss_bytes(),
    }


def rank_signal_diagnostics(
    pbe_values: Sequence[Decimal],
    map_values: ArrayLike,
    composition_keys: Sequence[str],
) -> dict[str, Any]:
    values = np.asarray(map_values, dtype=np.float64)
    if values.shape != (len(pbe_values),) or len(composition_keys) != values.size:
        raise ValueError("Signal inputs do not align")
    pbe_float = np.asarray([float(value) for value in pbe_values], dtype=np.float64)
    left, right = np.triu_indices(values.size, k=1)
    unique = sorted(set(pbe_values))
    rank_by_value = {value: rank for rank, value in enumerate(unique)}
    ranks = np.asarray([rank_by_value[value] for value in pbe_values], dtype=np.int32)
    difference = ranks[left] - ranks[right]
    keep = difference != 0
    margins = np.where(difference[keep] > 0, 1.0, -1.0) * (
        values[left[keep]] - values[right[keep]]
    )
    ordering = sorted(
        range(values.size),
        key=lambda index: (pbe_values[index], composition_keys[index]),
    )
    decile_count = int(math.ceil(values.size / 10.0))
    bottom = np.asarray(ordering[:decile_count], dtype=np.int64)
    top = np.asarray(ordering[-decile_count:], dtype=np.int64)
    return {
        "node_count": int(values.size),
        "spearman_pbe_rank_vs_map_rank": float(
            spearmanr(pbe_float, values).statistic
        ),
        "strict_pair_ordering_accuracy": float(np.mean(margins > 0.0)),
        "strict_pair_count": int(margins.size),
        "map_standard_deviation": float(np.std(values, ddof=0)),
        "map_range": float(np.ptp(values)),
        "decile_count": decile_count,
        "top_decile_minus_bottom_decile_map_contrast": float(
            np.mean(values[top]) - np.mean(values[bottom])
        ),
    }


class ImplicitMenzSystem:
    """A_t operator and residual-checked SPD influence solves."""

    def __init__(
        self,
        q0: sparse.spmatrix,
        bank: CompactPairBank,
        observed_nodes: ArrayLike,
        *,
        observation_precision: float = 400.0,
        residual_tolerance: float = 1e-9,
    ) -> None:
        self.q0 = sparse.csr_matrix(q0, dtype=np.float64)
        self.bank = bank
        self.adjacency = CompleteMinusTieAdjacency(bank, self.q0.shape[0])
        self.observation_diagonal = np.zeros(self.q0.shape[0], dtype=np.float64)
        self.observation_diagonal[np.asarray(observed_nodes, dtype=np.int64)] = (
            observation_precision
        )
        self.residual_tolerance = float(residual_tolerance)
        diagonal = self.q0.diagonal() + self.observation_diagonal
        self.preconditioner = LinearOperator(
            self.q0.shape,
            matvec=lambda values: np.asarray(values, dtype=np.float64) / diagonal,
            rmatvec=lambda values: np.asarray(values, dtype=np.float64) / diagonal,
            dtype=np.float64,
        )
        self.operator = LinearOperator(
            self.q0.shape,
            matvec=self.matvec,
            rmatvec=self.matvec,
            dtype=np.float64,
        )

    def matvec(self, values: ArrayLike) -> FloatArray:
        vector = np.asarray(values, dtype=np.float64)
        return (
            np.asarray(self.q0 @ vector, dtype=np.float64)
            - 0.25 * self.adjacency.matvec(vector)
            + self.observation_diagonal * vector
        )

    def solve(self, rhs: ArrayLike) -> tuple[FloatArray, dict[str, Any]]:
        vector = np.asarray(rhs, dtype=np.float64)
        denominator = float(np.linalg.norm(vector))
        if denominator == 0.0:
            return np.zeros_like(vector), {
                "iterations": 0,
                "relative_residual": 0.0,
                "seconds": 0.0,
            }
        iterations = 0

        def callback(_: FloatArray) -> None:
            nonlocal iterations
            iterations += 1

        started = time.perf_counter()
        solution, info = cg(
            self.operator,
            vector,
            M=self.preconditioner,
            rtol=min(self.residual_tolerance * 0.05, 1e-11),
            atol=0.0,
            maxiter=20000,
            callback=callback,
        )
        seconds = time.perf_counter() - started
        relative_residual = float(
            np.linalg.norm(self.matvec(solution) - vector) / denominator
        )
        if info != 0 or relative_residual > self.residual_tolerance:
            raise NumericalFailure(
                f"Menz CG failed: info={info}, residual={relative_residual}"
            )
        return np.asarray(solution, dtype=np.float64), {
            "iterations": iterations,
            "relative_residual": relative_residual,
            "seconds": seconds,
        }


@dataclass
class CompactActiveState:
    active_mask: BoolArray
    omitted_endpoint_degree: Int32Array

    @classmethod
    def empty(cls, bank: CompactPairBank) -> CompactActiveState:
        degree = np.bincount(
            bank.endpoint_pairs.ravel(), minlength=bank.support_count
        ).astype(np.int32)
        return cls(
            active_mask=np.zeros(bank.factor_count, dtype=bool),
            omitted_endpoint_degree=degree,
        )

    @property
    def active_count(self) -> int:
        return int(np.count_nonzero(self.active_mask))

    def active_indices(self) -> IntArray:
        return np.flatnonzero(self.active_mask).astype(np.int64)

    def omitted_indices(self) -> IntArray:
        return np.flatnonzero(~self.active_mask).astype(np.int64)

    def activate(self, factor_indices: ArrayLike, bank: CompactPairBank) -> None:
        indices = np.asarray(factor_indices, dtype=np.int64)
        if indices.size == 0:
            return
        if np.any(self.active_mask[indices]):
            raise ValueError("Factor activated twice")
        decrement = np.bincount(
            bank.endpoint_pairs[indices].ravel(), minlength=bank.support_count
        ).astype(np.int32)
        updated = self.omitted_endpoint_degree - decrement
        if np.any(updated < 0):
            raise NumericalFailure("Omitted degree became negative")
        self.active_mask[indices] = True
        self.omitted_endpoint_degree = updated


def stable_exact_activation_batch(
    omitted_factor_indices: ArrayLike,
    contributions: ArrayLike,
    *,
    active_gap: float,
    epsilon: float,
    rho: float,
) -> IntArray:
    omitted = np.asarray(omitted_factor_indices, dtype=np.int64)
    values = np.asarray(contributions, dtype=np.float64)
    if omitted.size == 0 or values.shape != omitted.shape:
        raise ValueError("Omitted factors and contributions must align")
    if float(np.min(values)) < -1e-10:
        raise NumericalFailure("A structural contribution is materially negative")
    values = np.maximum(values, 0.0)
    target = float(epsilon - active_gap)
    if target <= 0.0:
        raise NumericalFailure("Activation target is nonpositive")
    order = np.argsort(-values, kind="stable")
    ordered = values[order]
    required_removal = float(np.sum(ordered) - rho * target)
    if required_removal <= 0.0:
        raise NumericalFailure("Activation requested although safety target holds")
    cumulative = np.cumsum(ordered)
    count = int(np.searchsorted(cumulative, required_removal, side="left") + 1)
    return np.asarray(omitted[order[:count]], dtype=np.int64)


def _available_mask(action_count: int, observed_positions: ArrayLike) -> BoolArray:
    result = np.ones(action_count, dtype=bool)
    result[np.asarray(observed_positions, dtype=np.int64)] = False
    return result


def _stable_maximum(values: FloatArray, mask: BoolArray, keys: Sequence[str]) -> int:
    maximum = float(np.max(values[mask]))
    tied = np.flatnonzero(mask & (values == maximum))
    return min((int(index) for index in tied), key=lambda index: keys[index])


def run_adaptive_stage_probe(
    reference: FixedReferenceState,
    menz: ImplicitMenzSystem,
    bank: CompactPairBank,
    action_support_positions: ArrayLike,
    observed_action_positions: ArrayLike,
    action_keys: Sequence[str],
    incumbent: float,
    full_ei: ArrayLike,
    full_leader: int,
    previous_map: ArrayLike | None,
    *,
    epsilon: float,
    rho: float,
    max_stages: int,
    chunk_size: int,
    laplace_settings: dict[str, Any],
    guard: ResourceGuard,
    phase_name: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run the current activation loop from an empty decision-specific mask."""

    action_support = np.asarray(action_support_positions, dtype=np.int64)
    observed = np.asarray(observed_action_positions, dtype=np.int64)
    shadow_ei = np.asarray(full_ei, dtype=np.float64)
    state = CompactActiveState.empty(bank)
    available = _available_mask(len(action_keys), observed)
    warm_map = None if previous_map is None else np.asarray(previous_map, dtype=np.float64)
    phase_started = time.perf_counter()
    stage_records: list[dict[str, Any]] = []
    cumulative_energy_work = 0
    cumulative_hessian_work = 0
    cumulative_energy_calls = 0
    cumulative_hessian_calls = 0
    maximum_contribution_sum_error = 0.0
    activation_stage = 0
    final_map = reference.mean.copy()

    while True:
        guard.check(phase_name, phase_started=phase_started)
        active_indices = state.active_indices()
        if active_indices.size == 0:
            action_ei = gaussian_expected_improvement(
                reference.mean[action_support], reference.action_variances, incumbent
            )
            leader = select_unobserved_action(action_ei, observed, action_keys)
            current_map = reference.mean.copy()
            stage_fit_seconds = 0.0
        else:
            timed_bank = TimedCompactFactorBank(
                bank, active_indices, chunk_size=chunk_size
            )
            laplace = fit_compact_laplace(
                reference,
                timed_bank,
                reference.mean if warm_map is None else warm_map,
                action_support,
                retry_map=np.zeros(bank.support_count, dtype=np.float64),
                guard=guard,
                phase_name=phase_name,
                **laplace_settings,
            )
            current_map = laplace.map
            warm_map = current_map
            final_map = current_map
            action_ei = gaussian_expected_improvement(
                current_map[action_support], laplace.action_variances, incumbent
            )
            leader = select_unobserved_action(action_ei, observed, action_keys)
            diagnostics = laplace.diagnostics
            cumulative_energy_work += int(
                diagnostics["factor_energy_gradient_element_work"]
            )
            cumulative_hessian_work += int(
                diagnostics["factor_hessian_element_work"]
            )
            cumulative_energy_calls += int(diagnostics["factor_energy_gradient_calls"])
            cumulative_hessian_calls += int(diagnostics["factor_hessian_calls"])
            stage_fit_seconds = float(diagnostics["total_conditioning_seconds"])

        load = bank.weight * state.omitted_endpoint_degree.astype(np.float64)
        rhs = np.zeros(menz.q0.shape[0], dtype=np.float64)
        rhs[bank.support_nodes] = load
        influence, load_solve = menz.solve(rhs)
        leader_support = int(action_support[leader])
        psi = np.full(len(action_keys), -np.inf, dtype=np.float64)
        psi[available] = (
            action_ei[available]
            - action_ei[leader]
            + influence[bank.support_nodes[action_support[available]]]
            + influence[int(bank.support_nodes[leader_support])]
        )
        challenger_mask = available.copy()
        challenger_mask[leader] = False
        challenger = _stable_maximum(psi, challenger_mask, action_keys)
        active_gap = float(action_ei[challenger] - action_ei[leader])
        structural_bound = float(
            influence[int(bank.support_nodes[action_support[challenger]])]
            + influence[int(bank.support_nodes[leader_support])]
        )
        envelope = float(psi[challenger])
        shadow_regret = float(shadow_ei[full_leader] - shadow_ei[leader])
        record: dict[str, Any] = {
            "record_kind": "stage",
            "epsilon": epsilon,
            "stage": activation_stage,
            "active_factor_count": state.active_count,
            "active_factor_fraction": state.active_count / bank.factor_count,
            "active_leader": int(leader),
            "worst_challenger": int(challenger),
            "active_ei_gap": active_gap,
            "b_struct": structural_bound,
            "max_psi": envelope,
            "shadow_full_leader": int(full_leader),
            "shadow_full_action_agreement": bool(leader == full_leader),
            "shadow_full_laplace_ei_regret": shadow_regret,
            "cumulative_factor_energy_gradient_calls": cumulative_energy_calls,
            "cumulative_factor_energy_gradient_work": cumulative_energy_work,
            "cumulative_factor_hessian_calls": cumulative_hessian_calls,
            "cumulative_factor_hessian_work": cumulative_hessian_work,
            "stage_fit_conditioning_seconds": stage_fit_seconds,
            "load_solve_seconds": load_solve["seconds"],
            "load_solve_iterations": load_solve["iterations"],
            "load_solve_relative_residual": load_solve["relative_residual"],
            "cumulative_conditioning_seconds": time.perf_counter() - phase_started,
            "activated_factor_count_after_stage": 0,
            "certified": envelope <= epsilon,
            "full_bank_fallback": False,
            "peak_rss_bytes": peak_rss_bytes(),
        }
        stage_records.append(record)
        if envelope <= epsilon:
            return stage_records, {
                "certified": True,
                "full_bank_fallback": False,
                "pre_fallback_active_count": state.active_count,
                "pre_fallback_active_fraction": state.active_count / bank.factor_count,
                "stage_count": activation_stage,
                "selected_action": int(leader),
                "shadow_full_action_agreement": bool(leader == full_leader),
                "shadow_full_laplace_ei_regret": shadow_regret,
                "total_conditioning_seconds": time.perf_counter() - phase_started,
                "factor_energy_gradient_work": cumulative_energy_work,
                "factor_hessian_work": cumulative_hessian_work,
                "maximum_contribution_sum_error": maximum_contribution_sum_error,
                "map": final_map,
            }

        if activation_stage >= max_stages:
            pre_fallback_count = state.active_count
            full_bank = TimedCompactFactorBank(bank, None, chunk_size=chunk_size)
            fallback = fit_compact_laplace(
                reference,
                full_bank,
                reference.mean if warm_map is None else warm_map,
                action_support,
                retry_map=np.zeros(bank.support_count, dtype=np.float64),
                guard=guard,
                phase_name=phase_name,
                **laplace_settings,
            )
            fallback_ei = gaussian_expected_improvement(
                fallback.map[action_support], fallback.action_variances, incumbent
            )
            fallback_leader = select_unobserved_action(
                fallback_ei, observed, action_keys
            )
            diagnostics = fallback.diagnostics
            cumulative_energy_work += int(
                diagnostics["factor_energy_gradient_element_work"]
            )
            cumulative_hessian_work += int(
                diagnostics["factor_hessian_element_work"]
            )
            cumulative_energy_calls += int(diagnostics["factor_energy_gradient_calls"])
            cumulative_hessian_calls += int(diagnostics["factor_hessian_calls"])
            fallback_regret = float(
                shadow_ei[full_leader] - shadow_ei[fallback_leader]
            )
            stage_records.append(
                {
                    "record_kind": "fallback",
                    "epsilon": epsilon,
                    "stage": activation_stage,
                    "active_factor_count": bank.factor_count,
                    "active_factor_fraction": 1.0,
                    "active_leader": int(fallback_leader),
                    "worst_challenger": "",
                    "active_ei_gap": "",
                    "b_struct": 0.0,
                    "max_psi": "",
                    "shadow_full_leader": int(full_leader),
                    "shadow_full_action_agreement": bool(
                        fallback_leader == full_leader
                    ),
                    "shadow_full_laplace_ei_regret": fallback_regret,
                    "cumulative_factor_energy_gradient_calls": cumulative_energy_calls,
                    "cumulative_factor_energy_gradient_work": cumulative_energy_work,
                    "cumulative_factor_hessian_calls": cumulative_hessian_calls,
                    "cumulative_factor_hessian_work": cumulative_hessian_work,
                    "stage_fit_conditioning_seconds": diagnostics[
                        "total_conditioning_seconds"
                    ],
                    "load_solve_seconds": 0.0,
                    "load_solve_iterations": 0,
                    "load_solve_relative_residual": 0.0,
                    "cumulative_conditioning_seconds": time.perf_counter()
                    - phase_started,
                    "activated_factor_count_after_stage": 0,
                    "certified": False,
                    "full_bank_fallback": True,
                    "peak_rss_bytes": peak_rss_bytes(),
                }
            )
            return stage_records, {
                "certified": False,
                "full_bank_fallback": True,
                "pre_fallback_active_count": pre_fallback_count,
                "pre_fallback_active_fraction": pre_fallback_count
                / bank.factor_count,
                "stage_count": activation_stage,
                "selected_action": int(fallback_leader),
                "shadow_full_action_agreement": bool(
                    fallback_leader == full_leader
                ),
                "shadow_full_laplace_ei_regret": fallback_regret,
                "total_conditioning_seconds": time.perf_counter() - phase_started,
                "factor_energy_gradient_work": cumulative_energy_work,
                "factor_hessian_work": cumulative_hessian_work,
                "maximum_contribution_sum_error": maximum_contribution_sum_error,
                "map": fallback.map,
            }

        omitted = state.omitted_indices()
        pair_rhs = np.zeros(menz.q0.shape[0], dtype=np.float64)
        pair_rhs[int(bank.support_nodes[action_support[challenger]])] += 1.0
        pair_rhs[int(bank.support_nodes[leader_support])] += 1.0
        influence_row, pair_solve = menz.solve(pair_rhs)
        selected_pairs = bank.endpoint_pairs[omitted]
        contributions = bank.weight * (
            influence_row[bank.support_nodes[selected_pairs[:, 0]]]
            + influence_row[bank.support_nodes[selected_pairs[:, 1]]]
        )
        contribution_error = abs(float(np.sum(contributions)) - structural_bound)
        maximum_contribution_sum_error = max(
            maximum_contribution_sum_error, contribution_error
        )
        if contribution_error > 1e-8 * max(1.0, abs(structural_bound)):
            raise NumericalFailure(
                f"Implicit contributions do not sum to B_struct: {contribution_error}"
            )
        batch = stable_exact_activation_batch(
            omitted,
            contributions,
            active_gap=active_gap,
            epsilon=epsilon,
            rho=rho,
        )
        record["pair_solve_seconds"] = pair_solve["seconds"]
        record["pair_solve_iterations"] = pair_solve["iterations"]
        record["pair_solve_relative_residual"] = pair_solve["relative_residual"]
        record["activated_factor_count_after_stage"] = int(batch.size)
        state.activate(batch, bank)
        activation_stage += 1


def regression_against_dense_500(
    q0: sparse.spmatrix,
    bank: CompactPairBank,
    action_nodes: ArrayLike,
    *,
    residual_tolerance: float = 1e-9,
) -> dict[str, Any]:
    """Regress implicit solves against the prior 500-support dense C routine."""

    if bank.support_count != 500:
        raise ValueError("The direct regression is defined for m=500")
    observed_nodes = np.asarray(action_nodes, dtype=np.int64)[:4]
    adjacency = CompleteMinusTieAdjacency(bank, sparse.csr_matrix(q0).shape[0])
    a_t = sparse.csc_matrix(
        sparse.csr_matrix(q0)
        - 0.25 * adjacency.explicit_sparse()
        + sparse.diags(
            np.bincount(
                observed_nodes,
                weights=np.full(observed_nodes.size, 400.0),
                minlength=sparse.csr_matrix(q0).shape[0],
            )
        )
    )
    direct = splu(a_t)
    rhs = np.zeros((a_t.shape[0], bank.support_count), dtype=np.float64)
    rhs[bank.support_nodes, np.arange(bank.support_count)] = 1.0
    c_support = np.asarray(direct.solve(rhs)[bank.support_nodes], dtype=np.float64)
    state = CompactActiveState.empty(bank)
    state.activate(np.arange(0, bank.factor_count, 11, dtype=np.int64), bank)
    h_support = bank.weight * state.omitted_endpoint_degree.astype(np.float64)
    old_w = c_support @ h_support
    old_v = c_support[:, 0] + c_support[:, 1]
    system = ImplicitMenzSystem(
        q0,
        bank,
        observed_nodes,
        residual_tolerance=residual_tolerance,
    )
    full_h = np.zeros(a_t.shape[0], dtype=np.float64)
    full_h[bank.support_nodes] = h_support
    new_w, w_diag = system.solve(full_h)
    pair_rhs = np.zeros(a_t.shape[0], dtype=np.float64)
    pair_rhs[int(bank.support_nodes[0])] = 1.0
    pair_rhs[int(bank.support_nodes[1])] = 1.0
    new_v, v_diag = system.solve(pair_rhs)
    w_error = float(np.max(np.abs(old_w - new_w[bank.support_nodes])))
    v_error = float(np.max(np.abs(old_v - new_v[bank.support_nodes])))
    if max(w_error, v_error) > 1e-8:
        raise NumericalFailure("Implicit 500-support Menz regression failed")
    return {
        "support_count": 500,
        "observed_action_count": int(observed_nodes.size),
        "old_dense_support_inverse_formed_for_regression_only": True,
        "routine_dense_inverse_formed": False,
        "w_max_abs_error": w_error,
        "v_max_abs_error": v_error,
        "w_iterative_solve": w_diag,
        "v_iterative_solve": v_diag,
    }


__all__ = [
    "CompactActiveState",
    "CompactLaplaceState",
    "CompactPairBank",
    "CompleteMinusTieAdjacency",
    "FixedReferenceState",
    "ImplicitMenzSystem",
    "LocalResourceBlocked",
    "ResourceGuard",
    "SupportPrecisionReference",
    "TimedCompactFactorBank",
    "build_compact_pair_bank",
    "construct_support_precision_reference",
    "fit_compact_laplace",
    "fit_compact_map_only",
    "implicit_theory_diagnostics",
    "peak_rss_bytes",
    "rank_signal_diagnostics",
    "regression_against_dense_500",
    "run_adaptive_stage_probe",
    "stable_exact_activation_batch",
    "update_fixed_reference_state",
]
