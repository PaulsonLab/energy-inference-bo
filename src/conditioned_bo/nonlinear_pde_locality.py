"""Reusable numerical machinery for the E2 locality stress test.

The deployable selective methods in this module use only the current Gaussian
reference state, active nonlinear-PDE factors, and structural metadata.  FULL
information is accepted only by explicitly diagnostic helpers.  The Menz
comparison matrix is always kept sparse and is applied through sparse solves;
``A^{-1}`` is never materialized.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import math
import resource
import time
from typing import Any, Callable, Literal, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import linalg, sparse, special
from scipy.sparse import linalg as sparse_linalg

from .nonlinear_pde_influence import (
    DEFAULT_PARAMETERS,
    NonlinearPDEParameters,
    build_nonlinear_pde_comparison,
    factor_supports,
    omitted_factor_load,
    ranked_omitted_contributions,
    solve_comparison,
)


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]
MethodName = Literal[
    "FULL",
    "ADAPTIVE_INFLUENCE",
    "DYNAMIC_GEOMETRIC_SHELL",
    "STATIC_INFLUENCE",
    "FIXED_CHALLENGER",
]


@dataclass
class FactorWork:
    """Unambiguous factor-level work accumulated by one method."""

    factor_energy_evaluations: int = 0
    factor_gradient_elements: int = 0
    factor_hessian_elements: int = 0
    sparse_comparison_solves: int = 0

    def add(self, other: "FactorWork") -> None:
        self.factor_energy_evaluations += other.factor_energy_evaluations
        self.factor_gradient_elements += other.factor_gradient_elements
        self.factor_hessian_elements += other.factor_hessian_elements
        self.sparse_comparison_solves += other.sparse_comparison_solves

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class Problem:
    grid_size: int
    truth: FloatArray
    initial_mean: FloatArray
    source: FloatArray
    precision: sparse.csr_matrix
    supports: tuple[IntArray, ...]
    action_indices: IntArray
    action_coordinates: IntArray


@dataclass(frozen=True)
class BOState:
    grid_size: int
    checkpoint_queries: int
    checkpoint_label: str
    observed_indices: IntArray
    observed_values: FloatArray
    reference_mean: FloatArray
    reference_precision: sparse.csr_matrix
    reference_samples: FloatArray
    action_indices: IntArray
    action_coordinates: IntArray

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        for value in (
            self.observed_indices,
            self.observed_values,
            self.reference_mean,
            self.reference_precision.indptr,
            self.reference_precision.indices,
            self.reference_precision.data,
            self.reference_samples,
            self.action_indices,
        ):
            digest.update(np.ascontiguousarray(value).view(np.uint8))
        digest.update(str(self.checkpoint_queries).encode("ascii"))
        return digest.hexdigest()


@dataclass(frozen=True)
class InferenceResult:
    acquisition: FloatArray
    gap: FloatArray
    mc_bound: FloatArray
    leader_local: int
    leader_index: int
    ess_fraction: float
    proposal: str
    split_half_action_agreement: bool
    split_half_max_acquisition_difference: float
    laplace_diagnostics: dict[str, Any]
    inference_seconds: float


@dataclass(frozen=True)
class StageResult:
    stage: int
    active_count: int
    active_indices: tuple[int, ...]
    leader_index: int
    challenger_index: int
    sparse_gap: float
    structural_bound: float
    inference_bound: float
    envelope: float
    ess_fraction: float
    proposal: str
    activated_indices: tuple[int, ...]
    stopped: bool
    inference_seconds: float
    challenger_seconds: float
    cumulative_seconds: float
    work: dict[str, int]


@dataclass(frozen=True)
class MethodResult:
    method: MethodName
    action_index: int
    final_active_indices: tuple[int, ...]
    full_fallback: bool
    stages: tuple[StageResult, ...]
    inference_seconds: float
    challenger_seconds: float
    total_seconds: float
    peak_rss_bytes: int
    work: dict[str, int]
    final_ess_fraction: float
    final_proposal: str
    audit: dict[str, Any]


def peak_rss_bytes() -> int:
    """Return process peak RSS in bytes on macOS/Linux."""

    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # macOS reports bytes; Linux reports KiB.
    return value if value > 10_000_000 else value * 1024


def derive_prospective_seed(replicate: int) -> int:
    """Derive the frozen E2 seed exactly from the requested SHA-256 label."""

    if replicate < 0:
        raise ValueError("replicate must be nonnegative")
    label = f"E2_LOCALITY_STRESS_V1:{replicate}".encode("utf-8")
    raw = int.from_bytes(hashlib.sha256(label).digest()[:8], "big")
    return raw % (2**32 - 1)


def _path_laplacian_eigendecomposition(grid_size: int) -> tuple[FloatArray, FloatArray]:
    diagonal = np.full(grid_size, 2.0)
    diagonal[[0, -1]] = 1.0
    matrix = np.diag(diagonal)
    matrix += np.diag(-np.ones(grid_size - 1), 1)
    matrix += np.diag(-np.ones(grid_size - 1), -1)
    return np.linalg.eigh(matrix)


def sample_base_gaussian(
    grid_size: int,
    mean: ArrayLike,
    sample_count: int,
    rng: np.random.Generator,
    parameters: NonlinearPDEParameters = DEFAULT_PARAMETERS,
) -> FloatArray:
    """Sample ``N(mean, (q0 I + qL L)^{-1})`` via grid eigenvectors."""

    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    eigenvalues_1d, eigenvectors = _path_laplacian_eigendecomposition(grid_size)
    eigenvalues = (
        parameters.q0
        + parameters.q_laplacian
        * (eigenvalues_1d[:, None] + eigenvalues_1d[None, :])
    )
    white = rng.standard_normal((sample_count, grid_size, grid_size))
    scaled = white / np.sqrt(eigenvalues)[None, :, :]
    left = np.einsum("ia,sab->sib", eigenvectors, scaled, optimize=True)
    samples = np.einsum("sib,jb->sij", left, eigenvectors, optimize=True)
    return samples.reshape(sample_count, grid_size * grid_size) + np.asarray(mean)


def _base_covariance_diagonal(
    grid_size: int,
    parameters: NonlinearPDEParameters = DEFAULT_PARAMETERS,
) -> FloatArray:
    eigenvalues_1d, eigenvectors = _path_laplacian_eigendecomposition(grid_size)
    eigenvalues = (
        parameters.q0
        + parameters.q_laplacian
        * (eigenvalues_1d[:, None] + eigenvalues_1d[None, :])
    )
    squared = eigenvectors**2
    diagonal = np.einsum(
        "ia,jb,ab->ij", squared, squared, 1.0 / eigenvalues, optimize=True
    )
    return diagonal.ravel()


def _two_peak_fields(grid_size: int, right_bias: float = 0.05) -> tuple[FloatArray, FloatArray]:
    coordinates = np.arange(grid_size) - (grid_size - 1) / 2.0
    x_grid, y_grid = np.meshgrid(coordinates, coordinates, indexing="ij")
    left = np.exp(-((x_grid + 2.5) ** 2 + y_grid**2) / (2.0 * 1.3**2))
    right = np.exp(-((x_grid - 2.5) ** 2 + y_grid**2) / (2.0 * 1.3**2))
    truth = left + 0.96 * right
    mean = left + (0.96 + right_bias) * right
    return truth.ravel(), mean.ravel()


def _action_region(grid_size: int) -> tuple[IntArray, IntArray]:
    center = (grid_size - 1) / 2.0
    rows = range(max(0, int(center) - 6), min(grid_size, int(center) + 7))
    columns = range(max(0, int(center) - 4), min(grid_size, int(center) + 5))
    coordinates = np.asarray([(i, j) for i in rows for j in columns], dtype=np.int64)
    indices = coordinates[:, 0] * grid_size + coordinates[:, 1]
    return indices.astype(np.int64), coordinates


def build_problem(
    grid_size: int,
    source_seed: int,
    *,
    source_perturbation_scale: float,
    parameters: NonlinearPDEParameters = DEFAULT_PARAMETERS,
) -> Problem:
    """Build a seeded manufactured field without changing accepted PDE constants."""

    parameters.validate()
    _, derivative_bounds, _, _, _ = build_nonlinear_pde_comparison(
        grid_size, parameters
    )
    del derivative_bounds
    precision = build_nonlinear_pde_comparison(grid_size, parameters)[0]
    base_truth, initial_mean = _two_peak_fields(grid_size)
    rng = np.random.default_rng(source_seed)
    perturbation = sample_base_gaussian(
        grid_size, np.zeros(grid_size * grid_size), 1, rng, parameters
    )[0]
    perturbation -= perturbation.mean()
    scale = float(np.std(perturbation))
    if not np.isfinite(scale) or scale <= 0.0:
        raise RuntimeError("invalid seeded source perturbation")
    truth = base_truth + source_perturbation_scale * perturbation / scale
    supports = factor_supports(grid_size)
    source = np.empty(grid_size * grid_size, dtype=float)
    for center, support in enumerate(supports):
        source[center] = (
            truth[center]
            - parameters.coupling * truth[support[1:]].sum()
            + parameters.nonlinearity * np.sin(truth[center])
        )
    action_indices, action_coordinates = _action_region(grid_size)
    return Problem(
        grid_size=grid_size,
        truth=truth,
        initial_mean=initial_mean,
        source=source,
        precision=precision,
        supports=supports,
        action_indices=action_indices,
        action_coordinates=action_coordinates,
    )


def _posterior_components(
    problem: Problem,
    observed_indices: IntArray,
    observed_values: FloatArray,
    observation_noise_variance: float,
) -> tuple[FloatArray, sparse.csr_matrix, FloatArray, FloatArray]:
    n_sites = problem.grid_size**2
    base_diagonal = _base_covariance_diagonal(problem.grid_size)
    if observed_indices.size == 0:
        return (
            problem.initial_mean.copy(),
            problem.precision.copy(),
            base_diagonal,
            np.empty((n_sites, 0)),
        )
    basis = np.zeros((n_sites, observed_indices.size))
    basis[observed_indices, np.arange(observed_indices.size)] = 1.0
    covariance_columns = np.asarray(
        sparse_linalg.spsolve(problem.precision.tocsc(), basis), dtype=float
    )
    observation_covariance = covariance_columns[observed_indices, :]
    noisy_covariance = observation_covariance + observation_noise_variance * np.eye(
        observed_indices.size
    )
    residual = observed_values - problem.initial_mean[observed_indices]
    coefficients = np.linalg.solve(noisy_covariance, residual)
    mean = problem.initial_mean + covariance_columns @ coefficients
    solved = np.linalg.solve(noisy_covariance, covariance_columns.T)
    variance = base_diagonal - np.einsum(
        "ij,ji->i", covariance_columns, solved, optimize=True
    )
    variance = np.maximum(variance, 0.0)
    observation_precision = 1.0 / observation_noise_variance
    precision = problem.precision + sparse.csr_matrix(
        (
            np.full(observed_indices.size, observation_precision),
            (observed_indices, observed_indices),
        ),
        shape=problem.precision.shape,
    )
    return mean, precision.tocsr(), variance, covariance_columns


def _condition_reference_samples(
    prior_samples: FloatArray,
    problem: Problem,
    observed_indices: IntArray,
    observed_values: FloatArray,
    covariance_columns: FloatArray,
    observation_noise_variance: float,
    rng: np.random.Generator,
) -> FloatArray:
    if observed_indices.size == 0:
        return prior_samples.copy()
    observation_covariance = covariance_columns[observed_indices, :]
    noisy_covariance = observation_covariance + observation_noise_variance * np.eye(
        observed_indices.size
    )
    proposal_noise = math.sqrt(observation_noise_variance) * rng.standard_normal(
        (prior_samples.shape[0], observed_indices.size)
    )
    residual = observed_values[None, :] - (
        prior_samples[:, observed_indices] + proposal_noise
    )
    coefficients = np.linalg.solve(noisy_covariance, residual.T).T
    return prior_samples + coefficients @ covariance_columns.T


def gaussian_expected_improvement(
    mean: ArrayLike, variance: ArrayLike, incumbent: float
) -> FloatArray:
    mean_array = np.asarray(mean, dtype=float)
    sigma = np.sqrt(np.maximum(np.asarray(variance, dtype=float), 0.0))
    delta = mean_array - incumbent
    result = np.maximum(delta, 0.0)
    positive = sigma > 0.0
    z_value = delta[positive] / sigma[positive]
    result[positive] = (
        delta[positive] * special.ndtr(z_value)
        + sigma[positive] * np.exp(-0.5 * z_value**2) / math.sqrt(2.0 * math.pi)
    )
    return result


def _initial_design(action_indices: IntArray, action_coordinates: IntArray, count: int) -> IntArray:
    if count < 1 or count > action_indices.size:
        raise ValueError("invalid initialization size")
    center = action_coordinates.mean(axis=0)
    first = int(np.argmin(np.sum((action_coordinates - center) ** 2, axis=1)))
    selected = [first]
    while len(selected) < count:
        distances = np.min(
            np.sum(
                (action_coordinates[:, None, :] - action_coordinates[selected][None, :, :])
                ** 2,
                axis=2,
            ),
            axis=1,
        )
        distances[selected] = -1.0
        selected.append(int(np.argmax(distances)))
    return action_indices[np.asarray(selected, dtype=int)]


def build_common_bo_states(
    problem: Problem,
    *,
    initialization_size: int,
    total_queries: int,
    checkpoint_queries: Sequence[int],
    observation_noise_variance: float,
    reference_sample_count: int,
    trajectory_seed: int,
    incumbent: float,
) -> tuple[BOState, ...]:
    """Build method-independent ordinary-reference EI states."""

    checkpoints = tuple(int(value) for value in checkpoint_queries)
    if checkpoints[0] != initialization_size or checkpoints[-1] != total_queries:
        raise ValueError("checkpoints must include initialization and final query counts")
    observed = list(
        _initial_design(problem.action_indices, problem.action_coordinates, initialization_size)
    )
    labels = {checkpoints[0]: "early", checkpoints[1]: "middle", checkpoints[2]: "late"}
    rng = np.random.default_rng(trajectory_seed)
    prior_samples = sample_base_gaussian(
        problem.grid_size,
        problem.initial_mean,
        reference_sample_count,
        rng,
    )
    states: list[BOState] = []
    while True:
        observed_indices = np.asarray(observed, dtype=np.int64)
        observed_values = problem.truth[observed_indices]
        mean, precision, variance, covariance_columns = _posterior_components(
            problem,
            observed_indices,
            observed_values,
            observation_noise_variance,
        )
        reference_samples = _condition_reference_samples(
            prior_samples,
            problem,
            observed_indices,
            observed_values,
            covariance_columns,
            observation_noise_variance,
            rng,
        )
        query_count = observed_indices.size
        if query_count in checkpoints:
            states.append(
                BOState(
                    grid_size=problem.grid_size,
                    checkpoint_queries=query_count,
                    checkpoint_label=labels[query_count],
                    observed_indices=observed_indices.copy(),
                    observed_values=observed_values.copy(),
                    reference_mean=mean,
                    reference_precision=precision,
                    reference_samples=reference_samples,
                    action_indices=problem.action_indices.copy(),
                    action_coordinates=problem.action_coordinates.copy(),
                )
            )
        if query_count >= total_queries:
            break
        acquisitions = gaussian_expected_improvement(
            mean[problem.action_indices],
            variance[problem.action_indices],
            max(float(observed_values.max()), incumbent),
        )
        already_observed = np.isin(problem.action_indices, observed_indices)
        acquisitions[already_observed] = -np.inf
        next_local = int(np.argmax(acquisitions))
        observed.append(int(problem.action_indices[next_local]))
    if len(states) != len(checkpoints):
        raise RuntimeError("failed to construct every frozen checkpoint")
    return tuple(states)


def factor_energy_sum(
    samples: FloatArray,
    problem: Problem,
    factor_indices: ArrayLike,
    work: FactorWork | None = None,
    parameters: NonlinearPDEParameters = DEFAULT_PARAMETERS,
) -> FloatArray:
    indices = np.asarray(factor_indices, dtype=np.int64)
    result = np.zeros(samples.shape[0], dtype=float)
    for factor_index in indices:
        support = problem.supports[int(factor_index)]
        center = int(support[0])
        residual = (
            samples[:, center]
            - parameters.coupling * samples[:, support[1:]].sum(axis=1)
            + parameters.nonlinearity * np.sin(samples[:, center])
            - problem.source[int(factor_index)]
        )
        result += parameters.gamma * (
            np.logaddexp(residual / parameters.tau, -residual / parameters.tau)
            - math.log(2.0)
        )
    if work is not None:
        work.factor_energy_evaluations += int(samples.shape[0] * indices.size)
    return result


def effective_sample_size_fraction(log_weights: ArrayLike) -> float:
    values = np.asarray(log_weights, dtype=float)
    values = values - np.max(values)
    weights = np.exp(values)
    return float((weights.sum() ** 2 / np.dot(weights, weights)) / weights.size)


def _weighted_acquisition(
    samples: FloatArray,
    action_indices: IntArray,
    incumbent: float,
    log_weights: FloatArray,
    delta_mc: float,
) -> tuple[FloatArray, FloatArray, FloatArray, int, float, bool, float]:
    shifted = log_weights - np.max(log_weights)
    weights = np.exp(shifted)
    normalized = weights / weights.sum()
    utility = np.maximum(samples[:, action_indices] - incumbent, 0.0)
    acquisition = normalized @ utility
    leader_local = int(np.argmax(acquisition))
    gaps = utility - utility[:, leader_local, None]
    gap = normalized @ gaps
    influence = weights[:, None] * (gaps - gap[None, :]) / weights.mean()
    z_critical = special.ndtri(1.0 - delta_mc / (2.0 * action_indices.size))
    mc_bound = z_critical * influence.std(axis=0, ddof=1) / math.sqrt(samples.shape[0])
    midpoint = samples.shape[0] // 2
    half_acquisitions: list[FloatArray] = []
    for selection in (slice(0, midpoint), slice(midpoint, samples.shape[0])):
        half_log = log_weights[selection]
        half_weights = np.exp(half_log - np.max(half_log))
        half_weights /= half_weights.sum()
        half_acquisitions.append(half_weights @ utility[selection])
    agreement = int(np.argmax(half_acquisitions[0])) == int(np.argmax(half_acquisitions[1]))
    maximum_difference = float(np.max(np.abs(half_acquisitions[0] - half_acquisitions[1])))
    return (
        acquisition,
        gap,
        mc_bound,
        leader_local,
        effective_sample_size_fraction(log_weights),
        agreement,
        maximum_difference,
    )


def reference_snis_inference(
    state: BOState,
    problem: Problem,
    active_mask: BoolArray,
    *,
    incumbent: float,
    delta_mc: float,
    work: FactorWork,
) -> InferenceResult:
    """Archived GP-reference SNIS active-target inference."""

    start = time.perf_counter()
    active = np.flatnonzero(active_mask)
    energies = factor_energy_sum(state.reference_samples, problem, active, work)
    values = _weighted_acquisition(
        state.reference_samples,
        state.action_indices,
        incumbent,
        -energies,
        delta_mc,
    )
    acquisition, gap, bound, leader_local, ess, agreement, difference = values
    return InferenceResult(
        acquisition=acquisition,
        gap=gap,
        mc_bound=bound,
        leader_local=leader_local,
        leader_index=int(state.action_indices[leader_local]),
        ess_fraction=ess,
        proposal="GP_REFERENCE_SNIS",
        split_half_action_agreement=agreement,
        split_half_max_acquisition_difference=difference,
        laplace_diagnostics={},
        inference_seconds=time.perf_counter() - start,
    )


def _target_value_gradient_hessian(
    value: FloatArray,
    state: BOState,
    problem: Problem,
    active_indices: IntArray,
    work: FactorWork,
    *,
    need_hessian: bool,
    parameters: NonlinearPDEParameters = DEFAULT_PARAMETERS,
) -> tuple[float, FloatArray, sparse.csr_matrix | None]:
    delta = value - state.reference_mean
    objective = 0.5 * float(delta @ (state.reference_precision @ delta))
    gradient = np.asarray(state.reference_precision @ delta, dtype=float)
    rows: list[int] = []
    columns: list[int] = []
    entries: list[float] = []
    for factor_index in active_indices:
        support = problem.supports[int(factor_index)]
        center = int(support[0])
        local = value[support]
        residual = (
            local[0]
            - parameters.coupling * local[1:].sum()
            + parameters.nonlinearity * math.sin(local[0])
            - problem.source[int(factor_index)]
        )
        scaled = residual / parameters.tau
        tanh = math.tanh(scaled)
        objective += parameters.gamma * (
            np.logaddexp(scaled, -scaled) - math.log(2.0)
        )
        residual_gradient = np.full(support.size, -parameters.coupling)
        residual_gradient[0] = 1.0 + parameters.nonlinearity * math.cos(local[0])
        gradient[support] += parameters.gradient_scale * tanh * residual_gradient
        work.factor_energy_evaluations += 1
        work.factor_gradient_elements += int(support.size)
        if need_hessian:
            local_hessian = (
                parameters.outer_curvature_scale
                * (1.0 - tanh**2)
                * np.outer(residual_gradient, residual_gradient)
            )
            local_hessian[0, 0] += (
                parameters.gradient_scale
                * tanh
                * (-parameters.nonlinearity * math.sin(local[0]))
            )
            row_grid, column_grid = np.meshgrid(support, support, indexing="ij")
            rows.extend(row_grid.ravel().tolist())
            columns.extend(column_grid.ravel().tolist())
            entries.extend(local_hessian.ravel().tolist())
            work.factor_hessian_elements += int(support.size**2)
    if not need_hessian:
        return objective, gradient, None
    additions = sparse.coo_matrix(
        (entries, (rows, columns)), shape=state.reference_precision.shape
    ).tocsr()
    return objective, gradient, (state.reference_precision + additions).tocsr()


def laplace_mode(
    state: BOState,
    problem: Problem,
    active_mask: BoolArray,
    *,
    work: FactorWork,
    initial: FloatArray | None = None,
    gradient_tolerance: float,
    maximum_iterations: int,
) -> tuple[FloatArray, sparse.csr_matrix, dict[str, Any]]:
    active = np.flatnonzero(active_mask)
    mode = state.reference_mean.copy() if initial is None else np.asarray(initial).copy()
    accepted_steps = 0
    for iteration in range(maximum_iterations):
        objective, gradient, hessian = _target_value_gradient_hessian(
            mode, state, problem, active, work, need_hessian=True
        )
        assert hessian is not None
        gradient_norm = float(np.linalg.norm(gradient, ord=np.inf))
        if gradient_norm <= gradient_tolerance:
            return mode, hessian, {
                "converged": True,
                "iterations": iteration,
                "gradient_inf_norm": gradient_norm,
                "accepted_steps": accepted_steps,
            }
        step = np.asarray(sparse_linalg.spsolve(hessian.tocsc(), gradient), dtype=float)
        directional = float(gradient @ step)
        step_size = 1.0
        accepted = False
        for _ in range(16):
            candidate = mode - step_size * step
            candidate_value, _, _ = _target_value_gradient_hessian(
                candidate, state, problem, active, work, need_hessian=False
            )
            if candidate_value <= objective - 1.0e-4 * step_size * directional:
                mode = candidate
                accepted = True
                accepted_steps += 1
                break
            step_size *= 0.5
        if not accepted:
            raise RuntimeError("Laplace Newton line search failed")
    objective, gradient, hessian = _target_value_gradient_hessian(
        mode, state, problem, active, work, need_hessian=True
    )
    del objective
    assert hessian is not None
    gradient_norm = float(np.linalg.norm(gradient, ord=np.inf))
    if gradient_norm > gradient_tolerance:
        raise RuntimeError("Laplace mode failed the frozen gradient tolerance")
    return mode, hessian, {
        "converged": True,
        "iterations": maximum_iterations,
        "gradient_inf_norm": gradient_norm,
        "accepted_steps": accepted_steps,
    }


def laplace_snis_inference(
    state: BOState,
    problem: Problem,
    active_mask: BoolArray,
    *,
    incumbent: float,
    delta_mc: float,
    sample_count: int,
    proposal_seed: int,
    proposal_inflation: float,
    work: FactorWork,
    gradient_tolerance: float,
    maximum_iterations: int,
) -> InferenceResult:
    """Prototype-compatible Laplace-proposal SNIS for difficult targets."""

    start = time.perf_counter()
    mode, hessian, diagnostics = laplace_mode(
        state,
        problem,
        active_mask,
        work=work,
        gradient_tolerance=gradient_tolerance,
        maximum_iterations=maximum_iterations,
    )
    dense_hessian = hessian.toarray()
    factor = np.linalg.cholesky(dense_hessian / proposal_inflation)
    rng = np.random.default_rng(proposal_seed)
    white = rng.standard_normal((sample_count, mode.size))
    samples = mode + linalg.solve_triangular(
        factor.T, white.T, lower=False, check_finite=False
    ).T
    active = np.flatnonzero(active_mask)
    energies = factor_energy_sum(samples, problem, active, work)
    reference_delta = samples - state.reference_mean
    log_target = -0.5 * np.einsum(
        "bi,bi->b",
        reference_delta,
        (state.reference_precision @ reference_delta.T).T,
        optimize=True,
    ) - energies
    proposal_delta = samples - mode
    log_proposal = -0.5 * np.einsum(
        "bi,bi->b",
        proposal_delta,
        (hessian @ proposal_delta.T).T / proposal_inflation,
        optimize=True,
    )
    log_weights = log_target - log_proposal
    values = _weighted_acquisition(
        samples,
        state.action_indices,
        incumbent,
        log_weights,
        delta_mc,
    )
    acquisition, gap, bound, leader_local, ess, agreement, difference = values
    diagnostics = {
        **diagnostics,
        "dense_hessian_bytes": int(dense_hessian.nbytes),
        "proposal_inflation": proposal_inflation,
        "sample_count": sample_count,
    }
    return InferenceResult(
        acquisition=acquisition,
        gap=gap,
        mc_bound=bound,
        leader_local=leader_local,
        leader_index=int(state.action_indices[leader_local]),
        ess_fraction=ess,
        proposal="LAPLACE_SNIS",
        split_half_action_agreement=agreement,
        split_half_max_acquisition_difference=difference,
        laplace_diagnostics=diagnostics,
        inference_seconds=time.perf_counter() - start,
    )


def infer_with_escalation(
    state: BOState,
    problem: Problem,
    active_mask: BoolArray,
    *,
    incumbent: float,
    delta_mc: float,
    minimum_reference_ess_fraction: float,
    laplace_sample_count: int,
    proposal_seed: int,
    proposal_inflation: float,
    work: FactorWork,
    gradient_tolerance: float,
    maximum_iterations: int,
) -> InferenceResult:
    reference = reference_snis_inference(
        state,
        problem,
        active_mask,
        incumbent=incumbent,
        delta_mc=delta_mc,
        work=work,
    )
    if reference.ess_fraction >= minimum_reference_ess_fraction:
        return reference
    laplace = laplace_snis_inference(
        state,
        problem,
        active_mask,
        incumbent=incumbent,
        delta_mc=delta_mc,
        sample_count=laplace_sample_count,
        proposal_seed=proposal_seed,
        proposal_inflation=proposal_inflation,
        work=work,
        gradient_tolerance=gradient_tolerance,
        maximum_iterations=maximum_iterations,
    )
    return InferenceResult(
        **{
            **asdict(laplace),
            "laplace_diagnostics": {
                **laplace.laplace_diagnostics,
                "reference_ess_fraction_before_escalation": reference.ess_fraction,
            },
        }
    )


def factor_support_distance(
    grid_size: int,
    support: ArrayLike,
    decision_indices: Sequence[int],
) -> int:
    """Minimum four-neighbor graph distance from a residual support to a pair."""

    support_indices = np.asarray(support, dtype=np.int64)
    if support_indices.size == 0 or len(decision_indices) == 0:
        raise ValueError("support and decision region must be nonempty")
    support_coordinates = np.column_stack(np.divmod(support_indices, grid_size))
    decision_coordinates = np.column_stack(
        np.divmod(np.asarray(decision_indices, dtype=np.int64), grid_size)
    )
    distances = np.abs(
        support_coordinates[:, None, :] - decision_coordinates[None, :, :]
    ).sum(axis=2)
    return int(distances.min())


def geometric_ranking(
    grid_size: int,
    supports: Sequence[IntArray],
    omitted_mask: BoolArray,
    leader_index: int,
    challenger_index: int,
) -> IntArray:
    """Deterministic distance-only ranking used by M2 and oracle geometry."""

    omitted = np.flatnonzero(omitted_mask)
    distances = np.asarray(
        [
            factor_support_distance(
                grid_size, supports[int(index)], (leader_index, challenger_index)
            )
            for index in omitted
        ],
        dtype=np.int64,
    )
    order = np.lexsort((omitted, distances))
    return omitted[order]


def _comparison_for_state(
    state: BOState,
    parameters: NonlinearPDEParameters,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    base_precision, derivative_bounds, _, _, base_matrix = (
        build_nonlinear_pde_comparison(state.grid_size, parameters)
    )
    extra_diagonal = state.reference_precision.diagonal() - base_precision.diagonal()
    matrix = base_matrix + sparse.diags(extra_diagonal, format="csr")
    return derivative_bounds, matrix.tocsr()


def run_selective_method(
    method: MethodName,
    state: BOState,
    problem: Problem,
    *,
    epsilon: float,
    batch_size: int,
    maximum_refinement_stages: int,
    incumbent: float,
    delta_mc: float,
    minimum_reference_ess_fraction: float,
    laplace_sample_count: int,
    proposal_seed: int,
    proposal_inflation: float,
    gradient_tolerance: float,
    maximum_laplace_iterations: int,
    parameters: NonlinearPDEParameters = DEFAULT_PARAMETERS,
) -> MethodResult:
    """Run M0--M4 from an empty mask with cumulative within-decision activation."""

    if method not in {
        "FULL",
        "ADAPTIVE_INFLUENCE",
        "DYNAMIC_GEOMETRIC_SHELL",
        "STATIC_INFLUENCE",
        "FIXED_CHALLENGER",
    }:
        raise ValueError(f"unknown method: {method}")
    total_start = time.perf_counter()
    n_factors = problem.grid_size**2
    active = np.zeros(n_factors, dtype=bool)
    if method == "FULL":
        active[:] = True
    derivative_bounds, comparison = _comparison_for_state(state, parameters)
    work = FactorWork()
    stages: list[StageResult] = []
    static_ranking: IntArray | None = None
    fixed_challenger: int | None = None
    full_fallback = False
    inference_total = 0.0
    challenger_total = 0.0
    influence_selection_calls = 0
    geometric_selection_calls = 0
    # The extra iteration after the refinement budget is reserved for an
    # explicit FULL decision when the preceding stage activated the fallback.
    for stage in range(maximum_refinement_stages + 2):
        inference = infer_with_escalation(
            state,
            problem,
            active,
            incumbent=incumbent,
            delta_mc=delta_mc,
            minimum_reference_ess_fraction=minimum_reference_ess_fraction,
            laplace_sample_count=laplace_sample_count,
            proposal_seed=proposal_seed + stage,
            proposal_inflation=proposal_inflation,
            work=work,
            gradient_tolerance=gradient_tolerance,
            maximum_iterations=maximum_laplace_iterations,
        )
        inference_total += inference.inference_seconds
        challenger_start = time.perf_counter()
        omitted = ~active
        if not np.any(omitted):
            structural = np.zeros(state.action_indices.size)
        else:
            load = omitted_factor_load(
                derivative_bounds, omitted, parameters.gamma, parameters.tau
            )
            transported = solve_comparison(comparison, load)
            work.sparse_comparison_solves += 1
            structural = transported[state.action_indices] + transported[
                inference.leader_index
            ]
            structural[inference.leader_local] = 0.0
        envelope = inference.gap + structural + inference.mc_bound
        envelope[inference.leader_local] = 0.0
        worst_local = int(np.argmax(envelope))
        worst_challenger = int(state.action_indices[worst_local])
        if fixed_challenger is None:
            fixed_challenger = worst_challenger
        if method == "FIXED_CHALLENGER":
            matches = np.flatnonzero(state.action_indices == fixed_challenger)
            challenger_local = int(matches[0])
        else:
            challenger_local = worst_local
        challenger = int(state.action_indices[challenger_local])
        envelope_value = float(envelope[challenger_local])
        stopped = method == "FULL" or full_fallback or envelope_value <= epsilon
        activated: IntArray = np.empty(0, dtype=np.int64)
        if not stopped:
            if stage >= maximum_refinement_stages:
                activated = np.flatnonzero(omitted)
                active[:] = True
                full_fallback = True
            elif method == "DYNAMIC_GEOMETRIC_SHELL":
                ranking = geometric_ranking(
                    state.grid_size,
                    problem.supports,
                    omitted,
                    inference.leader_index,
                    challenger,
                )
                geometric_selection_calls += 1
                activated = ranking[:batch_size]
                active[activated] = True
            else:
                if method == "STATIC_INFLUENCE" and static_ranking is not None:
                    ranking = static_ranking[np.isin(static_ranking, np.flatnonzero(omitted))]
                else:
                    scores = ranked_omitted_contributions(
                        comparison,
                        derivative_bounds,
                        omitted,
                        challenger,
                        inference.leader_index,
                        parameters.gamma,
                        parameters.tau,
                    )
                    work.sparse_comparison_solves += 1
                    influence_selection_calls += 1
                    indices = np.flatnonzero(omitted)
                    ranking = indices[np.lexsort((indices, -scores[indices]))]
                    if method == "STATIC_INFLUENCE" and static_ranking is None:
                        static_ranking = ranking.copy()
                activated = ranking[:batch_size]
                active[activated] = True
        challenger_seconds = time.perf_counter() - challenger_start
        challenger_total += challenger_seconds
        stages.append(
            StageResult(
                stage=stage,
                active_count=int(active.sum()) if full_fallback else int(active.sum() - activated.size),
                active_indices=tuple(np.flatnonzero(active if activated.size == 0 else (active & ~np.isin(np.arange(n_factors), activated))).tolist()),
                leader_index=inference.leader_index,
                challenger_index=challenger,
                sparse_gap=float(inference.gap[challenger_local]),
                structural_bound=float(structural[challenger_local]),
                inference_bound=float(inference.mc_bound[challenger_local]),
                envelope=envelope_value,
                ess_fraction=inference.ess_fraction,
                proposal=inference.proposal,
                activated_indices=tuple(int(value) for value in activated),
                stopped=stopped,
                inference_seconds=inference.inference_seconds,
                challenger_seconds=challenger_seconds,
                cumulative_seconds=time.perf_counter() - total_start,
                work=work.to_dict().copy(),
            )
        )
        if stopped:
            break
        if full_fallback:
            # Explicitly perform the returned FULL decision after fallback.
            continue
    else:
        raise RuntimeError("selective method exceeded the refinement guard")
    final_inference = inference
    total_seconds = time.perf_counter() - total_start
    return MethodResult(
        method=method,
        action_index=final_inference.leader_index,
        final_active_indices=tuple(np.flatnonzero(active).tolist()),
        full_fallback=full_fallback,
        stages=tuple(stages),
        inference_seconds=inference_total,
        challenger_seconds=challenger_total,
        total_seconds=total_seconds,
        peak_rss_bytes=peak_rss_bytes(),
        work=work.to_dict(),
        final_ess_fraction=final_inference.ess_fraction,
        final_proposal=final_inference.proposal,
        audit={
            "initial_active_count": 0 if method != "FULL" else n_factors,
            "mask_reset_at_state_start": method != "FULL",
            "cumulative_activation_within_decision": True,
            "influence_selection_calls": influence_selection_calls,
            "geometric_selection_calls": geometric_selection_calls,
            "m2_selection_consulted_influence_scores": False,
            "state_fingerprint": state.fingerprint(),
            "full_shadow_information_used": False,
        },
    )


def evaluate_fixed_subset(
    state: BOState,
    problem: Problem,
    factor_indices: ArrayLike,
    *,
    incumbent: float,
    delta_mc: float,
) -> InferenceResult:
    """Evaluate a diagnostic fixed subset without structural selection."""

    active = np.zeros(problem.grid_size**2, dtype=bool)
    active[np.asarray(factor_indices, dtype=np.int64)] = True
    return reference_snis_inference(
        state,
        problem,
        active,
        incumbent=incumbent,
        delta_mc=delta_mc,
        work=FactorWork(),
    )


def oracle_geometric_prefix(
    state: BOState,
    problem: Problem,
    *,
    full_action_index: int,
    full_challenger_index: int,
    full_acquisition: FloatArray,
    batch_size: int,
    regret_threshold: float,
    incumbent: float,
    delta_mc: float,
) -> dict[str, Any]:
    """Hindsight-only geometric diagnostic isolated from deployable methods."""

    omitted = np.ones(problem.grid_size**2, dtype=bool)
    ranking = geometric_ranking(
        state.grid_size,
        problem.supports,
        omitted,
        full_action_index,
        full_challenger_index,
    )
    selected: list[int] = []
    path: list[dict[str, Any]] = []
    full_best_local = int(np.argmax(full_acquisition))
    for stop in range(0, ranking.size + batch_size, batch_size):
        selected = ranking[: min(stop, ranking.size)].tolist()
        inference = evaluate_fixed_subset(
            state,
            problem,
            selected,
            incumbent=incumbent,
            delta_mc=delta_mc,
        )
        selected_local = inference.leader_local
        regret = float(full_acquisition[full_best_local] - full_acquisition[selected_local])
        agrees = inference.leader_index == full_action_index
        path.append(
            {
                "active_count": len(selected),
                "action_index": inference.leader_index,
                "full_action_agreement": agrees,
                "full_acquisition_regret": regret,
            }
        )
        if agrees or regret <= regret_threshold:
            break
    return {
        "method": "ORACLE_GEOMETRIC_PREFIX",
        "deployable": False,
        "uses_full_information": True,
        "feeds_deployable_methods": False,
        "full_action_index": full_action_index,
        "full_challenger_index": full_challenger_index,
        "selected_factor_indices": selected,
        "path": path,
    }


def random_matched_subsets(
    state: BOState,
    problem: Problem,
    *,
    matched_count: int,
    subset_count: int,
    random_seed: int,
    incumbent: float,
    delta_mc: float,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(random_seed)
    rows: list[dict[str, Any]] = []
    for replicate in range(subset_count):
        subset = np.sort(
            rng.choice(problem.grid_size**2, size=matched_count, replace=False)
        )
        inference = evaluate_fixed_subset(
            state,
            problem,
            subset,
            incumbent=incumbent,
            delta_mc=delta_mc,
        )
        rows.append(
            {
                "random_replicate": replicate,
                "factor_indices": subset.tolist(),
                "active_count": matched_count,
                "action_index": inference.leader_index,
                "ess_fraction": inference.ess_fraction,
            }
        )
    return rows


__all__ = [
    "BOState",
    "FactorWork",
    "InferenceResult",
    "MethodResult",
    "Problem",
    "build_common_bo_states",
    "build_problem",
    "derive_prospective_seed",
    "effective_sample_size_fraction",
    "evaluate_fixed_subset",
    "factor_energy_sum",
    "factor_support_distance",
    "gaussian_expected_improvement",
    "geometric_ranking",
    "infer_with_escalation",
    "laplace_snis_inference",
    "oracle_geometric_prefix",
    "peak_rss_bytes",
    "random_matched_subsets",
    "reference_snis_inference",
    "run_selective_method",
    "sample_base_gaussian",
]
