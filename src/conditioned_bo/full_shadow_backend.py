"""Development-only FULL-shadow sampling backends for nonlinear E2.

This module is deliberately isolated from the deployable factor-selection
methods and the timed ``FULL`` baseline.  It evaluates the exact existing
FULL target and is only permitted to produce held-out action-quality shadows.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import linalg, sparse

from .nonlinear_pde_influence import DEFAULT_PARAMETERS
from .nonlinear_pde_locality import (
    BOState,
    FactorWork,
    Problem,
    factor_energy_sum,
    laplace_mode,
)


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ProposalSpec:
    """One prospectively ordered Gaussian proposal candidate."""

    name: str
    kind: str
    curvature_lambda: float | None
    covariance_inflation: float | None
    tie_break_rank: int


@dataclass(frozen=True)
class GaussianPrecisionProposal:
    """A Gaussian represented by its sparse precision and dense Cholesky."""

    spec: ProposalSpec
    mean: FloatArray
    precision: sparse.csr_matrix
    precision_cholesky: FloatArray
    log_determinant_precision: float
    spd_check_passed: bool
    minimum_cholesky_diagonal: float


@dataclass(frozen=True)
class LaplaceContext:
    mode: FloatArray
    target_hessian: sparse.csr_matrix
    diagnostics: dict[str, Any]
    work: dict[str, int]


@dataclass(frozen=True)
class SNISBatch:
    """One independent production or pilot batch, including pooling terms."""

    proposal_name: str
    proposal_seed: int
    sample_count: int
    acquisition: FloatArray
    action_local: int
    action_index: int
    top_five_local: NDArray[np.int64]
    ess_fraction: float
    ess_absolute: float
    log_weight_standard_deviation: float
    maximum_normalized_weight: float
    log_weights: FloatArray
    utility: FloatArray
    wall_seconds: float
    work: dict[str, int]


@dataclass(frozen=True)
class IndependenceMHChain:
    """Retained scalar/action traces from one exact independence-MH chain."""

    chain_index: int
    initialization: str
    proposal_seed: int
    uniform_seed: int
    burn_in: int
    retained_count: int
    acceptance_fraction: float
    accepted_transitions: int
    total_transitions: int
    utility: FloatArray
    target_energy: FloatArray
    factor_energy: FloatArray
    wall_seconds: float
    work: dict[str, int]


@dataclass(frozen=True)
class ReferenceGaussianDirectionSampler:
    """Exact centered sampler for the current Gaussian BO reference."""

    problem: Problem
    observed_indices: NDArray[np.int64]
    covariance_columns: FloatArray
    noisy_observation_covariance: FloatArray
    observation_noise_variance: float
    base_eigenvectors: FloatArray
    base_precision_eigenvalues: FloatArray


@dataclass(frozen=True)
class EllipticalSliceChain:
    """Retained traces and factor work from one elliptical-slice chain."""

    chain_index: int
    initialization: str
    proposal_seed: int
    uniform_seed: int
    burn_in: int
    retained_count: int
    acceptance_fraction: float
    accepted_transitions: int
    total_transitions: int
    utility: FloatArray
    target_energy: FloatArray
    factor_energy: FloatArray
    wall_seconds: float
    work: dict[str, int]
    likelihood_evaluations: int
    maximum_likelihood_evaluations_one_transition: int
    initial_angles: FloatArray


def curvature_tempered_precision(
    reference_precision: sparse.spmatrix | ArrayLike,
    target_hessian: sparse.spmatrix | ArrayLike,
    curvature_lambda: float,
) -> sparse.csr_matrix:
    """Return exactly ``Q_t + lambda * (H - Q_t)``."""

    if not 0.0 <= curvature_lambda <= 1.0:
        raise ValueError("curvature_lambda must lie in [0, 1]")
    reference = sparse.csr_matrix(reference_precision, dtype=float)
    hessian = sparse.csr_matrix(target_hessian, dtype=float)
    if reference.shape != hessian.shape or reference.shape[0] != reference.shape[1]:
        raise ValueError("reference precision and Hessian must be matching squares")
    result = reference + curvature_lambda * (hessian - reference)
    result = 0.5 * (result + result.T)
    result.eliminate_zeros()
    return result.tocsr()


def proposal_precision(
    spec: ProposalSpec,
    reference_precision: sparse.spmatrix,
    target_hessian: sparse.spmatrix,
) -> sparse.csr_matrix:
    if spec.kind == "CURVATURE_TEMPERED":
        if spec.curvature_lambda is None:
            raise ValueError("curvature-tempered proposal is missing lambda")
        return curvature_tempered_precision(
            reference_precision, target_hessian, spec.curvature_lambda
        )
    if spec.kind == "BASELINE_INFLATED_LAPLACE":
        if spec.covariance_inflation is None or spec.covariance_inflation <= 0.0:
            raise ValueError("baseline proposal requires positive inflation")
        result = sparse.csr_matrix(target_hessian, dtype=float)
        result = result / spec.covariance_inflation
        result = 0.5 * (result + result.T)
        result.eliminate_zeros()
        return result.tocsr()
    raise ValueError(f"unknown proposal kind: {spec.kind}")


def prepare_gaussian_proposal(
    spec: ProposalSpec,
    mode: ArrayLike,
    reference_precision: sparse.spmatrix,
    target_hessian: sparse.spmatrix,
) -> GaussianPrecisionProposal:
    """Build a proposal and mechanically require a successful SPD Cholesky."""

    precision = proposal_precision(spec, reference_precision, target_hessian)
    dense = precision.toarray()
    try:
        cholesky = np.linalg.cholesky(dense)
    except np.linalg.LinAlgError as error:
        raise RuntimeError(f"proposal {spec.name} is not SPD") from error
    diagonal = np.diag(cholesky)
    if not np.all(np.isfinite(diagonal)) or np.any(diagonal <= 0.0):
        raise RuntimeError(f"proposal {spec.name} failed its SPD check")
    return GaussianPrecisionProposal(
        spec=spec,
        mean=np.asarray(mode, dtype=float).copy(),
        precision=precision,
        precision_cholesky=cholesky,
        log_determinant_precision=float(2.0 * np.log(diagonal).sum()),
        spd_check_passed=True,
        minimum_cholesky_diagonal=float(diagonal.min()),
    )


def gaussian_log_density(
    samples: ArrayLike,
    mean: ArrayLike,
    precision: sparse.spmatrix | ArrayLike,
    *,
    precision_cholesky: FloatArray | None = None,
) -> FloatArray:
    """Evaluate an exactly normalized multivariate Gaussian log density."""

    values = np.asarray(samples, dtype=float)
    if values.ndim == 1:
        values = values[None, :]
    center = np.asarray(mean, dtype=float)
    if values.ndim != 2 or center.shape != (values.shape[1],):
        raise ValueError("sample and mean dimensions do not match")
    matrix = sparse.csr_matrix(precision, dtype=float)
    if matrix.shape != (center.size, center.size):
        raise ValueError("precision dimension does not match the samples")
    if precision_cholesky is None:
        precision_cholesky = np.linalg.cholesky(matrix.toarray())
    diagonal = np.diag(np.asarray(precision_cholesky, dtype=float))
    if np.any(diagonal <= 0.0):
        raise ValueError("precision Cholesky must have a positive diagonal")
    log_determinant = float(2.0 * np.log(diagonal).sum())
    delta = values - center
    quadratic = np.einsum(
        "bi,bi->b", delta, (matrix @ delta.T).T, optimize=True
    )
    dimension = center.size
    return np.asarray(
        0.5 * log_determinant
        - 0.5 * dimension * math.log(2.0 * math.pi)
        - 0.5 * quadratic,
        dtype=float,
    )


def sample_gaussian_precision(
    proposal: GaussianPrecisionProposal,
    sample_count: int,
    rng: np.random.Generator,
) -> FloatArray:
    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    white = rng.standard_normal((sample_count, proposal.mean.size))
    displacement = linalg.solve_triangular(
        proposal.precision_cholesky.T,
        white.T,
        lower=False,
        check_finite=False,
    ).T
    return proposal.mean + displacement


def unnormalized_full_log_target(
    samples: ArrayLike,
    state: BOState,
    problem: Problem,
    work: FactorWork,
) -> FloatArray:
    """Evaluate the unchanged FULL target up to one additive constant."""

    values = np.asarray(samples, dtype=float)
    if values.ndim == 1:
        values = values[None, :]
    if values.shape[1] != state.reference_mean.size:
        raise ValueError("sample dimension does not match the BO state")
    log_target, _, _ = full_target_components(values, state, problem, work)
    return log_target


def full_target_components(
    samples: ArrayLike,
    state: BOState,
    problem: Problem,
    work: FactorWork,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return log target, target energy, and full factor energy."""

    values = np.asarray(samples, dtype=float)
    if values.ndim == 1:
        values = values[None, :]
    delta = values - state.reference_mean
    gaussian_energy = 0.5 * np.einsum(
        "bi,bi->b",
        delta,
        (state.reference_precision @ delta.T).T,
        optimize=True,
    )
    factors = factor_energy_sum(
        values,
        problem,
        np.arange(problem.grid_size**2, dtype=np.int64),
        work,
    )
    target_energy = gaussian_energy + factors
    return -target_energy, target_energy, factors


def full_factor_energy_vectorized(
    samples: ArrayLike,
    problem: Problem,
    work: FactorWork | None = None,
) -> FloatArray:
    """Evaluate all unchanged PDE factors without a Python loop over sites."""

    values = np.asarray(samples, dtype=float)
    if values.ndim == 1:
        values = values[None, :]
    grid_size = problem.grid_size
    if values.ndim != 2 or values.shape[1] != grid_size**2:
        raise ValueError("samples do not match the nonlinear-PDE grid")
    fields = values.reshape(values.shape[0], grid_size, grid_size)
    neighbor_sum = np.zeros_like(fields)
    neighbor_sum[:, 1:, :] += fields[:, :-1, :]
    neighbor_sum[:, :-1, :] += fields[:, 1:, :]
    neighbor_sum[:, :, 1:] += fields[:, :, :-1]
    neighbor_sum[:, :, :-1] += fields[:, :, 1:]
    residual = (
        fields
        - DEFAULT_PARAMETERS.coupling * neighbor_sum
        + DEFAULT_PARAMETERS.nonlinearity * np.sin(fields)
        - problem.source.reshape(1, grid_size, grid_size)
    )
    scaled = residual / DEFAULT_PARAMETERS.tau
    energy = DEFAULT_PARAMETERS.gamma * np.sum(
        np.logaddexp(scaled, -scaled) - math.log(2.0), axis=(1, 2)
    )
    if work is not None:
        work.factor_energy_evaluations += int(values.shape[0] * grid_size**2)
    return np.asarray(energy, dtype=float)


def prepare_reference_direction_sampler(
    state: BOState,
    problem: Problem,
    *,
    observation_noise_variance: float,
) -> ReferenceGaussianDirectionSampler:
    """Prepare the existing exact Matheron-style reference sampler."""

    if observation_noise_variance <= 0.0:
        raise ValueError("observation noise variance must be positive")
    observed = np.asarray(state.observed_indices, dtype=np.int64)
    basis = np.zeros((problem.grid_size**2, observed.size), dtype=float)
    basis[observed, np.arange(observed.size)] = 1.0
    covariance_columns = np.asarray(
        sparse.linalg.spsolve(problem.precision.tocsc(), basis), dtype=float
    )
    observation_covariance = covariance_columns[observed, :]
    noisy = observation_covariance + observation_noise_variance * np.eye(
        observed.size
    )
    grid_size = problem.grid_size
    diagonal = np.full(grid_size, 2.0)
    diagonal[[0, -1]] = 1.0
    path_laplacian = np.diag(diagonal)
    path_laplacian += np.diag(-np.ones(grid_size - 1), 1)
    path_laplacian += np.diag(-np.ones(grid_size - 1), -1)
    eigenvalues_1d, eigenvectors = np.linalg.eigh(path_laplacian)
    precision_eigenvalues = (
        DEFAULT_PARAMETERS.q0
        + DEFAULT_PARAMETERS.q_laplacian
        * (eigenvalues_1d[:, None] + eigenvalues_1d[None, :])
    )
    return ReferenceGaussianDirectionSampler(
        problem=problem,
        observed_indices=observed,
        covariance_columns=covariance_columns,
        noisy_observation_covariance=noisy,
        observation_noise_variance=observation_noise_variance,
        base_eigenvectors=eigenvectors,
        base_precision_eigenvalues=precision_eigenvalues,
    )


def sample_reference_direction(
    sampler: ReferenceGaussianDirectionSampler,
    rng: np.random.Generator,
) -> FloatArray:
    """Draw exactly from ``N(0, state.reference_precision^-1)``."""

    grid_size = sampler.problem.grid_size
    white = rng.standard_normal((grid_size, grid_size))
    scaled = white / np.sqrt(sampler.base_precision_eigenvalues)
    prior_grid = sampler.base_eigenvectors @ scaled @ sampler.base_eigenvectors.T
    prior = prior_grid.ravel()
    noise = math.sqrt(sampler.observation_noise_variance) * rng.standard_normal(
        sampler.observed_indices.size
    )
    residual = -(prior[sampler.observed_indices] + noise)
    coefficients = np.linalg.solve(sampler.noisy_observation_covariance, residual)
    return np.asarray(prior + sampler.covariance_columns @ coefficients, dtype=float)


def build_full_laplace_context(
    state: BOState,
    problem: Problem,
    *,
    gradient_tolerance: float,
    maximum_iterations: int,
) -> LaplaceContext:
    work = FactorWork()
    mode, hessian, diagnostics = laplace_mode(
        state,
        problem,
        np.ones(problem.grid_size**2, dtype=bool),
        work=work,
        gradient_tolerance=gradient_tolerance,
        maximum_iterations=maximum_iterations,
    )
    return LaplaceContext(
        mode=mode,
        target_hessian=hessian,
        diagnostics=diagnostics,
        work=work.to_dict(),
    )


def run_snis_batch(
    state: BOState,
    problem: Problem,
    proposal: GaussianPrecisionProposal,
    *,
    incumbent: float,
    sample_count: int,
    proposal_seed: int,
) -> SNISBatch:
    """Draw and evaluate one fresh independent FULL-shadow SNIS batch."""

    started = time.perf_counter()
    work = FactorWork()
    samples = sample_gaussian_precision(
        proposal, sample_count, np.random.default_rng(proposal_seed)
    )
    log_target = unnormalized_full_log_target(samples, state, problem, work)
    log_proposal = gaussian_log_density(
        samples,
        proposal.mean,
        proposal.precision,
        precision_cholesky=proposal.precision_cholesky,
    )
    log_weights = log_target - log_proposal
    shifted = log_weights - np.max(log_weights)
    weights = np.exp(shifted)
    normalized = weights / weights.sum()
    utility = np.maximum(samples[:, state.action_indices] - incumbent, 0.0)
    acquisition = normalized @ utility
    order = np.argsort(-acquisition, kind="stable")
    ess_absolute = float(weights.sum() ** 2 / np.dot(weights, weights))
    return SNISBatch(
        proposal_name=proposal.spec.name,
        proposal_seed=int(proposal_seed),
        sample_count=int(sample_count),
        acquisition=np.asarray(acquisition, dtype=float),
        action_local=int(order[0]),
        action_index=int(state.action_indices[int(order[0])]),
        top_five_local=np.asarray(order[:5], dtype=np.int64),
        ess_fraction=ess_absolute / sample_count,
        ess_absolute=ess_absolute,
        log_weight_standard_deviation=float(np.std(log_weights, ddof=1)),
        maximum_normalized_weight=float(normalized.max()),
        log_weights=np.asarray(log_weights, dtype=float),
        utility=np.asarray(utility, dtype=float),
        wall_seconds=time.perf_counter() - started,
        work=work.to_dict(),
    )


def top_five(batch: SNISBatch, state: BOState) -> list[dict[str, Any]]:
    return [
        {
            "rank": rank,
            "action_index": int(state.action_indices[int(local)]),
            "acquisition": float(batch.acquisition[int(local)]),
        }
        for rank, local in enumerate(batch.top_five_local, start=1)
    ]


def batch_public_summary(batch: SNISBatch, state: BOState) -> dict[str, Any]:
    """Drop samples/pooling terms so pilot observations cannot be reused."""

    return {
        "proposal_name": batch.proposal_name,
        "proposal_seed": batch.proposal_seed,
        "sample_count": batch.sample_count,
        "action_index": batch.action_index,
        "top_five": top_five(batch, state),
        "ess_fraction": batch.ess_fraction,
        "ess_absolute": batch.ess_absolute,
        "log_weight_standard_deviation": batch.log_weight_standard_deviation,
        "maximum_normalized_weight": batch.maximum_normalized_weight,
        "wall_seconds": batch.wall_seconds,
        "work": batch.work,
    }


def select_proposal_from_pilots(
    pilot_records: Sequence[Mapping[str, Any]],
    proposal_specs: Sequence[ProposalSpec],
) -> ProposalSpec:
    """Choose highest pilot ESS, breaking exact ties by configured order."""

    ranks = {spec.name: spec.tie_break_rank for spec in proposal_specs}
    specs = {spec.name: spec for spec in proposal_specs}
    if len(ranks) != len(proposal_specs):
        raise ValueError("proposal names must be unique")
    if set(ranks) != {str(record["proposal_name"]) for record in pilot_records}:
        raise ValueError("pilot records do not match the candidate proposal set")
    best = max(
        pilot_records,
        key=lambda record: (
            float(record["ess_fraction"]),
            -ranks[str(record["proposal_name"])],
        ),
    )
    return specs[str(best["proposal_name"])]


def pooled_batch(batch_a: SNISBatch, batch_b: SNISBatch, state: BOState) -> SNISBatch:
    """Pool independent batches from the same proposal for diagnostics only."""

    if batch_a.proposal_name != batch_b.proposal_name:
        raise ValueError("only batches from the same proposal may be pooled")
    log_weights = np.concatenate((batch_a.log_weights, batch_b.log_weights))
    utility = np.concatenate((batch_a.utility, batch_b.utility), axis=0)
    shifted = log_weights - np.max(log_weights)
    weights = np.exp(shifted)
    normalized = weights / weights.sum()
    acquisition = normalized @ utility
    order = np.argsort(-acquisition, kind="stable")
    ess_absolute = float(weights.sum() ** 2 / np.dot(weights, weights))
    return SNISBatch(
        proposal_name=batch_a.proposal_name,
        proposal_seed=-1,
        sample_count=int(log_weights.size),
        acquisition=np.asarray(acquisition),
        action_local=int(order[0]),
        action_index=int(state.action_indices[int(order[0])]),
        top_five_local=np.asarray(order[:5], dtype=np.int64),
        ess_fraction=ess_absolute / log_weights.size,
        ess_absolute=ess_absolute,
        log_weight_standard_deviation=float(np.std(log_weights, ddof=1)),
        maximum_normalized_weight=float(normalized.max()),
        log_weights=log_weights,
        utility=utility,
        wall_seconds=batch_a.wall_seconds + batch_b.wall_seconds,
        work={
            key: int(batch_a.work[key] + batch_b.work[key]) for key in batch_a.work
        },
    )


def compare_independent_batches(
    batch_a: SNISBatch,
    batch_b: SNISBatch,
    state: BOState,
) -> dict[str, Any]:
    acquisition_a = batch_a.acquisition
    acquisition_b = batch_b.acquisition
    union = np.unique(np.concatenate((batch_a.top_five_local, batch_b.top_five_local)))
    regret_a_under_b = float(
        max(0.0, acquisition_b[batch_b.action_local] - acquisition_b[batch_a.action_local])
    )
    regret_b_under_a = float(
        max(0.0, acquisition_a[batch_a.action_local] - acquisition_a[batch_b.action_local])
    )
    return {
        "action_agreement": batch_a.action_index == batch_b.action_index,
        "batch_actions": [batch_a.action_index, batch_b.action_index],
        "maximum_acquisition_vector_difference": float(
            np.max(np.abs(acquisition_a - acquisition_b))
        ),
        "maximum_top_five_union_difference": float(
            np.max(np.abs(acquisition_a[union] - acquisition_b[union]))
        ),
        "batch_a_action_regret_under_batch_b": regret_a_under_b,
        "batch_b_action_regret_under_batch_a": regret_b_under_a,
        "maximum_reciprocal_action_regret": max(
            regret_a_under_b, regret_b_under_a
        ),
        "batch_a": batch_public_summary(batch_a, state),
        "batch_b": batch_public_summary(batch_b, state),
    }


def compare_batch_to_reference(
    batch: SNISBatch,
    reference: SNISBatch,
) -> dict[str, float | int]:
    """Compare one fresh batch to a prior high-fidelity diagnostic estimate."""

    regret_batch_under_reference = float(
        max(
            0.0,
            reference.acquisition[reference.action_local]
            - reference.acquisition[batch.action_local],
        )
    )
    regret_reference_under_batch = float(
        max(
            0.0,
            batch.acquisition[batch.action_local]
            - batch.acquisition[reference.action_local],
        )
    )
    return {
        "batch_action": batch.action_index,
        "reference_action": reference.action_index,
        "maximum_acquisition_vector_difference": float(
            np.max(np.abs(batch.acquisition - reference.acquisition))
        ),
        "batch_action_regret_under_reference": regret_batch_under_reference,
        "reference_action_regret_under_batch": regret_reference_under_batch,
        "maximum_reciprocal_action_regret": max(
            regret_batch_under_reference, regret_reference_under_batch
        ),
    }


def independence_mh_log_acceptance(
    current_log_weight: float, proposed_log_weight: float
) -> float:
    """Return the exact log acceptance probability for independence MH."""

    if not np.isfinite(current_log_weight) or not np.isfinite(proposed_log_weight):
        raise ValueError("independence-MH log weights must be finite")
    return float(min(0.0, proposed_log_weight - current_log_weight))


def independence_mh_accept(
    current_log_weight: float,
    proposed_log_weight: float,
    uniform: float,
) -> bool:
    if not 0.0 < uniform <= 1.0:
        raise ValueError("uniform variate must lie in (0, 1]")
    return bool(
        math.log(uniform)
        <= independence_mh_log_acceptance(current_log_weight, proposed_log_weight)
    )


def run_independence_mh_chain(
    state: BOState,
    problem: Problem,
    proposal: GaussianPrecisionProposal,
    *,
    incumbent: float,
    chain_index: int,
    initial_state: ArrayLike,
    initialization: str,
    burn_in: int,
    retained_count: int,
    proposal_seed: int,
    uniform_seed: int,
    proposal_evaluation_block_size: int,
) -> IndependenceMHChain:
    """Run one exact chain while batching expensive proposal evaluations."""

    if burn_in < 0 or retained_count < 2:
        raise ValueError("invalid burn-in or retained chain length")
    if proposal_evaluation_block_size < 1:
        raise ValueError("proposal block size must be positive")
    started = time.perf_counter()
    transition_count = burn_in + retained_count
    work = FactorWork()
    initial = np.asarray(initial_state, dtype=float)
    if initial.shape != proposal.mean.shape:
        raise ValueError("initial state has the wrong dimension")
    initial_log_target, initial_target_energy, initial_factor_energy = (
        full_target_components(initial, state, problem, work)
    )
    initial_log_proposal = gaussian_log_density(
        initial,
        proposal.mean,
        proposal.precision,
        precision_cholesky=proposal.precision_cholesky,
    )
    current_log_weight = float(initial_log_target[0] - initial_log_proposal[0])
    current_target_energy = float(initial_target_energy[0])
    current_factor_energy = float(initial_factor_energy[0])
    current_utility = np.maximum(initial[state.action_indices] - incumbent, 0.0)

    proposal_log_weights = np.empty(transition_count, dtype=float)
    proposal_target_energy = np.empty(transition_count, dtype=float)
    proposal_factor_energy = np.empty(transition_count, dtype=float)
    proposal_utility = np.empty(
        (transition_count, state.action_indices.size), dtype=float
    )
    proposal_rng = np.random.default_rng(proposal_seed)
    for start in range(0, transition_count, proposal_evaluation_block_size):
        stop = min(start + proposal_evaluation_block_size, transition_count)
        samples = sample_gaussian_precision(proposal, stop - start, proposal_rng)
        log_target, target_energy, factor_energy = full_target_components(
            samples, state, problem, work
        )
        log_proposal = gaussian_log_density(
            samples,
            proposal.mean,
            proposal.precision,
            precision_cholesky=proposal.precision_cholesky,
        )
        proposal_log_weights[start:stop] = log_target - log_proposal
        proposal_target_energy[start:stop] = target_energy
        proposal_factor_energy[start:stop] = factor_energy
        proposal_utility[start:stop] = np.maximum(
            samples[:, state.action_indices] - incumbent, 0.0
        )

    uniforms = np.random.default_rng(uniform_seed).random(transition_count)
    retained_utility = np.empty(
        (retained_count, state.action_indices.size), dtype=float
    )
    retained_target_energy = np.empty(retained_count, dtype=float)
    retained_factor_energy = np.empty(retained_count, dtype=float)
    accepted = 0
    retained_index = 0
    for transition in range(transition_count):
        proposed_log_weight = float(proposal_log_weights[transition])
        if independence_mh_accept(
            current_log_weight, proposed_log_weight, float(uniforms[transition])
        ):
            current_log_weight = proposed_log_weight
            current_target_energy = float(proposal_target_energy[transition])
            current_factor_energy = float(proposal_factor_energy[transition])
            current_utility = proposal_utility[transition].copy()
            accepted += 1
        if transition >= burn_in:
            retained_utility[retained_index] = current_utility
            retained_target_energy[retained_index] = current_target_energy
            retained_factor_energy[retained_index] = current_factor_energy
            retained_index += 1
    if retained_index != retained_count:
        raise RuntimeError("independence-MH retention accounting failed")
    return IndependenceMHChain(
        chain_index=chain_index,
        initialization=initialization,
        proposal_seed=proposal_seed,
        uniform_seed=uniform_seed,
        burn_in=burn_in,
        retained_count=retained_count,
        acceptance_fraction=accepted / transition_count,
        accepted_transitions=accepted,
        total_transitions=transition_count,
        utility=retained_utility,
        target_energy=retained_target_energy,
        factor_energy=retained_factor_energy,
        wall_seconds=time.perf_counter() - started,
        work=work.to_dict(),
    )


def split_rhat(chains: ArrayLike) -> float:
    """Classic split R-hat for equal-length independent scalar chains."""

    values = np.asarray(chains, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 4:
        raise ValueError("split R-hat requires at least two chains of length four")
    if not np.all(np.isfinite(values)):
        raise ValueError("split R-hat input must be finite")
    half = values.shape[1] // 2
    if half < 2:
        raise ValueError("split chains must retain at least two draws")
    split = np.concatenate((values[:, :half], values[:, -half:]), axis=0)
    within = float(np.mean(np.var(split, axis=1, ddof=1)))
    between = float(half * np.var(np.mean(split, axis=1), ddof=1))
    if within == 0.0:
        return 1.0 if between == 0.0 else float("inf")
    variance = (half - 1.0) / half * within + between / half
    return float(math.sqrt(max(variance / within, 0.0)))


def _autocovariance_fft(values: FloatArray) -> FloatArray:
    centered = np.asarray(values, dtype=float) - float(np.mean(values))
    count = centered.size
    fft_size = 1 << (2 * count - 1).bit_length()
    transformed = np.fft.rfft(centered, n=fft_size)
    covariance = np.fft.irfft(transformed * np.conjugate(transformed), n=fft_size)[
        :count
    ]
    return np.asarray(covariance / np.arange(count, 0, -1), dtype=float)


def autocorrelation_effective_sample_size(chains: ArrayLike) -> float:
    """Autocorrelation ESS using Geyer's positive/monotone paired sequence."""

    values = np.asarray(chains, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 4:
        raise ValueError("ESS requires at least two chains of length four")
    if not np.all(np.isfinite(values)):
        raise ValueError("ESS input must be finite")
    autocovariances = np.asarray([_autocovariance_fft(chain) for chain in values])
    variance = float(np.mean(autocovariances[:, 0]))
    total = values.size
    if variance <= 0.0:
        return float(total)
    rho = np.mean(autocovariances, axis=0) / variance
    paired: list[float] = []
    for lag in range(1, rho.size - 1, 2):
        pair = float(rho[lag] + rho[lag + 1])
        if pair < 0.0:
            break
        if paired:
            pair = min(pair, paired[-1])
        paired.append(pair)
    integrated_time = max(1.0, 1.0 + 2.0 * float(np.sum(paired)))
    return float(min(total, total / integrated_time))


def elliptical_slice_transition(
    current: ArrayLike,
    reference_mean: ArrayLike,
    reference_direction: ArrayLike,
    current_log_likelihood: float,
    log_likelihood: Any,
    rng: np.random.Generator,
    *,
    maximum_bracket_evaluations: int,
) -> tuple[FloatArray, float, int, float]:
    """Perform one standard elliptical-slice transition with random bracket."""

    if maximum_bracket_evaluations < 1:
        raise ValueError("maximum bracket evaluations must be positive")
    value = np.asarray(current, dtype=float)
    mean = np.asarray(reference_mean, dtype=float)
    direction = np.asarray(reference_direction, dtype=float)
    if value.shape != mean.shape or direction.shape != mean.shape:
        raise ValueError("ellipse vectors must have matching dimensions")
    log_slice = float(current_log_likelihood + math.log(float(rng.random())))
    initial_angle = float(rng.uniform(0.0, 2.0 * math.pi))
    angle = initial_angle
    angle_min = angle - 2.0 * math.pi
    angle_max = angle
    centered = value - mean
    for evaluation in range(1, maximum_bracket_evaluations + 1):
        candidate = (
            mean + centered * math.cos(angle) + direction * math.sin(angle)
        )
        candidate_log_likelihood = float(log_likelihood(candidate))
        if not np.isfinite(candidate_log_likelihood):
            raise RuntimeError("elliptical-slice likelihood became nonfinite")
        if candidate_log_likelihood >= log_slice:
            return (
                np.asarray(candidate, dtype=float),
                candidate_log_likelihood,
                evaluation,
                initial_angle,
            )
        if angle < 0.0:
            angle_min = angle
        else:
            angle_max = angle
        angle = float(rng.uniform(angle_min, angle_max))
    raise RuntimeError("elliptical-slice bracket failed to find the slice")


def run_elliptical_slice_chain(
    state: BOState,
    problem: Problem,
    reference_sampler: ReferenceGaussianDirectionSampler,
    *,
    incumbent: float,
    chain_index: int,
    initial_state: ArrayLike,
    initialization: str,
    burn_in: int,
    retained_count: int,
    direction_seed: int,
    slice_seed: int,
    maximum_bracket_evaluations: int,
) -> EllipticalSliceChain:
    """Run one FULL elliptical-slice chain using the exact Gaussian reference."""

    if burn_in < 0 or retained_count < 2:
        raise ValueError("invalid burn-in or retained chain length")
    started = time.perf_counter()
    current = np.asarray(initial_state, dtype=float).copy()
    if current.shape != state.reference_mean.shape:
        raise ValueError("initial state has the wrong dimension")
    work = FactorWork()
    current_factor_energy = float(
        full_factor_energy_vectorized(current, problem, work)[0]
    )
    current_log_likelihood = -current_factor_energy
    total_transitions = burn_in + retained_count
    retained_utility = np.empty(
        (retained_count, state.action_indices.size), dtype=float
    )
    retained_target_energy = np.empty(retained_count, dtype=float)
    retained_factor_energy = np.empty(retained_count, dtype=float)
    initial_angles = np.empty(total_transitions, dtype=float)
    likelihood_evaluations = 1
    maximum_evaluations = 0
    direction_rng = np.random.default_rng(direction_seed)
    slice_rng = np.random.default_rng(slice_seed)
    retained_index = 0

    def evaluate(candidate: FloatArray) -> float:
        return -float(full_factor_energy_vectorized(candidate, problem, work)[0])

    for transition in range(total_transitions):
        direction = sample_reference_direction(reference_sampler, direction_rng)
        current, current_log_likelihood, evaluations, initial_angle = (
            elliptical_slice_transition(
                current,
                state.reference_mean,
                direction,
                current_log_likelihood,
                evaluate,
                slice_rng,
                maximum_bracket_evaluations=maximum_bracket_evaluations,
            )
        )
        initial_angles[transition] = initial_angle
        likelihood_evaluations += evaluations
        maximum_evaluations = max(maximum_evaluations, evaluations)
        current_factor_energy = -current_log_likelihood
        if transition >= burn_in:
            retained_utility[retained_index] = np.maximum(
                current[state.action_indices] - incumbent, 0.0
            )
            delta = current - state.reference_mean
            gaussian_energy = 0.5 * float(
                delta @ (state.reference_precision @ delta)
            )
            retained_factor_energy[retained_index] = current_factor_energy
            retained_target_energy[retained_index] = (
                gaussian_energy + current_factor_energy
            )
            retained_index += 1
    if retained_index != retained_count:
        raise RuntimeError("elliptical-slice retention accounting failed")
    return EllipticalSliceChain(
        chain_index=chain_index,
        initialization=initialization,
        proposal_seed=direction_seed,
        uniform_seed=slice_seed,
        burn_in=burn_in,
        retained_count=retained_count,
        acceptance_fraction=1.0,
        accepted_transitions=total_transitions,
        total_transitions=total_transitions,
        utility=retained_utility,
        target_energy=retained_target_energy,
        factor_energy=retained_factor_energy,
        wall_seconds=time.perf_counter() - started,
        work=work.to_dict(),
        likelihood_evaluations=likelihood_evaluations,
        maximum_likelihood_evaluations_one_transition=maximum_evaluations,
        initial_angles=initial_angles,
    )


def _top_actions_from_acquisition(
    acquisition: FloatArray, state: BOState, count: int
) -> list[dict[str, Any]]:
    if count < 1:
        raise ValueError("top-action count must be positive")
    order = np.argsort(-acquisition, kind="stable")[:count]
    return [
        {
            "rank": rank,
            "action_index": int(state.action_indices[int(local)]),
            "acquisition": float(acquisition[int(local)]),
        }
        for rank, local in enumerate(order, start=1)
    ]


def aggregate_independence_mh_chains(
    chains: Sequence[IndependenceMHChain | EllipticalSliceChain],
    state: BOState,
    *,
    group_a_chains: Sequence[int],
    group_b_chains: Sequence[int],
    backend_name: str = "LAPLACE_INDEPENDENCE_MH",
    diagnostic_top_action_count: int = 5,
    strict_gate_top_action_count: int = 5,
) -> dict[str, Any]:
    """Compute convergence and decision diagnostics without confusing IS ESS."""

    if len(chains) < 4:
        raise ValueError("at least four independent chains are required")
    if diagnostic_top_action_count < strict_gate_top_action_count:
        raise ValueError(
            "diagnostic top-action count cannot be smaller than the strict gate count"
        )
    if strict_gate_top_action_count < 2:
        raise ValueError("strict gate needs at least a leader and challenger")
    ordered = sorted(chains, key=lambda chain: chain.chain_index)
    if [chain.chain_index for chain in ordered] != list(range(len(ordered))):
        raise ValueError("chain indices must be consecutive from zero")
    retained = {chain.retained_count for chain in ordered}
    if len(retained) != 1:
        raise ValueError("chains must have equal retained lengths")
    group_a = [ordered[index] for index in group_a_chains]
    group_b = [ordered[index] for index in group_b_chains]
    if set(group_a_chains) & set(group_b_chains):
        raise ValueError("independent chain groups must be disjoint")
    if sorted(tuple(group_a_chains) + tuple(group_b_chains)) != list(
        range(len(ordered))
    ):
        raise ValueError("chain groups must partition every chain")

    chain_acquisitions = [np.mean(chain.utility, axis=0) for chain in ordered]
    pooled_acquisition = np.mean(chain_acquisitions, axis=0)
    group_a_acquisition = np.mean(
        [chain_acquisitions[index] for index in group_a_chains], axis=0
    )
    group_b_acquisition = np.mean(
        [chain_acquisitions[index] for index in group_b_chains], axis=0
    )
    pooled_order = np.argsort(-pooled_acquisition, kind="stable")
    group_a_order = np.argsort(-group_a_acquisition, kind="stable")
    group_b_order = np.argsort(-group_b_acquisition, kind="stable")
    strict_candidate_union = set(
        pooled_order[:strict_gate_top_action_count].tolist()
    )
    strict_candidate_union.update(
        group_a_order[:strict_gate_top_action_count].tolist()
    )
    strict_candidate_union.update(
        group_b_order[:strict_gate_top_action_count].tolist()
    )
    candidate_union = set(pooled_order[:diagnostic_top_action_count].tolist())
    candidate_union.update(
        group_a_order[:diagnostic_top_action_count].tolist()
    )
    candidate_union.update(
        group_b_order[:diagnostic_top_action_count].tolist()
    )
    for acquisition in chain_acquisitions:
        order = np.argsort(-acquisition, kind="stable")
        strict_candidate_union.update(
            order[:strict_gate_top_action_count].tolist()
        )
        candidate_union.update(order[:diagnostic_top_action_count].tolist())
    candidate_locals = sorted(candidate_union)
    leader_local = int(pooled_order[0])

    scalar_diagnostics: list[dict[str, Any]] = []

    def add_diagnostic(
        observable: str,
        values: FloatArray,
        *,
        observable_type: str,
        action_index: int | None = None,
        challenger_index: int | None = None,
        required_for_strict_gate: bool = True,
    ) -> None:
        ess = autocorrelation_effective_sample_size(values)
        flattened = values.ravel()
        scalar_diagnostics.append(
            {
                "observable": observable,
                "observable_type": observable_type,
                "action_index": action_index,
                "challenger_index": challenger_index,
                "mean": float(np.mean(flattened)),
                "standard_deviation": float(np.std(flattened, ddof=1)),
                "split_rhat": split_rhat(values),
                "autocorrelation_ess": ess,
                "mcse": float(np.std(flattened, ddof=1) / math.sqrt(ess)),
                "required_for_strict_gate": required_for_strict_gate,
            }
        )

    add_diagnostic(
        "target_energy",
        np.asarray([chain.target_energy for chain in ordered]),
        observable_type="TARGET_ENERGY",
    )
    add_diagnostic(
        "full_factor_energy",
        np.asarray([chain.factor_energy for chain in ordered]),
        observable_type="FULL_FACTOR_ENERGY",
    )
    for local in candidate_locals:
        action_index = int(state.action_indices[local])
        utility = np.asarray([chain.utility[:, local] for chain in ordered])
        add_diagnostic(
            f"utility_action_{action_index}",
            utility,
            observable_type="TOP_ACTION_UTILITY",
            action_index=action_index,
            required_for_strict_gate=local in strict_candidate_union,
        )
        if local != leader_local:
            gap = np.asarray(
                [
                    chain.utility[:, leader_local] - chain.utility[:, local]
                    for chain in ordered
                ]
            )
            add_diagnostic(
                f"gap_{int(state.action_indices[leader_local])}_vs_{action_index}",
                gap,
                observable_type="LEADER_CHALLENGER_GAP",
                action_index=int(state.action_indices[leader_local]),
                challenger_index=action_index,
                required_for_strict_gate=local in strict_candidate_union,
            )

    action_a = int(group_a_order[0])
    action_b = int(group_b_order[0])
    regret_a_under_b = float(
        max(
            0.0,
            group_b_acquisition[action_b] - group_b_acquisition[action_a],
        )
    )
    regret_b_under_a = float(
        max(
            0.0,
            group_a_acquisition[action_a] - group_a_acquisition[action_b],
        )
    )
    work_keys = ordered[0].work.keys()
    return {
        "backend": backend_name,
        "importance_weight_ess_reported_as_mcmc_ess": False,
        "chain_count": len(ordered),
        "burn_in": ordered[0].burn_in,
        "retained_per_chain": ordered[0].retained_count,
        "acceptance_fractions": [chain.acceptance_fraction for chain in ordered],
        "median_chain_acceptance": float(
            np.median([chain.acceptance_fraction for chain in ordered])
        ),
        "chainwise_actions": [
            int(state.action_indices[int(np.argmax(acquisition))])
            for acquisition in chain_acquisitions
        ],
        "pooled_action": int(state.action_indices[leader_local]),
        "pooled_acquisition": pooled_acquisition.tolist(),
        "pooled_top_five": _top_actions_from_acquisition(
            pooled_acquisition, state, 5
        ),
        "pooled_top_actions": _top_actions_from_acquisition(
            pooled_acquisition, state, diagnostic_top_action_count
        ),
        "group_a_chains": list(group_a_chains),
        "group_b_chains": list(group_b_chains),
        "group_a_action": int(state.action_indices[action_a]),
        "group_b_action": int(state.action_indices[action_b]),
        "group_action_agreement": action_a == action_b,
        "group_a_acquisition": group_a_acquisition.tolist(),
        "group_b_acquisition": group_b_acquisition.tolist(),
        "group_a_top_five": _top_actions_from_acquisition(
            group_a_acquisition, state, 5
        ),
        "group_b_top_five": _top_actions_from_acquisition(
            group_b_acquisition, state, 5
        ),
        "group_a_top_actions": _top_actions_from_acquisition(
            group_a_acquisition, state, diagnostic_top_action_count
        ),
        "group_b_top_actions": _top_actions_from_acquisition(
            group_b_acquisition, state, diagnostic_top_action_count
        ),
        "maximum_group_acquisition_vector_difference": float(
            np.max(np.abs(group_a_acquisition - group_b_acquisition))
        ),
        "group_a_action_regret_under_group_b": regret_a_under_b,
        "group_b_action_regret_under_group_a": regret_b_under_a,
        "maximum_reciprocal_group_action_regret": max(
            regret_a_under_b, regret_b_under_a
        ),
        "candidate_top_action_union": [
            int(state.action_indices[local]) for local in candidate_locals
        ],
        "diagnostic_top_action_count": diagnostic_top_action_count,
        "strict_gate_top_action_count": strict_gate_top_action_count,
        "strict_gate_candidate_top_action_union": [
            int(state.action_indices[local])
            for local in sorted(strict_candidate_union)
        ],
        "leader_challenger_gap_observables": [
            row["observable"]
            for row in scalar_diagnostics
            if row["observable_type"] == "LEADER_CHALLENGER_GAP"
        ],
        "maximum_split_rhat": float(
            max(
                row["split_rhat"]
                for row in scalar_diagnostics
                if row["required_for_strict_gate"]
            )
        ),
        "maximum_split_rhat_all_diagnostics": float(
            max(row["split_rhat"] for row in scalar_diagnostics)
        ),
        "minimum_leader_challenger_gap_ess": float(
            min(
                row["autocorrelation_ess"]
                for row in scalar_diagnostics
                if row["observable_type"] == "LEADER_CHALLENGER_GAP"
                and row["required_for_strict_gate"]
            )
        ),
        "minimum_diagnostic_leader_challenger_gap_ess": float(
            min(
                row["autocorrelation_ess"]
                for row in scalar_diagnostics
                if row["observable_type"] == "LEADER_CHALLENGER_GAP"
            )
        ),
        "scalar_diagnostics": scalar_diagnostics,
        "chain_diagnostics": [
            {
                "chain_index": chain.chain_index,
                "initialization": chain.initialization,
                "proposal_seed": chain.proposal_seed,
                "uniform_seed": chain.uniform_seed,
                "burn_in": chain.burn_in,
                "retained_count": chain.retained_count,
                "acceptance_fraction": chain.acceptance_fraction,
                "accepted_transitions": chain.accepted_transitions,
                "total_transitions": chain.total_transitions,
                "wall_seconds": chain.wall_seconds,
                "work": chain.work,
                "likelihood_evaluations": getattr(
                    chain, "likelihood_evaluations", None
                ),
                "maximum_likelihood_evaluations_one_transition": getattr(
                    chain, "maximum_likelihood_evaluations_one_transition", None
                ),
                "initial_angle_min": (
                    float(np.min(chain.initial_angles))
                    if isinstance(chain, EllipticalSliceChain)
                    else None
                ),
                "initial_angle_max": (
                    float(np.max(chain.initial_angles))
                    if isinstance(chain, EllipticalSliceChain)
                    else None
                ),
                "action_index": int(
                    state.action_indices[
                        int(np.argmax(chain_acquisitions[chain.chain_index]))
                    ]
                ),
            }
            for chain in ordered
        ],
        "wall_seconds": float(sum(chain.wall_seconds for chain in ordered)),
        "factor_work": {
            key: int(sum(chain.work[key] for chain in ordered)) for key in work_keys
        },
    }


def shadow_only_contract() -> dict[str, bool]:
    return {
        "development_only": True,
        "held_out_full_shadow_only": True,
        "feeds_deployable_factor_selection": False,
        "included_in_timed_full_baseline": False,
    }


__all__ = [
    "EllipticalSliceChain",
    "GaussianPrecisionProposal",
    "IndependenceMHChain",
    "LaplaceContext",
    "ProposalSpec",
    "ReferenceGaussianDirectionSampler",
    "SNISBatch",
    "batch_public_summary",
    "build_full_laplace_context",
    "aggregate_independence_mh_chains",
    "autocorrelation_effective_sample_size",
    "compare_batch_to_reference",
    "compare_independent_batches",
    "curvature_tempered_precision",
    "elliptical_slice_transition",
    "full_factor_energy_vectorized",
    "gaussian_log_density",
    "full_target_components",
    "independence_mh_accept",
    "independence_mh_log_acceptance",
    "pooled_batch",
    "prepare_gaussian_proposal",
    "prepare_reference_direction_sampler",
    "proposal_precision",
    "run_snis_batch",
    "run_independence_mh_chain",
    "run_elliptical_slice_chain",
    "sample_gaussian_precision",
    "sample_reference_direction",
    "select_proposal_from_pilots",
    "shadow_only_contract",
    "split_rhat",
    "top_five",
    "unnormalized_full_log_target",
]
