"""Rigorous finite-sample inference certification for reflection symmetry.

This module instantiates the inference term in the finite-grid T3 action
certificate.  Gaussian precision ``Q`` is used only for the inference
Lipschitz geometry; the Menz comparison matrix ``A`` remains confined to the
structural bound supplied by :mod:`conditioned_bo.symmetry_influence`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg
from scipy.special import ndtr

from conditioned_bo.symmetry_influence import (
    ei_block_decision_footprint,
    solve_comparison,
)


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
ReferenceDraw = Callable[[np.random.Generator, int], FloatArray]


@dataclass(frozen=True)
class SamplerChunkDiagnostic:
    """Diagnostics for one generated rejection-sampling proposal chunk."""

    chunk_index: int
    proposals_generated: int
    proposals_consumed: int
    accepted_candidates: int
    accepted_retained: int
    cumulative_proposals_generated: int
    cumulative_proposals_consumed: int
    cumulative_accepted_retained: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class RejectionSampleResult:
    """Exact retained samples and complete proposal accounting."""

    samples: FloatArray
    proposals_generated: int
    proposals_consumed: int
    accepted_candidates: int
    acceptance_rate: float
    chunks: tuple[SamplerChunkDiagnostic, ...]


class ProposalCapExceeded(RuntimeError):
    """Raised when an exact requested batch cannot be completed under its cap."""

    def __init__(
        self,
        *,
        requested_samples: int,
        accepted_samples: int,
        proposals_generated: int,
        proposals_consumed: int,
        accepted_candidates: int,
        chunks: tuple[SamplerChunkDiagnostic, ...],
    ) -> None:
        super().__init__(
            "proposal cap reached before the exact requested sample count "
            f"({accepted_samples}/{requested_samples})"
        )
        self.requested_samples = requested_samples
        self.accepted_samples = accepted_samples
        self.proposals_generated = proposals_generated
        self.proposals_consumed = proposals_consumed
        self.accepted_candidates = accepted_candidates
        self.chunks = chunks


@dataclass(frozen=True)
class BatchClaim:
    """The active-set/leader pair to which a certification batch is bound."""

    batch_id: str
    active_set_hash: str
    leader_index: int


class CertificationBatchRegistry:
    """Enforce one certification use for every fresh batch ID."""

    def __init__(self) -> None:
        self._claims: dict[str, BatchClaim] = {}

    @property
    def claims(self) -> tuple[BatchClaim, ...]:
        return tuple(self._claims.values())

    def claim(
        self,
        batch_id: str,
        active_factors: Sequence[int],
        leader_index: int,
    ) -> BatchClaim:
        if not batch_id:
            raise ValueError("batch_id must be nonempty")
        if batch_id in self._claims:
            previous = self._claims[batch_id]
            raise RuntimeError(
                "certification batch reuse is forbidden: "
                f"{batch_id} was already bound to "
                f"({previous.active_set_hash}, {previous.leader_index})"
            )
        claim = BatchClaim(
            batch_id=batch_id,
            active_set_hash=active_set_sha256(active_factors),
            leader_index=int(leader_index),
        )
        self._claims[batch_id] = claim
        return claim


@dataclass(frozen=True)
class CertificationRoundResult:
    """All finite-grid components for one fresh certification batch."""

    batch_id: str
    active_set_hash: str
    leader_index: int
    worst_index: int
    accepted_samples: int
    active_acquisition: FloatArray
    estimated_gaps: FloatArray
    lipschitz_constants: FloatArray
    inference_bounds: FloatArray
    structural_bounds: FloatArray
    optimistic_bounds: FloatArray
    u_cert: float

    def challenger_rows(self, actions: ArrayLike) -> list[dict[str, Any]]:
        action_values = np.asarray(actions, dtype=float)
        if action_values.shape != self.estimated_gaps.shape:
            raise ValueError("actions must match the certification grid")
        rows: list[dict[str, Any]] = []
        for index, action in enumerate(action_values):
            rows.append(
                {
                    "action_index": index,
                    "action": float(action),
                    "is_leader": index == self.leader_index,
                    "is_maximizer": index == self.worst_index,
                    "active_acquisition": float(self.active_acquisition[index]),
                    "estimated_gap": float(self.estimated_gaps[index]),
                    "lambda_q": float(self.lipschitz_constants[index]),
                    "b_infer": float(self.inference_bounds[index]),
                    "b_struct": float(self.structural_bounds[index]),
                    "psi": float(self.optimistic_bounds[index]),
                }
            )
        return rows


def _validated_factor_parameters(
    n_factors: int,
    gamma: ArrayLike,
    tau: ArrayLike,
) -> tuple[FloatArray, FloatArray]:
    gamma_array = np.broadcast_to(np.asarray(gamma, dtype=float), (n_factors,))
    tau_array = np.broadcast_to(np.asarray(tau, dtype=float), (n_factors,))
    if np.any(gamma_array < 0.0) or not np.all(np.isfinite(gamma_array)):
        raise ValueError("gamma must be finite and nonnegative")
    if np.any(tau_array <= 0.0) or not np.all(np.isfinite(tau_array)):
        raise ValueError("tau must be finite and positive")
    return gamma_array, tau_array


def _validated_active_factors(
    active_factors: Sequence[int],
    n_factors: int,
) -> IntArray:
    active = np.asarray(tuple(active_factors), dtype=np.int64)
    if active.size and (active.min() < 0 or active.max() >= n_factors):
        raise ValueError("active factor index out of range")
    if np.unique(active).size != active.size:
        raise ValueError("active_factors must not contain duplicates")
    return active


def symmetry_active_energy_batch(
    samples: ArrayLike,
    blocks: ArrayLike,
    active_factors: Sequence[int],
    gamma: ArrayLike,
    tau: ArrayLike,
) -> FloatArray:
    """Evaluate the active symmetry energy stably for a sample matrix."""

    values = np.asarray(samples, dtype=float)
    block_array = np.asarray(blocks, dtype=np.int64)
    if values.ndim != 2:
        raise ValueError("samples must have shape (n_samples, latent_dimension)")
    if block_array.ndim != 2 or block_array.shape[1] != 2:
        raise ValueError("blocks must have shape (n_factors, 2)")
    if block_array.size and (
        block_array.min() < 0 or block_array.max() >= values.shape[1]
    ):
        raise ValueError("block coordinate out of range")
    active = _validated_active_factors(active_factors, block_array.shape[0])
    gamma_array, tau_array = _validated_factor_parameters(
        block_array.shape[0], gamma, tau
    )
    if active.size == 0:
        return np.zeros(values.shape[0], dtype=float)

    selected_blocks = block_array[active]
    differences = (
        values[:, selected_blocks[:, 1]] - values[:, selected_blocks[:, 0]]
    )
    scaled = differences / tau_array[active]
    if not np.all(np.isfinite(scaled)):
        raise ValueError("scaled symmetry differences must be finite")
    log_cosh = np.logaddexp(scaled, -scaled) - np.log(2.0)
    log_cosh = np.maximum(log_cosh, 0.0)
    return np.asarray(log_cosh @ gamma_array[active], dtype=float)


def draw_ou_reference_ar1(
    rng: np.random.Generator,
    n_samples: int,
    mean: ArrayLike,
    correlation: float,
) -> FloatArray:
    """Draw exact no-jitter samples from an equally spaced OU reference."""

    mean_array = np.asarray(mean, dtype=float)
    if mean_array.ndim != 1 or not np.all(np.isfinite(mean_array)):
        raise ValueError("mean must be a finite vector")
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    if not 0.0 < correlation < 1.0:
        raise ValueError("correlation must lie strictly between zero and one")

    deviations = rng.standard_normal((n_samples, mean_array.size))
    innovation_scale = float(np.sqrt(1.0 - correlation * correlation))
    for column in range(1, mean_array.size):
        deviations[:, column] = (
            correlation * deviations[:, column - 1]
            + innovation_scale * deviations[:, column]
        )
    deviations += mean_array
    return np.asarray(deviations, dtype=float)


def rejection_sample_symmetry_target(
    *,
    rng: np.random.Generator,
    reference_draw: ReferenceDraw,
    n_accepted: int,
    proposal_chunk_size: int,
    proposal_cap: int,
    blocks: ArrayLike,
    active_factors: Sequence[int],
    gamma: ArrayLike,
    tau: ArrayLike,
) -> RejectionSampleResult:
    """Draw exact i.i.d. active-target samples by reference rejection.

    Whole generated chunks are included in computational proposal accounting.
    ``proposals_consumed`` counts the proposal prefix through the final retained
    acceptance, while ``accepted_candidates`` includes all acceptances evaluated
    in every generated chunk.
    """

    if n_accepted < 1:
        raise ValueError("n_accepted must be positive")
    if proposal_chunk_size < 1 or proposal_cap < 1:
        raise ValueError("proposal sizes and cap must be positive")

    retained: FloatArray | None = None
    retained_count = 0
    proposals_generated = 0
    proposals_consumed = 0
    accepted_candidates = 0
    chunk_rows: list[SamplerChunkDiagnostic] = []

    while retained_count < n_accepted and proposals_generated < proposal_cap:
        chunk_size = min(
            proposal_chunk_size,
            proposal_cap - proposals_generated,
        )
        proposals = np.asarray(reference_draw(rng, chunk_size), dtype=float)
        if proposals.ndim != 2 or proposals.shape[0] != chunk_size:
            raise ValueError(
                "reference_draw must return shape (requested, latent_dimension)"
            )
        if retained is None:
            retained = np.empty((n_accepted, proposals.shape[1]), dtype=float)
        elif proposals.shape[1] != retained.shape[1]:
            raise ValueError("reference_draw changed latent dimension")

        energies = symmetry_active_energy_batch(
            proposals, blocks, active_factors, gamma, tau
        )
        log_uniforms = np.log(rng.random(chunk_size))
        accepted_indices = np.flatnonzero(log_uniforms <= -energies)
        accepted_in_chunk = int(accepted_indices.size)
        accepted_candidates += accepted_in_chunk
        need = n_accepted - retained_count
        retained_indices = accepted_indices[:need]
        retained_in_chunk = int(retained_indices.size)
        if retained_in_chunk:
            retained[retained_count : retained_count + retained_in_chunk] = proposals[
                retained_indices
            ]
            retained_count += retained_in_chunk

        consumed_in_chunk = chunk_size
        if retained_count == n_accepted:
            consumed_in_chunk = int(retained_indices[-1]) + 1
        proposals_generated += chunk_size
        proposals_consumed += consumed_in_chunk
        chunk_rows.append(
            SamplerChunkDiagnostic(
                chunk_index=len(chunk_rows),
                proposals_generated=chunk_size,
                proposals_consumed=consumed_in_chunk,
                accepted_candidates=accepted_in_chunk,
                accepted_retained=retained_in_chunk,
                cumulative_proposals_generated=proposals_generated,
                cumulative_proposals_consumed=proposals_consumed,
                cumulative_accepted_retained=retained_count,
            )
        )

    if retained_count != n_accepted or retained is None:
        raise ProposalCapExceeded(
            requested_samples=n_accepted,
            accepted_samples=retained_count,
            proposals_generated=proposals_generated,
            proposals_consumed=proposals_consumed,
            accepted_candidates=accepted_candidates,
            chunks=tuple(chunk_rows),
        )

    return RejectionSampleResult(
        samples=retained,
        proposals_generated=proposals_generated,
        proposals_consumed=proposals_consumed,
        accepted_candidates=accepted_candidates,
        acceptance_rate=accepted_candidates / proposals_generated,
        chunks=tuple(chunk_rows),
    )


def q_inverse_norm(
    precision: ArrayLike | sparse.spmatrix,
    vector: ArrayLike,
) -> float:
    """Return ``sqrt(v.T Q^{-1} v)`` using a linear solve."""

    values = np.asarray(vector, dtype=float)
    if values.ndim != 1:
        raise ValueError("vector must be one-dimensional")
    if sparse.issparse(precision):
        solution = sparse_linalg.spsolve(sparse.csc_matrix(precision), values)
    else:
        matrix = np.asarray(precision, dtype=float)
        solution = np.linalg.solve(matrix, values)
    quadratic = float(values @ np.asarray(solution, dtype=float))
    tolerance = 1e-12 * max(1.0, float(values @ values))
    if quadratic < -tolerance:
        raise ValueError("precision solve produced a negative quadratic form")
    return float(np.sqrt(max(0.0, quadratic)))


def ei_gap_q_lipschitz(
    precision: ArrayLike | sparse.spmatrix,
    action_coefficients: ArrayLike,
    leader_coefficients: ArrayLike,
    *,
    self_comparison: bool = False,
) -> float:
    """Return the locked whitened EI-gap Lipschitz constant."""

    action = np.asarray(action_coefficients, dtype=float)
    leader = np.asarray(leader_coefficients, dtype=float)
    if action.ndim != 1 or action.shape != leader.shape:
        raise ValueError("action and leader coefficients must be matching vectors")
    if self_comparison:
        return 0.0
    return max(
        q_inverse_norm(precision, action),
        q_inverse_norm(precision, leader),
        q_inverse_norm(precision, action - leader),
    )


def inference_confidence_radius(
    lipschitz_constant: ArrayLike,
    n_samples: int,
    action_count: int,
    r_max: int,
    delta: float,
) -> FloatArray:
    """Return the exact two-sided simultaneous finite-sample radius."""

    lipschitz = np.asarray(lipschitz_constant, dtype=float)
    if np.any(lipschitz < 0.0) or not np.all(np.isfinite(lipschitz)):
        raise ValueError("lipschitz_constant must be finite and nonnegative")
    if n_samples < 1 or action_count < 1 or r_max < 1:
        raise ValueError("sample, action, and round counts must be positive")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie strictly between zero and one")
    log_constant = np.log(2.0 * action_count * r_max / delta)
    return np.asarray(
        lipschitz * np.sqrt(2.0 * log_constant / n_samples), dtype=float
    )


def rao_blackwellized_ei_curve(
    samples: ArrayLike,
    latent_mean: ArrayLike,
    action_means: ArrayLike,
    action_coefficients: ArrayLike,
    conditional_variances: ArrayLike,
    incumbent: float,
    *,
    sample_chunk_size: int = 25_000,
) -> FloatArray:
    """Compute a conditional-EI sample mean without materializing all rows."""

    values = np.asarray(samples, dtype=float)
    latent_mean_array = np.asarray(latent_mean, dtype=float)
    action_mean_array = np.asarray(action_means, dtype=float)
    coefficients = np.asarray(action_coefficients, dtype=float)
    variances = np.asarray(conditional_variances, dtype=float)
    if values.ndim != 2:
        raise ValueError("samples must be a matrix")
    if latent_mean_array.shape != (values.shape[1],):
        raise ValueError("latent_mean has the wrong shape")
    if coefficients.ndim != 2 or coefficients.shape[1] != values.shape[1]:
        raise ValueError("action_coefficients have the wrong shape")
    if action_mean_array.shape != (coefficients.shape[0],):
        raise ValueError("action_means have the wrong shape")
    if variances.shape != action_mean_array.shape or np.any(variances < 0.0):
        raise ValueError("conditional_variances have the wrong shape")
    if sample_chunk_size < 1:
        raise ValueError("sample_chunk_size must be positive")

    coefficient_matrix = sparse.csr_matrix(coefficients)
    totals = np.zeros(coefficients.shape[0], dtype=float)
    sigma = np.sqrt(variances)
    positive_variance = variances > 0.0
    for start in range(0, values.shape[0], sample_chunk_size):
        stop = min(values.shape[0], start + sample_chunk_size)
        centered = values[start:stop] - latent_mean_array
        conditional_means = np.asarray(
            coefficient_matrix.dot(centered.T).T, dtype=float
        )
        conditional_means += action_mean_array
        improvements = np.empty_like(conditional_means)
        if np.any(positive_variance):
            centered_improvement = (
                conditional_means[:, positive_variance] - incumbent
            )
            z = centered_improvement / sigma[positive_variance]
            density = np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)
            improvements[:, positive_variance] = (
                centered_improvement * ndtr(z)
                + sigma[positive_variance] * density
            )
        if np.any(~positive_variance):
            improvements[:, ~positive_variance] = np.maximum(
                conditional_means[:, ~positive_variance] - incumbent,
                0.0,
            )
        totals += improvements.sum(axis=0)
    return totals / values.shape[0]


def certify_symmetry_grid_round(
    *,
    samples: ArrayLike,
    batch_id: str,
    registry: CertificationBatchRegistry,
    active_factors: Sequence[int],
    actions: ArrayLike,
    leader_index: int,
    latent_mean: ArrayLike,
    action_means: ArrayLike,
    action_coefficients: ArrayLike,
    conditional_variances: ArrayLike,
    incumbent: float,
    precision_q: ArrayLike | sparse.spmatrix,
    reflection_blocks: ArrayLike,
    comparison_matrix_a: ArrayLike | sparse.spmatrix,
    omitted_load: ArrayLike,
    r_max: int,
    delta: float,
    sample_chunk_size: int = 25_000,
) -> CertificationRoundResult:
    """Construct one exhaustive finite-grid T3 certificate from a fresh batch."""

    sample_values = np.asarray(samples, dtype=float)
    action_values = np.asarray(actions, dtype=float)
    coefficients = np.asarray(action_coefficients, dtype=float)
    block_array = np.asarray(reflection_blocks, dtype=np.int64)
    if sample_values.ndim != 2 or sample_values.shape[0] < 1:
        raise ValueError("samples must be a nonempty matrix")
    if action_values.ndim != 1 or coefficients.shape[0] != action_values.size:
        raise ValueError("actions and action coefficients must agree")
    if not 0 <= leader_index < action_values.size:
        raise ValueError("leader_index is out of range")

    claim = registry.claim(batch_id, active_factors, leader_index)
    acquisition = rao_blackwellized_ei_curve(
        sample_values,
        latent_mean,
        action_means,
        coefficients,
        conditional_variances,
        incumbent,
        sample_chunk_size=sample_chunk_size,
    )
    gaps = acquisition - acquisition[leader_index]

    leader_coefficients = coefficients[leader_index]
    lipschitz = np.empty(action_values.size, dtype=float)
    structural = np.empty(action_values.size, dtype=float)
    transported_load = solve_comparison(comparison_matrix_a, omitted_load)
    for action_index, action_coefficient in enumerate(coefficients):
        self_comparison = action_index == leader_index
        lipschitz[action_index] = ei_gap_q_lipschitz(
            precision_q,
            action_coefficient,
            leader_coefficients,
            self_comparison=self_comparison,
        )
        if self_comparison:
            structural[action_index] = 0.0
        else:
            footprint = ei_block_decision_footprint(
                action_coefficient,
                leader_coefficients,
                block_array,
            )
            structural[action_index] = float(footprint @ transported_load)

    inference = inference_confidence_radius(
        lipschitz,
        sample_values.shape[0],
        action_values.size,
        r_max,
        delta,
    )
    gaps[leader_index] = 0.0
    inference[leader_index] = 0.0
    structural[leader_index] = 0.0
    optimistic = gaps + inference + structural
    optimistic[leader_index] = 0.0
    worst_index = int(np.argmax(optimistic))
    return CertificationRoundResult(
        batch_id=batch_id,
        active_set_hash=claim.active_set_hash,
        leader_index=int(leader_index),
        worst_index=worst_index,
        accepted_samples=sample_values.shape[0],
        active_acquisition=acquisition,
        estimated_gaps=gaps,
        lipschitz_constants=lipschitz,
        inference_bounds=inference,
        structural_bounds=structural,
        optimistic_bounds=optimistic,
        u_cert=float(optimistic[worst_index]),
    )


def active_set_sha256(active_factors: Sequence[int]) -> str:
    """Hash the canonical sorted active factor set."""

    canonical = sorted(int(index) for index in active_factors)
    if len(canonical) != len(set(canonical)):
        raise ValueError("active_factors must not contain duplicates")
    payload = json.dumps(canonical, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def make_batch_id(
    *,
    config_sha256: str,
    round_index: int,
    child_seed_state: dict[str, Any],
    active_factors: Sequence[int],
    leader_index: int,
) -> str:
    """Create a deterministic auditable ID for a fresh certification batch."""

    payload = {
        "config_sha256": config_sha256,
        "round_index": int(round_index),
        "child_seed_state": child_seed_state,
        "active_set_hash": active_set_sha256(active_factors),
        "leader_index": int(leader_index),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return f"cert-r{round_index:02d}-{digest[:20]}"


__all__ = [
    "BatchClaim",
    "CertificationBatchRegistry",
    "CertificationRoundResult",
    "ProposalCapExceeded",
    "RejectionSampleResult",
    "SamplerChunkDiagnostic",
    "active_set_sha256",
    "certify_symmetry_grid_round",
    "draw_ou_reference_ar1",
    "ei_gap_q_lipschitz",
    "inference_confidence_radius",
    "make_batch_id",
    "q_inverse_norm",
    "rao_blackwellized_ei_curve",
    "rejection_sample_symmetry_target",
    "symmetry_active_energy_batch",
]
