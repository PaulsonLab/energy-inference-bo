"""Reusable structural influence bounds for matched preference factors.

The module contains only factor metadata and Menz-comparison calculations.
Conditioned inference lives in :mod:`conditioned_bo.preference_bo`; omitted
preference likelihoods are never evaluated by the screening functions here.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import expit


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class PreferenceFactorMetadata:
    """Support and block membership for one pairwise preference factor."""

    factor_index: int
    support: tuple[int, int]
    block_index: int


@dataclass(frozen=True)
class ComparisonDiagnostics:
    """Basic SPD diagnostics for a dense, symmetric comparison matrix."""

    minimum_eigenvalue: float
    maximum_eigenvalue: float
    condition_number: float
    is_spd: bool


def _validated_pair(endpoint_pair: Sequence[int], dimension: int) -> tuple[int, int]:
    pair = tuple(int(index) for index in endpoint_pair)
    if len(pair) != 2 or pair[0] == pair[1]:
        raise ValueError("endpoint_pair must contain two distinct indices")
    if min(pair) < 0 or max(pair) >= dimension:
        raise ValueError("preference endpoint is outside the latent vector")
    return pair


def _validated_sign(sign: int) -> int:
    value = int(sign)
    if value not in (-1, 1):
        raise ValueError("preference sign must be -1 or +1")
    return value


def logistic_preference_energy(
    latent: ArrayLike,
    endpoint_pair: Sequence[int],
    sign: int,
    temperature: float,
) -> float:
    """Evaluate ``log(1 + exp(-z))`` stably with ``numpy.logaddexp``."""

    values = np.asarray(latent, dtype=float)
    if values.ndim != 1:
        raise ValueError("latent must be one-dimensional")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    left, right = _validated_pair(endpoint_pair, values.size)
    signed_margin = _validated_sign(sign) * (values[left] - values[right])
    return float(np.logaddexp(0.0, -signed_margin / temperature))


def logistic_preference_gradient(
    latent: ArrayLike,
    endpoint_pair: Sequence[int],
    sign: int,
    temperature: float,
) -> FloatArray:
    """Return the exact full-coordinate logistic preference gradient."""

    values = np.asarray(latent, dtype=float)
    if values.ndim != 1:
        raise ValueError("latent must be one-dimensional")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    left, right = _validated_pair(endpoint_pair, values.size)
    preference_sign = _validated_sign(sign)
    z = preference_sign * (values[left] - values[right]) / temperature
    q = float(expit(-z))
    coefficient = -preference_sign * q / temperature
    gradient = np.zeros(values.size, dtype=float)
    gradient[left] = coefficient
    gradient[right] = -coefficient
    return gradient


def logistic_preference_hessian(
    latent: ArrayLike,
    endpoint_pair: Sequence[int],
    sign: int,
    temperature: float,
) -> FloatArray:
    """Return the exact positive-semidefinite full-coordinate Hessian."""

    values = np.asarray(latent, dtype=float)
    if values.ndim != 1:
        raise ValueError("latent must be one-dimensional")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    left, right = _validated_pair(endpoint_pair, values.size)
    preference_sign = _validated_sign(sign)
    z = preference_sign * (values[left] - values[right]) / temperature
    q = float(expit(-z))
    curvature = q * (1.0 - q) / temperature**2
    direction = np.zeros(values.size, dtype=float)
    direction[left] = 1.0
    direction[right] = -1.0
    return curvature * np.outer(direction, direction)


def preference_blocks(grid_size: int = 17) -> tuple[IntArray, ...]:
    """Return the central singleton and symmetric two-coordinate blocks."""

    if grid_size < 3 or grid_size % 2 != 1:
        raise ValueError("grid_size must be an odd integer of at least three")
    center = grid_size // 2
    return (np.asarray([center], dtype=np.int64),) + tuple(
        np.asarray([center - radius, center + radius], dtype=np.int64)
        for radius in range(1, center + 1)
    )


def factor_block_metadata(
    endpoint_pairs: ArrayLike,
    blocks: Sequence[ArrayLike],
) -> tuple[PreferenceFactorMetadata, ...]:
    """Assign every preference support to its unique containing block."""

    pairs = np.asarray(endpoint_pairs, dtype=np.int64)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("endpoint_pairs must have shape (n_factors, 2)")
    block_arrays = tuple(np.asarray(block, dtype=np.int64) for block in blocks)
    metadata: list[PreferenceFactorMetadata] = []
    for factor_index, pair in enumerate(pairs):
        support = (int(pair[0]), int(pair[1]))
        containing = [
            block_index
            for block_index, block in enumerate(block_arrays)
            if set(support).issubset(set(int(index) for index in block))
        ]
        if len(containing) != 1:
            raise ValueError(
                "every preference factor must lie in exactly one proposed block"
            )
        metadata.append(
            PreferenceFactorMetadata(
                factor_index=factor_index,
                support=support,
                block_index=containing[0],
            )
        )
    return tuple(metadata)


def preference_block_sensitivity(temperature: float) -> float:
    """Return the exact block gradient bound ``sqrt(2) / temperature``."""

    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    return float(np.sqrt(2.0) / temperature)


def factor_sensitivity_matrix(
    metadata: Sequence[PreferenceFactorMetadata],
    n_blocks: int,
    temperature: float,
) -> FloatArray:
    """Construct rows ``L(e_j)`` from factor/block metadata."""

    if n_blocks < 1:
        raise ValueError("n_blocks must be positive")
    sensitivity = preference_block_sensitivity(temperature)
    matrix = np.zeros((len(metadata), n_blocks), dtype=float)
    for item in metadata:
        if not 0 <= item.block_index < n_blocks:
            raise ValueError("factor block index is out of range")
        matrix[item.factor_index, item.block_index] = sensitivity
    return matrix


def comparison_matrix_from_precision(
    precision: ArrayLike,
    blocks: Sequence[ArrayLike],
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Build the dense Menz comparison matrix from scalar-GP precision."""

    matrix_q = np.asarray(precision, dtype=float)
    if matrix_q.ndim != 2 or matrix_q.shape[0] != matrix_q.shape[1]:
        raise ValueError("precision must be square")
    if not np.allclose(matrix_q, matrix_q.T, rtol=1e-12, atol=1e-12):
        raise ValueError("precision must be symmetric")
    block_arrays = tuple(np.asarray(block, dtype=np.int64) for block in blocks)
    n_blocks = len(block_arrays)
    rho = np.empty(n_blocks, dtype=float)
    kappa = np.zeros((n_blocks, n_blocks), dtype=float)
    for block_index, block in enumerate(block_arrays):
        if block.ndim != 1 or block.size < 1:
            raise ValueError("each block must be a nonempty index vector")
        if block.min() < 0 or block.max() >= matrix_q.shape[0]:
            raise ValueError("block coordinate is outside the precision")
        within = matrix_q[np.ix_(block, block)]
        rho[block_index] = float(np.linalg.eigvalsh(within).min())
        for other_index in range(block_index):
            other = block_arrays[other_index]
            cross = matrix_q[np.ix_(block, other)]
            coupling = float(np.linalg.norm(cross, ord=2))
            kappa[block_index, other_index] = coupling
            kappa[other_index, block_index] = coupling
    if np.any(rho <= 0.0):
        raise ValueError("within-block precision curvature must be positive")
    comparison = np.diag(rho) - kappa
    return comparison, rho, kappa


def comparison_diagnostics(matrix: ArrayLike) -> ComparisonDiagnostics:
    """Return eigenvalue and condition diagnostics for the comparison matrix."""

    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("matrix must be square")
    if not np.allclose(values, values.T, rtol=1e-12, atol=1e-12):
        raise ValueError("comparison matrix must be symmetric")
    eigenvalues = np.linalg.eigvalsh(values)
    minimum = float(eigenvalues[0])
    maximum = float(eigenvalues[-1])
    return ComparisonDiagnostics(
        minimum_eigenvalue=minimum,
        maximum_eigenvalue=maximum,
        condition_number=float(maximum / minimum) if minimum > 0.0 else np.inf,
        is_spd=minimum > 0.0,
    )


def _action_block_index(action_index: int, blocks: Sequence[ArrayLike]) -> int:
    containing = [
        block_index
        for block_index, block in enumerate(blocks)
        if int(action_index) in set(int(index) for index in np.asarray(block))
    ]
    if len(containing) != 1:
        raise ValueError("action must lie in exactly one block")
    return containing[0]


def ei_decision_footprint(
    action_index: int,
    leader_index: int,
    blocks: Sequence[ArrayLike],
) -> FloatArray:
    """Return the frozen block footprint for a finite-grid EI gap."""

    footprint = np.zeros(len(blocks), dtype=float)
    if int(action_index) == int(leader_index):
        return footprint
    action_block = _action_block_index(action_index, blocks)
    leader_block = _action_block_index(leader_index, blocks)
    if action_block == leader_block:
        footprint[action_block] = np.sqrt(2.0)
    else:
        footprint[action_block] = 1.0
        footprint[leader_block] = 1.0
    return footprint


def omitted_factor_load(
    factor_sensitivities: ArrayLike,
    omitted_mask: ArrayLike,
) -> FloatArray:
    """Aggregate ``h_U`` from preference metadata only."""

    sensitivities = np.asarray(factor_sensitivities, dtype=float)
    omitted = np.asarray(omitted_mask, dtype=bool)
    if sensitivities.ndim != 2 or omitted.shape != (sensitivities.shape[0],):
        raise ValueError("factor sensitivities and omitted mask do not match")
    if not np.any(omitted):
        return np.zeros(sensitivities.shape[1], dtype=float)
    return np.asarray(sensitivities[omitted].sum(axis=0), dtype=float)


def solve_comparison(matrix: ArrayLike, right_hand_side: ArrayLike) -> FloatArray:
    """Solve a comparison system without forming an explicit inverse."""

    values = np.asarray(matrix, dtype=float)
    rhs = np.asarray(right_hand_side, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("matrix must be square")
    if rhs.shape[0] != values.shape[0]:
        raise ValueError("right_hand_side has the wrong leading dimension")
    return np.asarray(np.linalg.solve(values, rhs), dtype=float)


def structural_bound(
    matrix: ArrayLike,
    decision_footprint: ArrayLike,
    omitted_load: ArrayLike,
) -> float:
    """Evaluate ``L(F).T A^{-1} h_U`` through a linear solve."""

    footprint = np.asarray(decision_footprint, dtype=float)
    load = np.asarray(omitted_load, dtype=float)
    if footprint.shape != load.shape:
        raise ValueError("decision footprint and omitted load must match")
    if not np.any(load):
        return 0.0
    return float(footprint @ solve_comparison(matrix, load))


def ranked_omitted_contributions(
    matrix: ArrayLike,
    decision_footprint: ArrayLike,
    factor_sensitivities: ArrayLike,
    omitted_mask: ArrayLike,
) -> FloatArray:
    """Return factor contributions for the current worst challenger."""

    footprint = np.asarray(decision_footprint, dtype=float)
    sensitivities = np.asarray(factor_sensitivities, dtype=float)
    omitted = np.asarray(omitted_mask, dtype=bool)
    if sensitivities.ndim != 2 or omitted.shape != (sensitivities.shape[0],):
        raise ValueError("factor sensitivities and omitted mask do not match")
    if sensitivities.shape[1] != footprint.size:
        raise ValueError("factor and decision block dimensions do not match")
    transported = solve_comparison(matrix, footprint)
    contributions = np.asarray(sensitivities @ transported, dtype=float)
    return np.where(omitted, contributions, -np.inf)


__all__ = [
    "ComparisonDiagnostics",
    "PreferenceFactorMetadata",
    "comparison_diagnostics",
    "comparison_matrix_from_precision",
    "ei_decision_footprint",
    "factor_block_metadata",
    "factor_sensitivity_matrix",
    "logistic_preference_energy",
    "logistic_preference_gradient",
    "logistic_preference_hessian",
    "omitted_factor_load",
    "preference_block_sensitivity",
    "preference_blocks",
    "ranked_omitted_contributions",
    "solve_comparison",
    "structural_bound",
]
