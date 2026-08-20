"""Reflection-symmetry influence bounds for an OU Gaussian reference model.

The functions in this module implement the concrete quantities used by the
reflection-symmetry specialization of the T2 omitted-factor bound.  They do
not perform conditioned inference and never require evaluating omitted
factors.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg
from scipy.special import ndtr


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def ou_ar1_precision(
    n_points: int,
    spacing: float,
    lengthscale: float,
) -> sparse.csr_matrix:
    """Return the exact precision of an equally spaced unit-variance OU process.

    The covariance is ``K[i, j] = exp(-abs(i-j) * spacing / lengthscale)``.
    The returned precision is the analytic tridiagonal AR(1) precision, so no
    dense covariance inversion or numerical thresholding is involved.
    """

    if n_points < 2:
        raise ValueError("n_points must be at least two")
    if spacing <= 0.0 or lengthscale <= 0.0:
        raise ValueError("spacing and lengthscale must be positive")

    q = float(np.exp(-spacing / lengthscale))
    denominator = 1.0 - q * q
    diagonal = np.full(n_points, (1.0 + q * q) / denominator)
    diagonal[[0, -1]] = 1.0 / denominator
    off_diagonal = np.full(n_points - 1, -q / denominator)
    return sparse.diags(
        (off_diagonal, diagonal, off_diagonal),
        offsets=(-1, 0, 1),
        format="csr",
    )


def reflection_blocks(n_factors: int) -> IntArray:
    """Return inner-to-outer index pairs for a left-to-right latent grid."""

    if n_factors < 1:
        raise ValueError("n_factors must be positive")
    radii = np.arange(n_factors, dtype=np.int64)
    return np.column_stack((n_factors - 1 - radii, n_factors + radii))


def _dense_block(
    matrix: ArrayLike | sparse.spmatrix,
    row_indices: IntArray,
    column_indices: IntArray,
) -> FloatArray:
    if sparse.issparse(matrix):
        return np.asarray(
            matrix[np.ix_(row_indices, column_indices)].toarray(), dtype=float
        )
    dense = np.asarray(matrix, dtype=float)
    return dense[np.ix_(row_indices, column_indices)]


def block_comparison_components(
    precision: ArrayLike | sparse.spmatrix,
    blocks: ArrayLike,
) -> tuple[FloatArray, FloatArray]:
    """Compute Menz one-block constants ``rho`` and cross couplings ``kappa``.

    ``rho[i]`` is the minimum eigenvalue of the within-block precision and
    ``kappa[i, j]`` is the operator norm of the corresponding cross block.
    """

    block_array = np.asarray(blocks, dtype=np.int64)
    if block_array.ndim != 2:
        raise ValueError("blocks must be a two-dimensional index array")

    n_blocks = block_array.shape[0]
    rho = np.empty(n_blocks, dtype=float)
    kappa = np.zeros((n_blocks, n_blocks), dtype=float)

    for i, block_i in enumerate(block_array):
        within = _dense_block(precision, block_i, block_i)
        rho[i] = float(np.linalg.eigvalsh(within).min())
        for j in range(i + 1, n_blocks):
            cross = _dense_block(precision, block_i, block_array[j])
            coupling = float(np.linalg.norm(cross, ord=2))
            kappa[i, j] = coupling
            kappa[j, i] = coupling

    return rho, kappa


def comparison_matrix(
    rho: ArrayLike,
    kappa: ArrayLike,
) -> sparse.csr_matrix:
    """Construct ``A`` with diagonal ``rho`` and off-diagonal ``-kappa``."""

    rho_array = np.asarray(rho, dtype=float)
    kappa_array = np.asarray(kappa, dtype=float)
    if rho_array.ndim != 1:
        raise ValueError("rho must be one-dimensional")
    if kappa_array.shape != (rho_array.size, rho_array.size):
        raise ValueError("kappa must be square with dimension len(rho)")
    if np.any(rho_array <= 0.0) or np.any(kappa_array < 0.0):
        raise ValueError("rho must be positive and kappa must be nonnegative")
    if not np.allclose(kappa_array, kappa_array.T):
        raise ValueError("kappa must be symmetric")
    if not np.allclose(np.diag(kappa_array), 0.0):
        raise ValueError("kappa must have a zero diagonal")

    matrix = sparse.diags(rho_array, format="csr") - sparse.csr_matrix(
        kappa_array
    )
    matrix.eliminate_zeros()
    return matrix


def ou_symmetry_comparison(
    n_factors: int,
    spacing: float,
    lengthscale: float,
) -> tuple[sparse.csr_matrix, IntArray, FloatArray, FloatArray, sparse.csr_matrix]:
    """Build the OU precision, reflection blocks, components, and comparison matrix."""

    precision = ou_ar1_precision(2 * n_factors, spacing, lengthscale)
    blocks = reflection_blocks(n_factors)
    rho, kappa = block_comparison_components(precision, blocks)
    matrix = comparison_matrix(rho, kappa)
    return precision, blocks, rho, kappa, matrix


def row_dominance_margins(matrix: ArrayLike | sparse.spmatrix) -> FloatArray:
    """Return ``A_ii - sum_{j != i} |A_ij|`` for every row."""

    dense = matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)
    dense = np.asarray(dense, dtype=float)
    return np.diag(dense) - (np.abs(dense).sum(axis=1) - np.abs(np.diag(dense)))


def solve_comparison(
    matrix: ArrayLike | sparse.spmatrix,
    right_hand_side: ArrayLike,
) -> FloatArray:
    """Solve a comparison-matrix system without forming its inverse."""

    rhs = np.asarray(right_hand_side, dtype=float)
    if sparse.issparse(matrix):
        solution = sparse_linalg.spsolve(sparse.csc_matrix(matrix), rhs)
    else:
        solution = np.linalg.solve(np.asarray(matrix, dtype=float), rhs)
    return np.asarray(solution, dtype=float)


def factor_sensitivity(gamma: ArrayLike, tau: ArrayLike) -> FloatArray:
    """Return the exact block gradient bound ``sqrt(2) * gamma / tau``."""

    gamma_array, tau_array = np.broadcast_arrays(
        np.asarray(gamma, dtype=float), np.asarray(tau, dtype=float)
    )
    if np.any(gamma_array < 0.0) or np.any(tau_array <= 0.0):
        raise ValueError("gamma must be nonnegative and tau must be positive")
    return np.asarray(np.sqrt(2.0) * gamma_array / tau_array, dtype=float)


def omitted_factor_load(
    active_mask: ArrayLike,
    gamma: ArrayLike,
    tau: ArrayLike,
) -> FloatArray:
    """Aggregate local factor sensitivities over omitted reflection factors."""

    active = np.asarray(active_mask, dtype=bool)
    sensitivity = np.broadcast_to(factor_sensitivity(gamma, tau), active.shape)
    return np.where(active, 0.0, sensitivity).astype(float)


def symmetry_logcosh_energy(
    block_value: ArrayLike,
    gamma: float,
    tau: float,
) -> float:
    """Evaluate ``gamma * log(cosh((y_right-y_left)/tau))`` stably."""

    block = np.asarray(block_value, dtype=float)
    if block.shape != (2,):
        raise ValueError("block_value must have shape (2,)")
    if gamma < 0.0 or tau <= 0.0:
        raise ValueError("gamma must be nonnegative and tau must be positive")
    z = float((block[1] - block[0]) / tau)
    return float(gamma * (np.logaddexp(z, -z) - np.log(2.0)))


def symmetry_logcosh_gradient(
    block_value: ArrayLike,
    gamma: float,
    tau: float,
) -> FloatArray:
    """Return the two-coordinate factor gradient."""

    block = np.asarray(block_value, dtype=float)
    if block.shape != (2,):
        raise ValueError("block_value must have shape (2,)")
    if gamma < 0.0 or tau <= 0.0:
        raise ValueError("gamma must be nonnegative and tau must be positive")
    direction = np.array([-1.0, 1.0])
    z = float(direction @ block / tau)
    return np.asarray((gamma / tau) * np.tanh(z) * direction, dtype=float)


def symmetry_logcosh_hessian(
    block_value: ArrayLike,
    gamma: float,
    tau: float,
) -> FloatArray:
    """Return the positive-semidefinite two-coordinate factor Hessian."""

    block = np.asarray(block_value, dtype=float)
    if block.shape != (2,):
        raise ValueError("block_value must have shape (2,)")
    if gamma < 0.0 or tau <= 0.0:
        raise ValueError("gamma must be nonnegative and tau must be positive")
    direction = np.array([-1.0, 1.0])
    z = float(direction @ block / tau)
    sech_squared = 1.0 - np.tanh(z) ** 2
    return np.asarray(
        (gamma / tau**2) * sech_squared * np.outer(direction, direction),
        dtype=float,
    )


def ou_covariance_vector(
    action: float,
    latent_points: ArrayLike,
    lengthscale: float,
) -> FloatArray:
    """Return OU covariances between one action and latent grid points."""

    if lengthscale <= 0.0:
        raise ValueError("lengthscale must be positive")
    points = np.asarray(latent_points, dtype=float)
    return np.asarray(np.exp(-np.abs(float(action) - points) / lengthscale))


def ei_action_coefficients(
    action: float,
    latent_points: ArrayLike,
    lengthscale: float,
    precision: ArrayLike | sparse.spmatrix,
) -> tuple[FloatArray, float]:
    """Return ``a_x=K^{-1}k_Yx`` and the OU conditional variance.

    The conditional variance is computed from the same covariance/precision
    pair as the coefficients.  A tiny negative roundoff residual is clipped to
    zero; a material inconsistency raises an error.
    """

    covariance = ou_covariance_vector(action, latent_points, lengthscale)
    coefficients = np.asarray(precision @ covariance, dtype=float).reshape(-1)
    variance = float(1.0 - covariance @ coefficients)
    if variance < -1e-10:
        raise ValueError("precision and covariance are inconsistent")
    return coefficients, max(0.0, variance)


def conditional_expected_improvement(
    conditional_mean: ArrayLike,
    conditional_variance: float,
    incumbent: float,
) -> FloatArray:
    """Evaluate conditional EI, including the zero-variance positive part."""

    mean = np.asarray(conditional_mean, dtype=float)
    if conditional_variance < 0.0:
        raise ValueError("conditional_variance must be nonnegative")
    if conditional_variance == 0.0:
        return np.maximum(mean - incumbent, 0.0)

    sigma = float(np.sqrt(conditional_variance))
    centered = mean - incumbent
    z = centered / sigma
    density = np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)
    return np.asarray(centered * ndtr(z) + sigma * density, dtype=float)


def conditional_ei_gradient(
    conditional_mean: ArrayLike,
    conditional_variance: float,
    incumbent: float,
    coefficients: ArrayLike,
) -> FloatArray:
    """Return the gradient of conditional EI with respect to the latent vector.

    At zero variance the weak derivative is used almost everywhere; at the kink
    itself the symmetric smooth-approximation limit ``1/2`` is returned.
    """

    mean = np.asarray(conditional_mean, dtype=float)
    coefficient_array = np.asarray(coefficients, dtype=float)
    if conditional_variance < 0.0:
        raise ValueError("conditional_variance must be nonnegative")
    if conditional_variance == 0.0:
        slope = np.where(mean > incumbent, 1.0, np.where(mean < incumbent, 0.0, 0.5))
    else:
        slope = ndtr((mean - incumbent) / np.sqrt(conditional_variance))
    return np.asarray(np.expand_dims(slope, axis=-1) * coefficient_array, dtype=float)


def _block_norms(vector: ArrayLike, blocks: ArrayLike) -> FloatArray:
    values = np.asarray(vector, dtype=float)
    block_array = np.asarray(blocks, dtype=np.int64)
    return np.asarray(np.linalg.norm(values[block_array], axis=1), dtype=float)


def ei_block_decision_footprint(
    action_coefficients: ArrayLike,
    leader_coefficients: ArrayLike,
    blocks: ArrayLike,
) -> FloatArray:
    """Return the conservative block sensitivity for one conditional-EI gap."""

    action = np.asarray(action_coefficients, dtype=float)
    leader = np.asarray(leader_coefficients, dtype=float)
    if action.shape != leader.shape:
        raise ValueError("action and leader coefficient vectors must match")
    return np.maximum.reduce(
        (
            _block_norms(action, blocks),
            _block_norms(leader, blocks),
            _block_norms(action - leader, blocks),
        )
    )


def log_exponential_block_footprint(
    action_coefficients: ArrayLike,
    leader_coefficients: ArrayLike,
    blocks: ArrayLike,
    beta: float,
) -> FloatArray:
    """Return the archived log-exponential-utility action-tilt footprint."""

    if beta < 0.0:
        raise ValueError("beta must be nonnegative")
    return beta * _block_norms(
        np.asarray(action_coefficients) - np.asarray(leader_coefficients), blocks
    )


def structural_bound(
    matrix: ArrayLike | sparse.spmatrix,
    decision_footprint: ArrayLike,
    omitted_load: ArrayLike,
) -> float:
    """Evaluate ``d.T @ A^{-1} @ h`` through a linear solve."""

    decision = np.asarray(decision_footprint, dtype=float)
    load = np.asarray(omitted_load, dtype=float)
    if decision.shape != load.shape:
        raise ValueError("decision_footprint and omitted_load must match")
    transported_load = solve_comparison(matrix, load)
    return float(decision @ transported_load)


def ranked_omitted_contributions(
    matrix: ArrayLike | sparse.spmatrix,
    decision_footprint: ArrayLike,
    active_mask: ArrayLike,
    gamma: ArrayLike,
    tau: ArrayLike,
) -> FloatArray:
    """Return per-factor contributions used to select the next omitted factor."""

    decision = np.asarray(decision_footprint, dtype=float)
    active = np.asarray(active_mask, dtype=bool)
    if decision.shape != active.shape:
        raise ValueError("decision_footprint and active_mask must match")
    transported_decision = solve_comparison(matrix, decision)
    local_load = np.broadcast_to(factor_sensitivity(gamma, tau), active.shape)
    scores = local_load * transported_decision
    return np.where(active, -np.inf, scores).astype(float)


def active_set_mask(n_factors: int, active: Sequence[int]) -> NDArray[np.bool_]:
    """Convert an index sequence into a validated Boolean active-set mask."""

    mask = np.zeros(n_factors, dtype=bool)
    indices = np.asarray(tuple(active), dtype=np.int64)
    if indices.size and (indices.min() < 0 or indices.max() >= n_factors):
        raise ValueError("active factor index out of range")
    mask[indices] = True
    return mask
