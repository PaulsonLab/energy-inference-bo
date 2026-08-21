"""Structural influence bounds for the nonlinear-PDE factor family.

This module implements the scalar-block Menz comparison matrix used by the
archived nonlinear reaction--diffusion prototype.  Every screening quantity is
computed from Gaussian-reference and factor metadata; no omitted residual or
factor-energy evaluation is required.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class NonlinearPDEParameters:
    """Parameters of the Gaussian reference and nonlinear residual factors."""

    q0: float = 3.5
    q_laplacian: float = 0.6
    coupling: float = 0.12
    nonlinearity: float = 0.25
    gamma: float = 0.08
    tau: float = 0.30

    def validate(self) -> None:
        if self.q0 <= 0.0 or self.q_laplacian < 0.0:
            raise ValueError("q0 must be positive and q_laplacian nonnegative")
        if self.coupling < 0.0 or self.nonlinearity < 0.0:
            raise ValueError("coupling and nonlinearity must be nonnegative")
        if self.gamma < 0.0 or self.tau <= 0.0:
            raise ValueError("gamma must be nonnegative and tau positive")

    @property
    def center_derivative_bound(self) -> float:
        return 1.0 + self.nonlinearity

    @property
    def outer_curvature_scale(self) -> float:
        return self.gamma / self.tau**2

    @property
    def gradient_scale(self) -> float:
        return self.gamma / self.tau

    @property
    def negative_curvature(self) -> float:
        return self.gamma * self.nonlinearity / self.tau


DEFAULT_PARAMETERS = NonlinearPDEParameters()


def _validate_grid_size(grid_size: int) -> None:
    if grid_size < 2:
        raise ValueError("grid_size must be at least two")


def grid_neighbors(grid_size: int, index: int) -> IntArray:
    """Return nonperiodic four-neighbors in the archived notebook order."""

    _validate_grid_size(grid_size)
    n_sites = grid_size * grid_size
    if index < 0 or index >= n_sites:
        raise ValueError("index is outside the grid")
    row, column = divmod(index, grid_size)
    neighbors: list[int] = []
    for neighbor_row, neighbor_column in (
        (row - 1, column),
        (row + 1, column),
        (row, column - 1),
        (row, column + 1),
    ):
        if 0 <= neighbor_row < grid_size and 0 <= neighbor_column < grid_size:
            neighbors.append(neighbor_row * grid_size + neighbor_column)
    return np.asarray(neighbors, dtype=np.int64)


def factor_supports(grid_size: int) -> tuple[IntArray, ...]:
    """Return each residual support with its center as the first coordinate."""

    _validate_grid_size(grid_size)
    return tuple(
        np.concatenate((np.asarray([center], dtype=np.int64), grid_neighbors(grid_size, center)))
        for center in range(grid_size * grid_size)
    )


def sparse_grid_precision(
    grid_size: int,
    q0: float,
    q_laplacian: float,
) -> sparse.csr_matrix:
    """Construct ``Q = q0 I + q_laplacian L`` on a nonperiodic square grid."""

    _validate_grid_size(grid_size)
    if q0 <= 0.0 or q_laplacian < 0.0:
        raise ValueError("q0 must be positive and q_laplacian nonnegative")

    one_dimensional_degree = np.full(grid_size, 2.0)
    one_dimensional_degree[[0, -1]] = 1.0
    path_laplacian = sparse.diags(
        (
            -np.ones(grid_size - 1),
            one_dimensional_degree,
            -np.ones(grid_size - 1),
        ),
        offsets=(-1, 0, 1),
        format="csr",
    )
    identity = sparse.eye(grid_size, format="csr")
    grid_laplacian = sparse.kron(identity, path_laplacian, format="csr") + sparse.kron(
        path_laplacian, identity, format="csr"
    )
    precision = q0 * sparse.eye(grid_size * grid_size, format="csr")
    precision = precision + q_laplacian * grid_laplacian
    precision.eliminate_zeros()
    return precision.tocsr()


def factor_derivative_bound_matrix(
    grid_size: int,
    coupling: float,
    nonlinearity: float,
) -> sparse.csr_matrix:
    """Return absolute residual-gradient bounds by factor and latent block."""

    _validate_grid_size(grid_size)
    if coupling < 0.0 or nonlinearity < 0.0:
        raise ValueError("coupling and nonlinearity must be nonnegative")
    supports = factor_supports(grid_size)
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    for factor_index, support in enumerate(supports):
        rows.extend([factor_index] * support.size)
        columns.extend(support.tolist())
        values.append(1.0 + nonlinearity)
        values.extend([coupling] * (support.size - 1))
    matrix = sparse.coo_matrix(
        (values, (rows, columns)),
        shape=(grid_size * grid_size, grid_size * grid_size),
    )
    return matrix.tocsr()


def block_poincare_constants(
    precision: sparse.spmatrix,
    gamma: float,
    nonlinearity: float,
    tau: float,
) -> FloatArray:
    """Return ``rho_i = Q_ii - gamma * nonlinearity / tau``."""

    if gamma < 0.0 or nonlinearity < 0.0 or tau <= 0.0:
        raise ValueError("gamma/nonlinearity must be nonnegative and tau positive")
    rho = np.asarray(precision.diagonal(), dtype=float) - gamma * nonlinearity / tau
    if np.any(rho <= 0.0):
        raise ValueError("the conservative block Poincare constants are not positive")
    return rho


def complete_cross_block_couplings(
    precision: sparse.spmatrix,
    derivative_bounds: sparse.spmatrix,
    gamma: float,
    tau: float,
) -> sparse.csr_matrix:
    """Compute every overlap-aware ``kappa_ik`` for distinct scalar blocks."""

    if gamma < 0.0 or tau <= 0.0:
        raise ValueError("gamma must be nonnegative and tau positive")
    if precision.shape != derivative_bounds.shape:
        raise ValueError("precision and derivative_bounds must have matching shapes")

    absolute_precision = sparse.csr_matrix(precision, dtype=float).copy()
    absolute_precision.data = np.abs(absolute_precision.data)
    absolute_precision.setdiag(0.0)
    absolute_precision.eliminate_zeros()

    overlap = derivative_bounds.T @ derivative_bounds
    overlap = sparse.csr_matrix((gamma / tau**2) * overlap, dtype=float)
    overlap.setdiag(0.0)
    overlap.eliminate_zeros()

    kappa = absolute_precision + overlap
    kappa.eliminate_zeros()
    return kappa.tocsr()


def comparison_matrix(rho: ArrayLike, kappa: sparse.spmatrix) -> sparse.csr_matrix:
    """Construct the sparse Menz matrix with diagonal ``rho`` and ``-kappa``."""

    rho_array = np.asarray(rho, dtype=float)
    if rho_array.ndim != 1 or kappa.shape != (rho_array.size, rho_array.size):
        raise ValueError("rho and kappa dimensions do not match")
    if np.any(rho_array <= 0.0) or np.any(kappa.data < 0.0):
        raise ValueError("rho must be positive and kappa nonnegative")
    if kappa.diagonal().any():
        raise ValueError("kappa must have a zero diagonal")
    matrix = sparse.diags(rho_array, format="csr") - sparse.csr_matrix(kappa)
    matrix.eliminate_zeros()
    return matrix.tocsr()


def build_nonlinear_pde_comparison(
    grid_size: int,
    parameters: NonlinearPDEParameters = DEFAULT_PARAMETERS,
) -> tuple[
    sparse.csr_matrix,
    sparse.csr_matrix,
    FloatArray,
    sparse.csr_matrix,
    sparse.csr_matrix,
]:
    """Build ``Q``, derivative bounds, ``rho``, ``kappa``, and ``A``."""

    parameters.validate()
    precision = sparse_grid_precision(
        grid_size, parameters.q0, parameters.q_laplacian
    )
    derivative_bounds = factor_derivative_bound_matrix(
        grid_size, parameters.coupling, parameters.nonlinearity
    )
    rho = block_poincare_constants(
        precision, parameters.gamma, parameters.nonlinearity, parameters.tau
    )
    kappa = complete_cross_block_couplings(
        precision, derivative_bounds, parameters.gamma, parameters.tau
    )
    matrix = comparison_matrix(rho, kappa)
    return precision, derivative_bounds, rho, kappa, matrix


def row_dominance_margins(matrix: sparse.spmatrix) -> FloatArray:
    """Return ``A_ii - sum_{k != i}|A_ik|`` for every row."""

    csr = sparse.csr_matrix(matrix, dtype=float)
    diagonal = np.asarray(csr.diagonal(), dtype=float)
    absolute = csr.copy()
    absolute.data = np.abs(absolute.data)
    row_sums = np.asarray(absolute.sum(axis=1)).ravel()
    return diagonal - (row_sums - np.abs(diagonal))


def diagonal_dominance_threshold(
    parameters: NonlinearPDEParameters = DEFAULT_PARAMETERS,
) -> float:
    """Return the sufficient lower threshold on ``q0`` from the proof."""

    parameters.validate()
    alpha = parameters.outer_curvature_scale
    return float(
        parameters.negative_curvature
        + 8.0
        * alpha
        * parameters.center_derivative_bound
        * parameters.coupling
        + 12.0 * alpha * parameters.coupling**2
    )


def analytic_interior_row_margin(
    parameters: NonlinearPDEParameters = DEFAULT_PARAMETERS,
) -> float:
    """Return the domain-independent interior diagonal-dominance margin."""

    return float(parameters.q0 - diagonal_dominance_threshold(parameters))


def factor_sensitivity_matrix(
    derivative_bounds: sparse.spmatrix,
    gamma: float,
    tau: float,
) -> sparse.csr_matrix:
    """Return block sensitivities ``L_i(e_j)`` for all residual factors."""

    if gamma < 0.0 or tau <= 0.0:
        raise ValueError("gamma must be nonnegative and tau positive")
    return sparse.csr_matrix((gamma / tau) * derivative_bounds, dtype=float)


def omitted_factor_load(
    derivative_bounds: sparse.spmatrix,
    omitted_mask: ArrayLike,
    gamma: float,
    tau: float,
) -> FloatArray:
    """Return ``h_U`` using factor metadata only."""

    omitted = np.asarray(omitted_mask, dtype=bool)
    if omitted.shape != (derivative_bounds.shape[0],):
        raise ValueError("omitted_mask has the wrong shape")
    sensitivities = factor_sensitivity_matrix(derivative_bounds, gamma, tau)
    if not np.any(omitted):
        return np.zeros(derivative_bounds.shape[1], dtype=float)
    return np.asarray(sensitivities[omitted].sum(axis=0)).ravel()


def ei_decision_footprint(
    n_sites: int,
    action_index: int,
    leader_index: int,
) -> FloatArray:
    """Return the two-coordinate EI-gap sensitivity for discretized actions."""

    if n_sites < 1:
        raise ValueError("n_sites must be positive")
    if not 0 <= action_index < n_sites or not 0 <= leader_index < n_sites:
        raise ValueError("action index is outside the latent field")
    footprint = np.zeros(n_sites, dtype=float)
    if action_index != leader_index:
        footprint[action_index] = 1.0
        footprint[leader_index] = 1.0
    return footprint


def solve_comparison(
    matrix: sparse.spmatrix,
    right_hand_side: ArrayLike,
) -> FloatArray:
    """Solve a sparse comparison system without materializing ``A^{-1}``."""

    rhs = np.asarray(right_hand_side, dtype=float)
    if rhs.shape[0] != matrix.shape[0]:
        raise ValueError("right_hand_side has the wrong leading dimension")
    solution = sparse_linalg.spsolve(sparse.csc_matrix(matrix), rhs)
    return np.asarray(solution, dtype=float)


def structural_screening_bound(
    matrix: sparse.spmatrix,
    derivative_bounds: sparse.spmatrix,
    omitted_mask: ArrayLike,
    action_index: int,
    leader_index: int,
    gamma: float,
    tau: float,
) -> float:
    """Evaluate the structural EI bound using metadata and a sparse solve."""

    load = omitted_factor_load(derivative_bounds, omitted_mask, gamma, tau)
    transported_load = solve_comparison(matrix, load)
    footprint = ei_decision_footprint(matrix.shape[0], action_index, leader_index)
    return float(footprint @ transported_load)


def ranked_omitted_contributions(
    matrix: sparse.spmatrix,
    derivative_bounds: sparse.spmatrix,
    omitted_mask: ArrayLike,
    action_index: int,
    leader_index: int,
    gamma: float,
    tau: float,
) -> FloatArray:
    """Return metadata-only factor contributions for one EI comparison."""

    omitted = np.asarray(omitted_mask, dtype=bool)
    if omitted.shape != (derivative_bounds.shape[0],):
        raise ValueError("omitted_mask has the wrong shape")
    footprint = ei_decision_footprint(matrix.shape[0], action_index, leader_index)
    transported_decision = solve_comparison(matrix, footprint)
    sensitivities = factor_sensitivity_matrix(derivative_bounds, gamma, tau)
    contributions = np.asarray(sensitivities @ transported_decision).ravel()
    return np.where(omitted, contributions, -np.inf)


def stable_logcosh(value: ArrayLike) -> FloatArray:
    """Evaluate ``log(cosh(value))`` without overflow."""

    values = np.asarray(value, dtype=float)
    return np.asarray(np.logaddexp(values, -values) - np.log(2.0), dtype=float)


def residual_value(
    local_values: ArrayLike,
    source_value: float,
    coupling: float,
    nonlinearity: float,
) -> float:
    """Evaluate one residual; the center is the first local coordinate."""

    values = np.asarray(local_values, dtype=float)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("local_values must contain a center and at least one neighbor")
    return float(
        values[0]
        - coupling * values[1:].sum()
        + nonlinearity * np.sin(values[0])
        - source_value
    )


def residual_gradient(
    local_values: ArrayLike,
    coupling: float,
    nonlinearity: float,
) -> FloatArray:
    """Return the exact local gradient of one nonlinear residual."""

    values = np.asarray(local_values, dtype=float)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("local_values must contain a center and at least one neighbor")
    gradient = np.full(values.size, -coupling, dtype=float)
    gradient[0] = 1.0 + nonlinearity * np.cos(values[0])
    return gradient


def residual_hessian(local_values: ArrayLike, nonlinearity: float) -> FloatArray:
    """Return the exact local Hessian of one nonlinear residual."""

    values = np.asarray(local_values, dtype=float)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("local_values must contain a center and at least one neighbor")
    hessian = np.zeros((values.size, values.size), dtype=float)
    hessian[0, 0] = -nonlinearity * np.sin(values[0])
    return hessian


def factor_energy(
    local_values: ArrayLike,
    source_value: float,
    parameters: NonlinearPDEParameters = DEFAULT_PARAMETERS,
) -> float:
    """Evaluate one nonlinear-PDE ``gamma logcosh(r/tau)`` factor."""

    parameters.validate()
    residual = residual_value(
        local_values,
        source_value,
        parameters.coupling,
        parameters.nonlinearity,
    )
    return float(parameters.gamma * stable_logcosh(residual / parameters.tau))


def factor_gradient(
    local_values: ArrayLike,
    source_value: float,
    parameters: NonlinearPDEParameters = DEFAULT_PARAMETERS,
) -> FloatArray:
    """Return the exact local factor gradient."""

    parameters.validate()
    residual = residual_value(
        local_values,
        source_value,
        parameters.coupling,
        parameters.nonlinearity,
    )
    gradient = residual_gradient(
        local_values, parameters.coupling, parameters.nonlinearity
    )
    return np.asarray(
        parameters.gradient_scale * np.tanh(residual / parameters.tau) * gradient,
        dtype=float,
    )


def factor_hessian_terms(
    local_values: ArrayLike,
    source_value: float,
    parameters: NonlinearPDEParameters = DEFAULT_PARAMETERS,
) -> tuple[FloatArray, FloatArray]:
    """Return the positive outer-product and nonlinear chain-rule terms."""

    parameters.validate()
    residual = residual_value(
        local_values,
        source_value,
        parameters.coupling,
        parameters.nonlinearity,
    )
    gradient = residual_gradient(
        local_values, parameters.coupling, parameters.nonlinearity
    )
    hessian = residual_hessian(local_values, parameters.nonlinearity)
    tanh_value = float(np.tanh(residual / parameters.tau))
    positive_term = (
        parameters.outer_curvature_scale
        * (1.0 - tanh_value**2)
        * np.outer(gradient, gradient)
    )
    nonlinear_term = parameters.gradient_scale * tanh_value * hessian
    return np.asarray(positive_term), np.asarray(nonlinear_term)


def factor_hessian(
    local_values: ArrayLike,
    source_value: float,
    parameters: NonlinearPDEParameters = DEFAULT_PARAMETERS,
) -> FloatArray:
    """Return the exact two-term local factor Hessian."""

    positive_term, nonlinear_term = factor_hessian_terms(
        local_values, source_value, parameters
    )
    return positive_term + nonlinear_term


def extreme_eigenvalues(matrix: sparse.spmatrix) -> tuple[float, float]:
    """Compute the smallest and largest eigenvalues of a symmetric sparse matrix."""

    csr = sparse.csr_matrix(matrix, dtype=float)
    if csr.shape[0] < 4:
        eigenvalues = np.linalg.eigvalsh(csr.toarray())
        return float(eigenvalues[0]), float(eigenvalues[-1])
    initial = np.ones(csr.shape[0], dtype=float)
    smallest = sparse_linalg.eigsh(
        csr, k=1, which="SA", v0=initial, tol=1e-13, return_eigenvectors=False
    )[0]
    largest = sparse_linalg.eigsh(
        csr, k=1, which="LA", v0=initial, tol=1e-13, return_eigenvectors=False
    )[0]
    return float(smallest), float(largest)


def maximum_nonzeros_per_row(matrix: sparse.spmatrix) -> int:
    """Return the largest CSR row width."""

    csr = sparse.csr_matrix(matrix)
    return int(np.diff(csr.indptr).max())
