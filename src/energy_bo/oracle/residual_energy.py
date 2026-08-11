"""A deliberately small, normalized scalar residual-energy model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from numpy.polynomial.hermite import hermgauss
from scipy.integrate import cumulative_trapezoid, quad

LOG_SQRT_2PI = float(0.5 * np.log(2.0 * np.pi))


@dataclass(frozen=True)
class ResidualFitSummary:
    nll: float
    penalty: float
    iterations: int


class RBFResidualEnergy:
    """Normalized ``N(0,1) exp(-R_phi)`` density with a fixed energy gauge.

    The RBF basis expectation under the N(0,1) reference is subtracted exactly.
    This removes the constant-energy direction from the parameterization; a scalar
    normalizer is still explicitly retained in every log-density evaluation.
    """

    def __init__(
        self,
        centers: torch.Tensor | None = None,
        bandwidth: float = 0.8,
        quadrature_points: int = 64,
        l2_precision: float = 10.0,
    ) -> None:
        if bandwidth <= 0.0:
            raise ValueError("bandwidth must be positive")
        if quadrature_points < 8:
            raise ValueError("at least eight quadrature nodes are required")
        self.centers = (
            torch.linspace(-3.0, 3.0, 9, dtype=torch.double)
            if centers is None
            else centers.detach().clone().to(dtype=torch.double)
        )
        self.bandwidth = float(bandwidth)
        self.quadrature_points = int(quadrature_points)
        self.l2_precision = float(l2_precision)
        if self.l2_precision < 0.0:
            raise ValueError("l2_precision must be non-negative")
        nodes, weights = hermgauss(self.quadrature_points)
        self._quadrature_nodes = torch.tensor(np.sqrt(2.0) * nodes, dtype=torch.double)
        self._quadrature_weights = torch.tensor(weights / np.sqrt(np.pi), dtype=torch.double)
        self.reference_basis_mean = self._reference_basis_expectation()
        self.phi = torch.zeros_like(self.centers, dtype=torch.double)

    @property
    def parameter_count(self) -> int:
        return int(self.centers.numel())

    def _reference_basis_expectation(self) -> torch.Tensor:
        h2 = self.bandwidth**2
        return (
            self.bandwidth
            / np.sqrt(1.0 + h2)
            * torch.exp(-torch.square(self.centers) / (2.0 * (1.0 + h2)))
        )

    def basis(self, z: torch.Tensor) -> torch.Tensor:
        z = z.to(dtype=torch.double)
        raw = torch.exp(
            -0.5 * torch.square((z.unsqueeze(-1) - self.centers) / self.bandwidth)
        )
        return raw - self.reference_basis_mean

    def energy(self, z: torch.Tensor, phi: torch.Tensor | None = None) -> torch.Tensor:
        coefficients = self.phi if phi is None else phi
        return self.basis(z) @ coefficients

    def log_normalizer(self, phi: torch.Tensor | None = None) -> torch.Tensor:
        coefficients = self.phi if phi is None else phi
        log_weights = torch.log(self._quadrature_weights)
        return torch.logsumexp(log_weights - self.energy(self._quadrature_nodes, coefficients), dim=0)

    def log_prob(self, z: torch.Tensor, phi: torch.Tensor | None = None) -> torch.Tensor:
        z = z.to(dtype=torch.double)
        return -0.5 * torch.square(z) - LOG_SQRT_2PI - self.energy(z, phi) - self.log_normalizer(phi)

    def density(self, z: torch.Tensor, phi: torch.Tensor | None = None) -> torch.Tensor:
        return torch.exp(self.log_prob(z, phi))

    def fit(self, samples: torch.Tensor, max_iter: int = 250) -> ResidualFitSummary:
        """Fit the zero-initialized model by MAP full-batch L-BFGS."""
        samples = samples.detach().to(dtype=torch.double).reshape(-1)
        if samples.numel() < 2:
            raise ValueError("at least two residual samples are required")
        phi = torch.nn.Parameter(torch.zeros_like(self.phi))
        optimizer = torch.optim.LBFGS(
            [phi], max_iter=max_iter, tolerance_grad=1e-10, tolerance_change=1e-12,
            line_search_fn="strong_wolfe"
        )
        calls = 0

        def closure() -> torch.Tensor:
            nonlocal calls
            optimizer.zero_grad()
            negative_log_likelihood = -torch.sum(self.log_prob(samples, phi))
            penalty = 0.5 * self.l2_precision * torch.sum(torch.square(phi))
            objective = negative_log_likelihood + penalty
            objective.backward()
            calls += 1
            return objective

        optimizer.step(closure)
        self.phi = phi.detach().clone()
        nll = float((-torch.sum(self.log_prob(samples))).item())
        penalty = float((0.5 * self.l2_precision * torch.sum(torch.square(self.phi))).item())
        return ResidualFitSummary(nll=nll, penalty=penalty, iterations=calls)

    def density_numpy(self, z: np.ndarray | float) -> np.ndarray:
        z_array = np.asarray(z, dtype=float)
        tensor = torch.as_tensor(z_array.reshape(-1), dtype=torch.double)
        values = self.density(tensor).detach().cpu().numpy().reshape(z_array.shape)
        return values

    def log_density_numpy(self, z: np.ndarray | float) -> np.ndarray:
        z_array = np.asarray(z, dtype=float)
        tensor = torch.as_tensor(z_array.reshape(-1), dtype=torch.double)
        values = self.log_prob(tensor).detach().cpu().numpy().reshape(z_array.shape)
        return values

    def moments(self) -> tuple[float, float]:
        mean, _ = quad(
            lambda z: z * float(self.density_numpy(z)),
            -10.0,
            10.0,
            epsabs=1e-10,
            epsrel=1e-10,
            limit=300,
        )
        second, _ = quad(
            lambda z: z * z * float(self.density_numpy(z)),
            -10.0,
            10.0,
            epsabs=1e-10,
            epsrel=1e-10,
            limit=300,
        )
        return float(mean), float(second - mean * mean)

    def tail_probability(self, threshold: float) -> float:
        probability, _ = quad(
            lambda z: float(self.density_numpy(z)),
            threshold,
            10.0,
            epsabs=1e-10,
            epsrel=1e-10,
            limit=300,
        )
        return float(probability)

    def standardized_ei(
        self, threshold: np.ndarray | float, integration_points: int = 16_001
    ) -> np.ndarray:
        """Evaluate E[(Z-t)+] from a dense independent integration grid.

        Gauss-Hermite quadrature remains the model normalizer. This separate grid
        avoids a moving ReLU kink on the 64 fixed Hermite nodes when reporting EI.
        """
        thresholds = np.asarray(threshold, dtype=float)
        grid = np.linspace(-10.0, 10.0, integration_points)
        density = self.density_numpy(grid)
        cumulative_mass = cumulative_trapezoid(density, grid, initial=0.0)
        cumulative_first = cumulative_trapezoid(grid * density, grid, initial=0.0)
        tail_mass = cumulative_mass[-1] - cumulative_mass
        tail_first = cumulative_first[-1] - cumulative_first
        clipped = np.clip(thresholds, grid[0], grid[-1])
        mass_at_t = np.interp(clipped, grid, tail_mass)
        first_at_t = np.interp(clipped, grid, tail_first)
        result = first_at_t - thresholds * mass_at_t
        return np.where(thresholds >= grid[-1], 0.0, np.maximum(result, 0.0))

    def expected_improvement(
        self, mean: np.ndarray, scale: np.ndarray, best_f: float
    ) -> np.ndarray:
        mean = np.asarray(mean, dtype=float)
        scale = np.asarray(scale, dtype=float)
        if np.any(scale <= 0.0):
            raise ValueError("scale must be positive")
        threshold = (best_f - mean) / scale
        return scale * self.standardized_ei(threshold)
