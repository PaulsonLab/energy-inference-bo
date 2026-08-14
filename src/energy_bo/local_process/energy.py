"""Convex unary and pairwise corrections to local Gaussian conditionals."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import torch
from numpy.polynomial.hermite import hermgauss

CENTERS = torch.linspace(-3.0, 3.0, 7, dtype=torch.double)
BANDWIDTH = 0.8
LOG_SQRT_2PI = 0.5 * math.log(2 * math.pi)


def centered_rbf_basis(z: torch.Tensor, centers: torch.Tensor = CENTERS, bandwidth: float = BANDWIDTH) -> torch.Tensor:
    """Reference-centered RBFs evaluated in a standard-normal coordinate."""
    z = torch.as_tensor(z, dtype=torch.double)
    centers = centers.to(z.device)
    raw = torch.exp(-0.5 * ((z[..., None] - centers) / bandwidth).square())
    mean = bandwidth / math.sqrt(1 + bandwidth**2) * torch.exp(-centers.square() / (2 * (1 + bandwidth**2)))
    return raw - mean


def neighbor_summary(values: torch.Tensor, neighbors: torch.Tensor, mask: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    basis = centered_rbf_basis(values[neighbors])
    return (basis * weights[..., None] * mask[..., None]).sum(-2)


@dataclass(frozen=True)
class FitSummary:
    objective: float
    nll: float
    penalty: float
    iterations: int
    converged: bool
    seconds: float
    objective_gradient_seconds: float


class LocalEnergyModel:
    def __init__(self, pairwise: bool, *, l2_precision: float = 10.0, quadrature_points: int = 64) -> None:
        self.pairwise = bool(pairwise)
        self.l2_precision = float(l2_precision)
        nodes, weights = hermgauss(quadrature_points)
        self.nodes = torch.tensor(math.sqrt(2) * nodes, dtype=torch.double)
        self.weights = torch.tensor(weights / math.sqrt(math.pi), dtype=torch.double)
        self.unary = torch.zeros(7, dtype=torch.double)
        self.pair = torch.zeros((7, 7), dtype=torch.double) if pairwise else None

    @property
    def parameter_count(self) -> int:
        return 56 if self.pairwise else 7

    @property
    def is_identity(self) -> bool:
        return bool(torch.count_nonzero(self.unary) == 0 and (self.pair is None or torch.count_nonzero(self.pair) == 0))

    def to(self, device: torch.device | str) -> "LocalEnergyModel":
        self.nodes, self.weights, self.unary = self.nodes.to(device), self.weights.to(device), self.unary.to(device)
        if self.pair is not None: self.pair = self.pair.to(device)
        return self

    def _unpack(self, parameter: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor | None]:
        if parameter is None: return self.unary, self.pair
        return parameter[:7], parameter[7:].reshape(7, 7) if self.pairwise else None

    def correction(
        self,
        y: torch.Tensor,
        mean: torch.Tensor,
        scale: torch.Tensor,
        summary: torch.Tensor,
        parameter: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Energy correction in the local reference coordinate z=(y-mean)/scale."""
        unary, pair = self._unpack(parameter)
        basis = centered_rbf_basis((y - mean) / scale)
        result = torch.einsum("...k,k->...", basis, unary)
        if pair is not None:
            result = result + torch.einsum("...k,kl,...l->...", basis, pair, summary)
        return result

    def log_normalizer(self, mean: torch.Tensor, scale: torch.Tensor, summary: torch.Tensor, parameter: torch.Tensor | None = None) -> torch.Tensor:
        if parameter is None and self.is_identity:
            return torch.zeros_like(mean)
        y = mean[..., None] + scale[..., None] * self.nodes.to(mean.device)
        correction = self.correction(y, mean[..., None], scale[..., None], summary[..., None, :], parameter)
        return torch.logsumexp(self.weights.to(mean.device).log() - correction, dim=-1)

    def log_prob(self, y: torch.Tensor, mean: torch.Tensor, scale: torch.Tensor, summary: torch.Tensor, parameter: torch.Tensor | None = None) -> torch.Tensor:
        z = (y - mean) / scale
        return -0.5 * z.square() - scale.log() - LOG_SQRT_2PI - self.correction(y, mean, scale, summary, parameter) - self.log_normalizer(mean, scale, summary, parameter)

    def objective(self, y: torch.Tensor, mean: torch.Tensor, scale: torch.Tensor, summary: torch.Tensor, parameter: torch.Tensor) -> torch.Tensor:
        nll = -self.log_prob(y, mean, scale, summary, parameter).sum()
        return nll + 0.5 * self.l2_precision * parameter.square().sum()

    def fit(self, y: torch.Tensor, mean: torch.Tensor, scale: torch.Tensor, summary: torch.Tensor, *, max_iter: int = 250, initial: torch.Tensor | None = None) -> FitSummary:
        device = y.device
        parameter = torch.nn.Parameter(torch.zeros(self.parameter_count, dtype=torch.double, device=device) if initial is None else initial.detach().double().to(device).clone())
        optimizer = torch.optim.LBFGS([parameter], max_iter=max_iter, tolerance_grad=1e-10, tolerance_change=1e-12, line_search_fn="strong_wolfe")
        calls = 0; objective_seconds = 0.0; start = time.perf_counter()
        def closure() -> torch.Tensor:
            nonlocal calls, objective_seconds
            if device.type == "cuda": torch.cuda.synchronize(device)
            closure_start = time.perf_counter()
            optimizer.zero_grad(); value = self.objective(y, mean, scale, summary, parameter); value.backward(); calls += 1
            if device.type == "cuda": torch.cuda.synchronize(device)
            objective_seconds += time.perf_counter() - closure_start
            return value
        optimizer.step(closure)
        if device.type == "cuda": torch.cuda.synchronize(device)
        seconds = time.perf_counter() - start
        unary, pair = self._unpack(parameter.detach()); self.unary = unary.clone(); self.pair = None if pair is None else pair.clone()
        nll = float(-self.log_prob(y, mean, scale, summary).sum())
        penalty = float(0.5 * self.l2_precision * parameter.detach().square().sum())
        gradient = torch.autograd.grad(self.objective(y, mean, scale, summary, parameter), parameter)[0]
        return FitSummary(nll + penalty, nll, penalty, calls, bool(torch.isfinite(parameter).all() and gradient.norm() < 1e-5), seconds, objective_seconds)

    def quadrature(self, mean: torch.Tensor, scale: torch.Tensor, summary: torch.Tensor, *, points: int = 160, tail: float = 10.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        nodes_np, weights_np = np.polynomial.legendre.leggauss(points)
        nodes = torch.tensor(nodes_np, dtype=torch.double, device=mean.device)
        weights = torch.tensor(weights_np, dtype=torch.double, device=mean.device)
        z = tail * nodes
        y = mean[..., None] + scale[..., None] * z
        logp = self.log_prob(y, mean[..., None], scale[..., None], summary[..., None, :])
        mass = tail * scale[..., None] * weights * torch.exp(logp)
        return y, mass, logp

    def moments(self, mean: torch.Tensor, scale: torch.Tensor, summary: torch.Tensor, points: int = 160) -> tuple[torch.Tensor, torch.Tensor]:
        if self.is_identity: return mean, scale.square()
        y, mass, _ = self.quadrature(mean, scale, summary, points=points)
        first = (mass * y).sum(-1); second = (mass * y.square()).sum(-1)
        return first, (second - first.square()).clamp_min(0)

    def cdf(self, value: torch.Tensor, mean: torch.Tensor, scale: torch.Tensor, summary: torch.Tensor, points: int = 160) -> torch.Tensor:
        if self.is_identity: return 0.5 * (1 + torch.erf((value - mean) / scale / math.sqrt(2)))
        nodes_np, weights_np = np.polynomial.legendre.leggauss(points)
        nodes = torch.tensor(nodes_np, dtype=torch.double, device=mean.device); weights = torch.tensor(weights_np, dtype=torch.double, device=mean.device)
        lower = mean - 10 * scale; upper = value.clamp(min=lower, max=mean + 10 * scale)
        y = lower[..., None] + 0.5 * (upper - lower)[..., None] * (nodes + 1)
        integral = 0.5 * (upper - lower) * torch.sum(weights * torch.exp(self.log_prob(y, mean[..., None], scale[..., None], summary[..., None, :])), -1)
        return torch.where(value <= lower, 0, torch.where(value >= mean + 10 * scale, 1, integral)).clamp(0, 1)

    def sample(self, mean: torch.Tensor, scale: torch.Tensor, summary: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        if self.is_identity:
            return mean + scale * torch.randn(mean.shape,dtype=torch.double,device=mean.device,generator=generator)
        uniform=torch.rand(mean.shape,dtype=torch.double,device=mean.device,generator=generator)
        lower,upper=mean-10*scale,mean+10*scale
        for _ in range(64):
            middle=.5*(lower+upper); go_right=self.cdf(middle,mean,scale,summary)<uniform
            lower=torch.where(go_right,middle,lower); upper=torch.where(go_right,upper,middle)
        return .5*(lower+upper)

    def expected_improvement(self, mean: torch.Tensor, scale: torch.Tensor, summary: torch.Tensor, best: float, points: int = 128) -> torch.Tensor:
        if self.is_identity:
            delta = mean - best; z = delta / scale
            return (delta * 0.5 * (1 + torch.erf(z / math.sqrt(2))) + scale * torch.exp(-0.5 * z.square()) / math.sqrt(2 * math.pi)).clamp_min(0)
        return self.expected_improvement_quadrature(mean, scale, summary, best, points=points)

    def expected_improvement_quadrature(self, mean: torch.Tensor, scale: torch.Tensor, summary: torch.Tensor, best: float, points: int = 128) -> torch.Tensor:
        """Direct free-energy-ratio EI, retained independently of the identity shortcut."""
        nodes_np, weights_np = np.polynomial.legendre.leggauss(points)
        nodes = torch.tensor(nodes_np, dtype=torch.double, device=mean.device); weights = torch.tensor(weights_np, dtype=torch.double, device=mean.device)
        lower = torch.maximum(torch.as_tensor(best, dtype=torch.double, device=mean.device), mean - 10 * scale); upper = mean + 10 * scale
        width = (upper - lower).clamp_min(0); y = lower[..., None] + 0.5 * width[..., None] * (nodes + 1)
        log_terms = weights.log() + torch.log((y - best).clamp_min(torch.finfo(torch.double).tiny)) + self.log_prob(y, mean[..., None], scale[..., None], summary[..., None, :])
        return torch.exp(math.log(0.5) + width.clamp_min(torch.finfo(torch.double).tiny).log() + torch.logsumexp(log_terms, -1)).where(width > 0, 0)

    def correction_kl(self, mean: torch.Tensor, scale: torch.Tensor, summary: torch.Tensor) -> torch.Tensor:
        """KL(corrected conditional || local Gaussian reference)."""
        if self.is_identity:
            return torch.zeros_like(mean)
        y = mean[..., None] + scale[..., None] * self.nodes.to(mean.device)
        correction = self.correction(y, mean[..., None], scale[..., None], summary[..., None, :])
        log_normalizer = self.log_normalizer(mean, scale, summary)[..., None]
        log_ratio = -correction - log_normalizer
        ratio = torch.exp(log_ratio)
        return (self.weights.to(mean.device) * ratio * log_ratio).sum(-1).clamp_min(0)
