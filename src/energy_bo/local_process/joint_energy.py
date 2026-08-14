"""Convex bivariate residual energies relative to a correlated Gaussian."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import torch
from numpy.polynomial.hermite import hermgauss

from .energy import centered_rbf_basis


@dataclass(frozen=True)
class BivariateFitSummary:
    objective: float
    nll: float
    penalty: float
    iterations: int
    converged: bool
    gradient_norm: float
    seconds: float


def _upper_indices() -> tuple[torch.Tensor, torch.Tensor]:
    return torch.triu_indices(7, 7)


def unpack_symmetric(unique: torch.Tensor) -> torch.Tensor:
    """Map 28 orthonormal upper-triangle coordinates to a symmetric matrix."""
    if unique.shape[-1] != 28:
        raise ValueError("symmetric coordinate must have length 28")
    row, column = _upper_indices()
    row, column = row.to(unique.device), column.to(unique.device)
    matrix = torch.zeros((*unique.shape[:-1], 7, 7), dtype=unique.dtype, device=unique.device)
    diagonal = row == column
    matrix[..., row[diagonal], column[diagonal]] = unique[..., diagonal]
    scaled = unique[..., ~diagonal] / math.sqrt(2)
    matrix[..., row[~diagonal], column[~diagonal]] = scaled
    matrix[..., column[~diagonal], row[~diagonal]] = scaled
    return matrix


class BivariateEnergyModel:
    def __init__(self, pairwise: bool, *, rho: float = 0.5, l2_precision: float = 10.0, quadrature_points: int = 32) -> None:
        self.pairwise = bool(pairwise)
        self.rho = float(rho)
        self.l2_precision = float(l2_precision)
        self.quadrature_points = int(quadrature_points)
        nodes, weights = hermgauss(quadrature_points)
        standard = torch.tensor(math.sqrt(2) * nodes, dtype=torch.double)
        one_weights = torch.tensor(weights / math.sqrt(math.pi), dtype=torch.double)
        first, second = torch.meshgrid(standard, standard, indexing="ij")
        self.base_nodes = torch.stack((first.reshape(-1), (self.rho * first + math.sqrt(1 - self.rho**2) * second).reshape(-1)), -1)
        self.base_weights = torch.outer(one_weights, one_weights).reshape(-1)
        self.parameter = torch.zeros(self.parameter_count, dtype=torch.double)
        self._refresh_features()

    @property
    def parameter_count(self) -> int:
        return 35 if self.pairwise else 7

    @property
    def is_identity(self) -> bool:
        return bool(torch.count_nonzero(self.parameter) == 0)

    def _refresh_features(self) -> None:
        basis = centered_rbf_basis(self.base_nodes)
        self.node_unary = basis[:, 0] + basis[:, 1]
        self.node_pair = self.symmetric_features(basis[:, 0], basis[:, 1])
        self.pair_center = torch.einsum("q,qk->k", self.base_weights, self.node_pair)
        self.node_pair = self.node_pair - self.pair_center

    def to(self, device: torch.device | str) -> "BivariateEnergyModel":
        for name in ("base_nodes", "base_weights", "node_unary", "node_pair", "pair_center", "parameter"):
            setattr(self, name, getattr(self, name).to(device))
        return self

    @staticmethod
    def symmetric_features(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        row, column = _upper_indices()
        row, column = row.to(first.device), column.to(first.device)
        diagonal = row == column
        result = torch.empty((*first.shape[:-1], 28), dtype=first.dtype, device=first.device)
        result[..., diagonal] = first[..., row[diagonal]] * second[..., column[diagonal]]
        result[..., ~diagonal] = (
            first[..., row[~diagonal]] * second[..., column[~diagonal]]
            + first[..., column[~diagonal]] * second[..., row[~diagonal]]
        ) / math.sqrt(2)
        return result

    def features(self, z: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        basis = centered_rbf_basis(z)
        unary = basis[..., 0, :] + basis[..., 1, :]
        if not self.pairwise:
            return unary
        pair = self.symmetric_features(basis[..., 0, :], basis[..., 1, :]) - self.pair_center.to(z.device)
        return torch.cat((unary, context[..., None] * pair), -1)

    def correction(self, z: torch.Tensor, context: torch.Tensor, parameter: torch.Tensor | None = None) -> torch.Tensor:
        parameter = self.parameter if parameter is None else parameter
        return torch.einsum("...k,k->...", self.features(z, context), parameter)

    def node_correction(self, context: torch.Tensor, parameter: torch.Tensor | None = None) -> torch.Tensor:
        parameter = self.parameter if parameter is None else parameter
        unary = self.node_unary.to(parameter.device) @ parameter[:7]
        if not self.pairwise:
            return unary.expand((*context.shape, len(unary)))
        pair = self.node_pair.to(parameter.device) @ parameter[7:]
        return unary + context[..., None] * pair

    def log_normalizer(self, context: torch.Tensor, parameter: torch.Tensor | None = None) -> torch.Tensor:
        context = torch.as_tensor(context, dtype=torch.double, device=self.parameter.device if parameter is None else parameter.device)
        if parameter is None and self.is_identity:
            return torch.zeros_like(context)
        flat = context.reshape(-1)
        unique, inverse = torch.unique(flat, sorted=True, return_inverse=True)
        correction = self.node_correction(unique, parameter)
        values = torch.logsumexp(self.base_weights.to(correction.device).log() - correction, -1)
        return values[inverse].reshape(context.shape)

    def base_log_prob(self, z: torch.Tensor) -> torch.Tensor:
        quadratic = (z[..., 0].square() - 2 * self.rho * z[..., 0] * z[..., 1] + z[..., 1].square()) / (1 - self.rho**2)
        return -math.log(2 * math.pi) - 0.5 * math.log(1 - self.rho**2) - 0.5 * quadratic

    def log_prob(self, z: torch.Tensor, context: torch.Tensor, parameter: torch.Tensor | None = None) -> torch.Tensor:
        return self.base_log_prob(z) - self.correction(z, context, parameter) - self.log_normalizer(context, parameter)

    def objective(self, z: torch.Tensor, context: torch.Tensor, parameter: torch.Tensor) -> torch.Tensor:
        nll = -self.log_prob(z, context, parameter).sum()
        return nll + 0.5 * self.l2_precision * parameter.square().sum()

    def fit(self, z: torch.Tensor, context: torch.Tensor, *, max_iter: int = 250, initial: torch.Tensor | None = None) -> BivariateFitSummary:
        device = z.device
        parameter = torch.nn.Parameter(torch.zeros(self.parameter_count, dtype=torch.double, device=device) if initial is None else initial.detach().double().to(device).clone())
        optimizer = torch.optim.LBFGS([parameter], max_iter=max_iter, tolerance_grad=1e-10, tolerance_change=1e-12, line_search_fn="strong_wolfe")
        calls = 0
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()

        def closure() -> torch.Tensor:
            nonlocal calls
            optimizer.zero_grad()
            value = self.objective(z, context, parameter)
            value.backward()
            calls += 1
            return value

        optimizer.step(closure)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        self.parameter = parameter.detach().clone()
        nll = float(-self.log_prob(z, context).sum())
        penalty = float(0.5 * self.l2_precision * self.parameter.square().sum())
        gradient = torch.autograd.grad(self.objective(z, context, parameter), parameter)[0]
        norm = float(gradient.norm())
        return BivariateFitSummary(nll + penalty, nll, penalty, calls, bool(torch.isfinite(parameter).all() and norm < 1e-5), norm, time.perf_counter() - start)

    def normalized_node_weights(self, context: torch.Tensor) -> torch.Tensor:
        correction = self.node_correction(context)
        log_weight = self.base_weights.to(correction.device).log() - correction
        return torch.softmax(log_weight, -1)

    def moments(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        weights = self.normalized_node_weights(context)
        nodes = self.base_nodes.to(weights.device)
        means = torch.einsum("...q,qj->...j", weights, nodes)
        second = torch.einsum("...q,qj->...j", weights, nodes.square())
        cross = torch.einsum("...q,q->...", weights, nodes[:, 0] * nodes[:, 1])
        variance = second - means.square()
        covariance = cross - means[..., 0] * means[..., 1]
        correlation = covariance / (variance[..., 0] * variance[..., 1]).sqrt().clamp_min(1e-15)
        return means, variance, correlation

    def qei(self, mean: torch.Tensor, scale: torch.Tensor, context: torch.Tensor, best: float, *, chunk: int = 32, points: int | None = None, tail: float = 9.0) -> torch.Tensor:
        """Stable qEI integration split along the two max-utility regions."""
        mean, scale, context = mean.double(), scale.double(), context.double()
        points = self.quadrature_points if points is None else int(points)
        nodes_np, weights_np = np.polynomial.legendre.leggauss(points)
        nodes = torch.tensor(nodes_np, dtype=torch.double, device=mean.device)
        weights = torch.tensor(weights_np, dtype=torch.double, device=mean.device)
        outputs: list[torch.Tensor] = []
        for start in range(0, len(mean), chunk):
            stop = start + chunk
            batch_mean, batch_scale, batch_context = mean[start:stop], scale[start:stop], context[start:stop]
            total = torch.zeros(len(batch_mean), dtype=torch.double, device=mean.device)
            for focus in (0, 1):
                other = 1 - focus
                lower = (best - batch_mean[:, focus]) / batch_scale[:, focus]
                width = (tail - lower).clamp_min(0)
                focus_z = lower[:, None] + 0.5 * width[:, None] * (nodes + 1)
                focus_weight = 0.5 * width[:, None] * weights
                focus_y = batch_mean[:, focus, None] + batch_scale[:, focus, None] * focus_z
                boundary = (focus_y - batch_mean[:, other, None]) / batch_scale[:, other, None]
                other_width = (boundary.clamp(max=tail) + tail).clamp_min(0)
                other_z = -tail + 0.5 * other_width[..., None] * (nodes + 1)
                other_weight = 0.5 * other_width[..., None] * weights
                pair = torch.empty((*other_z.shape, 2), dtype=torch.double, device=mean.device)
                pair[..., focus] = focus_z[..., None]
                pair[..., other] = other_z
                expanded_context = batch_context[:, None, None].expand_as(other_z)
                density = torch.exp(self.log_prob(pair, expanded_context))
                utility = (focus_y - best).clamp_min(0)
                total = total + (focus_weight[..., None] * other_weight * density * utility[..., None]).sum((-2, -1))
            outputs.append(total)
        return torch.cat(outputs)

    def marginal_log_prob(self, value: torch.Tensor, context: torch.Tensor, *, conditional_points: int = 64) -> torch.Tensor:
        nodes_np, weights_np = hermgauss(conditional_points)
        epsilon = torch.tensor(math.sqrt(2) * nodes_np, dtype=torch.double, device=value.device)
        weights = torch.tensor(weights_np / math.sqrt(math.pi), dtype=torch.double, device=value.device)
        second = self.rho * value[..., None] + math.sqrt(1 - self.rho**2) * epsilon
        pair = torch.stack((value[..., None].expand_as(second), second), -1)
        expanded_context = context[..., None].expand_as(second)
        log_ratio = torch.logsumexp(weights.log() - self.correction(pair, expanded_context), -1) - self.log_normalizer(context)
        return -0.5 * value.square() - 0.5 * math.log(2 * math.pi) + log_ratio

    def marginal_metrics(self, context: torch.Tensor, best: float, *, points: int = 64) -> dict[str, torch.Tensor]:
        nodes_np, weights_np = hermgauss(points)
        nodes = torch.tensor(math.sqrt(2) * nodes_np, dtype=torch.double, device=context.device)
        weights = torch.tensor(weights_np / math.sqrt(math.pi), dtype=torch.double, device=context.device)
        values = nodes.expand((*context.shape, len(nodes)))
        contexts = context[..., None].expand_as(values)
        log_marginal = self.marginal_log_prob(values, contexts)
        log_phi = -0.5 * values.square() - 0.5 * math.log(2 * math.pi)
        ratio = torch.exp(log_marginal - log_phi)
        marginal_kl = (weights * (log_phi - log_marginal)).sum(-1).clamp_min(0)
        mean = (weights * ratio * values).sum(-1)
        variance = (weights * ratio * values.square()).sum(-1) - mean.square()
        q1 = self.marginal_q1_ei(context, best, points=max(128, 2 * points))
        return {"kl": marginal_kl, "mean": mean, "variance": variance, "q1_ei": q1}

    def marginal_q1_ei(self, context: torch.Tensor, best: float, *, points: int = 128, upper: float = 9.0) -> torch.Tensor:
        nodes_np, weights_np = np.polynomial.legendre.leggauss(points)
        nodes = torch.tensor(nodes_np, dtype=torch.double, device=context.device)
        weights = torch.tensor(weights_np, dtype=torch.double, device=context.device)
        lower = torch.as_tensor(best, dtype=torch.double, device=context.device)
        value = lower + 0.5 * (upper - lower) * (nodes + 1)
        expanded_value = value.expand((*context.shape, points))
        expanded_context = context[..., None].expand_as(expanded_value)
        density = torch.exp(self.marginal_log_prob(expanded_value, expanded_context))
        return 0.5 * (upper - lower) * (weights * (expanded_value - best) * density).sum(-1)

    def marginal_cdf(self, value: torch.Tensor, context: torch.Tensor, *, points: int = 128, lower: float = -9.0) -> torch.Tensor:
        nodes_np, weights_np = np.polynomial.legendre.leggauss(points)
        nodes = torch.tensor(nodes_np, dtype=torch.double, device=value.device)
        weights = torch.tensor(weights_np, dtype=torch.double, device=value.device)
        width = (value - lower).clamp_min(0)
        locations = lower + 0.5 * width[..., None] * (nodes + 1)
        density = torch.exp(self.marginal_log_prob(locations, context[..., None].expand_as(locations)))
        return (0.5 * width * (weights * density).sum(-1)).clamp(0, 1)
