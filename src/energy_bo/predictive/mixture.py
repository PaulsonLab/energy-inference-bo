"""Batched scalar Gaussian-mixture marginals.

The leading dimension indexes mixture components and the second dimension indexes
independent design points.  Task 03A uses equal weights, but explicit weights keep
the numerical checks reusable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class GaussianMixtureMarginals:
    means: torch.Tensor
    variances: torch.Tensor
    weights: torch.Tensor | None = None

    def __post_init__(self) -> None:
        means = self.means.detach().double().clone()
        variances = self.variances.detach().double().clone()
        if means.ndim == 1:
            means = means[:, None]
        if variances.ndim == 1:
            variances = variances[:, None]
        if means.ndim != 2 or means.shape != variances.shape or means.shape[0] < 1:
            raise ValueError("means and variances must have shape [components, points]")
        if not torch.isfinite(means).all() or not torch.isfinite(variances).all():
            raise ValueError("mixture parameters must be finite")
        if not torch.all(variances > 0):
            raise ValueError("component variances must be positive")
        weights = (
            torch.full((means.shape[0],), 1.0 / means.shape[0], dtype=torch.double, device=means.device)
            if self.weights is None
            else self.weights.detach().to(dtype=torch.double, device=means.device).reshape(-1).clone()
        )
        if weights.shape != (means.shape[0],) or not torch.all(weights > 0):
            raise ValueError("one positive weight is required per component")
        weights = weights / weights.sum()
        object.__setattr__(self, "means", means)
        object.__setattr__(self, "variances", variances)
        object.__setattr__(self, "weights", weights)

    @property
    def component_count(self) -> int:
        return self.means.shape[0]

    @property
    def point_count(self) -> int:
        return self.means.shape[1]

    @property
    def mean(self) -> torch.Tensor:
        return torch.einsum("m,mn->n", self.weights, self.means)

    @property
    def variance(self) -> torch.Tensor:
        second = torch.einsum(
            "m,mn->n", self.weights, self.variances + self.means.square()
        )
        return (second - self.mean.square()).clamp_min(torch.finfo(torch.double).eps)

    @property
    def disagreement_fraction(self) -> torch.Tensor:
        between = torch.einsum(
            "m,mn->n", self.weights, (self.means - self.mean[None, :]).square()
        )
        return between / self.variance.clamp_min(1e-15)

    def _values(self, value: torch.Tensor | float) -> torch.Tensor:
        values = torch.as_tensor(value, dtype=torch.double, device=self.means.device)
        if values.ndim == 0:
            values = values.expand(self.point_count)
        values = values.reshape(-1)
        if values.numel() not in (1, self.point_count):
            raise ValueError("value must be scalar or have one entry per point")
        return values.expand(self.point_count)

    def log_prob(self, value: torch.Tensor | float) -> torch.Tensor:
        values = self._values(value)
        log_component = (
            -0.5 * ((values[None, :] - self.means).square() / self.variances)
            - 0.5 * self.variances.log()
            - 0.5 * math.log(2.0 * math.pi)
        )
        return torch.logsumexp(self.weights.log()[:, None] + log_component, dim=0)

    def cdf(self, value: torch.Tensor | float) -> torch.Tensor:
        values = self._values(value)
        z = (values[None, :] - self.means) / self.variances.sqrt()
        component = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
        return torch.einsum("m,mn->n", self.weights, component)

    def ppf(self, probability: torch.Tensor | float, iterations: int = 80) -> torch.Tensor:
        probability = torch.as_tensor(probability, dtype=torch.double)
        if probability.ndim == 0:
            probability = probability.expand(self.point_count)
        probability = probability.reshape(-1).expand(self.point_count)
        if not torch.all((probability > 0) & (probability < 1)):
            raise ValueError("probabilities must be strictly between zero and one")
        scale = self.variances.sqrt()
        lower = (self.means - 12.0 * scale).min(dim=0).values
        upper = (self.means + 12.0 * scale).max(dim=0).values
        for _ in range(iterations):
            middle = 0.5 * (lower + upper)
            go_right = self.cdf(middle) < probability
            lower = torch.where(go_right, middle, lower)
            upper = torch.where(go_right, upper, middle)
        return 0.5 * (lower + upper)

    def sample(self, sample_count: int, generator: torch.Generator) -> torch.Tensor:
        if sample_count < 1:
            raise ValueError("sample_count must be positive")
        components = torch.multinomial(
            self.weights, sample_count * self.point_count, replacement=True, generator=generator
        ).reshape(sample_count, self.point_count)
        point = torch.arange(self.point_count, device=self.means.device)[None, :].expand(sample_count, -1)
        means = self.means[components, point]
        scales = self.variances[components, point].sqrt()
        return means + scales * torch.randn(
            (sample_count, self.point_count), dtype=torch.double, device=self.means.device, generator=generator
        )

    def expected_improvement(self, best_f: float | torch.Tensor) -> torch.Tensor:
        best = torch.as_tensor(best_f, dtype=torch.double, device=self.means.device)
        scale = self.variances.sqrt()
        improvement = self.means - best
        z = improvement / scale
        pdf = torch.exp(-0.5 * z.square()) / math.sqrt(2.0 * math.pi)
        cdf = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
        component = (improvement * cdf + scale * pdf).clamp_min(0.0)
        return torch.einsum("m,mn->n", self.weights, component)

    def subset(self, indices: torch.Tensor | list[int]) -> "GaussianMixtureMarginals":
        index = torch.as_tensor(indices, dtype=torch.long, device=self.means.device)
        return GaussianMixtureMarginals(self.means[:, index], self.variances[:, index], self.weights)

    def to(self, device: torch.device | str) -> "GaussianMixtureMarginals":
        return GaussianMixtureMarginals(self.means.to(device), self.variances.to(device), self.weights.to(device))
