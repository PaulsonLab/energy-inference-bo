"""PIT-space scalar predictive corrections with deterministic one-dimensional EI."""

from __future__ import annotations

import math

import numpy as np
import torch

from .mixture import GaussianMixtureMarginals
from .residuals import ResidualDistribution, normal_log_prob


class PITCorrectedPredictive:
    """Apply a normalized residual law to Gaussian-mixture PIT coordinates."""

    def __init__(
        self,
        reference: GaussianMixtureMarginals,
        residual: ResidualDistribution,
        context: torch.Tensor,
    ) -> None:
        context = torch.as_tensor(context, dtype=torch.double, device=reference.means.device)
        if context.ndim != 2 or context.shape[0] != reference.point_count:
            raise ValueError("context must have one row per predictive marginal")
        self.reference = reference
        self.residual = residual
        self.context = context

    def _pit_z(self, value: torch.Tensor | float) -> torch.Tensor:
        probability = self.reference.cdf(value).clamp(1e-15, 1.0 - 1e-15)
        return torch.special.ndtri(probability)

    def log_prob(self, value: torch.Tensor | float) -> torch.Tensor:
        if self.residual.is_identity:
            return self.reference.log_prob(value)
        z = self._pit_z(value)
        return (
            self.reference.log_prob(value)
            + self.residual.log_prob(z, self.context)
            - normal_log_prob(z)
        )

    def cdf(self, value: torch.Tensor | float) -> torch.Tensor:
        if self.residual.is_identity:
            return self.reference.cdf(value)
        return self.residual.cdf(self._pit_z(value), self.context)

    def sample(self, sample_count: int, generator: torch.Generator) -> torch.Tensor:
        if self.residual.is_identity:
            return self.reference.sample(sample_count, generator)
        context = self.context[None, :, :].expand(sample_count, -1, -1)
        z = self.residual.sample(context, generator)
        probability = (0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))).clamp(
            1e-14, 1.0 - 1e-14
        )
        rows = []
        for point in range(self.reference.point_count):
            marginal = self.reference.subset([point])
            repeated = GaussianMixtureMarginals(
                marginal.means.expand(-1, sample_count),
                marginal.variances.expand(-1, sample_count),
                marginal.weights,
            )
            rows.append(repeated.ppf(probability[:, point]).reshape(-1))
        return torch.stack(rows, dim=1)

    def _reference_cdf_at_component_values(self, values: torch.Tensor) -> torch.Tensor:
        """Evaluate F0 at values shaped [source_component, point, quadrature]."""
        means = self.reference.means[:, None, :, None]
        scales = self.reference.variances.sqrt()[:, None, :, None]
        standardized = (values[None, :, :, :] - means) / scales
        cdf = 0.5 * (1.0 + torch.erf(standardized / math.sqrt(2.0)))
        return torch.einsum("r,rmnq->mnq", self.reference.weights, cdf)

    def expected_improvement(
        self,
        best_f: float | torch.Tensor,
        *,
        quadrature_points: int = 96,
        tail_limit: float = 10.0,
    ) -> torch.Tensor:
        """Evaluate q=1 EI by component-wise Gauss-Legendre tail integration."""
        if self.residual.is_identity:
            return self.reference.expected_improvement(best_f)
        nodes_np, weights_np = np.polynomial.legendre.leggauss(quadrature_points)
        nodes = torch.tensor(nodes_np, dtype=torch.double, device=self.reference.means.device)
        weights = torch.tensor(weights_np, dtype=torch.double, device=self.reference.means.device)
        means = self.reference.means
        scales = self.reference.variances.sqrt()
        threshold = (torch.as_tensor(best_f, dtype=torch.double, device=means.device) - means) / scales
        lower = threshold.clamp(min=-tail_limit, max=tail_limit)
        width = (tail_limit - lower).clamp_min(0.0)
        t = lower[:, :, None] + 0.5 * width[:, :, None] * (nodes + 1.0)
        y = means[:, :, None] + scales[:, :, None] * t
        probability = self._reference_cdf_at_component_values(y).clamp(1e-15, 1 - 1e-15)
        z = torch.special.ndtri(probability)
        context = self.context[None, :, None, :].expand(
            self.reference.component_count, -1, quadrature_points, -1
        )
        log_ratio = self.residual.log_prob(z, context) - normal_log_prob(z)
        normal_density = torch.exp(normal_log_prob(t))
        improvement = (y - torch.as_tensor(best_f, dtype=torch.double, device=y.device)).clamp_min(0)
        component_integral = 0.5 * width * torch.sum(
            weights * improvement * normal_density * torch.exp(log_ratio), dim=-1
        )
        return torch.einsum("m,mn->n", self.reference.weights, component_integral).clamp_min(0)

    def probability_improvement(self, best_f: float | torch.Tensor) -> torch.Tensor:
        return (1.0 - self.cdf(best_f)).clamp(0, 1)

    def moments(self, *, quadrature_points: int = 96, tail_limit: float = 10.0) -> tuple[torch.Tensor, torch.Tensor]:
        if self.residual.is_identity:
            return self.reference.mean, self.reference.variance
        nodes_np, weights_np = np.polynomial.legendre.leggauss(quadrature_points)
        nodes=torch.tensor(nodes_np,dtype=torch.double,device=self.reference.means.device); weights=torch.tensor(weights_np,dtype=torch.double,device=self.reference.means.device)
        t=tail_limit*nodes; y=self.reference.means[:,:,None]+self.reference.variances.sqrt()[:,:,None]*t
        probability=self._reference_cdf_at_component_values(y).clamp(1e-15,1-1e-15); z=torch.special.ndtri(probability)
        context=self.context[None,:,None,:].expand(self.reference.component_count,-1,quadrature_points,-1)
        ratio=torch.exp(self.residual.log_prob(z,context)-normal_log_prob(z)); density=torch.exp(normal_log_prob(t))
        first=tail_limit*torch.sum(weights*y*density*ratio,dim=-1); second=tail_limit*torch.sum(weights*y.square()*density*ratio,dim=-1)
        mean=torch.einsum("m,mn->n",self.reference.weights,first); raw2=torch.einsum("m,mn->n",self.reference.weights,second)
        return mean,(raw2-mean.square()).clamp_min(0)
