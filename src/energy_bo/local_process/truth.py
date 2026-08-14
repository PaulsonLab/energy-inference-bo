"""Sequential Gaussian, warped, and explicit-interaction local oracle truths."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from .energy import LocalEnergyModel, centered_rbf_basis, neighbor_summary
from .geometry import LocalGeometry, build_geometry

U_TRUE = torch.tensor([0, -1, 0, 2, 0, -1, 0], dtype=torch.double) / math.sqrt(6)
V_TRUE = torch.tensor([0, 0, -1, 0, 1, 0, 0], dtype=torch.double) / math.sqrt(2)


def _interaction_parameter() -> torch.Tensor:
    parameter = torch.zeros(56, dtype=torch.double)
    parameter[7:] = (2.0 * torch.outer(U_TRUE, V_TRUE)).reshape(-1)
    return parameter


def inverse_grid_sample(model: LocalEnergyModel, mean: torch.Tensor, scale: torch.Tensor, summary: torch.Tensor, uniform: torch.Tensor, points: int = 4097) -> torch.Tensor:
    z = torch.linspace(-10, 10, points, dtype=torch.double, device=mean.device)
    y = mean + scale * z
    log_normalizer = model.log_normalizer(mean, scale, summary)
    logp = (
        -0.5 * z.square()
        - scale.log()
        - 0.5 * math.log(2 * math.pi)
        - model.correction(y, mean.expand_as(y), scale.expand_as(y), summary.expand(points, -1))
        - log_normalizer
    )
    density = torch.exp(logp)
    increments = 0.5 * (density[1:] + density[:-1]) * (y[1:] - y[:-1])
    cdf = torch.cat((torch.zeros(1, dtype=torch.double, device=y.device), increments.cumsum(0)))
    cdf = cdf / cdf[-1]
    idx = torch.searchsorted(cdf, uniform).clamp(1, points - 1)
    fraction = (uniform - cdf[idx - 1]) / (cdf[idx] - cdf[idx - 1]).clamp_min(1e-15)
    return y[idx - 1] + fraction * (y[idx] - y[idx - 1])


@dataclass(frozen=True)
class LocalOracle:
    regime: str
    geometry: LocalGeometry
    values: torch.Tensor
    latent: torch.Tensor
    alpha: float = 0.6

    def conditional(self, geometry: LocalGeometry, source_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, LocalEnergyModel | None]:
        if self.regime == "W":
            latent_source = torch.log1p(self.alpha * source_values) / self.alpha
            mean = geometry.means(latent_source); scale = geometry.variances.sqrt()
            summary = torch.zeros((geometry.count, 7), dtype=torch.double, device=mean.device)
            return mean, scale, summary, None
        mean = geometry.means(source_values); scale = geometry.variances.sqrt()
        summary = neighbor_summary(source_values, geometry.neighbors, geometry.mask, geometry.similarity_weights)
        model = None
        if self.regime == "I":
            model = LocalEnergyModel(True).to(mean.device); unary, pair = model._unpack(_interaction_parameter().to(mean.device)); model.unary, model.pair = unary, pair
        return mean, scale, summary, model

    def log_prob(self, y: torch.Tensor, geometry: LocalGeometry, source_values: torch.Tensor) -> torch.Tensor:
        mean, scale, summary, model = self.conditional(geometry, source_values)
        while mean.ndim < y.ndim:
            mean, scale, summary = mean[..., None], scale[..., None], summary[..., None, :]
        if self.regime == "W":
            latent = torch.log1p(self.alpha * y) / self.alpha
            z = (latent - mean) / scale
            return -0.5*z.square() - scale.log() - 0.5*math.log(2*math.pi) - self.alpha*latent
        if model is None:
            z=(y-mean)/scale; return -0.5*z.square()-scale.log()-0.5*math.log(2*math.pi)
        return model.log_prob(y, mean, scale, summary)

    def expected_improvement(self, geometry: LocalGeometry, source_values: torch.Tensor, best: float) -> torch.Tensor:
        mean, scale, summary, model = self.conditional(geometry, source_values)
        if self.regime == "W":
            threshold = torch.log1p(self.alpha * torch.as_tensor(best, dtype=torch.double)) / self.alpha
            first_z=(mean+self.alpha*scale.square()-threshold)/scale; second_z=(mean-threshold)/scale
            cdf=lambda z: .5*(1+torch.erf(z/math.sqrt(2)))
            return ((torch.exp(self.alpha*mean+.5*self.alpha**2*scale.square())*cdf(first_z)-torch.exp(self.alpha*threshold)*cdf(second_z))/self.alpha).clamp_min(0)
        return (model or LocalEnergyModel(False)).expected_improvement(mean, scale, summary, best)

    def cdf(self, value: torch.Tensor, geometry: LocalGeometry, source_values: torch.Tensor) -> torch.Tensor:
        mean,scale,summary,model=self.conditional(geometry,source_values)
        if self.regime=="W":
            latent=torch.log1p(self.alpha*value)/self.alpha
            return .5*(1+torch.erf((latent-mean)/scale/math.sqrt(2)))
        return (model or LocalEnergyModel(False)).cdf(value,mean,scale,summary)

    def moments(self, geometry: LocalGeometry, source_values: torch.Tensor) -> tuple[torch.Tensor,torch.Tensor]:
        mean,scale,summary,model=self.conditional(geometry,source_values)
        if self.regime=="W":
            first=torch.exp(self.alpha*mean+.5*self.alpha**2*scale.square())
            second=torch.exp(2*self.alpha*mean+2*self.alpha**2*scale.square())
            return (first-1)/self.alpha,(second-first.square())/self.alpha**2
        return (model or LocalEnergyModel(False)).moments(mean,scale,summary)


def generate_oracle(x: torch.Tensor, regime: str, seed: int, m: int = 8, geometry: LocalGeometry | None = None) -> LocalOracle:
    if regime not in {"G", "W", "I"}: raise ValueError("regime must be G, W, or I")
    geometry = build_geometry(x, m=m) if geometry is None else geometry
    if not torch.equal(geometry.x, x.detach().double().cpu()): raise ValueError("shared geometry does not match x")
    generator = torch.Generator().manual_seed(40_000 + seed + {"G":0,"W":10_000,"I":20_000}[regime])
    uniforms = torch.rand(len(x), dtype=torch.double, generator=generator)
    normals = torch.randn(len(x), dtype=torch.double, generator=generator)
    values = torch.zeros(len(x), dtype=torch.double); latent = torch.zeros_like(values)
    interaction = LocalEnergyModel(True); unary, pair = interaction._unpack(_interaction_parameter()); interaction.unary, interaction.pair = unary, pair
    for i in range(len(x)):
        coeff=geometry.coefficients[i]; neighbor=geometry.neighbors[i]; mask=geometry.mask[i]; scale=geometry.variances[i].sqrt()
        if regime == "W":
            mean=(coeff*latent[neighbor]*mask).sum(); latent[i]=mean+scale*normals[i]; values[i]=torch.expm1(.6*latent[i])/.6
        else:
            mean=(coeff*values[neighbor]*mask).sum()
            if regime == "G": values[i]=mean+scale*normals[i]
            else:
                summary=neighbor_summary(values, neighbor[None], mask[None], geometry.similarity_weights[i:i+1])[0]
                values[i]=inverse_grid_sample(interaction,mean,scale,summary,uniforms[i])
            latent[i]=values[i]
    return LocalOracle(regime, geometry, values, latent)
