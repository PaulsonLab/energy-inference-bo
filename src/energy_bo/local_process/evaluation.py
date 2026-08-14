"""Independent scalar integration and decision metrics for local processes."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from scipy.stats import spearmanr

from energy_bo.metrics import top_fraction_overlap

from .energy import LocalEnergyModel
from .geometry import LocalGeometry
from .truth import LocalOracle


@dataclass(frozen=True)
class ConditionalGrid:
    y: torch.Tensor
    mass: torch.Tensor
    log_density: torch.Tensor


def oracle_grid(oracle: LocalOracle, geometry: LocalGeometry, source: torch.Tensor, *, points: int = 160) -> ConditionalGrid:
    mean, scale, summary, truth_model = oracle.conditional(geometry, source)
    nodes_np, weights_np = np.polynomial.legendre.leggauss(points)
    nodes = torch.tensor(nodes_np, dtype=torch.double, device=mean.device)
    weights = torch.tensor(weights_np, dtype=torch.double, device=mean.device)
    if oracle.regime == "W":
        latent = mean[..., None] + 10 * scale[..., None] * nodes
        y = torch.expm1(oracle.alpha * latent) / oracle.alpha
        normal_mass = 10 * weights * torch.exp(-0.5 * (10 * nodes).square()) / math.sqrt(2 * math.pi)
        mass = normal_mass.expand_as(y)
        log_density = oracle.log_prob(y, geometry, source)
    else:
        model = truth_model or LocalEnergyModel(False).to(mean.device)
        y, mass, log_density = model.quadrature(mean, scale, summary, points=points)
    mass = mass / mass.sum(-1, keepdim=True)
    return ConditionalGrid(y, mass, log_density)


def model_grid(model: LocalEnergyModel, mean: torch.Tensor, scale: torch.Tensor, summary: torch.Tensor, *, points: int = 160) -> ConditionalGrid:
    y, mass, log_density = model.quadrature(mean, scale, summary, points=points)
    mass = mass / mass.sum(-1, keepdim=True)
    return ConditionalGrid(y, mass, log_density)


def _weighted_quantile(grid: ConditionalGrid, levels: tuple[float, ...]) -> torch.Tensor:
    cdf = grid.mass.cumsum(-1)
    output = []
    for level in levels:
        index = torch.searchsorted(cdf.contiguous(), torch.full((cdf.shape[0], 1), level, dtype=torch.double, device=cdf.device)).squeeze(-1).clamp(max=cdf.shape[-1] - 1)
        output.append(grid.y.gather(-1, index[:, None]).squeeze(-1))
    return torch.stack(output, -1)


def predictive_metrics(
    oracle: LocalOracle,
    geometry: LocalGeometry,
    source: torch.Tensor,
    model: LocalEnergyModel,
    reference_mean: torch.Tensor,
    reference_scale: torch.Tensor,
    summary: torch.Tensor,
    *,
    points: int = 160,
) -> dict[str, float]:
    truth = oracle_grid(oracle, geometry, source, points=points)
    repeated_mean = reference_mean[:, None].expand_as(truth.y)
    repeated_scale = reference_scale[:, None].expand_as(truth.y)
    repeated_summary = summary[:, None, :].expand(-1, points, -1)
    model_log = model.log_prob(truth.y, repeated_mean, repeated_scale, repeated_summary)
    true_nll = float(-(truth.mass * model_log).sum(-1).mean())
    true_entropy = float(-(truth.mass * truth.log_density).sum(-1).mean())
    truth_mean = (truth.mass * truth.y).sum(-1)
    truth_second = (truth.mass * truth.y.square()).sum(-1)
    model_mean, model_variance = model.moments(reference_mean, reference_scale, summary, points=points)
    levels = (0.5, 0.9, 0.95, 0.99)
    quantiles = _weighted_quantile(truth, levels)
    repeated_context = summary[:, None, :].expand(-1, len(levels), -1)
    model_cdf = model.cdf(quantiles, reference_mean[:, None], reference_scale[:, None], repeated_context, points=points)
    # Expected CRPS on the common truth grid: E|X-Y| - .5 E|X-X'|.
    model_density = torch.exp(model_log)
    dy = torch.zeros_like(truth.y)
    dy[:, 1:-1] = 0.5 * (truth.y[:, 2:] - truth.y[:, :-2])
    dy[:, 0] = truth.y[:, 1] - truth.y[:, 0]; dy[:, -1] = truth.y[:, -1] - truth.y[:, -2]
    model_mass = (model_density * dy).clamp_min(0); model_mass /= model_mass.sum(-1, keepdim=True)
    crps_values = []
    for start in range(0, len(truth.y), 128):
        y = truth.y[start:start+128]; tm = truth.mass[start:start+128]; mm = model_mass[start:start+128]
        distance = (y[:, :, None] - y[:, None, :]).abs()
        cross = torch.einsum("bi,bij,bj->b", tm, distance, mm)
        self_term = torch.einsum("bi,bij,bj->b", mm, distance, mm)
        crps_values.append(cross - 0.5 * self_term)
    return {
        "conditional_cross_entropy": true_nll,
        "conditional_kl": max(0.0, true_nll - true_entropy),
        "expected_crps": float(torch.cat(crps_values).mean()),
        "mean_rmse": float(torch.mean((model_mean - truth_mean).square()).sqrt()),
        "variance_rmse": float(torch.mean((model_variance - (truth_second - truth_mean.square())).square()).sqrt()),
        **{f"quantile_{int(level*100):02d}_cdf_error": float((model_cdf[:, i] - level).abs().mean()) for i, level in enumerate(levels)},
    }


def oracle_predictive_metrics(oracle: LocalOracle, geometry: LocalGeometry, source: torch.Tensor, *, points: int = 160) -> dict[str, float]:
    truth = oracle_grid(oracle, geometry, source, points=points)
    entropy = float(-(truth.mass * truth.log_density).sum(-1).mean())
    crps=[]
    for start in range(0,len(truth.y),128):
        y=truth.y[start:start+128]; mass=truth.mass[start:start+128]
        distance=(y[:,:,None]-y[:,None,:]).abs()
        crps.append(0.5*torch.einsum("bi,bij,bj->b",mass,distance,mass))
    return {"conditional_cross_entropy":entropy,"conditional_kl":0.0,"expected_crps":float(torch.cat(crps).mean()),"mean_rmse":0.0,"variance_rmse":0.0,"quantile_50_cdf_error":0.0,"quantile_90_cdf_error":0.0,"quantile_95_cdf_error":0.0,"quantile_99_cdf_error":0.0,"average_correction_kl":0.0}


def decision_metrics(model_ei: torch.Tensor, true_ei: torch.Tensor, candidates: torch.Tensor) -> dict[str, float | int]:
    model_ei, true_ei = model_ei.detach().cpu(), true_ei.detach().cpu()
    chosen, optimum = int(torch.argmax(model_ei)), int(torch.argmax(true_ei))
    peak, selected = float(true_ei[optimum]), float(true_ei[chosen])
    absolute = peak - selected
    rank = spearmanr(model_ei.numpy(), true_ei.numpy()).statistic
    return {
        "ei_max_abs_error": float((model_ei - true_ei).abs().max()),
        "ei_max_relative_error": float(((model_ei - true_ei).abs() / true_ei.clamp_min(1e-15)).max()),
        "ei_spearman": float(rank) if np.isfinite(rank) else 1.0,
        "top5_overlap": float(top_fraction_overlap(true_ei.numpy(), model_ei.numpy(), 0.05)),
        "chosen_index": chosen,
        "oracle_index": optimum,
        "chosen_distance": float(torch.linalg.vector_norm(candidates[chosen, :2] - candidates[optimum, :2])),
        "true_ei_optimum": peak,
        "true_ei_at_selection": selected,
        "absolute_regret": absolute,
        "normalized_regret": absolute / max(peak, 1e-15),
    }
