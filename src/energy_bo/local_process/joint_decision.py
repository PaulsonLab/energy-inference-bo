"""Prospectively paired q=2 decision panels and tie-aware metrics."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from scipy.stats import spearmanr
from torch.quasirandom import SobolEngine

from energy_bo.metrics import top_fraction_overlap


@dataclass(frozen=True)
class PairedBatchPanel:
    mean: torch.Tensor
    scale: torch.Tensor
    context: torch.Tensor
    pair_index: torch.Tensor


def paired_batch_panel(count: int = 128, seed: int = 40_405) -> PairedBatchPanel:
    base = SobolEngine(4, scramble=True, seed=seed).draw(count).double()
    mean = -0.25 + 0.5 * base[:, :2]
    scale = 0.8 + 0.4 * base[:, 2:]
    mean = mean.repeat_interleave(2, 0)
    scale = scale.repeat_interleave(2, 0)
    context = torch.tensor([0.0, 1.0], dtype=torch.double).repeat(count)
    pair_index = torch.arange(count, dtype=torch.long).repeat_interleave(2)
    return PairedBatchPanel(mean, scale, context, pair_index)


def maximizer_set(values: torch.Tensor) -> torch.Tensor:
    tolerance = 1e-12 * max(1.0, float(values.max()))
    return torch.where(values >= values.max() - tolerance)[0]


def tie_aware_decision_metrics(estimate: torch.Tensor, truth: torch.Tensor) -> dict[str, float | int]:
    estimate, truth = estimate.detach().double().cpu(), truth.detach().double().cpu()
    selected = maximizer_set(estimate)
    optimum = float(truth.max())
    selected_truth = truth[selected]
    rank = spearmanr(estimate.numpy(), truth.numpy()).statistic
    return {
        "maximizer_count": int(len(selected)),
        "tie_aware_regret": float((optimum - selected_truth.mean()) / max(optimum, 1e-15)),
        "optimistic_regret": float((optimum - selected_truth.max()) / max(optimum, 1e-15)),
        "pessimistic_regret": float((optimum - selected_truth.min()) / max(optimum, 1e-15)),
        "spearman": float(rank) if rank == rank else 0.0,
        "top10_overlap": float(top_fraction_overlap(truth.numpy(), estimate.numpy(), 0.10)),
    }


def paired_endpoint_metrics(estimate: torch.Tensor, truth: torch.Tensor) -> dict[str, float]:
    estimate = estimate.reshape(-1, 2).detach().double().cpu()
    truth = truth.reshape(-1, 2).detach().double().cpu()
    truth_delta = truth[:, 1] - truth[:, 0]
    estimate_delta = estimate[:, 1] - estimate[:, 0]
    contrast = truth_delta.abs() / truth.max(-1).values.clamp_min(1e-15)
    significant = contrast >= 0.01
    accuracy = (torch.sign(estimate_delta[significant]) == torch.sign(truth_delta[significant])).double()
    return {
        "oracle_significant_fraction": float(significant.double().mean()),
        "median_oracle_contrast": float(contrast.median()),
        "choice_accuracy": float(accuracy.mean()) if bool(significant.any()) else float("nan"),
    }
