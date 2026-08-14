"""Teacher-free decision panels for the Task 04A-D local diagnostic."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch.quasirandom import SobolEngine


@dataclass(frozen=True)
class CandidateSets:
    construction: torch.Tensor
    verification: torch.Tensor


@dataclass(frozen=True)
class CounterfactualPanel:
    base_indices: torch.Tensor
    low_summary: torch.Tensor
    high_summary: torch.Tensor


def active_candidate_sets(dimension: int, seed: int, *, sobol_count: int = 4096, grid_size: int = 65) -> CandidateSets:
    """Sobol plus active grid construction set and an independent Sobol verifier."""
    if dimension < 2:
        raise ValueError("dimension must be at least two")

    def embed(active: torch.Tensor) -> torch.Tensor:
        result = torch.full((len(active), dimension), 0.5, dtype=torch.double)
        result[:, :2] = active.double()
        return result

    sobol = SobolEngine(2, scramble=True, seed=30_000 + seed).draw(sobol_count)
    axis = torch.linspace(0.0, 1.0, grid_size, dtype=torch.double)
    grid = torch.cartesian_prod(axis, axis)
    verification = SobolEngine(2, scramble=True, seed=40_000 + seed).draw(sobol_count)
    return CandidateSets(torch.cat((embed(sobol), embed(grid))), embed(verification))


def context_scores(summary: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    return summary.double() @ direction.double().to(summary.device)


def select_real_pairs(
    g0_ei: torch.Tensor,
    summary: torch.Tensor,
    direction: torch.Tensor,
    *,
    maximum_pairs: int = 64,
    minimum_fraction: float = 0.2,
    relative_tolerance: float = 0.005,
) -> torch.Tensor:
    """Greedily pair low/high contexts using reference quantities only."""
    g0 = g0_ei.detach().double().cpu()
    scores = context_scores(summary.detach().double().cpu(), direction.detach().double().cpu())
    if g0.ndim != 1 or len(g0) != len(scores):
        raise ValueError("g0_ei and summary must have matching candidate counts")
    eligible = g0 >= minimum_fraction * g0.max()
    low_cut, high_cut = torch.quantile(scores[eligible], torch.tensor([0.1, 0.9], dtype=torch.double))
    low = torch.where(eligible & (scores <= low_cut))[0]
    high = torch.where(eligible & (scores >= high_cut))[0]
    proposals: list[tuple[float, float, int, int]] = []
    high_values = g0[high]
    order = torch.argsort(high_values, stable=True)
    high, high_values = high[order], high_values[order]
    for low_index in low.tolist():
        position = int(torch.searchsorted(high_values, g0[low_index]))
        for candidate_position in range(max(0, position - 2), min(len(high), position + 3)):
            high_index = int(high[candidate_position])
            denominator = max(float(g0[low_index]), float(g0[high_index]), 1e-15)
            relative = abs(float(g0[low_index] - g0[high_index])) / denominator
            gap = float(scores[high_index] - scores[low_index])
            if relative <= relative_tolerance and gap >= float(high_cut - low_cut):
                proposals.append((-gap, relative, low_index, high_index))
    proposals.sort()
    used: set[int] = set()
    selected: list[tuple[int, int]] = []
    for _, _, low_index, high_index in proposals:
        if low_index in used or high_index in used:
            continue
        selected.append((low_index, high_index))
        used.update((low_index, high_index))
        if len(selected) == maximum_pairs:
            break
    return torch.tensor(selected, dtype=torch.long).reshape(-1, 2)


def build_counterfactual_panel(
    g0_ei: torch.Tensor,
    summary: torch.Tensor,
    direction: torch.Tensor,
    *,
    count: int = 32,
) -> CounterfactualPanel:
    """Hold base mean/scale fixed while swapping observed low/high summaries."""
    g0 = g0_ei.detach().double().cpu()
    summaries = summary.detach().double().cpu()
    scores = context_scores(summaries, direction.detach().double().cpu())
    top = torch.where(g0 >= torch.quantile(g0, 0.8))[0]
    top = top[torch.argsort(g0[top], stable=True)]
    positions = torch.linspace(0, len(top) - 1, count, dtype=torch.double).round().long()
    base = top[positions]
    low_value, high_value = torch.quantile(scores, torch.tensor([0.1, 0.9], dtype=torch.double))
    low_index = int(torch.argmin((scores - low_value).abs()))
    high_index = int(torch.argmin((scores - high_value).abs()))
    return CounterfactualPanel(base, summaries[low_index].clone(), summaries[high_index].clone())


def pairwise_metrics(model_ei: torch.Tensor, oracle_ei: torch.Tensor, pairs: torch.Tensor) -> dict[str, float | int]:
    model, oracle, pairs = model_ei.detach().double().cpu(), oracle_ei.detach().double().cpu(), pairs.long().cpu()
    if len(pairs) == 0:
        return {"pair_count": 0, "choice_accuracy": math.nan, "margin_weighted_regret": math.nan, "median_normalized_contrast": math.nan}
    model_delta = model[pairs[:, 1]] - model[pairs[:, 0]]
    oracle_delta = oracle[pairs[:, 1]] - oracle[pairs[:, 0]]
    non_tie = oracle_delta.abs() > 1e-15
    correct = torch.sign(model_delta[non_tie]) == torch.sign(oracle_delta[non_tie])
    denominator = torch.maximum(oracle[pairs[:, 0]], oracle[pairs[:, 1]]).clamp_min(1e-15)
    normalized = oracle_delta.abs() / denominator
    total_margin = oracle_delta.abs().sum().clamp_min(1e-15)
    wrong_margin = oracle_delta.abs()[torch.sign(model_delta) != torch.sign(oracle_delta)].sum()
    return {
        "pair_count": len(pairs),
        "choice_accuracy": float(correct.double().mean()) if bool(non_tie.any()) else 1.0,
        "margin_weighted_regret": float(wrong_margin / total_margin),
        "median_normalized_contrast": float(normalized.median()),
    }


def counterfactual_metrics(low: torch.Tensor, high: torch.Tensor, oracle_low: torch.Tensor, oracle_high: torch.Tensor) -> dict[str, float]:
    estimate_delta = high.detach().double().cpu() - low.detach().double().cpu()
    oracle_low, oracle_high = oracle_low.detach().double().cpu(), oracle_high.detach().double().cpu()
    oracle_delta = oracle_high - oracle_low
    normalized = oracle_delta.abs() / torch.maximum(oracle_low, oracle_high).clamp_min(1e-15)
    significant = normalized >= 0.01
    sign_accuracy = (torch.sign(estimate_delta[significant]) == torch.sign(oracle_delta[significant])).double()
    relative_error = (estimate_delta[significant] - oracle_delta[significant]).abs() / oracle_delta[significant].abs().clamp_min(1e-15)
    return {
        "oracle_significant_fraction": float(significant.double().mean()),
        "sign_accuracy": float(sign_accuracy.mean()) if bool(significant.any()) else math.nan,
        "median_relative_contrast_error": float(relative_error.median()) if bool(significant.any()) else math.nan,
        "median_normalized_oracle_contrast": float(normalized.median()),
    }


def within_one_percent(model_ei: torch.Tensor, oracle_ei: torch.Tensor) -> bool:
    chosen = int(torch.argmax(model_ei))
    return bool(oracle_ei[chosen] >= 0.99 * oracle_ei.max())
