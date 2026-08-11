"""Finite-candidate decision metrics and low-rank diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from scipy.stats import spearmanr

from energy_bo.metrics import top_fraction_overlap


def _vector(value: torch.Tensor, name: str) -> torch.Tensor:
    result = value.detach().to(dtype=torch.double, device="cpu").reshape(-1)
    if result.numel() < 1 or not torch.isfinite(result).all():
        raise ValueError(f"{name} must be a finite nonempty vector")
    return result


def acquisition_metrics(
    teacher: torch.Tensor,
    approximate: torch.Tensor,
    *,
    epsilon: float = 1e-15,
) -> dict[str, float | int | bool]:
    """Compare two acquisition vectors and check the finite-candidate regret bound."""

    teacher = _vector(teacher, "teacher")
    approximate = _vector(approximate, "approximate")
    if teacher.shape != approximate.shape:
        raise ValueError("teacher and approximate acquisitions must have equal shape")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    teacher_index = int(torch.argmax(teacher))
    approximate_index = int(torch.argmax(approximate))
    peak = float(teacher[teacher_index])
    absolute_regret = float(teacher[teacher_index] - teacher[approximate_index])
    denominator = max(abs(peak), epsilon)
    error = (teacher - approximate).abs()
    delta = float(error.max())
    bound_slack = 2.0 * delta - absolute_regret
    teacher_np = teacher.numpy()
    approximate_np = approximate.numpy()
    rank = spearmanr(teacher_np, approximate_np).statistic
    return {
        "teacher_index": teacher_index,
        "approximate_index": approximate_index,
        "exact_index_agreement": teacher_index == approximate_index,
        "absolute_decision_regret": absolute_regret,
        "normalized_decision_regret": absolute_regret / denominator,
        "max_absolute_error": delta,
        "normalized_max_error": delta / denominator,
        "rmse": float(torch.mean((teacher - approximate).square()).sqrt()),
        "spearman": float(rank) if np.isfinite(rank) else 1.0,
        "top5_overlap": float(top_fraction_overlap(teacher_np, approximate_np, fraction=0.05)),
        "twice_delta": 2.0 * delta,
        "regret_bound_slack": bound_slack,
        "regret_bound_pass": bound_slack >= -1e-12,
    }


def signature_spectrum(
    matrix: torch.Tensor,
    *,
    column_standardize: bool = False,
) -> dict[str, Any]:
    """Center a particle-by-feature matrix and summarize its squared singular spectrum.

    Acquisition signatures use one global RMS scale so candidate importance is not
    altered. Structural coordinates use per-column population standardization.
    """

    matrix = matrix.detach().to(dtype=torch.double, device="cpu")
    if matrix.ndim != 2 or min(matrix.shape) < 1 or not torch.isfinite(matrix).all():
        raise ValueError("matrix must be a finite nonempty matrix")
    centered = matrix - matrix.mean(dim=0, keepdim=True)
    if column_standardize:
        scale = centered.square().mean(dim=0).sqrt().clamp_min(1e-15)
        centered = centered / scale
        scaling = "column_population_standard_deviation"
    else:
        scale = centered.square().mean().sqrt().clamp_min(1e-15)
        centered = centered / scale
        scaling = "global_centered_rms"
    singular_values = torch.linalg.svdvals(centered)
    squared = singular_values.square()
    total = squared.sum()
    if float(total) == 0.0:
        fractions = torch.zeros_like(squared)
        cumulative = torch.zeros_like(squared)
        entropy_rank = 0.0
        stable_rank = 0.0
    else:
        fractions = squared / total
        cumulative = fractions.cumsum(dim=0)
        positive = fractions[fractions > 0]
        entropy_rank = float(torch.exp(-(positive * positive.log()).sum()))
        stable_rank = float(total / squared[0])

    def rank_at(threshold: float) -> int:
        if float(total) == 0.0:
            return 0
        return int(torch.searchsorted(cumulative, torch.tensor(threshold, dtype=torch.double))) + 1

    return {
        "scaling": scaling,
        "rows": matrix.shape[0],
        "columns": matrix.shape[1],
        "singular_values": [float(value) for value in singular_values],
        "explained_squared_fraction": [float(value) for value in fractions],
        "cumulative_explained_squared_fraction": [float(value) for value in cumulative],
        "entropy_effective_rank": entropy_rank,
        "stable_rank": stable_rank,
        "rank90": rank_at(0.90),
        "rank95": rank_at(0.95),
        "rank99": rank_at(0.99),
    }
