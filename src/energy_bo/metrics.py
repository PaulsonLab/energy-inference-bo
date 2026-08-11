"""Small metric helpers shared by the oracle and GP validation experiments."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
from scipy.integrate import trapezoid
from scipy.stats import spearmanr


def normalized_grid(values: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Normalize a non-negative one-dimensional density evaluated on ``x``."""
    values = np.asarray(values, dtype=float)
    x = np.asarray(x, dtype=float)
    normalizer = float(trapezoid(values, x))
    if not math.isfinite(normalizer) or normalizer <= 0.0:
        raise ValueError("grid values must have a positive finite integral")
    return values / normalizer


def top_fraction_overlap(
    truth: np.ndarray, estimate: np.ndarray, fraction: float = 0.05
) -> float:
    """Fraction of the true top set recovered by the estimated top set."""
    truth = np.asarray(truth)
    estimate = np.asarray(estimate)
    if truth.shape != estimate.shape:
        raise ValueError("truth and estimate must have the same shape")
    k = max(1, int(math.ceil(fraction * truth.size)))
    truth_top = set(np.argpartition(truth, -k)[-k:])
    estimate_top = set(np.argpartition(estimate, -k)[-k:])
    return len(truth_top & estimate_top) / k


def ei_curve_metrics(
    truth: np.ndarray, estimate: np.ndarray, x: np.ndarray
) -> dict[str, float]:
    """Decision-focused comparisons for two acquisition curves."""
    truth = np.asarray(truth, dtype=float)
    estimate = np.asarray(estimate, dtype=float)
    x = np.asarray(x, dtype=float)
    high_mask = truth >= 0.9 * float(np.max(truth))
    denominator = np.maximum(truth[high_mask], 1e-12)
    correlation = spearmanr(truth, estimate).statistic
    return {
        "max_abs_ei_error": float(np.max(np.abs(truth - estimate))),
        "max_relative_error_high_ei": float(
            np.max(np.abs(truth[high_mask] - estimate[high_mask]) / denominator)
        ),
        "spearman_rank_correlation": float(correlation),
        "top_5pct_overlap": top_fraction_overlap(truth, estimate),
        "truth_argmax_x": float(x[int(np.argmax(truth))]),
        "estimate_argmax_x": float(x[int(np.argmax(estimate))]),
    }


def effective_sample_size(weights: Iterable[float]) -> float:
    """Return the standard importance-sampling effective sample size."""
    weights = np.asarray(list(weights), dtype=float)
    total = float(np.sum(weights))
    if total <= 0.0 or not math.isfinite(total):
        return 0.0
    normalized = weights / total
    return float(1.0 / np.sum(np.square(normalized)))


def histogram_target_masses(
    density: np.ndarray, x: np.ndarray, edges: np.ndarray
) -> np.ndarray:
    """Integrate a grid density into bins whose edges lie on the grid."""
    density = normalized_grid(density, x)
    masses = np.empty(len(edges) - 1, dtype=float)
    for index, (left, right) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        mask = (x >= left) & (x <= right)
        masses[index] = trapezoid(density[mask], x[mask])
    return masses / np.sum(masses)
