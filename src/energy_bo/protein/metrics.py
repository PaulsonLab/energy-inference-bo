"""Probabilistic and finite-pool decision metrics for Task 05A."""

from __future__ import annotations

import math

import numpy as np
import torch
from scipy.stats import rankdata


def gaussian_nll(y: torch.Tensor, mean: torch.Tensor, variance: torch.Tensor) -> torch.Tensor:
    variance = variance.clamp_min(1e-15)
    return 0.5 * (math.log(2 * math.pi) + variance.log() + (y - mean).square() / variance)


def gaussian_crps(y: torch.Tensor, mean: torch.Tensor, variance: torch.Tensor) -> torch.Tensor:
    scale = variance.clamp_min(1e-15).sqrt()
    z = (y - mean) / scale
    phi = torch.exp(-0.5 * z.square()) / math.sqrt(2 * math.pi)
    cdf = 0.5 * (1 + torch.erf(z / math.sqrt(2)))
    return scale * (z * (2 * cdf - 1) + 2 * phi - 1 / math.sqrt(math.pi))


def spearman(predicted: torch.Tensor, truth: torch.Tensor) -> float:
    left = rankdata(predicted.detach().cpu().numpy())
    right = rankdata(truth.detach().cpu().numpy())
    if np.std(left) == 0 or np.std(right) == 0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _top_mask(values: torch.Tensor, fraction: float) -> torch.Tensor:
    count = max(1, int(math.ceil(fraction * len(values))))
    indices = torch.topk(values, count, largest=True, sorted=False).indices
    mask = torch.zeros(len(values), dtype=torch.bool)
    mask[indices] = True
    return mask


def top_recall(predicted: torch.Tensor, truth: torch.Tensor, fraction: float = 0.1) -> float:
    predicted_mask = _top_mask(predicted, fraction)
    true_mask = _top_mask(truth, fraction)
    return float((predicted_mask & true_mask).sum() / true_mask.sum())


def interval_calibration(
    y: torch.Tensor,
    mean: torch.Tensor,
    variance: torch.Tensor,
    selection_mean: torch.Tensor,
    fraction: float = 0.1,
) -> dict[str, float]:
    high = _top_mask(selection_mean, fraction)
    standardized = ((y[high] - mean[high]) / variance[high].clamp_min(1e-15).sqrt()).abs()
    levels = (0.5, 0.8, 0.9, 0.95)
    # Two-sided standard Normal central-interval quantiles.
    normal = torch.distributions.Normal(torch.tensor(0.0, dtype=torch.double), torch.tensor(1.0, dtype=torch.double))
    coverages = {}
    errors = []
    for level in levels:
        cutoff = normal.icdf(torch.tensor((1 + level) / 2, dtype=torch.double))
        coverage = float((standardized <= cutoff).double().mean())
        coverages[f"coverage_{int(level * 100)}"] = coverage
        errors.append(abs(coverage - level))
    coverages["interval_calibration_error"] = float(np.mean(errors))
    coverages["high_utility_count"] = int(high.sum())
    return coverages


def constant_gaussian_nll(train_y: torch.Tensor, test_y: torch.Tensor, selection: torch.Tensor) -> float:
    mean = train_y.mean()
    variance = train_y.var(unbiased=False).clamp_min(1e-12)
    return float(gaussian_nll(test_y[selection], mean.expand_as(test_y[selection]), variance.expand_as(test_y[selection])).mean())


def normalized_regret(selected_value: float, global_best: float, initial_best: float) -> float:
    denominator = global_best - initial_best
    if denominator <= 1e-15:
        return 0.0
    return float(max(0.0, global_best - selected_value) / denominator)


def offline_metrics(
    train_y: torch.Tensor,
    test_y: torch.Tensor,
    mean: torch.Tensor,
    variance: torch.Tensor,
    latent_log_ei: torch.Tensor,
) -> dict[str, float | int | bool]:
    high = _top_mask(mean, 0.1)
    calibration = interval_calibration(test_y, mean, variance, mean)
    selected = int(torch.argmax(latent_log_ei))
    global_best = float(torch.maximum(train_y.max(), test_y.max()))
    initial_best = float(train_y.max())
    top_decile_threshold = float(torch.quantile(test_y, 0.9))
    result: dict[str, float | int | bool] = {
        "nll": float(gaussian_nll(test_y, mean, variance).mean()),
        "crps": float(gaussian_crps(test_y, mean, variance).mean()),
        "rmse": float(torch.mean((test_y - mean).square()).sqrt()),
        "spearman": spearman(mean, test_y),
        "top10_recall": top_recall(mean, test_y),
        "high_utility_nll": float(gaussian_nll(test_y[high], mean[high], variance[high]).mean()),
        "constant_high_utility_nll": constant_gaussian_nll(train_y, test_y, high),
        "one_step_regret": normalized_regret(float(test_y[selected]), global_best, initial_best),
        "selected_top_decile": bool(float(test_y[selected]) >= top_decile_threshold),
        "selected_test_index": selected,
    }
    result.update(calibration)
    return result


def trajectory_metrics(values: list[float], global_best: float) -> dict[str, float | bool]:
    initial_best = values[0]
    denominator = global_best - initial_best
    regrets = [0.0 if denominator <= 1e-15 else max(0.0, global_best - value) / denominator for value in values]
    return {
        "regret_auc": float(np.trapz(regrets, dx=1.0) / max(len(regrets) - 1, 1)),
        "final_regret": float(regrets[-1]),
        "initial_best": float(initial_best),
        "final_best": float(values[-1]),
        "global_best": float(global_best),
    }
