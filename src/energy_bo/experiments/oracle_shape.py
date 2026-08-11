"""Part A/B: oracle shape value and regularized residual-energy learning."""

from __future__ import annotations

import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("artifacts/.matplotlib")))

import matplotlib
import numpy as np
import torch
from scipy.special import ndtr

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from energy_bo.metrics import ei_curve_metrics
from energy_bo.oracle.acquisition import (
    gaussian_ei_curve,
    oracle_ei_curve,
    residual_ei_curve,
)
from energy_bo.oracle.distributions import (
    ORACLE_MIXTURE,
    STANDARD_NORMAL,
    kl_true_to_model,
    normal_ei,
)
from energy_bo.oracle.residual_energy import RBFResidualEnergy
from energy_bo.oracle.scenarios import PREDETERMINED_SCENARIOS, scenario_grid


def _normal_pdf(z: np.ndarray | float, mean: float, scale: float) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    return np.exp(-0.5 * np.square((z - mean) / scale)) / (scale * np.sqrt(2.0 * np.pi))


def _save_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=sorted({key for row in rows for key in row}),
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def oracle_shape_rows(grid_points: int = 1001) -> list[dict[str, object]]:
    """Compare known mixture and Gaussian EI on the fixed scenarios."""
    x = scenario_grid(grid_points)
    rows: list[dict[str, object]] = []
    for scenario in PREDETERMINED_SCENARIOS:
        oracle = oracle_ei_curve(ORACLE_MIXTURE, scenario, x)
        gaussian = gaussian_ei_curve(scenario, x)
        quadrature_errors = []
        for index in np.linspace(0, grid_points - 1, 21, dtype=int):
            value, error = ORACLE_MIXTURE.expected_improvement_quadrature(
                float(scenario.mean(x[[index]])[0]),
                float(scenario.scale(x[[index]])[0]),
                scenario.best_f,
            )
            quadrature_errors.append(abs(value - oracle[index]))
            quadrature_errors.append(error)
        row: dict[str, object] = {
            "record_type": "oracle_gaussian_comparison",
            "scenario": scenario.name,
            "description": scenario.description,
            "quadrature_max_abs_discrepancy": max(quadrature_errors),
        }
        row.update(ei_curve_metrics(oracle, gaussian, x))
        rows.append(row)
    return rows


def _plot_oracle_curves(output_dir: Path, grid_points: int) -> None:
    x = scenario_grid(grid_points)
    figure, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    for axis, scenario in zip(axes, PREDETERMINED_SCENARIOS, strict=True):
        axis.plot(x, oracle_ei_curve(ORACLE_MIXTURE, scenario, x), label="Oracle mixture EI")
        axis.plot(x, gaussian_ei_curve(scenario, x), "--", label="Moment-matched Gaussian EI")
        axis.set_ylabel("EI")
        axis.set_title(scenario.name.replace("_", " "))
        axis.grid(alpha=0.25)
    axes[0].legend(loc="best")
    axes[-1].set_xlabel("x")
    figure.tight_layout()
    figure.savefig(output_dir / "oracle_true_vs_gaussian_ei.png", dpi=180)
    plt.close(figure)


def run_oracle_shape_experiment(
    output_dir: Path,
    seeds: tuple[int, ...] = (0, 1, 2),
    sample_sizes: tuple[int, ...] = (20, 50, 100, 200, 500),
    grid_points: int = 1001,
    heldout_count: int = 20_000,
) -> list[dict[str, object]]:
    """Fit every requested sample-size/seed combination and persist transparent rows."""
    output_dir.mkdir(parents=True, exist_ok=True)
    _plot_oracle_curves(output_dir, grid_points)
    x = scenario_grid(grid_points)
    heldout_generator = torch.Generator(device="cpu").manual_seed(991)
    heldout = ORACLE_MIXTURE.sample(heldout_count, heldout_generator).numpy()
    truth_tail = {threshold: ORACLE_MIXTURE.tail_probability(threshold) for threshold in (1.0, 1.5, 2.0)}
    rows: list[dict[str, object]] = []

    for sample_size in sample_sizes:
        for seed in seeds:
            generator = torch.Generator(device="cpu").manual_seed(seed)
            samples = ORACLE_MIXTURE.sample(sample_size, generator)
            sample_numpy = samples.numpy()
            mle_mean = float(np.mean(sample_numpy))
            mle_scale = float(np.sqrt(np.mean(np.square(sample_numpy - mle_mean))))
            residual = RBFResidualEnergy()
            fit = residual.fit(samples)
            residual_mean, residual_variance = residual.moments()

            model_specs = (
                (
                    "fixed_standard_normal",
                    lambda z: _normal_pdf(z, 0.0, 1.0),
                    0.0,
                    1.0,
                    lambda scenario: gaussian_ei_curve(scenario, x),
                ),
                (
                    "gaussian_mle",
                    lambda z: _normal_pdf(z, mle_mean, mle_scale),
                    mle_mean,
                    mle_scale**2,
                    lambda scenario: normal_ei(
                        scenario.mean(x) + scenario.scale(x) * mle_mean,
                        scenario.scale(x) * mle_scale,
                        scenario.best_f,
                    ),
                ),
                (
                    "residual_energy",
                    lambda z: residual.density_numpy(z),
                    residual_mean,
                    residual_variance,
                    lambda scenario: residual_ei_curve(residual, scenario, x),
                ),
            )

            for model_name, pdf, fitted_mean, fitted_variance, acquisition in model_specs:
                log_score = float(np.mean(np.log(np.maximum(pdf(heldout), 1e-300))))
                kl_value, kl_error = kl_true_to_model(ORACLE_MIXTURE, pdf)
                if model_name == "fixed_standard_normal":
                    model_tail = {threshold: STANDARD_NORMAL.tail_probability(threshold) for threshold in truth_tail}
                elif model_name == "gaussian_mle":
                    model_tail = {
                        threshold: float(1.0 - ndtr((threshold - mle_mean) / mle_scale))
                        for threshold in truth_tail
                    }
                else:
                    model_tail = {threshold: residual.tail_probability(threshold) for threshold in truth_tail}
                base_row: dict[str, object] = {
                    "record_type": "residual_model_metric",
                    "sample_size": sample_size,
                    "seed": seed,
                    "model": model_name,
                    "heldout_log_score": log_score,
                    "kl_true_to_model": kl_value,
                    "kl_quadrature_error_bound": kl_error,
                    "fitted_mean": fitted_mean,
                    "fitted_variance": fitted_variance,
                    "fit_nll": fit.nll if model_name == "residual_energy" else "",
                    "fit_penalty": fit.penalty if model_name == "residual_energy" else "",
                    "fit_lbfgs_closures": fit.iterations if model_name == "residual_energy" else "",
                }
                base_row.update(
                    {
                        f"tail_error_gt_{threshold:g}": model_tail[threshold] - truth_tail[threshold]
                        for threshold in truth_tail
                    }
                )
                rows.append(base_row)

                for scenario in PREDETERMINED_SCENARIOS:
                    truth_ei = oracle_ei_curve(ORACLE_MIXTURE, scenario, x)
                    acquisition_metrics: dict[str, object] = {
                        "record_type": "residual_ei_metric",
                        "sample_size": sample_size,
                        "seed": seed,
                        "model": model_name,
                        "scenario": scenario.name,
                    }
                    acquisition_metrics.update(ei_curve_metrics(truth_ei, acquisition(scenario), x))
                    rows.append(acquisition_metrics)

    _save_csv(output_dir / "oracle_shape_metrics.csv", rows)
    return rows
