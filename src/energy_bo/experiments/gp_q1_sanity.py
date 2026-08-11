"""Part D: report the exact-GP q=1 EI/augmented-marginal check."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from energy_bo.gp.exact_gp import run_gp_q1_sanity
from energy_bo.metrics import normalized_grid


def run_gp_q1_sanity_experiment(
    output_dir: Path, seed: int = 0, grid_points: int = 1001
) -> list[dict[str, object]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    result = run_gp_q1_sanity(seed=seed, grid_points=grid_points)
    manual_marginal = normalized_grid(result.manual_ei, result.x)
    rows = [
        {
            "record_type": "gp_q1_sanity",
            "seed": seed,
            "grid_points": grid_points,
            "best_f": result.best_f,
            "max_abs_manual_vs_quadrature_ei": float(
                np.max(np.abs(result.manual_ei - result.quadrature_ei))
            ),
            "max_abs_botorch_vs_manual_ei": float(
                np.max(np.abs(result.botorch_ei - result.manual_ei))
            ),
            "max_abs_augmented_vs_ei_marginal": float(
                np.max(np.abs(result.augmented_marginal - manual_marginal))
            ),
            "max_abs_m2_vs_ei_squared_marginal": float(
                np.max(
                    np.abs(
                        result.ei_squared_marginal
                        - normalized_grid(np.square(result.manual_ei), result.x)
                    )
                )
            ),
            "quadrature_error_bound": result.quadrature_error_bound,
            "manual_ei_argmax": result.manual_ei_argmax,
            "botorch_ei_argmax": result.botorch_ei_argmax,
            "log_ei_argmax": result.log_ei_argmax,
            "augmented_argmax": result.augmented_argmax,
            "m2_argmax": result.squared_argmax,
        }
    ]
    with (output_dir / "gp_q1_sanity_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows
