"""Part C: numerical augmented expected-utility and replicated-power checks."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy.integrate import trapezoid

from energy_bo.oracle.acquisition import oracle_ei_curve
from energy_bo.oracle.augmentation import (
    augmented_grid_marginal,
    importance_sample_augmented_oracle,
    oracle_augmented_marginal_by_quadrature,
)
from energy_bo.oracle.distributions import ORACLE_MIXTURE
from energy_bo.oracle.scenarios import TAIL_SENSITIVE, scenario_grid


def _save_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=sorted({key for row in rows for key in row}),
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def run_augmented_inference_experiment(
    output_dir: Path,
    seed: int = 17,
    grid_points: int = 1001,
    particle_counts: tuple[int, ...] = (2_000, 10_000, 30_000),
) -> list[dict[str, object]]:
    """Run direct identity and intentionally simple raw-weight IS checks."""
    output_dir.mkdir(parents=True, exist_ok=True)
    x = scenario_grid(grid_points)
    analytic_ei = oracle_ei_curve(ORACLE_MIXTURE, TAIL_SENSITIVE, x)
    rows: list[dict[str, object]] = []
    for replicas in (1, 2, 4):
        direct_marginal, quadrature_error = oracle_augmented_marginal_by_quadrature(
            ORACLE_MIXTURE, TAIL_SENSITIVE, x, replicas
        )
        predicted_marginal = augmented_grid_marginal(analytic_ei, x, replicas)
        rows.append(
            {
                "record_type": "augmented_grid_identity",
                "replicas": replicas,
                "max_abs_normalized_marginal_error": float(
                    np.max(np.abs(direct_marginal - predicted_marginal))
                ),
                "l1_normalized_marginal_error": float(
                    trapezoid(np.abs(direct_marginal - predicted_marginal), x)
                ),
                "direct_marginal_mode": float(x[int(np.argmax(direct_marginal))]),
                "predicted_mode": float(x[int(np.argmax(predicted_marginal))]),
                "largest_quadrature_error_bound": quadrature_error,
            }
        )
        for particle_count in particle_counts:
            result = importance_sample_augmented_oracle(
                ORACLE_MIXTURE,
                TAIL_SENSITIVE,
                x,
                replicas=replicas,
                particle_count=particle_count,
                seed=seed + 100 * replicas + particle_count,
            )
            rows.append(
                {
                    "record_type": "augmented_importance_sampling",
                    "replicas": result.replicas,
                    "particle_count": result.particle_count,
                    "l1_marginal_error": result.l1_marginal_error,
                    "max_bin_error": result.max_bin_error,
                    "inferred_mode": result.inferred_mode,
                    "exact_mode": result.exact_mode,
                    "effective_sample_size": result.effective_sample_size,
                    "effective_sample_fraction": result.effective_sample_fraction,
                    "nonzero_weight_fraction": result.nonzero_weight_fraction,
                }
            )
    _save_csv(output_dir / "augmented_inference_metrics.csv", rows)
    return rows
