"""Run the narrow T2-B reflection-symmetry EI non-vacuity validation.

This is not a full E1 reproduction and not an end-to-end rigorous certificate.
The structural bound is analytic; the active and full acquisition calculations
are self-normalized importance-sampling diagnostics on fixed action grids.
"""

from __future__ import annotations

import csv
import gc
import hashlib
import json
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from conditioned_bo.symmetry_influence import (
    conditional_expected_improvement,
    ei_action_coefficients,
    ei_block_decision_footprint,
    omitted_factor_load,
    ou_symmetry_comparison,
    ranked_omitted_contributions,
    row_dominance_margins,
    solve_comparison,
)


OUTPUT_DIRECTORY = Path(__file__).resolve().parent / "outputs" / "t2b_ei_validation"

FROZEN_CONFIG: dict[str, Any] = {
    "model": {
        "n_factors": 40,
        "spacing": 0.05,
        "lengthscale": 0.125,
        "gamma": 0.05,
        "tau": 0.5,
        "mean": "0.36 exp(-0.5 ((x+0.22)/0.095)^2) + 0.31 exp(-0.5 ((x-0.25)/0.18)^2)",
    },
    "decision": {
        "incumbent": 0.50,
        "action_min": -0.58,
        "action_max": -0.06,
        "screen_action_count": 401,
        "refined_action_count": 1601,
        "epsilon_ei": 0.01,
        "nonvacuity_minimum_omitted_fraction": 0.20,
    },
    "screening": {
        "seed": 123,
        "reference_samples": 80_000,
        "activation_batch_size": 3,
        "mc_diagnostic_batches": 8,
    },
    "heldout_validation": {
        "seeds": list(range(1000, 1008)),
        "reference_samples_per_seed": 20_000,
    },
    "interpretation": {
        "epsilon_rationale": (
            "A prospective absolute EI tolerance, fixed in code before any "
            "full-target evaluation; it is not the prototype's 0.03 "
            "log-acquisition tolerance."
        ),
        "incumbent_rationale": (
            "Fixed modestly above the maximum of the archived reference mean."
        ),
        "validation_scope": (
            "High-accuracy empirical held-out validation only; no rigorous "
            "finite-sample inference or continuous-action certificate."
        ),
    },
}


def reference_mean(x: np.ndarray) -> np.ndarray:
    """Archived two-peak reference mean."""

    values = np.asarray(x, dtype=float)
    return 0.36 * np.exp(-0.5 * ((values + 0.22) / 0.095) ** 2) + 0.31 * np.exp(
        -0.5 * ((values - 0.25) / 0.18) ** 2
    )


def latent_grid(n_factors: int, spacing: float) -> np.ndarray:
    radii = (np.arange(n_factors) + 0.5) * spacing
    return np.concatenate((-radii[::-1], radii))


def sample_ou_reference(
    seed: int,
    n_samples: int,
    points: np.ndarray,
    spacing: float,
    lengthscale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample the exact no-jitter AR(1) reference in spatial order."""

    rng = np.random.default_rng(seed)
    q = float(np.exp(-spacing / lengthscale))
    innovations = rng.standard_normal((n_samples, points.size))
    deviations = innovations
    innovation_scale = float(np.sqrt(1.0 - q * q))
    for column in range(1, points.size):
        deviations[:, column] = (
            q * deviations[:, column - 1]
            + innovation_scale * innovations[:, column]
        )
    mean = reference_mean(points)
    return mean + deviations, mean


class LazySymmetryFactors:
    """Evaluate only activated factor vectors until held-out validation."""

    def __init__(
        self,
        samples: np.ndarray,
        blocks: np.ndarray,
        gamma: float,
        tau: float,
    ) -> None:
        self.samples = samples
        self.blocks = blocks
        self.gamma = gamma
        self.tau = tau
        self._cache: dict[int, np.ndarray] = {}

    @property
    def evaluated_indices(self) -> tuple[int, ...]:
        return tuple(self._cache)

    def evaluate(self, factor_index: int) -> np.ndarray:
        if factor_index not in self._cache:
            left, right = self.blocks[factor_index]
            scaled_difference = (
                self.samples[:, right] - self.samples[:, left]
            ) / self.tau
            self._cache[factor_index] = self.gamma * (
                np.logaddexp(scaled_difference, -scaled_difference) - np.log(2.0)
            )
        return self._cache[factor_index]

    def log_weights(self, active: list[int]) -> np.ndarray:
        result = np.zeros(self.samples.shape[0], dtype=float)
        for factor_index in active:
            result -= self.evaluate(factor_index)
        return result


@dataclass(frozen=True)
class ActionConditionals:
    actions: np.ndarray
    coefficients: np.ndarray
    variances: np.ndarray
    means: np.ndarray


def build_action_conditionals(
    actions: np.ndarray,
    points: np.ndarray,
    lengthscale: float,
    precision: scipy.sparse.spmatrix,
) -> ActionConditionals:
    coefficients = np.empty((actions.size, points.size), dtype=float)
    variances = np.empty(actions.size, dtype=float)
    for index, action in enumerate(actions):
        coefficients[index], variances[index] = ei_action_coefficients(
            float(action), points, lengthscale, precision
        )
    if np.count_nonzero(np.abs(coefficients) > 2e-13, axis=1).max() > 2:
        raise RuntimeError("OU Markov coefficients unexpectedly lost local support")
    return ActionConditionals(
        actions=actions,
        coefficients=coefficients,
        variances=variances,
        means=reference_mean(actions),
    )


def conditional_ei_matrix(
    samples: np.ndarray,
    latent_mean: np.ndarray,
    conditionals: ActionConditionals,
    incumbent: float,
) -> np.ndarray:
    centered = samples - latent_mean
    values = np.empty((samples.shape[0], conditionals.actions.size), dtype=float)
    for action_index in range(conditionals.actions.size):
        coefficients = conditionals.coefficients[action_index]
        support = np.flatnonzero(np.abs(coefficients) > 2e-13)
        conditional_mean = conditionals.means[action_index] + (
            centered[:, support] @ coefficients[support]
        )
        values[:, action_index] = conditional_expected_improvement(
            conditional_mean,
            float(conditionals.variances[action_index]),
            incumbent,
        )
    return values


def normalized_weights(log_weights: np.ndarray) -> np.ndarray:
    shifted = log_weights - float(log_weights.max())
    weights = np.exp(shifted)
    return weights / weights.sum()


def effective_sample_size(weights: np.ndarray) -> float:
    normalized = weights / weights.sum()
    return min(float(weights.size), float(1.0 / np.square(normalized).sum()))


def acquisition_curve(
    ei_values: np.ndarray,
    normalized_importance_weights: np.ndarray,
) -> np.ndarray:
    return np.asarray(normalized_importance_weights @ ei_values, dtype=float)


def envelope_for_curve(
    acquisition: np.ndarray,
    conditionals: ActionConditionals,
    leader_index: int,
    blocks: np.ndarray,
    transported_omitted_load: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gaps = acquisition - acquisition[leader_index]
    bounds = np.empty(acquisition.size, dtype=float)
    leader_coefficients = conditionals.coefficients[leader_index]
    for action_index, coefficients in enumerate(conditionals.coefficients):
        footprint = ei_block_decision_footprint(
            coefficients, leader_coefficients, blocks
        )
        bounds[action_index] = float(footprint @ transported_omitted_load)
    # The self-gap is identically zero; the generic max-of-three footprint is
    # conservative but needlessly nonzero for this special comparison.
    bounds[leader_index] = 0.0
    envelope = gaps + bounds
    return gaps, bounds, envelope


def weighted_ei_curve_streaming(
    samples: np.ndarray,
    latent_mean: np.ndarray,
    conditionals: ActionConditionals,
    incumbent: float,
    weights: np.ndarray,
) -> np.ndarray:
    centered = samples - latent_mean
    acquisition = np.empty(conditionals.actions.size, dtype=float)
    for action_index in range(conditionals.actions.size):
        coefficients = conditionals.coefficients[action_index]
        support = np.flatnonzero(np.abs(coefficients) > 2e-13)
        conditional_mean = conditionals.means[action_index] + (
            centered[:, support] @ coefficients[support]
        )
        values = conditional_expected_improvement(
            conditional_mean,
            float(conditionals.variances[action_index]),
            incumbent,
        )
        acquisition[action_index] = float(weights @ values)
    return acquisition


def write_frozen_config() -> str:
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    config_path = OUTPUT_DIRECTORY / "frozen_config.json"
    serialized = json.dumps(FROZEN_CONFIG, indent=2, sort_keys=True) + "\n"
    if config_path.exists() and config_path.read_text() != serialized:
        raise RuntimeError("frozen_config.json differs from the prospective config")
    config_path.write_text(serialized)
    return hashlib.sha256(serialized.encode()).hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty result table")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def run() -> dict[str, Any]:
    start_time = time.perf_counter()
    config_hash = write_frozen_config()
    model = FROZEN_CONFIG["model"]
    decision = FROZEN_CONFIG["decision"]
    screening = FROZEN_CONFIG["screening"]
    validation = FROZEN_CONFIG["heldout_validation"]

    n_factors = int(model["n_factors"])
    spacing = float(model["spacing"])
    lengthscale = float(model["lengthscale"])
    gamma = float(model["gamma"])
    tau = float(model["tau"])
    incumbent = float(decision["incumbent"])
    epsilon = float(decision["epsilon_ei"])

    precision, blocks, rho, kappa, matrix = ou_symmetry_comparison(
        n_factors, spacing, lengthscale
    )
    dense_matrix = matrix.toarray()
    margins = row_dominance_margins(matrix)
    eigenvalues = np.linalg.eigvalsh(dense_matrix)
    solve_probe = np.linspace(0.1, 1.0, n_factors)
    solve_solution = solve_comparison(matrix, solve_probe)
    solve_residual = float(np.linalg.norm(dense_matrix @ solve_solution - solve_probe))

    actions = np.linspace(
        float(decision["action_min"]),
        float(decision["action_max"]),
        int(decision["screen_action_count"]),
    )
    points = latent_grid(n_factors, spacing)
    conditionals = build_action_conditionals(
        actions, points, lengthscale, precision
    )

    samples, latent_mean = sample_ou_reference(
        int(screening["seed"]),
        int(screening["reference_samples"]),
        points,
        spacing,
        lengthscale,
    )
    factor_cache = LazySymmetryFactors(samples, blocks, gamma, tau)
    ei_values = conditional_ei_matrix(samples, latent_mean, conditionals, incumbent)

    active: list[int] = []
    history: list[dict[str, Any]] = []
    final_weights: np.ndarray | None = None
    final_leader_index: int | None = None
    final_challenger_index: int | None = None
    final_curve: np.ndarray | None = None

    for round_index in range(n_factors + 1):
        active_mask = np.zeros(n_factors, dtype=bool)
        active_mask[active] = True
        log_weights = factor_cache.log_weights(active)
        weights = normalized_weights(log_weights)
        curve = acquisition_curve(ei_values, weights)
        leader_index = int(np.argmax(curve))

        omitted_load = omitted_factor_load(active_mask, gamma, tau)
        transported_load = solve_comparison(matrix, omitted_load)
        gaps, bounds, envelope = envelope_for_curve(
            curve, conditionals, leader_index, blocks, transported_load
        )
        challenger_scores = envelope.copy()
        challenger_scores[leader_index] = -np.inf
        challenger_index = int(np.argmax(challenger_scores))
        envelope_value = float(envelope[challenger_index])
        stop = envelope_value <= epsilon or len(active) == n_factors

        new_factors: list[int] = []
        if not stop:
            challenger_footprint = ei_block_decision_footprint(
                conditionals.coefficients[challenger_index],
                conditionals.coefficients[leader_index],
                blocks,
            )
            scores = ranked_omitted_contributions(
                matrix, challenger_footprint, active_mask, gamma, tau
            )
            candidates = [
                int(index)
                for index in np.argsort(-scores, kind="stable")
                if np.isfinite(scores[index])
            ]
            new_factors = candidates[: int(screening["activation_batch_size"])]

        row = {
            "round": round_index,
            "active_count": len(active),
            "active_set": json.dumps(active),
            "leader_index": leader_index,
            "leader_action": float(actions[leader_index]),
            "challenger_index": challenger_index,
            "challenger_action": float(actions[challenger_index]),
            "sparse_ei": float(curve[leader_index]),
            "sparse_gap": float(gaps[challenger_index]),
            "structural_bound": float(bounds[challenger_index]),
            "optimistic_envelope": envelope_value,
            "epsilon_ei": epsilon,
            "certification_margin": epsilon - envelope_value,
            "active_ess_fraction": effective_sample_size(weights) / weights.size,
            "newly_activated_factors": json.dumps(new_factors),
            "factor_vectors_evaluated": len(factor_cache.evaluated_indices),
            "stopped": stop,
        }
        history.append(row)
        print(
            f"[EI SCREEN] round={round_index} active={len(active):2d} "
            f"leader={actions[leader_index]:.6f} "
            f"envelope={envelope_value:.6f}"
        )

        if stop:
            final_weights = weights
            final_leader_index = leader_index
            final_challenger_index = challenger_index
            final_curve = curve
            break
        active.extend(new_factors)

    if (
        final_weights is None
        or final_leader_index is None
        or final_challenger_index is None
        or final_curve is None
    ):
        raise RuntimeError("screening failed to produce a final decision")
    if set(factor_cache.evaluated_indices) != set(active):
        raise RuntimeError("an omitted factor was evaluated during screening")

    # Monte Carlo diagnostic for the final active target.  This is not an
    # adaptivity-safe finite-sample inference bound.
    batch_leaders: list[float] = []
    batch_gaps: list[float] = []
    batch_count = int(screening["mc_diagnostic_batches"])
    for sample_indices in np.array_split(np.arange(samples.shape[0]), batch_count):
        batch_log_weights = factor_cache.log_weights(active)[sample_indices]
        batch_weights = normalized_weights(batch_log_weights)
        batch_curve = acquisition_curve(ei_values[sample_indices], batch_weights)
        batch_leaders.append(float(actions[int(np.argmax(batch_curve))]))
        batch_gaps.append(
            float(
                batch_curve[final_challenger_index]
                - batch_curve[final_leader_index]
            )
        )

    # A fixed grid-refinement diagnostic uses the already frozen active target.
    refined_actions = np.linspace(
        float(decision["action_min"]),
        float(decision["action_max"]),
        int(decision["refined_action_count"]),
    )
    refined_conditionals = build_action_conditionals(
        refined_actions, points, lengthscale, precision
    )
    refined_curve = weighted_ei_curve_streaming(
        samples,
        latent_mean,
        refined_conditionals,
        incumbent,
        final_weights,
    )
    refined_leader_index = int(np.argmax(refined_curve))
    final_mask = np.zeros(n_factors, dtype=bool)
    final_mask[active] = True
    refined_transport = solve_comparison(
        matrix, omitted_factor_load(final_mask, gamma, tau)
    )
    _, _, refined_envelope = envelope_for_curve(
        refined_curve,
        refined_conditionals,
        refined_leader_index,
        blocks,
        refined_transport,
    )
    refined_scores = refined_envelope.copy()
    refined_scores[refined_leader_index] = -np.inf
    refined_challenger_index = int(np.argmax(refined_scores))

    # Only now evaluate all factors, on independent reference samples.
    validation_rows: list[dict[str, Any]] = []
    frozen_action = float(actions[final_leader_index])
    for replicate_index, seed in enumerate(validation["seeds"]):
        heldout_samples, heldout_mean = sample_ou_reference(
            int(seed),
            int(validation["reference_samples_per_seed"]),
            points,
            spacing,
            lengthscale,
        )
        full_factors = LazySymmetryFactors(
            heldout_samples, blocks, gamma, tau
        )
        full_log_weights = full_factors.log_weights(list(range(n_factors)))
        full_weights = normalized_weights(full_log_weights)
        heldout_ei = conditional_ei_matrix(
            heldout_samples, heldout_mean, conditionals, incumbent
        )
        full_curve = acquisition_curve(heldout_ei, full_weights)
        full_leader_index = int(np.argmax(full_curve))
        regret = float(full_curve[full_leader_index] - full_curve[final_leader_index])
        validation_rows.append(
            {
                "replicate": replicate_index,
                "seed": int(seed),
                "reference_samples": int(validation["reference_samples_per_seed"]),
                "full_leader_index": full_leader_index,
                "full_leader_action": float(actions[full_leader_index]),
                "screened_action_index": final_leader_index,
                "screened_action": frozen_action,
                "action_distance": abs(
                    float(actions[full_leader_index]) - frozen_action
                ),
                "exact_grid_action_match": full_leader_index == final_leader_index,
                "full_ei_regret": regret,
                "within_epsilon": regret <= epsilon,
                "full_ess_fraction": effective_sample_size(full_weights)
                / full_weights.size,
                "full_factor_vectors_evaluated_after_screening": len(
                    full_factors.evaluated_indices
                ),
            }
        )
        print(
            f"[EI VALIDATION] replicate={replicate_index + 1}/"
            f"{len(validation['seeds'])} leader={actions[full_leader_index]:.6f} "
            f"regret={regret:.6g}"
        )
        del heldout_samples, heldout_ei, full_curve, full_weights, full_factors
        gc.collect()

    omitted_count = n_factors - len(active)
    minimum_omitted = int(
        np.ceil(
            float(decision["nonvacuity_minimum_omitted_fraction"]) * n_factors
        )
    )
    stopped_within_tolerance = bool(history[-1]["stopped"])
    structurally_nonvacuous = stopped_within_tolerance and omitted_count >= minimum_omitted
    heldout_within_tolerance = all(row["within_epsilon"] for row in validation_rows)
    numerical_pass = structurally_nonvacuous and heldout_within_tolerance
    blocker_verdict = "PASS" if numerical_pass else "ANALYTIC PASS / NUMERICAL ISSUE"

    validation_regrets = np.array(
        [row["full_ei_regret"] for row in validation_rows]
    )
    validation_leaders = np.array(
        [row["full_leader_action"] for row in validation_rows]
    )
    validation_ess = np.array(
        [row["full_ess_fraction"] for row in validation_rows]
    )
    wall_time = time.perf_counter() - start_time

    summary: dict[str, Any] = {
        "blocker_verdict": blocker_verdict,
        "analytic_status": "PROVED_FOR_REFLECTION_SYMMETRY_UNDER_STATED_ASSUMPTIONS",
        "numerical_status": "NONVACUOUS" if numerical_pass else "NUMERICAL_ISSUE",
        "end_to_end_certificate": False,
        "config_sha256": config_hash,
        "screening": {
            "active_factors": len(active),
            "active_factor_order": active,
            "active_factor_set": sorted(active),
            "omitted_factors_before_validation": omitted_count,
            "omitted_fraction_before_validation": omitted_count / n_factors,
            "leader_action": frozen_action,
            "challenger_action": float(actions[final_challenger_index]),
            "sparse_ei": float(final_curve[final_leader_index]),
            "sparse_gap": float(
                final_curve[final_challenger_index]
                - final_curve[final_leader_index]
            ),
            "structural_bound": float(history[-1]["structural_bound"]),
            "optimistic_envelope": float(history[-1]["optimistic_envelope"]),
            "epsilon_ei": epsilon,
            "certification_margin": float(history[-1]["certification_margin"]),
            "active_ess_fraction": float(history[-1]["active_ess_fraction"]),
            "factor_vector_evaluations_before_validation": len(
                factor_cache.evaluated_indices
            ),
            "factor_vector_evaluations_avoided_before_validation": omitted_count,
            "factor_scalar_evaluations_avoided_before_validation": omitted_count
            * samples.shape[0],
            "no_omitted_factor_evaluated_during_screening": True,
        },
        "monte_carlo_diagnostic": {
            "status": "EMPIRICAL_NOT_A_RIGOROUS_INFERENCE_BOUND",
            "active_batch_leader_min": min(batch_leaders),
            "active_batch_leader_max": max(batch_leaders),
            "final_gap_batch_standard_error": float(
                np.std(batch_gaps, ddof=1) / np.sqrt(batch_count)
            ),
        },
        "grid_diagnostic": {
            "status": "EMPIRICAL_NOT_A_CONTINUOUS_OPTIMIZATION_BOUND",
            "coarse_action_count": actions.size,
            "refined_action_count": refined_actions.size,
            "coarse_leader": frozen_action,
            "refined_leader": float(refined_actions[refined_leader_index]),
            "leader_drift": abs(
                float(refined_actions[refined_leader_index]) - frozen_action
            ),
            "refined_challenger": float(
                refined_actions[refined_challenger_index]
            ),
            "refined_optimistic_envelope": float(
                refined_envelope[refined_challenger_index]
            ),
        },
        "heldout_validation": {
            "status": "HIGH_ACCURACY_EMPIRICAL_VALIDATION",
            "independent_replicates": len(validation_rows),
            "samples_per_replicate": int(
                validation["reference_samples_per_seed"]
            ),
            "leader_min": float(validation_leaders.min()),
            "leader_median": float(np.median(validation_leaders)),
            "leader_max": float(validation_leaders.max()),
            "maximum_action_distance": float(
                max(row["action_distance"] for row in validation_rows)
            ),
            "exact_grid_match_fraction": float(
                np.mean([row["exact_grid_action_match"] for row in validation_rows])
            ),
            "within_epsilon_fraction": float(
                np.mean([row["within_epsilon"] for row in validation_rows])
            ),
            "median_full_ei_regret": float(np.median(validation_regrets)),
            "maximum_full_ei_regret": float(validation_regrets.max()),
            "full_ess_fraction_min": float(validation_ess.min()),
            "full_ess_fraction_max": float(validation_ess.max()),
        },
        "linear_system_diagnostics": {
            "q": float(np.exp(-spacing / lengthscale)),
            "rho_inner": float(rho[0]),
            "rho_middle": float(rho[1]),
            "rho_outer": float(rho[-1]),
            "kappa_adjacent": float(kappa[0, 1]),
            "minimum_row_dominance_margin": float(margins.min()),
            "lambda_min_A": float(eigenvalues.min()),
            "lambda_max_A": float(eigenvalues.max()),
            "condition_number_A": float(eigenvalues.max() / eigenvalues.min()),
            "solve_residual_l2": solve_residual,
        },
        "runtime": {
            "wall_seconds": wall_time,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "remaining_blockers": [
            "finite-sample adaptivity-safe inference error",
            "rigorous continuous-action optimization error",
        ],
    }

    write_csv(OUTPUT_DIRECTORY / "screening_history.csv", history)
    write_csv(OUTPUT_DIRECTORY / "heldout_validation.csv", validation_rows)
    (OUTPUT_DIRECTORY / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )

    results_markdown = f"""# T2-B Reflection-Symmetry EI Validation

Status: **{blocker_verdict}**

This run tests only whether the proved EI structural bound is numerically
non-vacuous in the archived OU/symmetry regime. It is not a full E1 reproduction
and not a rigorous end-to-end action certificate.

## Prospective protocol

- incumbent: `{incumbent:.2f}`;
- EI tolerance: `{epsilon:.3f}` (fixed before full-target evaluation and distinct
  from the archived `0.03` log-acquisition tolerance);
- screen: `{screening['reference_samples']}` reference samples, seed
  `{screening['seed']}`, `{actions.size}` actions, factor batches of
  `{screening['activation_batch_size']}`;
- non-vacuity criterion: certify while omitting at least
  `{100 * decision['nonvacuity_minimum_omitted_fraction']:.0f}%` of factors;
- validation: `{len(validation_rows)}` fresh replicates with
  `{validation['reference_samples_per_seed']}` samples each, evaluated only after
  the active set and action were frozen.

## Result

- active factors: `{len(active)}/{n_factors}`; omitted before validation:
  `{omitted_count}/{n_factors}` (`{100 * omitted_count / n_factors:.1f}%`);
- screened EI action: `{frozen_action:.6f}`;
- final sparse gap: `{summary['screening']['sparse_gap']:.8f}`;
- structural bound: `{summary['screening']['structural_bound']:.8f}`;
- optimistic envelope: `{summary['screening']['optimistic_envelope']:.8f}`
  versus tolerance `{epsilon:.5f}`;
- refined-grid leader drift: `{summary['grid_diagnostic']['leader_drift']:.6g}`;
- held-out full leaders: `{summary['heldout_validation']['leader_min']:.6f}` to
  `{summary['heldout_validation']['leader_max']:.6f}`;
- exact coarse-grid action matches:
  `{100 * summary['heldout_validation']['exact_grid_match_fraction']:.1f}%`;
- maximum held-out action distance:
  `{summary['heldout_validation']['maximum_action_distance']:.6g}`;
- maximum held-out EI regret at the screened action:
  `{summary['heldout_validation']['maximum_full_ei_regret']:.8g}`;
- held-out runs within EI tolerance:
  `{100 * summary['heldout_validation']['within_epsilon_fraction']:.1f}%`;
- full-target importance-sampling ESS:
  `{100 * summary['heldout_validation']['full_ess_fraction_min']:.1f}%` to
  `{100 * summary['heldout_validation']['full_ess_fraction_max']:.1f}%`.

No omitted factor was evaluated during screening. Full-factor evaluation used
fresh samples and occurred only after the screened action was frozen.

## Error separation

- The structural term is the proved analytic bound.
- Importance-sampling variability is an empirical diagnostic, not a rigorous
  finite-sample inference allowance.
- The 401-to-1601 grid comparison is a numerical diagnostic, not a continuous
  global-optimization certificate.

## Decision

The reflection-symmetry T2-B construction is **{blocker_verdict}**. The
nonlinear-PDE construction was subsequently proved. Rigorous finite-sample
inference and global-optimization allowances remain unresolved.
"""
    (OUTPUT_DIRECTORY / "RESULTS.md").write_text(results_markdown)
    print(f"[RESULT] reflection-symmetry T2-B: {blocker_verdict}")
    return summary


if __name__ == "__main__":
    run()
