"""Deterministic short-horizon Bellman reference for the two-regime model."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.polynomial.hermite import hermgauss

from causal_policy_bo.model import TwoRegimeConfig, TwoRegimeModel


@dataclass(frozen=True)
class ReferenceLevel:
    name: str
    action_points: int
    quadrature_points: int
    belief_points: int
    incumbent_points: int
    log_odds_limit: float = 30.0
    state_batch: int = 32

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ReferenceSolution:
    level: ReferenceLevel
    rows: list[dict[str, float | int | str]]
    actions: np.ndarray
    initial_q: dict[int, np.ndarray]


DESIGN_SCAN_LEVEL = ReferenceLevel("design_scan", 201, 15, 81, 101)
REFERENCE_LEVELS = (
    ReferenceLevel("coarse", 201, 15, 81, 101),
    ReferenceLevel("medium", 301, 21, 121, 151),
    ReferenceLevel("fine", 401, 31, 161, 201),
)
REFINEMENT_LEVEL = ReferenceLevel("refinement", 601, 41, 201, 251, state_batch=20)


def _quadrature(order: int) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = hermgauss(order)
    return np.sqrt(2.0) * nodes, weights / np.sqrt(np.pi)


def _state_grids(
    model: TwoRegimeModel, level: ReferenceLevel
) -> tuple[np.ndarray, np.ndarray]:
    log_odds = np.linspace(
        -level.log_odds_limit,
        level.log_odds_limit,
        level.belief_points,
        dtype=np.float64,
    )
    nodes, _ = _quadrature(level.quadrature_points)
    action_probe = np.linspace(0.0, 1.0, 4001, dtype=np.float64)
    maximum_response = max(
        float(np.max(model.response_numpy(action_probe, 0))),
        float(np.max(model.response_numpy(action_probe, 1))),
    )
    incumbent_maximum = maximum_response + model.config.noise_std * float(nodes[-1])
    unit = np.linspace(0.0, 1.0, level.incumbent_points, dtype=np.float64)
    incumbent = model.config.incumbent + (
        incumbent_maximum - model.config.incumbent
    ) * unit**2
    return log_odds, incumbent


def _bilinear(
    values: np.ndarray,
    log_odds_grid: np.ndarray,
    incumbent_grid: np.ndarray,
    query_log_odds: np.ndarray,
    query_incumbent: np.ndarray,
) -> np.ndarray:
    query_log_odds = np.clip(query_log_odds, log_odds_grid[0], log_odds_grid[-1])
    query_incumbent = np.clip(
        query_incumbent, incumbent_grid[0], incumbent_grid[-1]
    )

    log_step = log_odds_grid[1] - log_odds_grid[0]
    log_position = (query_log_odds - log_odds_grid[0]) / log_step
    log_lower = np.floor(log_position).astype(np.int64)
    log_lower = np.clip(log_lower, 0, log_odds_grid.size - 2)
    log_weight = log_position - log_lower

    incumbent_upper = np.searchsorted(
        incumbent_grid, query_incumbent, side="right"
    )
    incumbent_upper = np.clip(incumbent_upper, 1, incumbent_grid.size - 1)
    incumbent_lower = incumbent_upper - 1
    incumbent_span = incumbent_grid[incumbent_upper] - incumbent_grid[incumbent_lower]
    incumbent_weight = np.divide(
        query_incumbent - incumbent_grid[incumbent_lower],
        incumbent_span,
        out=np.zeros_like(query_incumbent, dtype=np.float64),
        where=incumbent_span > 0.0,
    )

    value00 = values[log_lower, incumbent_lower]
    value10 = values[log_lower + 1, incumbent_lower]
    value01 = values[log_lower, incumbent_upper]
    value11 = values[log_lower + 1, incumbent_upper]
    lower = value00 + log_weight * (value10 - value00)
    upper = value01 + log_weight * (value11 - value01)
    return lower + incumbent_weight * (upper - lower)


def _one_step_table(
    model: TwoRegimeModel,
    actions: np.ndarray,
    log_odds_grid: np.ndarray,
    incumbent_grid: np.ndarray,
    state_batch: int,
) -> np.ndarray:
    mesh_log_odds, mesh_incumbent = np.meshgrid(
        log_odds_grid, incumbent_grid, indexing="ij"
    )
    flat_log_odds = mesh_log_odds.ravel()
    flat_incumbent = mesh_incumbent.ravel()
    output = np.empty_like(flat_log_odds)
    mean0, mean1 = model.means_numpy(actions)
    for start in range(0, output.size, state_batch):
        stop = min(start + state_batch, output.size)
        incumbent = flat_incumbent[start:stop, None]
        probability1 = model.probability_m1(flat_log_odds[start:stop])[:, None]
        ei0 = model.normal_expected_improvement(
            mean0[None, :], incumbent, model.config.noise_std
        )
        ei1 = model.normal_expected_improvement(
            mean1[None, :], incumbent, model.config.noise_std
        )
        future = np.max((1.0 - probability1) * ei0 + probability1 * ei1, axis=1)
        output[start:stop] = (
            flat_incumbent[start:stop] - model.config.incumbent + future
        )
    return output.reshape(log_odds_grid.size, incumbent_grid.size)


def _transition_arrays(
    model: TwoRegimeModel, actions: np.ndarray, nodes: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    mean0, mean1 = model.means_numpy(actions)
    means = np.stack((mean0, mean1), axis=1)
    observations = means[:, :, None] + model.config.noise_std * nodes[None, None, :]
    likelihood_ratio = (
        (observations - mean0[:, None, None]) ** 2
        - (observations - mean1[:, None, None]) ** 2
    ) / (2.0 * model.config.noise_std**2)
    return observations, likelihood_ratio


def _expected_continuation(
    previous: np.ndarray,
    model: TwoRegimeModel,
    log_odds_grid: np.ndarray,
    incumbent_grid: np.ndarray,
    log_odds: np.ndarray,
    incumbent: np.ndarray,
    observations: np.ndarray,
    likelihood_ratio: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    next_log_odds = log_odds[:, None, None, None] + likelihood_ratio[None, :, :, :]
    next_incumbent = np.maximum(
        incumbent[:, None, None, None], observations[None, :, :, :]
    )
    continuation = _bilinear(
        previous,
        log_odds_grid,
        incumbent_grid,
        next_log_odds,
        next_incumbent,
    )
    probability1 = model.probability_m1(log_odds)
    regime_probability = np.stack((1.0 - probability1, probability1), axis=1)
    return np.sum(
        continuation
        * regime_probability[:, None, :, None]
        * weights[None, None, None, :],
        axis=(2, 3),
    )


def _bellman_table(
    previous: np.ndarray,
    model: TwoRegimeModel,
    actions: np.ndarray,
    log_odds_grid: np.ndarray,
    incumbent_grid: np.ndarray,
    observations: np.ndarray,
    likelihood_ratio: np.ndarray,
    weights: np.ndarray,
    state_batch: int,
) -> np.ndarray:
    mesh_log_odds, mesh_incumbent = np.meshgrid(
        log_odds_grid, incumbent_grid, indexing="ij"
    )
    flat_log_odds = mesh_log_odds.ravel()
    flat_incumbent = mesh_incumbent.ravel()
    output = np.empty_like(flat_log_odds)
    for start in range(0, output.size, state_batch):
        stop = min(start + state_batch, output.size)
        q_values = _expected_continuation(
            previous,
            model,
            log_odds_grid,
            incumbent_grid,
            flat_log_odds[start:stop],
            flat_incumbent[start:stop],
            observations,
            likelihood_ratio,
            weights,
        )
        output[start:stop] = np.max(q_values, axis=1)
    return output.reshape(log_odds_grid.size, incumbent_grid.size)


def solve_reference_level(
    config: TwoRegimeConfig, level: ReferenceLevel
) -> ReferenceSolution:
    model = TwoRegimeModel(config)
    actions = np.linspace(
        config.domain_low, config.domain_high, level.action_points, dtype=np.float64
    )
    nodes, weights = _quadrature(level.quadrature_points)
    log_odds_grid, incumbent_grid = _state_grids(model, level)
    observations, likelihood_ratio = _transition_arrays(model, actions, nodes)

    value1 = _one_step_table(
        model, actions, log_odds_grid, incumbent_grid, level.state_batch
    )
    value2 = _bellman_table(
        value1,
        model,
        actions,
        log_odds_grid,
        incumbent_grid,
        observations,
        likelihood_ratio,
        weights,
        level.state_batch,
    )

    initial_log_odds = np.array([config.prior_log_odds], dtype=np.float64)
    initial_incumbent = np.array([config.incumbent], dtype=np.float64)
    one_step_q = model.predictive_expected_improvement(
        actions, config.prior_log_odds, config.incumbent
    )
    two_step_q = _expected_continuation(
        value1,
        model,
        log_odds_grid,
        incumbent_grid,
        initial_log_odds,
        initial_incumbent,
        observations,
        likelihood_ratio,
        weights,
    )[0]
    three_step_q = _expected_continuation(
        value2,
        model,
        log_odds_grid,
        incumbent_grid,
        initial_log_odds,
        initial_incumbent,
        observations,
        likelihood_ratio,
        weights,
    )[0]
    q_by_horizon = {1: one_step_q, 2: two_step_q, 3: three_step_q}
    myopic_index = int(np.argmax(one_step_q))
    rows: list[dict[str, float | int | str]] = []
    for horizon, q_values in q_by_horizon.items():
        best_index = int(np.argmax(q_values))
        best_value = float(q_values[best_index])
        forced_value = float(q_values[myopic_index])
        rows.append(
            {
                "level": level.name,
                "horizon": horizon,
                "action_points": level.action_points,
                "quadrature_points": level.quadrature_points,
                "belief_points": level.belief_points,
                "incumbent_points": level.incumbent_points,
                "first_action": float(actions[best_index]),
                "value": best_value,
                "forced_myopic_action": float(actions[myopic_index]),
                "forced_myopic_value": forced_value,
                "forced_myopic_relative_loss": (
                    (best_value - forced_value) / best_value if best_value > 0.0 else 0.0
                ),
            }
        )
    return ReferenceSolution(level, rows, actions, q_by_horizon)


def planning_opportunity(
    solution: ReferenceSolution, model: TwoRegimeModel
) -> tuple[bool, dict[str, Any]]:
    rows = {int(row["horizon"]): row for row in solution.rows}
    one_step_action = float(rows[1]["first_action"])
    one_step_ei = model.predictive_expected_improvement(
        solution.actions, model.config.prior_log_odds, model.config.incumbent
    )
    diagnostic_mask = (solution.actions >= 0.45) & (solution.actions <= 0.55)
    diagnostic_ratio = float(np.max(one_step_ei[diagnostic_mask]) / np.max(one_step_ei))
    horizon_checks: dict[str, dict[str, float | bool]] = {}
    passes_any = False
    for horizon in (2, 3):
        action = float(rows[horizon]["first_action"])
        displacement = abs(action - one_step_action)
        relative_loss = float(rows[horizon]["forced_myopic_relative_loss"])
        check = {
            "first_action": action,
            "in_diagnostic_region": 0.45 <= action <= 0.55,
            "action_displacement": displacement,
            "forced_myopic_relative_loss": relative_loss,
            "passes": (
                0.45 <= action <= 0.55
                and displacement >= 0.10
                and relative_loss >= 0.02
            ),
        }
        horizon_checks[str(horizon)] = check
        passes_any = passes_any or bool(check["passes"])
    one_step_exploit = (0.15 <= one_step_action <= 0.35) or (
        0.65 <= one_step_action <= 0.85
    )
    diagnostics = {
        "one_step_action": one_step_action,
        "one_step_action_in_exploit_region": one_step_exploit,
        "diagnostic_to_optimal_ei_ratio": diagnostic_ratio,
        "diagnostic_immediate_reward_passes": diagnostic_ratio <= 0.25,
        "horizons": horizon_checks,
    }
    passes = one_step_exploit and diagnostic_ratio <= 0.25 and passes_any
    diagnostics["passes"] = passes
    return passes, diagnostics
