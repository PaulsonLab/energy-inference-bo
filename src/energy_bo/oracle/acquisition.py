"""Scalar expected-improvement helpers for the oracle and residual models."""

from __future__ import annotations

import numpy as np

from .distributions import GaussianMixture, normal_ei
from .residual_energy import RBFResidualEnergy
from .scenarios import OracleScenario


def oracle_ei_curve(
    distribution: GaussianMixture, scenario: OracleScenario, x: np.ndarray
) -> np.ndarray:
    return distribution.expected_improvement(scenario.mean(x), scenario.scale(x), scenario.best_f)


def gaussian_ei_curve(scenario: OracleScenario, x: np.ndarray) -> np.ndarray:
    return normal_ei(scenario.mean(x), scenario.scale(x), scenario.best_f)


def residual_ei_curve(
    model: RBFResidualEnergy, scenario: OracleScenario, x: np.ndarray
) -> np.ndarray:
    return model.expected_improvement(scenario.mean(x), scenario.scale(x), scenario.best_f)
