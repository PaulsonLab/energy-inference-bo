from __future__ import annotations

import numpy as np

from energy_bo.oracle.acquisition import oracle_ei_curve
from energy_bo.oracle.augmentation import (
    augmented_grid_marginal,
    oracle_augmented_marginal_by_quadrature,
)
from energy_bo.oracle.distributions import ORACLE_MIXTURE
from energy_bo.oracle.scenarios import TAIL_SENSITIVE, scenario_grid


def test_augmented_marginal_identity_for_m1() -> None:
    x = scenario_grid(101)
    expected_utility = oracle_ei_curve(ORACLE_MIXTURE, TAIL_SENSITIVE, x)
    direct, _ = oracle_augmented_marginal_by_quadrature(
        ORACLE_MIXTURE, TAIL_SENSITIVE, x, replicas=1
    )
    expected = augmented_grid_marginal(expected_utility, x, replicas=1)
    assert np.max(np.abs(direct - expected)) < 1e-10
    assert int(np.argmax(direct)) == int(np.argmax(expected))


def test_replicated_m2_identity_on_grid() -> None:
    x = scenario_grid(101)
    expected_utility = oracle_ei_curve(ORACLE_MIXTURE, TAIL_SENSITIVE, x)
    direct, _ = oracle_augmented_marginal_by_quadrature(
        ORACLE_MIXTURE, TAIL_SENSITIVE, x, replicas=2
    )
    expected = augmented_grid_marginal(expected_utility, x, replicas=2)
    assert np.max(np.abs(direct - expected)) < 1e-10
    assert int(np.argmax(direct)) == int(np.argmax(expected))
