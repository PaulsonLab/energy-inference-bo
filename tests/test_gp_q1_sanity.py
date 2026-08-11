from __future__ import annotations

import numpy as np
import pytest

from energy_bo.gp.exact_gp import GPSanityResult, run_gp_q1_sanity
from energy_bo.metrics import normalized_grid


@pytest.fixture(scope="module")
def gp_result() -> GPSanityResult:
    return run_gp_q1_sanity(seed=0, grid_points=151)


def test_exact_gp_analytic_ei_matches_quadrature(gp_result: GPSanityResult) -> None:
    assert np.max(np.abs(gp_result.manual_ei - gp_result.quadrature_ei)) < 1e-9
    # BoTorch's stable analytic implementation differs by 2.5e-8 at the
    # largest EI values in the resolved version; retain and report that value.
    assert np.max(np.abs(gp_result.botorch_ei - gp_result.manual_ei)) < 5e-8


def test_exact_gp_augmented_marginal_matches_ei_and_modes(gp_result: GPSanityResult) -> None:
    expected = normalized_grid(gp_result.manual_ei, gp_result.x)
    expected_m2 = normalized_grid(np.square(gp_result.manual_ei), gp_result.x)
    assert np.max(np.abs(gp_result.augmented_marginal - expected)) < 1e-9
    assert np.max(np.abs(gp_result.ei_squared_marginal - expected_m2)) < 1e-9
    assert len(
        {
            gp_result.manual_ei_argmax,
            gp_result.botorch_ei_argmax,
            gp_result.log_ei_argmax,
            gp_result.augmented_argmax,
        }
    ) == 1
