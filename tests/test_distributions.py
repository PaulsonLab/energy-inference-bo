from __future__ import annotations

import numpy as np

from energy_bo.oracle.distributions import ORACLE_MIXTURE, normal_ei


def test_oracle_mixture_has_unit_matched_moments() -> None:
    numerical_mean, numerical_variance = ORACLE_MIXTURE.moments_quadrature()
    assert abs(ORACLE_MIXTURE.mean) < 1e-14
    assert abs(ORACLE_MIXTURE.variance - 1.0) < 1e-14
    assert abs(numerical_mean) < 1e-11
    assert abs(numerical_variance - 1.0) < 1e-11


def test_mixture_analytic_ei_matches_adaptive_quadrature() -> None:
    means = np.array([-0.5, 0.0, 0.4, 0.8])
    scales = np.array([0.2, 0.5, 0.9, 1.1])
    best_f = 0.25
    analytic = ORACLE_MIXTURE.expected_improvement(means, scales, best_f)
    numerical = np.array(
        [
            ORACLE_MIXTURE.expected_improvement_quadrature(mean, scale, best_f)[0]
            for mean, scale in zip(means, scales, strict=True)
        ]
    )
    assert np.max(np.abs(analytic - numerical)) < 1e-10
    assert np.all(normal_ei(means, scales, best_f) > 0.0)
