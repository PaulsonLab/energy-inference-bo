from pathlib import Path

import numpy as np
import pytest

from decision_tilt.rare_mode import RareModeConfig, exact_landscape, mixture_at


CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "rare_mode_mechanism"
    / "config.json"
)


def test_frozen_protocol_satisfies_analytical_preflight() -> None:
    config = RareModeConfig.load(CONFIG_PATH)
    candidate_a = mixture_at(config.model["candidate_a"], config)
    candidate_b = mixture_at(config.model["candidate_b"], config)
    raw_a, raw_second_a = candidate_a.improvement_moments(config.model["incumbent"])
    raw_b, raw_second_b = candidate_b.improvement_moments(config.model["incumbent"])
    smooth_a, _ = candidate_a.softplus_moments(
        config.model["incumbent"],
        config.model["smooth_temperature"],
        order=config.numerics["gauss_hermite_order"],
    )
    smooth_b, _ = candidate_b.softplus_moments(
        config.model["incumbent"],
        config.model["smooth_temperature"],
        order=config.numerics["gauss_hermite_order"],
    )
    assert raw_b / raw_a >= config.gate["minimum_raw_ei_ratio_b_over_a"]
    assert smooth_b / smooth_a >= config.gate["minimum_smooth_acquisition_ratio_b_over_a"]
    assert raw_a**2 / raw_second_a > 0.4
    assert raw_b**2 / raw_second_b <= config.gate["maximum_candidate_b_ess_fraction"]
    assert candidate_b.tilted_component_weights(config.model["incumbent"])[1] >= 0.99


def test_frozen_exact_landscape_has_rare_mode_global_maximum() -> None:
    config = RareModeConfig.load(CONFIG_PATH)
    grid = np.linspace(*config.model["domain"], 4001)
    exact = exact_landscape(grid, config)
    maximizer = float(grid[int(np.argmax(exact["raw_first"]))])
    assert maximizer == pytest.approx(config.model["candidate_b"], abs=1e-12)


def test_candidate_b_predictive_belief_is_strongly_non_gaussian() -> None:
    config = RareModeConfig.load(CONFIG_PATH)
    shape = mixture_at(config.model["candidate_b"], config).standardized_shape()
    assert shape["skewness"] > 10.0
    assert shape["excess_kurtosis"] > 100.0
