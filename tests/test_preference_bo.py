from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from conditioned_bo.preference_bo import (
    REQUIRED_TRAJECTORY_FIELDS,
    NumericalSettings,
    analytic_expected_improvement,
    evaluate_gates,
    full_target_inference,
    gp_reference_posterior,
    importance_sample_preference_target,
    laplace_preference_mode,
    load_frozen_config,
    prepare_pilot_inputs,
    run_seed,
    run_standard_trajectory,
    select_unobserved_argmax,
    true_objective,
    validate_trajectory_rows,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPOSITORY_ROOT / "experiments/preference_bo/configs/minimal_pilot.json"
CONFIG = json.loads(CONFIG_PATH.read_text())
GRID = np.linspace(0.0, 1.0, 17)
PAIRS = np.asarray(CONFIG["preference_bank"]["endpoint_index_pairs"], dtype=int)


def _posterior():
    gp = CONFIG["gp_reference"]
    return gp_reference_posterior(
        GRID,
        observed_indices=CONFIG["scalar_observations"]["initial_action_indices"],
        observed_values=[0.1, -0.05, 0.2],
        kernel_amplitude=gp["kernel_amplitude"],
        kernel_lengthscale=gp["kernel_lengthscale"],
        observation_noise_standard_deviation=gp[
            "observation_noise_standard_deviation"
        ],
    )


def test_true_objective_frozen_grid_maximum() -> None:
    values = true_objective(GRID)
    assert int(np.argmax(values)) == 12
    np.testing.assert_allclose(values[12], 0.9882354306, atol=5e-11)


def test_zero_active_factors_reproduce_ordinary_gp_ei() -> None:
    posterior = _posterior()
    incumbent = 0.2
    analytic = analytic_expected_improvement(
        posterior.mean, np.diag(posterior.covariance), incumbent
    )
    result = importance_sample_preference_target(
        posterior=posterior,
        endpoint_pairs=PAIRS,
        signs=np.ones(PAIRS.shape[0], dtype=int),
        temperature=CONFIG["preference_bank"]["temperature"],
        active_factors=[],
        incumbent=incumbent,
        draws=30_000,
        rng=np.random.default_rng(12345),
        laplace_settings=CONFIG["laplace"],
    )
    assert result.active_factor_count == 0
    assert result.factor_likelihood_evaluations == 0
    assert result.ess_fraction > 0.999999
    np.testing.assert_allclose(result.acquisition, analytic, atol=0.0035)


def test_all_active_target_matches_full_target_wrapper_exactly() -> None:
    posterior = _posterior()
    kwargs = {
        "posterior": posterior,
        "endpoint_pairs": PAIRS,
        "signs": np.array([-1, 1, 1, -1, 1, -1, 1, 1]),
        "temperature": CONFIG["preference_bank"]["temperature"],
        "incumbent": 0.2,
        "draws": 4_096,
        "laplace_settings": CONFIG["laplace"],
    }
    active = importance_sample_preference_target(
        **kwargs,
        active_factors=range(PAIRS.shape[0]),
        rng=np.random.default_rng(987),
    )
    full = full_target_inference(**kwargs, rng=np.random.default_rng(987))
    np.testing.assert_array_equal(active.acquisition, full.acquisition)
    np.testing.assert_array_equal(active.normalized_weights, full.normalized_weights)
    assert active.factor_likelihood_evaluations == full.factor_likelihood_evaluations


def test_laplace_roundoff_safe_newton_step_meets_frozen_gradient_tolerance() -> None:
    config = load_frozen_config(CONFIG_PATH)
    seed_input = prepare_pilot_inputs(config)[1]
    objective = true_objective(GRID)
    initial = config["scalar_observations"]["initial_action_indices"]
    observations = [
        float(objective[index] + seed_input.scalar_noise[index]) for index in initial
    ]
    gp = config["gp_reference"]
    posterior = gp_reference_posterior(
        GRID,
        observed_indices=initial,
        observed_values=observations,
        kernel_amplitude=gp["kernel_amplitude"],
        kernel_lengthscale=gp["kernel_lengthscale"],
        observation_noise_standard_deviation=gp[
            "observation_noise_standard_deviation"
        ],
    )
    result = laplace_preference_mode(
        posterior,
        PAIRS,
        seed_input.preference_signs,
        config["preference_bank"]["temperature"],
        active_factors=[0, 4, 5, 6],
        laplace_settings=config["laplace"],
    )
    assert result.converged
    assert result.gradient_infinity_norm <= 1e-9


def test_banks_and_location_noise_are_prepared_once_and_shared() -> None:
    config = load_frozen_config(CONFIG_PATH)
    prepared = prepare_pilot_inputs(config)
    assert [item.seed for item in prepared] == list(range(12))
    assert all(item.generated_before_methods for item in prepared)
    for item in prepared:
        assert item.preference_signs.shape == (8,)
        assert item.scalar_noise.shape == (17,)
        assert set(np.unique(item.preference_signs)).issubset({-1, 1})
        expected_bank_hash = hashlib.sha256(item.preference_signs.tobytes()).hexdigest()
        expected_noise_hash = hashlib.sha256(item.scalar_noise.tobytes()).hexdigest()
        assert item.preference_bank_sha256 == expected_bank_hash
        assert item.scalar_noise_sha256 == expected_noise_hash


def test_all_methods_receive_identical_seed_inputs_and_scalar_data_are_not_screened() -> None:
    config = load_frozen_config(CONFIG_PATH)
    seed_input = prepare_pilot_inputs(config)[0]
    numerical = NumericalSettings(
        working_draws_per_batch_schedule=(64, 128),
        full_draws_schedule=(256, 512),
    )
    result = run_seed(
        config,
        seed_input,
        horizon=1,
        numerical_settings=numerical,
        config_sha256=hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest(),
    )
    rows = result.trajectory_rows
    assert {row["method"] for row in rows} == {"standard", "full", "adaptive"}
    assert {row["preference_bank_sha256"] for row in rows} == {
        seed_input.preference_bank_sha256
    }
    assert {row["scalar_noise_sha256"] for row in rows} == {
        seed_input.scalar_noise_sha256
    }
    assert {row["screened_factor_source"] for row in rows} == {
        "historical_preference_bank_only"
    }
    for row in rows:
        active = json.loads(row["active_factor_indices"])
        assert all(0 <= index < 8 for index in active)
        assert row["scalar_observation_count"] == 4
    full_row = next(row for row in rows if row["method"] == "full")
    adaptive_row = next(row for row in rows if row["method"] == "adaptive")
    assert full_row["inference_sample_count"] in {256, 512}
    assert full_row["heldout_full_target_sample_count"] is None
    assert adaptive_row["inference_sample_count"] in {128, 256}
    assert adaptive_row["heldout_full_target_sample_count"] in {256, 512}
    assert adaptive_row["heldout_full_target_ess_fraction"] > 0.0


def test_exhaustive_action_search_and_complete_standard_trajectory_never_repeat() -> None:
    acquisition = np.arange(17, dtype=float)
    assert select_unobserved_argmax(acquisition, observed_indices=[16, 15]) == 14

    config = load_frozen_config(CONFIG_PATH)
    seed_input = prepare_pilot_inputs(config)[1]
    rows = run_standard_trajectory(
        config,
        seed_input,
        config_sha256=hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest(),
    )
    selected = [int(row["selected_action_index"]) for row in rows]
    assert len(selected) == 6
    assert len(set(selected)) == 6
    assert not set(selected).intersection(
        CONFIG["scalar_observations"]["initial_action_indices"]
    )


def test_result_schema_contains_required_provenance_fields() -> None:
    config = load_frozen_config(CONFIG_PATH)
    seed_input = prepare_pilot_inputs(config)[2]
    rows = run_standard_trajectory(
        config,
        seed_input,
        config_sha256=hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest(),
    )
    validate_trajectory_rows(rows)
    assert REQUIRED_TRAJECTORY_FIELDS.issubset(rows[0])
    assert all(row["config_sha256"] for row in rows)
    assert all(row["starting_git_commit"] for row in rows)


def test_gate_evaluator_synthetic_pass_fail_p1_and_fail_p2() -> None:
    passing = evaluate_gates(
        {
            "standard": [3] * 12,
            "full": [2] * 12,
            "adaptive": [3] * 12,
        },
        [0.5] * 12,
    )
    assert passing["verdict"] == "PASS"
    assert passing["p1_pass"] and passing["p2_pass"]

    fail_p1 = evaluate_gates(
        {
            "standard": [2] * 12,
            "full": [2] * 12,
            "adaptive": [7] * 12,
        },
        [1.0] * 12,
    )
    assert fail_p1["verdict"] == "FAIL-P1"

    fail_p2 = evaluate_gates(
        {
            "standard": [4] * 12,
            "full": [2] * 12,
            "adaptive": [4] * 12,
        },
        [0.66] * 12,
    )
    assert fail_p2["p1_pass"]
    assert not fail_p2["p2_pass"]
    assert fail_p2["verdict"] == "FAIL-P2"
