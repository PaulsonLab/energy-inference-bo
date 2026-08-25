from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIRECTORY = (
    REPOSITORY_ROOT
    / "experiments/nonlinear_pde/outputs/ess_resolution_gate"
)


def _summary():
    return json.loads((OUTPUT_DIRECTORY / "summary.json").read_text())


def test_resolution_gate_required_outputs_and_no_validation_artifacts() -> None:
    required = {
        "diagnostic_config.json",
        "schedule_metrics.csv",
        "gap_mcse_metrics.csv",
        "summary.json",
        "RESULTS.md",
    }
    assert required.issubset(path.name for path in OUTPUT_DIRECTORY.iterdir())
    assert not (OUTPUT_DIRECTORY / "frozen_ess_schedule.json").exists()
    assert not (OUTPUT_DIRECTORY / "validation_metrics.csv").exists()


def test_terminal_classification_and_schedule_order_are_exact() -> None:
    summary = _summary()
    assert summary["terminal_classification"] == (
        "BACKEND_HEALTHY_REFERENCE_RULE_TOO_BRITTLE"
    )
    assert summary["strategy_case"] == "CASE_B_HEALTHY_NEAR_TIE_DECISIONS"
    assert summary["calibration_schedules_reached"] == ["S1", "S2", "S3"]
    assert summary["frozen_schedule"] is None
    assert summary["validation"] == []
    assert summary["fresh_final_validation_seed"] is None
    assert summary["fresh_final_validation_seed_classification"] == (
        "NOT DERIVED_OR_ACCESSED"
    )


def test_only_calibration_development_source_was_evaluated() -> None:
    summary = _summary()
    assert summary["calibration_source_seed"] == 2026082401
    assert summary["prospective_source_seeds_accessed"] is False
    assert summary["prospective_source_fields_constructed"] is False
    assert summary["scientific_preregistration_created_or_executed"] is False
    seeds = set(pd.read_csv(OUTPUT_DIRECTORY / "schedule_metrics.csv").source_seed)
    assert seeds == {2026082401}
    assert seeds.isdisjoint({4215109622, 1083605379, 4045758625})


def test_s3_has_healthy_n40_convergence_and_only_near_tie_failures() -> None:
    s3 = next(
        item for item in _summary()["calibration_schedules"]
        if item["schedule_id"] == "S3"
    )
    assert s3["burn_in"] == 4096
    assert s3["retained_per_chain"] == 32768
    assert s3["global_pass"] is False
    n40 = {state["checkpoint"]: state for state in s3["states"] if state["grid_size"] == 40}
    assert max(state["aggregate"]["maximum_split_rhat"] for state in n40.values()) < 1.0014
    assert min(
        state["aggregate"]["minimum_leader_challenger_gap_ess"]
        for state in n40.values()
    ) > 19800
    assert n40["early"]["gate"]["pass"] is True
    assert n40["middle"]["failure_mechanism"] == "FINITE_MC_NEAR_TIE"
    assert n40["late"]["failure_mechanism"] == "FINITE_MC_NEAR_TIE"
    assert n40["middle"]["aggregate"]["maximum_reciprocal_group_action_regret"] < 0.002
    assert n40["late"]["aggregate"]["maximum_reciprocal_group_action_regret"] < 0.002


def test_s3_n24_agrees_with_previously_reliable_reference() -> None:
    s3 = next(
        item for item in _summary()["calibration_schedules"]
        if item["schedule_id"] == "S3"
    )
    n24 = [state for state in s3["states"] if state["grid_size"] == 24]
    assert len(n24) == 3
    assert all(state["gate"]["pass"] for state in n24)
    assert all(state["gate"]["n24_reference_vector_pass"] for state in n24)
    assert all(state["gate"]["n24_reference_regret_pass"] for state in n24)


def test_gap_mcse_records_explain_the_two_s3_action_flips() -> None:
    gaps = pd.read_csv(OUTPUT_DIRECTORY / "gap_mcse_metrics.csv")
    s3 = gaps[
        (gaps.schedule_id == "S3")
        & (gaps.grid_size == 40)
        & (gaps.observable_type == "LEADER_CHALLENGER_GAP")
    ]
    middle = s3[(s3.checkpoint == "middle") & (s3.challenger_index == 859)]
    late = s3[(s3.checkpoint == "late") & (s3.challenger_index == 660)]
    assert len(middle) == 1
    assert len(late) == 1
    assert float(middle.iloc[0].absolute_estimate_over_mcse) < 0.57
    assert float(late.iloc[0].absolute_estimate_over_mcse) < 0.05


def test_runtime_projection_is_observed_work_based_and_local_feasible() -> None:
    runtime = _summary()["runtime_projection"]
    assert runtime["estimated_total_shadow_seconds_45_states"] > 0.0
    assert runtime["estimated_total_shadow_hours_45_states"] < 0.5
    assert runtime["expected_peak_ram_gb"] < 0.4
    assert runtime["suitable_for_16_gb_local_macbook"] is True
    assert runtime["shadow_excluded_from_scientific_timing"] is True


def test_results_state_exact_recommendation_and_preserve_old_classification() -> None:
    summary = _summary()
    assert summary["previous_backend_rescue_classification_preserved"] == (
        "FULL_REFERENCE_BACKEND_UNRESOLVED"
    )
    assert summary["recommended_backend"] == (
        "STANDARD_ELLIPTICAL_SLICE_FULL_PENDING_DECISION_ALIGNED_RULE_AUDIT"
    )
    results = (OUTPUT_DIRECTORY / "RESULTS.md").read_text()
    assert "Do not develop another sampler next" in results
    assert "no schedule was frozen" in results
    assert "fresh final development-validation seed was not derived or accessed" in results
