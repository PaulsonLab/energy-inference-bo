from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIRECTORY = (
    REPOSITORY_ROOT
    / "experiments/nonlinear_pde/outputs/full_shadow_backend_rescue"
)


def _summary():
    return json.loads((OUTPUT_DIRECTORY / "summary.json").read_text())


def test_backend_rescue_required_outputs_and_terminal_classification() -> None:
    required = {
        "backend_config.json",
        "proposal_pilot.csv",
        "production_snis.csv",
        "mcmc_chain_diagnostics.csv",
        "backend_validation.csv",
        "summary.json",
        "RESULTS.md",
    }
    assert required.issubset(path.name for path in OUTPUT_DIRECTORY.iterdir())
    summary = _summary()
    assert summary["terminal_backend_classification"] == (
        "FULL_REFERENCE_BACKEND_UNRESOLVED"
    )
    assert summary["recommended_backend"] is None
    assert summary["sampler_escalation_required"] is False


def test_only_two_permanent_development_source_seeds_were_used() -> None:
    summary = _summary()
    assert summary["calibration_source_seed"] == 2026082401
    assert summary["held_out_validation_source_seed"] == 3321078991
    assert summary["prospective_source_seeds_accessed"] is False
    assert summary["scientific_preregistration_created_or_executed"] is False
    sources = set(pd.read_csv(OUTPUT_DIRECTORY / "backend_validation.csv").source_seed)
    assert sources == {2026082401, 3321078991}
    assert sources.isdisjoint({4215109622, 1083605379, 4045758625})


def test_configuration_and_selection_hashes_are_exact() -> None:
    expected = {
        "backend_config.json": "408614b2c67411a0ad0ac59e5dfb973767f11ae127e9eb96b7c3422aed396258",
        "mcmc_config.json": "0fff47831b6f0a59fe93a1cd7cfecb775e1c5cfed0a887cc6f2ce26e0bd85d13",
        "elliptical_slice_config.json": "0a71fb13162259b7524e0309a7c806c4c4efa2f1fdf5b6ba6e5c4d1b07faf18c",
        "selected_proposals.json": "932e757d3a710e93e591a070eb45690970f0205b0789b1ba79d421cb7e612078",
    }
    for filename, digest in expected.items():
        assert hashlib.sha256((OUTPUT_DIRECTORY / filename).read_bytes()).hexdigest() == digest


def test_calibrated_snis_failed_held_out_gate_without_reusing_pilots() -> None:
    summary = _summary()["calibrated_snis"]
    assert summary["calibrated_snis_gate"] == {
        "held_out_n40_pass": False,
        "n24_health_and_prior_agreement_pass": False,
        "pass": False,
    }
    assert all(state["validation_pilot_run"] is False for state in summary["validation"])
    selected = {
        state["state_key"]: state["selected_proposal"] for state in summary["calibration"]
    }
    assert selected["n40_early"] == "BASELINE_LAPLACE_INFLATION_1.10"
    assert all(
        proposal == "CURVATURE_TEMPERED_LAMBDA_0.40"
        for state, proposal in selected.items()
        if state != "n40_early"
    )


def test_independence_mh_failed_calibration_before_held_out_validation() -> None:
    result = _summary()["independence_mh"]
    assert result["gate"] == {
        "calibration_pass": False,
        "held_out_validation_pass": False,
        "pass": False,
    }
    assert result["validation"] == []
    assert len(result["calibration_attempts"]) == 2
    assert all(
        state["aggregate"]["chain_diagnostics"][0]["acceptance_fraction"] == 0.0
        for attempt in result["calibration_attempts"]
        for state in attempt["states"]
    )


def test_elliptical_slice_failed_calibration_before_held_out_validation() -> None:
    result = _summary()["elliptical_slice"]
    assert result["gate"] == {
        "calibration_pass": False,
        "held_out_validation_pass": False,
        "pass": False,
    }
    assert result["validation"] == []
    assert len(result["calibration_attempts"]) == 2
    assert all(
        state["randomized_initial_angular_bracket"] is True
        and state["ellipse_reference"] == "EXACT_CURRENT_GAUSSIAN_BO_REFERENCE"
        for attempt in result["calibration_attempts"]
        for state in attempt["states"]
    )


def test_shadow_backend_never_feeds_timed_or_deployable_methods() -> None:
    contract = _summary()["shadow_only_contract"]
    assert contract["feeds_deployable_factor_selection"] is False
    assert contract["included_in_timed_full_baseline"] is False
    results = (OUTPUT_DIRECTORY / "RESULTS.md").read_text()
    assert "do not feed deployable selection or the timed `FULL` baseline" in results
    assert "Do not create a replacement E2 preregistration" in results
