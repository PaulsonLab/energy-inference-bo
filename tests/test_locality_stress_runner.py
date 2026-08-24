from __future__ import annotations

import hashlib
import json
import runpy
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIRECTORY = (
    REPOSITORY_ROOT
    / "experiments"
    / "nonlinear_pde"
    / "outputs"
    / "locality_stress_v1"
)
CONFIG_PATH = OUTPUT_DIRECTORY / "frozen_config.json"
EXPECTED_CONFIG_SHA256 = "0aa5d94108b617efba33900c25911e77439d074aa1f27d93d55c6c62bc160211"


def _runner():
    return runpy.run_path(
        str(REPOSITORY_ROOT / "experiments/nonlinear_pde/run_locality_stress.py")
    )


def test_frozen_config_hash_and_provenance_contract() -> None:
    actual = hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest()
    assert actual == EXPECTED_CONFIG_SHA256
    runner = _runner()
    assert runner["_config_sha256"]() == EXPECTED_CONFIG_SHA256
    required = json.loads(CONFIG_PATH.read_text())["output_schema"]["required"]
    assert {
        "summary.json",
        "state_metrics.csv",
        "stage_metrics.csv",
        "method_summary.csv",
        "resource_metrics.csv",
        "RESULTS.md",
    }.issubset(required)


def test_frozen_design_has_exactly_45_paired_states() -> None:
    config = json.loads(CONFIG_PATH.read_text())
    assert config["domain_sizes"] == [18, 24, 30, 36, 40]
    assert len(config["prospective_source_seeds"]) == 3
    assert config["bo_trajectory"]["checkpoint_queries"] == [4, 8, 12]
    assert 5 * 3 * 3 == 45


def test_development_seed_is_disjoint_and_random_seeds_are_independent() -> None:
    config = json.loads(CONFIG_PATH.read_text())
    prospective = set(config["prospective_source_seeds"])
    assert config["development"]["source_seed"] not in prospective
    random_seed = int(
        hashlib.sha256(b"E2_RANDOM_MATCHED:n24_r0_early_q4").hexdigest()[:8], 16
    )
    assert random_seed not in prospective


def test_frozen_methods_and_mask_lifecycle() -> None:
    config = json.loads(CONFIG_PATH.read_text())
    assert config["methods"] == [
        "FULL",
        "ADAPTIVE_INFLUENCE",
        "DYNAMIC_GEOMETRIC_SHELL",
        "STATIC_INFLUENCE",
        "FIXED_CHALLENGER",
        "RANDOM_MATCHED_M",
        "ORACLE_GEOMETRIC_PREFIX",
    ]
    assert config["selection"]["initial_mask"] == "empty for M1-M4 at every BO state"
    assert config["selection"]["activation_within_decision"] == "cumulative"
    assert config["selection"]["mask_carry_between_states"] is False


def test_tolerance_discrepancy_is_explicit_and_prospective_mode_is_guarded() -> None:
    config = json.loads(CONFIG_PATH.read_text())
    discrepancy = config["historical_tolerance_discrepancy"]
    assert discrepancy["primary_24x24_tolerance"] == 0.06
    assert discrepancy["archived_scaling_helper_default_tolerance"] == 0.075
    assert discrepancy["status"] == "PROSPECTIVE_RUN_REQUIRES_EXPLICIT_ACKNOWLEDGEMENT"
    source = (
        REPOSITORY_ROOT / "experiments/nonlinear_pde/run_locality_stress.py"
    ).read_text()
    assert "--acknowledge-scaling-epsilon-resolution" in source
    assert "use-primary-0.060" in source


def test_leakage_and_fairness_are_frozen() -> None:
    config = json.loads(CONFIG_PATH.read_text())
    assert config["diagnostics"]["oracle_is_deployable"] is False
    assert config["diagnostics"]["full_information_may_feed_deployable_methods"] is False
    assert config["bo_trajectory"]["paired_state_requirement"].startswith("byte-identical")
    assert config["random_baseline"]["subsets_per_state"] >= 10


def test_colab_runner_uses_reported_commit_and_exact_scientific_guard() -> None:
    notebook = json.loads(
        (REPOSITORY_ROOT / "experiments/nonlinear_pde/colab_locality_stress.ipynb").read_text()
    )
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    assert "PREREGISTRATION_SHA" in source
    assert "git\", \"checkout\", PREREGISTRATION_SHA" in source
    assert "--acknowledge-scaling-epsilon-resolution" in source
    assert "use-primary-0.060" in source
