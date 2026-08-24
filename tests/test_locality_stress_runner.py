from __future__ import annotations

import hashlib
import json
import runpy
from pathlib import Path

import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIRECTORY = (
    REPOSITORY_ROOT
    / "experiments"
    / "nonlinear_pde"
    / "outputs"
    / "locality_stress_v1"
)
CONFIG_PATH = OUTPUT_DIRECTORY / "frozen_config.json"
EXPECTED_CONFIG_SHA256 = "2717ece2e5581a7224e1a7a5cb5f69c8291c14ad0ea5183aff54154072b4748b"


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


def test_full_fidelity_profile_is_development_only_and_uses_scientific_budgets() -> None:
    config = json.loads(CONFIG_PATH.read_text())
    profile = json.loads(
        (OUTPUT_DIRECTORY / "development_full_fidelity_profile.json").read_text()
    )
    assert profile["development_source_seed"] == config["development"]["source_seed"]
    assert profile["prospective_seeds_evaluated"] is False
    assert profile["development_source_seed"] not in config["prospective_source_seeds"]
    assert [state["checkpoint"] for state in profile["state_profiles"]] == [
        "early",
        "late",
    ]
    assert {item["method"] for item in profile["method_profiles"]} == {
        "FULL",
        "ADAPTIVE_INFLUENCE",
        "DYNAMIC_GEOMETRIC_SHELL",
        "STATIC_INFLUENCE",
        "FIXED_CHALLENGER",
    }
    budgets = profile["scientific_budgets_used"]
    assert budgets["reference_sample_count"] == config["numerical"][
        "reference_sample_count"
    ]
    assert budgets["laplace_sample_count"] == config["numerical"][
        "laplace_sample_count"
    ]
    assert budgets["maximum_refinement_stages"] == 50
    assert budgets["initial_full_shadow_samples_per_batch"] == 8192
    assert all(
        state["full_shadow"]["attempts"][0]["sample_count_per_batch"] == 8192
        for state in profile["state_profiles"]
    )


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


def test_tolerance_discrepancy_is_resolved_before_prospective_execution() -> None:
    config = json.loads(CONFIG_PATH.read_text())
    discrepancy = config["historical_tolerance_discrepancy"]
    assert discrepancy["primary_24x24_tolerance"] == 0.06
    assert discrepancy["archived_scaling_helper_default_tolerance"] == 0.075
    assert discrepancy["frozen_choice"] == 0.06
    assert discrepancy["status"] == "RESOLVED_BEFORE_EXECUTION_USE_PRIMARY_0.060"
    assert "stricter prospective test" in discrepancy["interpretation"]
    source = (
        REPOSITORY_ROOT / "experiments/nonlinear_pde/run_locality_stress.py"
    ).read_text()
    assert "--acknowledge-scaling-epsilon-resolution" not in source
    assert config["selection"]["stopping_tolerance"] == 0.06


def test_superseded_preregistration_is_explicitly_unrun() -> None:
    config = json.loads(CONFIG_PATH.read_text())
    provenance = config["provenance"]
    assert provenance["supersedes_preregistration_sha"] == (
        "c976ab186a730e5ddfd2270ccf1d60577ee0d6b6"
    )
    assert provenance["superseded_preregistration_status"] == (
        "SUPERSEDED_BEFORE_EXECUTION"
    )
    assert provenance["superseded_scientific_results_observed"] is False


def test_full_reference_quality_filter_keeps_count_statistics_on_all_states() -> None:
    config = json.loads(CONFIG_PATH.read_text())
    table = pd.DataFrame(
        [
            {
                "state_id": "reliable",
                "method": method,
                "M": count,
                "full_reference_reliable": True,
                "full_acquisition_regret": regret,
                "full_fallback": False,
            }
            for method, count, regret in (
                ("ADAPTIVE_INFLUENCE", 4, 0.002),
                ("DYNAMIC_GEOMETRIC_SHELL", 8, 0.003),
            )
        ]
        + [
            {
                "state_id": "unreliable",
                "method": method,
                "M": count,
                "full_reference_reliable": False,
                "full_acquisition_regret": regret,
                "full_fallback": method == "DYNAMIC_GEOMETRIC_SHELL",
            }
            for method, count, regret in (
                ("ADAPTIVE_INFLUENCE", 9, 99.0),
                ("DYNAMIC_GEOMETRIC_SHELL", 3, 199.0),
            )
        ]
    )
    comparison = _runner()["_paired_comparison"](
        table, "DYNAMIC_GEOMETRIC_SHELL", config
    )
    assert comparison["wins"] == 1
    assert comparison["losses"] == 1
    assert comparison["full_reference_reliable_quality_states"] == 1
    assert comparison["adaptive_mean_regret"] == 0.002
    assert comparison["baseline_mean_regret"] == 0.003
    a4_quality = _runner()["_matched_quality_disadvantage"](
        table, "DYNAMIC_GEOMETRIC_SHELL", 0.01
    )
    assert a4_quality["full_reference_reliable_quality_states"] == 1
    assert a4_quality["worse_quality_or_fallback_fraction"] == 0.0


def test_leakage_and_fairness_are_frozen() -> None:
    config = json.loads(CONFIG_PATH.read_text())
    assert config["diagnostics"]["oracle_is_deployable"] is False
    assert config["diagnostics"]["full_information_may_feed_deployable_methods"] is False
    assert config["bo_trajectory"]["paired_state_requirement"].startswith("byte-identical")
    assert config["random_baseline"]["subsets_per_state"] >= 10


def test_colab_runner_uses_reported_replacement_commit() -> None:
    notebook = json.loads(
        (REPOSITORY_ROOT / "experiments/nonlinear_pde/colab_locality_stress.ipynb").read_text()
    )
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    assert "PREREGISTRATION_SHA" in source
    assert "git\", \"checkout\", PREREGISTRATION_SHA" in source
    assert "--acknowledge-scaling-epsilon-resolution" not in source
    assert '"--preregistration-sha", PREREGISTRATION_SHA' in source
