from __future__ import annotations

import runpy
from pathlib import Path

import numpy as np
import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def diagnostic():
    return runpy.run_path(
        str(
            REPOSITORY_ROOT
            / "experiments/nonlinear_pde/run_full_shadow_reliability_diagnostic.py"
        )
    )


def test_diagnostic_is_pinned_to_development_seed_and_replacement_freeze(
    diagnostic,
) -> None:
    config = diagnostic["_read_frozen_config"]()
    assert diagnostic["DEVELOPMENT_SOURCE_SEED"] == 2026082401
    assert diagnostic["DEVELOPMENT_SOURCE_SEED"] not in config[
        "prospective_source_seeds"
    ]
    assert diagnostic["FROZEN_PREREGISTRATION_SHA"] == (
        "0dcb76f5f2d053e098b472ac9984182b837295b5"
    )
    assert diagnostic["PRIMARY_SAMPLE_COUNTS"] == (8192, 16384, 32768)


def test_first_two_seed_pairs_reproduce_frozen_shadow_schedule(diagnostic) -> None:
    identifier = "n40_r-2_early_q4"
    state_seed = int(diagnostic["hashlib"].sha256(identifier.encode()).hexdigest()[:8], 16)
    assert diagnostic["_batch_seeds"](identifier, 8192) == (
        state_seed + 900_000,
        state_seed + 900_001,
    )
    assert diagnostic["_batch_seeds"](identifier, 16384) == (
        state_seed + 1_000_003,
        state_seed + 1_000_004,
    )


def test_diagnostic_batch_matches_frozen_laplace_snis_acquisition(diagnostic) -> None:
    locality = diagnostic["locality"]
    config = diagnostic["_read_frozen_config"]()
    problem = locality.build_problem(
        5,
        diagnostic["DEVELOPMENT_SOURCE_SEED"],
        source_perturbation_scale=config["source_field"]["perturbation_scale"],
    )
    state = locality.build_common_bo_states(
        problem,
        initialization_size=2,
        total_queries=4,
        checkpoint_queries=(2, 3, 4),
        observation_noise_variance=0.0025,
        reference_sample_count=64,
        trajectory_seed=731,
        incumbent=0.55,
    )[0]
    seed = 991
    sample_count = 64
    observed = diagnostic["_diagnostic_laplace_snis"](
        config,
        state,
        problem,
        sample_count=sample_count,
        proposal_seed=seed,
    )
    expected = locality.laplace_snis_inference(
        state,
        problem,
        np.ones(25, dtype=bool),
        incumbent=locality.state_incumbent(state),
        delta_mc=config["inference"]["delta_mc"],
        sample_count=sample_count,
        proposal_seed=seed,
        proposal_inflation=config["inference"]["laplace_proposal_inflation"],
        work=locality.FactorWork(),
        gradient_tolerance=config["inference"]["laplace_gradient_tolerance"],
        maximum_iterations=config["inference"]["laplace_maximum_iterations"],
    )
    np.testing.assert_array_equal(observed["acquisition"], expected.acquisition)
    assert observed["action_index"] == expected.leader_index
    assert observed["ess_fraction"] == expected.ess_fraction


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (
            {
                "converged": True,
                "finite": True,
                "ess_pass": False,
                "action_agreement": False,
                "vector_pass": False,
                "maximum_cross_regret": 1.0,
                "pooled_top_two_gap": 1.0,
            },
            "LOW_ESS",
        ),
        (
            {
                "converged": True,
                "finite": True,
                "ess_pass": True,
                "action_agreement": False,
                "vector_pass": False,
                "maximum_cross_regret": 0.001,
                "pooled_top_two_gap": 0.002,
            },
            "NEAR_TIE_ACTION_INSTABILITY",
        ),
        (
            {
                "converged": True,
                "finite": True,
                "ess_pass": True,
                "action_agreement": False,
                "vector_pass": True,
                "maximum_cross_regret": 0.02,
                "pooled_top_two_gap": 0.02,
            },
            "ACTION_INSTABILITY_WITH_MATERIAL_GAP",
        ),
        (
            {
                "converged": True,
                "finite": True,
                "ess_pass": True,
                "action_agreement": True,
                "vector_pass": False,
                "maximum_cross_regret": 0.0,
                "pooled_top_two_gap": 0.02,
            },
            "GLOBAL_VECTOR_DISAGREEMENT_ONLY",
        ),
        (
            {
                "converged": False,
                "finite": True,
                "ess_pass": True,
                "action_agreement": True,
                "vector_pass": True,
                "maximum_cross_regret": 0.0,
                "pooled_top_two_gap": 0.02,
            },
            "OTHER_NUMERICAL_FAILURE",
        ),
        (
            {
                "converged": True,
                "finite": True,
                "ess_pass": True,
                "action_agreement": True,
                "vector_pass": True,
                "maximum_cross_regret": 0.0,
                "pooled_top_two_gap": 0.02,
            },
            None,
        ),
    ],
)
def test_failure_classification_is_mechanical(diagnostic, values, expected) -> None:
    assert diagnostic["classify_failure"](
        **values,
        materiality_scale=0.01,
    ) == expected
