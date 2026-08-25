from __future__ import annotations

import hashlib
import json
from pathlib import Path
import runpy
from types import SimpleNamespace

import numpy as np
import pytest

from conditioned_bo import full_shadow_backend as backend


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = (
    REPOSITORY_ROOT
    / "experiments/nonlinear_pde/run_ess_resolution_gate.py"
)
OUTPUT_DIRECTORY = (
    REPOSITORY_ROOT
    / "experiments/nonlinear_pde/outputs/ess_resolution_gate"
)
CONFIG_PATH = OUTPUT_DIRECTORY / "diagnostic_config.json"
EXPECTED_CONFIG_SHA256 = (
    "8285522fb03c9b07ae4d87b5bdf2e7a13004abcdea7edb24f4e93aa74c63e194"
)


@pytest.fixture(scope="module")
def runner():
    return runpy.run_path(str(RUNNER_PATH))


def test_resolution_config_is_frozen_and_transition_is_unchanged() -> None:
    assert hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest() == EXPECTED_CONFIG_SHA256
    config = json.loads(CONFIG_PATH.read_text())
    transition = config["existing_transition_contract"]
    assert transition["backend"] == "ELLIPTICAL_SLICE_FULL"
    assert transition["chain_count"] == 8
    assert transition["thinning"] is False
    assert transition["transition_changes_permitted"] is False
    assert transition["group_a_chains"] == [0, 1, 2, 3]
    assert transition["group_b_chains"] == [4, 5, 6, 7]


def test_prior_standard_ess_transition_is_exact_on_fixed_tiny_fixture() -> None:
    current = np.asarray([0.2, -0.4])
    mean = np.asarray([0.1, -0.2])
    direction = np.asarray([0.7, 0.3])
    extra_precision = np.asarray([[1.3, 0.2], [0.2, 0.8]])

    def log_likelihood(value):
        return -0.5 * float(value @ extra_precision @ value)

    observed = backend.elliptical_slice_transition(
        current,
        mean,
        direction,
        log_likelihood(current),
        log_likelihood,
        np.random.default_rng(8127),
        maximum_bracket_evaluations=50,
    )
    np.testing.assert_array_equal(
        observed[0],
        np.asarray(
            [
                float.fromhex("0x1.a7ea50af000f5p-6"),
                float.fromhex("0x1.65f24cc3acad0p-7"),
            ]
        ),
    )
    assert observed[1] == float.fromhex("-0x1.1acd17494479ep-11")
    assert observed[2] == 1
    assert observed[3] == float.fromhex("0x1.8d676e05d047cp+1")


def test_s1_s2_s3_stream_labels_are_deterministic_and_independent(runner) -> None:
    observed = [
        runner["_stream_seed"](
            "calibration", 24, "early", schedule, "chain_0_slice"
        )
        for schedule in ("S1", "S2", "S3")
    ]
    repeated = [
        runner["_stream_seed"](
            "calibration", 24, "early", schedule, "chain_0_slice"
        )
        for schedule in ("S1", "S2", "S3")
    ]
    assert observed == repeated
    assert len(set(observed)) == 3
    assert set(observed).isdisjoint({4215109622, 1083605379, 4045758625})


def test_schedules_have_exact_lengths_and_no_thinning() -> None:
    config = json.loads(CONFIG_PATH.read_text())
    observed = [
        (
            item["schedule_id"],
            item["burn_in"],
            item["retained_per_chain"],
        )
        for item in config["calibration_schedules"]
    ]
    assert observed == [
        ("S1", 2048, 8192),
        ("S2", 2048, 16384),
        ("S3", 4096, 32768),
    ]
    assert config["existing_transition_contract"]["thinning"] is False


def test_calibration_cannot_access_prospective_sources_and_validation_is_gated(
    runner, tmp_path
) -> None:
    assert runner["_source_seed_for_role"]("calibration") == 2026082401
    assert 2026082401 not in {4215109622, 1083605379, 4045758625}
    with pytest.raises(ValueError):
        runner["_source_seed_for_role"]("scientific")
    with pytest.raises(RuntimeError, match="inaccessible before schedule freeze"):
        runner["_source_seed_for_role"](
            "validation", frozen_schedule_path=tmp_path / "missing.json"
        )


def test_validation_cannot_change_a_frozen_schedule(runner, tmp_path) -> None:
    config = runner["_read_config"]()
    schedule = config["calibration_schedules"][0]
    payload = runner["_schedule_payload"](schedule, config)
    frozen = {
        "status": "FROZEN_AFTER_GLOBAL_CALIBRATION_PASS",
        "diagnostic_config_sha256": EXPECTED_CONFIG_SHA256,
        "schedule": payload,
        "schedule_sha256": runner["_canonical_sha256"](payload),
        "final_validation": {
            "derivation_label": runner["FINAL_VALIDATION_LABEL"],
            "seed": runner["derive_final_validation_seed"](),
        },
    }
    path = tmp_path / "frozen.json"
    path.write_text(json.dumps(frozen))
    assert runner["_read_frozen_schedule"](path)["schedule"] == payload
    frozen["schedule"]["retained_per_chain"] += 1
    path.write_text(json.dumps(frozen))
    with pytest.raises(RuntimeError, match="payload hash"):
        runner["_read_frozen_schedule"](path)


def _iid_chain(chain_index: int, rng: np.random.Generator):
    retained = 6000
    utility = rng.normal(loc=[0.5, 0.3, 0.1], scale=0.7, size=(retained, 3))
    return SimpleNamespace(
        chain_index=chain_index,
        initialization="IID_FIXTURE",
        proposal_seed=100 + chain_index,
        uniform_seed=200 + chain_index,
        burn_in=0,
        retained_count=retained,
        acceptance_fraction=1.0,
        accepted_transitions=retained,
        total_transitions=retained,
        utility=utility,
        target_energy=rng.normal(size=retained),
        factor_energy=rng.normal(size=retained),
        wall_seconds=0.0,
        work={
            "factor_energy_evaluations": retained,
            "factor_gradient_evaluations": 0,
            "factor_hessian_evaluations": 0,
        },
    )


def test_group_diagnostics_are_disjoint_and_mcse_matches_iid_fixture() -> None:
    rng = np.random.default_rng(4412)
    chains = [_iid_chain(index, rng) for index in range(8)]
    state = SimpleNamespace(action_indices=np.asarray([10, 20, 30]))
    aggregate = backend.aggregate_independence_mh_chains(
        chains,
        state,
        group_a_chains=(0, 1, 2, 3),
        group_b_chains=(4, 5, 6, 7),
        diagnostic_top_action_count=3,
        strict_gate_top_action_count=3,
    )
    assert set(aggregate["group_a_chains"]).isdisjoint(
        aggregate["group_b_chains"]
    )
    assert aggregate["maximum_split_rhat"] <= 1.01
    for diagnostic in aggregate["scalar_diagnostics"]:
        expected = diagnostic["standard_deviation"] / np.sqrt(
            diagnostic["autocorrelation_ess"]
        )
        assert diagnostic["mcse"] == pytest.approx(expected, rel=1e-14)
    with pytest.raises(ValueError, match="disjoint"):
        backend.aggregate_independence_mh_chains(
            chains,
            state,
            group_a_chains=(0, 1, 2, 3),
            group_b_chains=(3, 4, 5, 6, 7),
            diagnostic_top_action_count=3,
            strict_gate_top_action_count=3,
        )


def test_prior_backend_rescue_outputs_remain_hash_unchanged() -> None:
    config = json.loads(CONFIG_PATH.read_text())
    prior = (
        REPOSITORY_ROOT
        / "experiments/nonlinear_pde/outputs/full_shadow_backend_rescue"
    )
    for filename, expected in config[
        "prior_backend_rescue_output_sha256"
    ].items():
        assert hashlib.sha256((prior / filename).read_bytes()).hexdigest() == expected


def test_resolution_shadow_cannot_feed_deployable_factor_selection() -> None:
    config = json.loads(CONFIG_PATH.read_text())
    scientific = config["preserved_scientific_contract"]
    assert scientific["shadow_feeds_deployable_factor_selection"] is False
    assert scientific["shadow_included_in_timed_full_baseline"] is False
    deployable = (
        REPOSITORY_ROOT / "src/conditioned_bo/nonlinear_pde_locality.py"
    ).read_text()
    assert "run_ess_resolution_gate" not in deployable
