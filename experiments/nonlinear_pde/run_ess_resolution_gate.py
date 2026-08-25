#!/usr/bin/env python3
"""Development-only chain-length gate for the existing standard ESS shadow.

The transition kernel is imported unchanged from ``full_shadow_backend``.
Only the three frozen chain-length schedules may be evaluated.  A fresh
development-validation source is inaccessible until a globally passing
calibration schedule has been written and hash-locked.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Sequence

import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from conditioned_bo import full_shadow_backend as backend  # noqa: E402
from conditioned_bo import nonlinear_pde_locality as locality  # noqa: E402


OUTPUT_DIRECTORY = (
    Path(__file__).resolve().parent / "outputs" / "ess_resolution_gate"
)
CONFIG_PATH = OUTPUT_DIRECTORY / "diagnostic_config.json"
FROZEN_SCHEDULE_PATH = OUTPUT_DIRECTORY / "frozen_ess_schedule.json"
EXPECTED_CONFIG_SHA256 = (
    "8285522fb03c9b07ae4d87b5bdf2e7a13004abcdea7edb24f4e93aa74c63e194"
)
PRIOR_RESCUE_DIRECTORY = (
    Path(__file__).resolve().parent / "outputs" / "full_shadow_backend_rescue"
)
PRIOR_BACKEND_CONFIG_PATH = PRIOR_RESCUE_DIRECTORY / "backend_config.json"
PRIOR_SELECTION_PATH = PRIOR_RESCUE_DIRECTORY / "selected_proposals.json"
PRIOR_SNIS_SUMMARY_PATH = PRIOR_RESCUE_DIRECTORY / "calibrated_snis_summary.json"

CALIBRATION_SEED = 2026082401
FINAL_VALIDATION_LABEL = "E2_STANDARD_ESS_FINAL_VALIDATE_V1"
PREVIOUS_DEVELOPMENT_SEEDS = (2026082401, 3321078991)
PROSPECTIVE_SOURCE_SEEDS = (4215109622, 1083605379, 4045758625)
GRID_SIZES = (24, 40)
CHECKPOINTS = ("early", "middle", "late")
CHECKPOINT_QUERIES = {"early": 4, "middle": 8, "late": 12}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_config() -> dict[str, Any]:
    if _sha256(CONFIG_PATH) != EXPECTED_CONFIG_SHA256:
        raise RuntimeError("ESS resolution configuration changed after freeze")
    config = json.loads(CONFIG_PATH.read_text())
    if config["starting_main_sha"] != "4219f5082063e8c714cf7f998221aa5125faa7e2":
        raise RuntimeError("starting main SHA changed in the frozen configuration")
    if config["development_sources"]["calibration"]["seed"] != CALIBRATION_SEED:
        raise RuntimeError("calibration source seed changed")
    if tuple(
        config["development_sources"]["prospective_source_seeds_forbidden"]
    ) != PROSPECTIVE_SOURCE_SEEDS:
        raise RuntimeError("prospective source-seed denylist changed")
    if CALIBRATION_SEED in PROSPECTIVE_SOURCE_SEEDS:
        raise RuntimeError("calibration seed overlaps a prospective source seed")
    prior_design = json.loads(PRIOR_BACKEND_CONFIG_PATH.read_text())[
        "state_design_unchanged"
    ]
    design = config["state_design"]
    required_equalities = {
        "grid_sizes": design["grid_sizes"],
        "checkpoints": design["checkpoints"],
        "checkpoint_queries": design["checkpoint_queries"],
        "ordinary_reference_sample_count": design[
            "ordinary_reference_sample_count"
        ],
        "observation_noise_variance": design["observation_noise_variance"],
        "source_perturbation_scale": design["source_perturbation_scale"],
        "incumbent_policy": design["incumbent_policy"],
    }
    for key, observed in required_equalities.items():
        if prior_design[key] != observed:
            raise RuntimeError(f"frozen state design changed: {key}")
    _verify_prior_outputs(config)
    return config


def _verify_prior_outputs(config: dict[str, Any]) -> None:
    for filename, expected in config[
        "prior_backend_rescue_output_sha256"
    ].items():
        if _sha256(PRIOR_RESCUE_DIRECTORY / filename) != expected:
            raise RuntimeError(f"prior backend-rescue output changed: {filename}")


def derive_final_validation_seed() -> int:
    raw = int.from_bytes(
        hashlib.sha256(FINAL_VALIDATION_LABEL.encode("utf-8")).digest()[:8],
        "big",
    )
    return raw % (2**32 - 1)


def _schedule_payload(schedule: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    transition = config["existing_transition_contract"]
    return {
        "schedule_id": schedule["schedule_id"],
        "burn_in": schedule["burn_in"],
        "retained_per_chain": schedule["retained_per_chain"],
        "chain_count": transition["chain_count"],
        "thinning": transition["thinning"],
        "stream_phase_label": schedule["stream_phase_label"],
        "group_a_chains": transition["group_a_chains"],
        "group_b_chains": transition["group_b_chains"],
        "maximum_bracket_evaluations_per_transition": transition[
            "maximum_bracket_evaluations_per_transition"
        ],
        "transition_backend": transition["backend"],
    }


def _read_frozen_schedule(path: Path = FROZEN_SCHEDULE_PATH) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError("validation source is inaccessible before schedule freeze")
    frozen = json.loads(path.read_text())
    if frozen["status"] != "FROZEN_AFTER_GLOBAL_CALIBRATION_PASS":
        raise RuntimeError("frozen schedule has an invalid status")
    if frozen["diagnostic_config_sha256"] != EXPECTED_CONFIG_SHA256:
        raise RuntimeError("frozen schedule has the wrong parent configuration")
    if _canonical_sha256(frozen["schedule"]) != frozen["schedule_sha256"]:
        raise RuntimeError("frozen ESS schedule payload hash is invalid")
    if frozen["final_validation"]["derivation_label"] != FINAL_VALIDATION_LABEL:
        raise RuntimeError("validation derivation label changed")
    if frozen["final_validation"]["seed"] != derive_final_validation_seed():
        raise RuntimeError("validation seed changed after schedule freeze")
    return frozen


def _source_seed_for_role(
    source_role: str, *, frozen_schedule_path: Path = FROZEN_SCHEDULE_PATH
) -> int:
    if source_role == "calibration":
        seed = CALIBRATION_SEED
    elif source_role == "validation":
        seed = int(
            _read_frozen_schedule(frozen_schedule_path)["final_validation"]["seed"]
        )
    else:
        raise ValueError("source role must be calibration or validation")
    if seed in PROSPECTIVE_SOURCE_SEEDS:
        raise RuntimeError("prospective E2 source seed access is forbidden")
    return seed


def _audit_final_validation_seed_before_freeze(seed: int) -> dict[str, Any]:
    if seed in PROSPECTIVE_SOURCE_SEEDS:
        raise RuntimeError("validation seed overlaps a prospective source seed")
    if seed in PREVIOUS_DEVELOPMENT_SEEDS:
        raise RuntimeError("validation seed overlaps a previous development seed")
    completed = subprocess.run(
        ["git", "grep", "-n", str(seed), "HEAD", "--", "."],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode not in (0, 1):
        raise RuntimeError("repository validation-seed audit failed")
    matches = [line for line in completed.stdout.splitlines() if line.strip()]
    if matches:
        raise RuntimeError(
            "fresh validation seed already appears in committed repository history"
        )
    return {
        "seed": seed,
        "derivation_label": FINAL_VALIDATION_LABEL,
        "derivation_digest_sha256": hashlib.sha256(
            FINAL_VALIDATION_LABEL.encode("utf-8")
        ).hexdigest(),
        "distinct_from_prospective_source_seeds": True,
        "distinct_from_previous_development_seeds": True,
        "committed_HEAD_literal_search_matches": matches,
        "prior_scientific_E2_use_found": False,
        "classification": "DEVELOPMENT ONLY",
    }


def _freeze_schedule(
    schedule: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    if FROZEN_SCHEDULE_PATH.exists():
        raise RuntimeError("a frozen ESS schedule already exists")
    validation_seed = derive_final_validation_seed()
    validation_audit = _audit_final_validation_seed_before_freeze(validation_seed)
    payload = _schedule_payload(schedule, config)
    frozen = {
        "diagnostic_id": config["diagnostic_id"],
        "status": "FROZEN_AFTER_GLOBAL_CALIBRATION_PASS",
        "development_only": True,
        "selected_from_calibration_seed": CALIBRATION_SEED,
        "diagnostic_config_sha256": EXPECTED_CONFIG_SHA256,
        "schedule": payload,
        "schedule_sha256": _canonical_sha256(payload),
        "final_validation": validation_audit,
        "validation_may_change_schedule": False,
    }
    FROZEN_SCHEDULE_PATH.write_text(
        json.dumps(frozen, indent=2, sort_keys=True) + "\n"
    )
    return _read_frozen_schedule()


def _trajectory_seed(grid_size: int) -> int:
    label = f"E2_LOCALITY_TRAJECTORY_V1:{grid_size}:-2"
    return int(hashlib.sha256(label.encode("utf-8")).hexdigest()[:8], 16)


def _stream_seed(
    source_role: str,
    grid_size: int,
    checkpoint: str,
    schedule_id: str,
    stream: str,
) -> int:
    label = (
        "E2_STANDARD_ESS_RESOLUTION_GATE_V1:"
        f"{source_role}:{grid_size}:{checkpoint}:ess_resolution_{schedule_id}:{stream}"
    )
    raw = int.from_bytes(hashlib.sha256(label.encode("utf-8")).digest()[:8], "big")
    seed = raw % (2**32 - 1)
    if seed in PROSPECTIVE_SOURCE_SEEDS:
        raise RuntimeError("an ESS RNG stream collided with a prospective source seed")
    return seed


def _reference_stream_seed(
    source_role: str, grid_size: int, checkpoint: str, batch: str
) -> int:
    label = (
        "E2_STANDARD_ESS_RESOLUTION_GATE_V1:"
        f"{source_role}:{grid_size}:{checkpoint}:n24_reference:{batch}"
    )
    raw = int.from_bytes(hashlib.sha256(label.encode("utf-8")).digest()[:8], "big")
    seed = raw % (2**32 - 1)
    if seed in PROSPECTIVE_SOURCE_SEEDS:
        raise RuntimeError("an n=24 reference stream collided with a prospective seed")
    return seed


def _state_key(grid_size: int, checkpoint: str) -> str:
    return f"n{grid_size}_{checkpoint}"


def _state_identifier(source_role: str, grid_size: int, checkpoint: str) -> str:
    return (
        f"{source_role}_n{grid_size}_r-2_{checkpoint}_"
        f"q{CHECKPOINT_QUERIES[checkpoint]}"
    )


def _build_development_state(
    config: dict[str, Any],
    source_role: str,
    grid_size: int,
    checkpoint: str,
) -> tuple[locality.Problem, locality.BOState]:
    source_seed = _source_seed_for_role(source_role)
    design = config["state_design"]
    problem = locality.build_problem(
        grid_size,
        source_seed,
        source_perturbation_scale=design["source_perturbation_scale"],
    )
    states = locality.build_common_bo_states(
        problem,
        initialization_size=design["checkpoint_queries"][0],
        total_queries=design["checkpoint_queries"][-1],
        checkpoint_queries=design["checkpoint_queries"],
        observation_noise_variance=design["observation_noise_variance"],
        reference_sample_count=design["ordinary_reference_sample_count"],
        trajectory_seed=_trajectory_seed(grid_size),
        incumbent=0.55,
    )
    matches = [state for state in states if state.checkpoint_label == checkpoint]
    if len(matches) != 1:
        raise RuntimeError("failed to resolve one frozen BO checkpoint")
    return problem, matches[0]


def _proposal_specs() -> tuple[backend.ProposalSpec, ...]:
    prior = json.loads(PRIOR_BACKEND_CONFIG_PATH.read_text())
    return tuple(
        backend.ProposalSpec(
            name=item["name"],
            kind=item["kind"],
            curvature_lambda=item["curvature_lambda"],
            covariance_inflation=item["covariance_inflation"],
            tie_break_rank=rank,
        )
        for rank, item in enumerate(
            prior["proposal_candidates_in_tie_break_order"]
        )
    )


def _selected_initialization_spec(
    grid_size: int, checkpoint: str
) -> backend.ProposalSpec:
    selection = json.loads(PRIOR_SELECTION_PATH.read_text())[
        "selected_proposals"
    ][_state_key(grid_size, checkpoint)]
    matches = [spec for spec in _proposal_specs() if spec.name == selection]
    if len(matches) != 1:
        raise RuntimeError("prior initialization proposal mapping is invalid")
    return matches[0]


def _baseline_spec() -> backend.ProposalSpec:
    return next(
        spec for spec in _proposal_specs() if spec.kind == "BASELINE_INFLATED_LAPLACE"
    )


def _laplace_context(
    state: locality.BOState, problem: locality.Problem
) -> backend.LaplaceContext:
    prior = json.loads(PRIOR_BACKEND_CONFIG_PATH.read_text())["laplace"]
    return backend.build_full_laplace_context(
        state,
        problem,
        gradient_tolerance=prior["gradient_tolerance"],
        maximum_iterations=prior["maximum_iterations"],
    )


def _prepare_initialization_proposal(
    grid_size: int,
    checkpoint: str,
    state: locality.BOState,
    context: backend.LaplaceContext,
) -> backend.GaussianPrecisionProposal:
    return backend.prepare_gaussian_proposal(
        _selected_initialization_spec(grid_size, checkpoint),
        context.mode,
        state.reference_precision,
        context.target_hessian,
    )


def _compare_acquisition_vectors(
    first: Sequence[float], second: Sequence[float], action_indices: Sequence[int]
) -> dict[str, Any]:
    acquisition_a = np.asarray(first, dtype=float)
    acquisition_b = np.asarray(second, dtype=float)
    indices = np.asarray(action_indices, dtype=int)
    local_a = int(np.argmax(acquisition_a))
    local_b = int(np.argmax(acquisition_b))
    regret_a_under_b = float(
        max(0.0, acquisition_b[local_b] - acquisition_b[local_a])
    )
    regret_b_under_a = float(
        max(0.0, acquisition_a[local_a] - acquisition_a[local_b])
    )
    return {
        "action_a": int(indices[local_a]),
        "action_b": int(indices[local_b]),
        "maximum_acquisition_vector_difference": float(
            np.max(np.abs(acquisition_a - acquisition_b))
        ),
        "action_a_regret_under_b": regret_a_under_b,
        "action_b_regret_under_a": regret_b_under_a,
        "maximum_reciprocal_action_regret": max(
            regret_a_under_b, regret_b_under_a
        ),
    }


def _committed_n24_reference(checkpoint: str) -> dict[str, Any]:
    summary = json.loads(PRIOR_SNIS_SUMMARY_PATH.read_text())
    state = next(
        item
        for item in summary["calibration"]
        if item["grid_size"] == 24 and item["checkpoint"] == checkpoint
    )
    return state["prior_n24_pooled_reference"]


def _fresh_validation_n24_reference(
    state: locality.BOState,
    problem: locality.Problem,
    context: backend.LaplaceContext,
    checkpoint: str,
) -> dict[str, Any]:
    proposal = backend.prepare_gaussian_proposal(
        _baseline_spec(),
        context.mode,
        state.reference_precision,
        context.target_hessian,
    )
    incumbent = locality.state_incumbent(state)
    batches = [
        backend.run_snis_batch(
            state,
            problem,
            proposal,
            incumbent=incumbent,
            sample_count=32768,
            proposal_seed=_reference_stream_seed(
                "validation", 24, checkpoint, batch
            ),
        )
        for batch in ("A", "B")
    ]
    pooled = backend.pooled_batch(batches[0], batches[1], state)
    return {
        "kind": "FRESH_VALIDATION_BASELINE_LAPLACE_SNIS_32768x2",
        "sample_count_per_batch": 32768,
        "batch_seeds": [batch.proposal_seed for batch in batches],
        "batch_ess_fractions": [batch.ess_fraction for batch in batches],
        "batch_actions": [batch.action_index for batch in batches],
        "pooled_action": pooled.action_index,
        "acquisition_vector": pooled.acquisition.tolist(),
        "wall_seconds": sum(batch.wall_seconds for batch in batches),
        "factor_work": {
            key: int(sum(batch.work[key] for batch in batches))
            for key in batches[0].work
        },
    }


def _schedule_from_id(config: dict[str, Any], schedule_id: str) -> dict[str, Any]:
    matches = [
        schedule
        for schedule in config["calibration_schedules"]
        if schedule["schedule_id"] == schedule_id
    ]
    if len(matches) != 1:
        raise RuntimeError("unknown frozen ESS schedule")
    return matches[0]


def _worker(
    source_role: str, grid_size: int, checkpoint: str, schedule_id: str
) -> dict[str, Any]:
    started = time.perf_counter()
    config = _read_config()
    schedule = _schedule_from_id(config, schedule_id)
    if source_role == "validation":
        frozen = _read_frozen_schedule()
        if frozen["schedule"] != _schedule_payload(schedule, config):
            raise RuntimeError("validation cannot alter the frozen ESS schedule")
    problem, state = _build_development_state(
        config, source_role, grid_size, checkpoint
    )
    context = _laplace_context(state, problem)
    initialization_proposal = _prepare_initialization_proposal(
        grid_size, checkpoint, state, context
    )
    reference_sampler = backend.prepare_reference_direction_sampler(
        state,
        problem,
        observation_noise_variance=config["state_design"][
            "observation_noise_variance"
        ],
    )
    chain_count = config["existing_transition_contract"]["chain_count"]
    initial_states = [context.mode.copy()]
    initializations = ["LAPLACE_MODE"]
    for chain_index in range(1, chain_count):
        seed = _stream_seed(
            source_role,
            grid_size,
            checkpoint,
            schedule_id,
            f"chain_{chain_index}_initial",
        )
        initial_states.append(
            backend.sample_gaussian_precision(
                initialization_proposal, 1, np.random.default_rng(seed)
            )[0]
        )
        initializations.append("INDEPENDENT_SELECTED_PROPOSAL_DRAW")

    chains: list[backend.EllipticalSliceChain] = []
    for chain_index in range(chain_count):
        chains.append(
            backend.run_elliptical_slice_chain(
                state,
                problem,
                reference_sampler,
                incumbent=locality.state_incumbent(state),
                chain_index=chain_index,
                initial_state=initial_states[chain_index],
                initialization=initializations[chain_index],
                burn_in=schedule["burn_in"],
                retained_count=schedule["retained_per_chain"],
                direction_seed=_stream_seed(
                    source_role,
                    grid_size,
                    checkpoint,
                    schedule_id,
                    f"chain_{chain_index}_reference_direction",
                ),
                slice_seed=_stream_seed(
                    source_role,
                    grid_size,
                    checkpoint,
                    schedule_id,
                    f"chain_{chain_index}_slice",
                ),
                maximum_bracket_evaluations=config[
                    "existing_transition_contract"
                ]["maximum_bracket_evaluations_per_transition"],
            )
        )

    diagnostics = config["additional_diagnostics_not_used_to_change_gate"]
    transition = config["existing_transition_contract"]
    aggregate = backend.aggregate_independence_mh_chains(
        chains,
        state,
        group_a_chains=transition["group_a_chains"],
        group_b_chains=transition["group_b_chains"],
        backend_name="ELLIPTICAL_SLICE_FULL",
        diagnostic_top_action_count=diagnostics["diagnostic_top_action_count"],
        strict_gate_top_action_count=diagnostics[
            "strict_gate_top_action_union_count"
        ],
    )
    total_transitions = sum(chain.total_transitions for chain in chains)
    total_likelihood_evaluations = sum(
        chain.likelihood_evaluations for chain in chains
    )
    aggregate["transition_acceptance_interpretation"] = (
        "one accepted slice state per transition by construction"
    )
    aggregate["reference_direction_backend"] = (
        "EXACT_GAUSSIAN_REFERENCE_MATHERON_SAMPLER"
    )
    aggregate["total_retained_draws"] = int(
        chain_count * schedule["retained_per_chain"]
    )
    aggregate["total_transitions"] = int(total_transitions)
    aggregate["total_likelihood_evaluations"] = int(total_likelihood_evaluations)
    aggregate["mean_likelihood_evaluations_per_transition"] = float(
        total_likelihood_evaluations / total_transitions
    )
    aggregate["factor_energy_evaluations_per_transition"] = float(
        aggregate["factor_work"]["factor_energy_evaluations"] / total_transitions
    )
    aggregate["maximum_likelihood_evaluations_one_transition"] = int(
        max(chain.maximum_likelihood_evaluations_one_transition for chain in chains)
    )
    aggregate["pooled_top_two_gap"] = float(
        aggregate["pooled_top_actions"][0]["acquisition"]
        - aggregate["pooled_top_actions"][1]["acquisition"]
    )

    n24_reference = None
    n24_comparison = None
    if grid_size == 24:
        if source_role == "calibration":
            n24_reference = _committed_n24_reference(checkpoint)
            reference_kind = "COMMITTED_RELIABLE_LAPLACE_SNIS_32768x2"
        else:
            n24_reference = _fresh_validation_n24_reference(
                state, problem, context, checkpoint
            )
            reference_kind = n24_reference["kind"]
        n24_comparison = _compare_acquisition_vectors(
            aggregate["pooled_acquisition"],
            n24_reference["acquisition_vector"],
            state.action_indices,
        )
        n24_comparison["reference_kind"] = reference_kind

    return {
        "backend": "ELLIPTICAL_SLICE_FULL",
        "source_role": source_role,
        "source_seed": _source_seed_for_role(source_role),
        "state_key": _state_key(grid_size, checkpoint),
        "state_id": _state_identifier(source_role, grid_size, checkpoint),
        "grid_size": grid_size,
        "checkpoint": checkpoint,
        "checkpoint_queries": state.checkpoint_queries,
        "state_fingerprint": state.fingerprint(),
        "state_specific_incumbent": locality.state_incumbent(state),
        "schedule_id": schedule_id,
        "schedule": _schedule_payload(schedule, config),
        "initialization_proposal": initialization_proposal.spec.name,
        "ellipse_reference": "EXACT_CURRENT_GAUSSIAN_BO_REFERENCE",
        "randomized_initial_angular_bracket": True,
        "thinning": False,
        "laplace_diagnostics": context.diagnostics,
        "laplace_work": context.work,
        "aggregate": aggregate,
        "n24_reference": n24_reference,
        "n24_reference_comparison": n24_comparison,
        "worker_wall_seconds": time.perf_counter() - started,
        "peak_rss_bytes": locality.peak_rss_bytes(),
    }


def _worker_subprocess(
    source_role: str, grid_size: int, checkpoint: str, schedule_id: str
) -> dict[str, Any]:
    completed = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--source-role",
            source_role,
            "--grid-size",
            str(grid_size),
            "--checkpoint",
            checkpoint,
            "--schedule-id",
            schedule_id,
        ],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def _state_gate(result: dict[str, Any], config: dict[str, Any]) -> dict[str, bool]:
    aggregate = result["aggregate"]
    gate = config["strict_state_gate"]
    components = {
        "rhat_pass": aggregate["maximum_split_rhat"]
        <= gate["maximum_split_rhat_all_required_observables"],
        "gap_ess_pass": aggregate["minimum_leader_challenger_gap_ess"]
        >= gate["minimum_pooled_mcmc_ess_each_required_leader_challenger_gap"],
        "group_action_pass": (
            aggregate["group_action_agreement"]
            if gate["require_independent_group_action_agreement"]
            else True
        ),
        "group_regret_pass": aggregate[
            "maximum_reciprocal_group_action_regret"
        ]
        <= gate["maximum_reciprocal_group_action_regret"],
        "group_vector_pass": aggregate[
            "maximum_group_acquisition_vector_difference"
        ]
        <= gate["maximum_group_acquisition_vector_difference"],
    }
    if result["grid_size"] == 24:
        comparison = result["n24_reference_comparison"]
        reference_gate = config["n24_reference_consistency_gate"]
        components["n24_reference_vector_pass"] = comparison[
            "maximum_acquisition_vector_difference"
        ] <= reference_gate["maximum_pooled_acquisition_vector_difference"]
        components["n24_reference_regret_pass"] = comparison[
            "maximum_reciprocal_action_regret"
        ] <= reference_gate["maximum_reciprocal_action_regret"]
    return {**components, "pass": all(components.values())}


def _gap_signal_to_mcse(result: dict[str, Any], challenger_action: int) -> float:
    matches = [
        row
        for row in result["aggregate"]["scalar_diagnostics"]
        if row["observable_type"] == "LEADER_CHALLENGER_GAP"
        and row["challenger_index"] == challenger_action
    ]
    if not matches:
        return float("inf")
    row = matches[0]
    if row["mcse"] == 0.0:
        return 0.0 if row["mean"] == 0.0 else float("inf")
    return abs(float(row["mean"])) / float(row["mcse"])


def _failure_mechanism(
    result: dict[str, Any], config: dict[str, Any]
) -> str:
    if result["gate"]["pass"]:
        return "PASS"
    aggregate = result["aggregate"]
    if not result["gate"]["rhat_pass"] or not result["gate"]["gap_ess_pass"]:
        return "MCMC_NONCONVERGENCE"
    regret = aggregate["maximum_reciprocal_group_action_regret"]
    if regret > 0.01:
        return "MATERIAL_DECISION_UNCERTAINTY"
    non_vector_components = {
        key: value
        for key, value in result["gate"].items()
        if key not in {"group_vector_pass", "pass"}
    }
    if (
        all(non_vector_components.values())
        and not result["gate"]["group_vector_pass"]
    ):
        return "GLOBAL_VECTOR_ONLY"
    pooled_action = aggregate["pooled_action"]
    disagreement_actions = {
        aggregate["group_a_action"],
        aggregate["group_b_action"],
    } - {pooled_action}
    ratios = [
        _gap_signal_to_mcse(result, action) for action in disagreement_actions
    ]
    threshold = config["additional_diagnostics_not_used_to_change_gate"][
        "near_tie_gap_signal_to_mcse_maximum"
    ]
    if regret <= 0.01 and ratios and max(ratios) <= threshold:
        return "FINITE_MC_NEAR_TIE"
    raise RuntimeError(
        "a longest-schedule failure did not match a frozen mechanism category"
    )


def _run_schedule(
    source_role: str, schedule: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    states: list[dict[str, Any]] = []
    for grid_size in GRID_SIZES:
        for checkpoint in CHECKPOINTS:
            state = _worker_subprocess(
                source_role, grid_size, checkpoint, schedule["schedule_id"]
            )
            state["gate"] = _state_gate(state, config)
            states.append(state)
            aggregate = state["aggregate"]
            print(
                f"[standard ESS {source_role}] {schedule['schedule_id']} "
                f"n={grid_size} {checkpoint}: "
                f"Rhat={aggregate['maximum_split_rhat']:.5f} "
                f"gapESS={aggregate['minimum_leader_challenger_gap_ess']:.0f} "
                f"groups=({aggregate['group_a_action']},"
                f"{aggregate['group_b_action']}) "
                f"diff={aggregate['maximum_group_acquisition_vector_difference']:.5f} "
                f"regret={aggregate['maximum_reciprocal_group_action_regret']:.5f} "
                f"pass={state['gate']['pass']}",
                flush=True,
            )
    return {
        **schedule,
        "source_role": source_role,
        "states": states,
        "global_pass": all(state["gate"]["pass"] for state in states),
    }


def _flatten_schedule_rows(
    schedules: Sequence[dict[str, Any]], *, phase: str
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for schedule in schedules:
        for state in schedule["states"]:
            aggregate = state["aggregate"]
            comparison = state.get("n24_reference_comparison") or {}
            rows.append(
                {
                    "phase": phase,
                    "source_role": state["source_role"],
                    "source_seed": state["source_seed"],
                    "schedule_id": state["schedule_id"],
                    "burn_in": state["schedule"]["burn_in"],
                    "retained_per_chain": state["schedule"]["retained_per_chain"],
                    "chain_count": aggregate["chain_count"],
                    "thinning": state["thinning"],
                    "grid_size": state["grid_size"],
                    "checkpoint": state["checkpoint"],
                    "state_id": state["state_id"],
                    "state_fingerprint": state["state_fingerprint"],
                    "state_specific_incumbent": state[
                        "state_specific_incumbent"
                    ],
                    "pooled_action": aggregate["pooled_action"],
                    "group_a_action": aggregate["group_a_action"],
                    "group_b_action": aggregate["group_b_action"],
                    "group_action_agreement": aggregate["group_action_agreement"],
                    "maximum_split_rhat": aggregate["maximum_split_rhat"],
                    "maximum_split_rhat_all_diagnostics": aggregate[
                        "maximum_split_rhat_all_diagnostics"
                    ],
                    "minimum_gap_ess": aggregate[
                        "minimum_leader_challenger_gap_ess"
                    ],
                    "minimum_diagnostic_gap_ess": aggregate[
                        "minimum_diagnostic_leader_challenger_gap_ess"
                    ],
                    "maximum_group_acquisition_vector_difference": aggregate[
                        "maximum_group_acquisition_vector_difference"
                    ],
                    "group_a_action_regret_under_group_b": aggregate[
                        "group_a_action_regret_under_group_b"
                    ],
                    "group_b_action_regret_under_group_a": aggregate[
                        "group_b_action_regret_under_group_a"
                    ],
                    "maximum_reciprocal_group_action_regret": aggregate[
                        "maximum_reciprocal_group_action_regret"
                    ],
                    "pooled_top_two_gap": aggregate["pooled_top_two_gap"],
                    "pooled_top_ten": json.dumps(
                        aggregate["pooled_top_actions"], sort_keys=True
                    ),
                    "group_a_acquisition": json.dumps(
                        aggregate["group_a_acquisition"]
                    ),
                    "group_b_acquisition": json.dumps(
                        aggregate["group_b_acquisition"]
                    ),
                    "pooled_acquisition": json.dumps(
                        aggregate["pooled_acquisition"]
                    ),
                    "total_retained_draws": aggregate["total_retained_draws"],
                    "total_transitions": aggregate["total_transitions"],
                    "total_factor_energy_evaluations": aggregate["factor_work"][
                        "factor_energy_evaluations"
                    ],
                    "total_likelihood_evaluations": aggregate[
                        "total_likelihood_evaluations"
                    ],
                    "mean_likelihood_evaluations_per_transition": aggregate[
                        "mean_likelihood_evaluations_per_transition"
                    ],
                    "factor_energy_evaluations_per_transition": aggregate[
                        "factor_energy_evaluations_per_transition"
                    ],
                    "worker_wall_seconds": state["worker_wall_seconds"],
                    "peak_rss_bytes": state["peak_rss_bytes"],
                    "n24_reference_kind": comparison.get("reference_kind"),
                    "n24_reference_vector_difference": comparison.get(
                        "maximum_acquisition_vector_difference"
                    ),
                    "n24_reference_reciprocal_regret": comparison.get(
                        "maximum_reciprocal_action_regret"
                    ),
                    **state["gate"],
                    "failure_mechanism": state.get("failure_mechanism"),
                }
            )
    return rows


def _gap_rows(
    schedules: Sequence[dict[str, Any]], *, phase: str
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for schedule in schedules:
        for state in schedule["states"]:
            pooled_top_ten = {
                item["action_index"] for item in state["aggregate"]["pooled_top_actions"]
            }
            for diagnostic in state["aggregate"]["scalar_diagnostics"]:
                if diagnostic["observable_type"] not in {
                    "TOP_ACTION_UTILITY",
                    "LEADER_CHALLENGER_GAP",
                }:
                    continue
                mcse = float(diagnostic["mcse"])
                mean = float(diagnostic["mean"])
                rows.append(
                    {
                        "phase": phase,
                        "source_role": state["source_role"],
                        "source_seed": state["source_seed"],
                        "schedule_id": state["schedule_id"],
                        "grid_size": state["grid_size"],
                        "checkpoint": state["checkpoint"],
                        **diagnostic,
                        "pooled_top_ten_member": (
                            diagnostic["action_index"] in pooled_top_ten
                            or diagnostic["challenger_index"] in pooled_top_ten
                        ),
                        "absolute_estimate_over_mcse": (
                            abs(mean) / mcse
                            if mcse > 0.0
                            else (0.0 if mean == 0.0 else float("inf"))
                        ),
                    }
                )
    return rows


def _runtime_projection(
    states: Sequence[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    median_24 = float(
        np.median(
            [state["worker_wall_seconds"] for state in states if state["grid_size"] == 24]
        )
    )
    median_40 = float(
        np.median(
            [state["worker_wall_seconds"] for state in states if state["grid_size"] == 40]
        )
    )
    exponent = math.log(median_40 / median_24) / math.log(40.0 / 24.0)
    projection = config["runtime_projection"]
    per_n = {
        str(n): float(median_24 * (n / 24.0) ** exponent)
        for n in projection["prospective_domain_sizes"]
    }
    total = float(
        projection["states_per_domain_size"] * sum(per_n.values())
    )
    peak = max(state["peak_rss_bytes"] for state in states)
    return {
        "observed_median_shadow_seconds_n24": median_24,
        "observed_median_shadow_seconds_n40": median_40,
        "observed_power_law_exponent": exponent,
        "estimated_shadow_seconds_per_n": per_n,
        "states_per_domain_size": projection["states_per_domain_size"],
        "estimated_total_shadow_seconds_45_states": total,
        "estimated_total_shadow_hours_45_states": total / 3600.0,
        "expected_peak_rss_bytes": peak,
        "expected_peak_ram_gb": peak / 1_000_000_000.0,
        "suitable_for_16_gb_local_macbook": peak
        < projection["local_ram_capacity_gb"] * 1_000_000_000,
        "shadow_excluded_from_scientific_timing": projection[
            "shadow_excluded_from_scientific_timing"
        ],
    }


def _strategy_decision(
    calibration: Sequence[dict[str, Any]],
    validation: Sequence[dict[str, Any]],
) -> tuple[str, str, str]:
    selected = next((item for item in calibration if item["global_pass"]), None)
    if selected is not None and validation and validation[0]["global_pass"]:
        return (
            "STANDARD_ESS_FULL_REFERENCE_PASS",
            "CASE_A_STRICT_GATE_PASS",
            "Use the frozen standard-ESS schedule as the future replacement E2 FULL-shadow backend; keep the exact strict MCMC/action/vector gate.",
        )
    decision_schedule = validation[0] if validation else calibration[-1]
    n40 = [state for state in decision_schedule["states"] if state["grid_size"] == 40]
    healthy = all(
        state["gate"]["rhat_pass"] and state["gate"]["gap_ess_pass"]
        for state in n40
    )
    regret_within_near_tie_scale = all(
        state["aggregate"]["maximum_reciprocal_group_action_regret"] <= 0.01
        for state in n40
    )
    if healthy and regret_within_near_tie_scale:
        return (
            "BACKEND_HEALTHY_REFERENCE_RULE_TOO_BRITTLE",
            "CASE_B_HEALTHY_NEAR_TIE_DECISIONS",
            "Do not develop another sampler next. Formulate and prospectively audit a decision-aligned FULL-reference rule based on acquisition-gap Monte Carlo uncertainty.",
        )
    if healthy:
        return (
            "STANDARD_ESS_MATERIAL_DECISION_UNCERTAINTY",
            "CASE_C_MATERIAL_DECISION_DISAGREEMENT",
            "Run one separate sampler-development decision for a fixed Gaussian-approximation/generalized elliptical-slice construction with exact residual correction.",
        )
    return (
        "STANDARD_ESS_MCMC_NONCONVERGENCE",
        "CASE_D_STANDARD_ESS_INADEQUATE",
        "Treat standard ESS as inadequate at n=40 and make a separate sampler-method decision.",
    )


def _environment() -> dict[str, str]:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }


def _write_results_markdown(summary: dict[str, Any]) -> None:
    lines = [
        "# E2 Standard-ESS Resolution Gate",
        "",
        f"Terminal classification: **`{summary['terminal_classification']}`**",
        "",
        "Status: **DEVELOPMENT ONLY — ZERO PROSPECTIVE E2 SOURCE ACCESS**",
        "",
        f"Starting fetched `main` SHA: `{summary['starting_main_sha']}`.",
        "The historical backend-rescue classification",
        "`FULL_REFERENCE_BACKEND_UNRESOLVED` remains unchanged for that prior protocol.",
        "No scientific preregistration was created or executed.",
        "",
        "## Schedules reached",
        "",
        "| schedule | burn-in | retained/chain | global calibration pass |",
        "|---|---:|---:|---|",
    ]
    for schedule in summary["calibration_schedules"]:
        lines.append(
            f"| {schedule['schedule_id']} | {schedule['burn_in']} | "
            f"{schedule['retained_per_chain']} | {schedule['global_pass']} |"
        )
    lines.extend(
        [
            "",
            "## Six-state calibration diagnostics",
            "",
            "| schedule | n | checkpoint | Rhat | min gap ESS | groups | vector diff | max regret | prior n=24 diff/regret | gate | mechanism |",
            "|---|---:|---|---:|---:|---|---:|---:|---|---|---|",
        ]
    )
    for schedule in summary["calibration_schedules"]:
        for state in schedule["states"]:
            aggregate = state["aggregate"]
            comparison = state.get("n24_reference_comparison")
            prior = (
                f"{comparison['maximum_acquisition_vector_difference']:.6f}/"
                f"{comparison['maximum_reciprocal_action_regret']:.6f}"
                if comparison
                else "—"
            )
            lines.append(
                f"| {schedule['schedule_id']} | {state['grid_size']} | "
                f"{state['checkpoint']} | {aggregate['maximum_split_rhat']:.6f} | "
                f"{aggregate['minimum_leader_challenger_gap_ess']:.1f} | "
                f"{aggregate['group_a_action']}/{aggregate['group_b_action']} | "
                f"{aggregate['maximum_group_acquisition_vector_difference']:.6f} | "
                f"{aggregate['maximum_reciprocal_group_action_regret']:.6f} | "
                f"{prior} | {state['gate']['pass']} | "
                f"{state.get('failure_mechanism') or '—'} |"
            )
    frozen = summary.get("frozen_schedule")
    lines.extend(["", "## Schedule freeze and held-out validation", ""])
    if frozen is None:
        lines.append(
            "No calibration schedule passed globally, so no schedule was frozen and the fresh final development-validation seed was not derived or accessed."
        )
    else:
        lines.extend(
            [
                f"First passing schedule: `{frozen['schedule']['schedule_id']}`; schedule payload SHA-256 `{frozen['schedule_sha256']}`.",
                f"Fresh validation seed: `{frozen['final_validation']['seed']}` (`DEVELOPMENT ONLY`).",
                "",
                "| n | checkpoint | Rhat | min gap ESS | groups | vector diff | max regret | gate |",
                "|---:|---|---:|---:|---|---:|---:|---|",
            ]
        )
        for state in summary["validation"][0]["states"]:
            aggregate = state["aggregate"]
            lines.append(
                f"| {state['grid_size']} | {state['checkpoint']} | "
                f"{aggregate['maximum_split_rhat']:.6f} | "
                f"{aggregate['minimum_leader_challenger_gap_ess']:.1f} | "
                f"{aggregate['group_a_action']}/{aggregate['group_b_action']} | "
                f"{aggregate['maximum_group_acquisition_vector_difference']:.6f} | "
                f"{aggregate['maximum_reciprocal_group_action_regret']:.6f} | "
                f"{state['gate']['pass']} |"
            )
    runtime = summary["runtime_projection"]
    decision_record = (
        summary["validation"][0]
        if summary["validation"]
        else next(
            schedule
            for schedule in summary["calibration_schedules"]
            if schedule["schedule_id"] == summary["decision_schedule_id"]
        )
    )
    lines.extend(
        [
            "",
            "## MCSE interpretation",
            "",
            summary["mcse_interpretation"],
            "Full group-A, group-B, and pooled acquisition vectors, pooled top-ten actions, per-action MCSE, and leader/top-ten gap MCSE are in the CSV and JSON records. These diagnostics did not alter the strict gate.",
            "",
            "## Runtime and memory",
            "",
            "| n | checkpoint | likelihood eval/transition | factor eval/transition | total factor eval | wall s | peak RSS GB |",
            "|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for state in decision_record["states"]:
        aggregate = state["aggregate"]
        lines.append(
            f"| {state['grid_size']} | {state['checkpoint']} | "
            f"{aggregate['mean_likelihood_evaluations_per_transition']:.4f} | "
            f"{aggregate['factor_energy_evaluations_per_transition']:.1f} | "
            f"{aggregate['factor_work']['factor_energy_evaluations']} | "
            f"{state['worker_wall_seconds']:.3f} | "
            f"{state['peak_rss_bytes'] / 1_000_000_000.0:.3f} |"
        )
    lines.extend(
        [
            "",
            f"Observed median state time was `{runtime['observed_median_shadow_seconds_n24']:.3f}` s at n=24 and `{runtime['observed_median_shadow_seconds_n40']:.3f}` s at n=40.",
            f"The observed-work projection for 45 shadows is `{runtime['estimated_total_shadow_seconds_45_states']:.1f}` s (`{runtime['estimated_total_shadow_hours_45_states']:.3f}` h).",
            f"Peak isolated-worker RSS was `{runtime['expected_peak_ram_gb']:.3f}` GB; 16 GB local suitability: `{runtime['suitable_for_16_gb_local_macbook']}`.",
            "High-fidelity shadow time remains excluded from the paper's adaptive-versus-FULL routine timing comparison.",
            "",
            "## Decision",
            "",
            f"Strategy case: **`{summary['strategy_case']}`**.",
            "",
            summary["recommended_next_action"],
            "",
            f"Recommended backend for a future replacement E2 preregistration: `{summary['recommended_backend']}`.",
            f"Recommended reliability rule to freeze now: {summary['recommended_reliability_rule']}",
            "",
            "No prospective source field, trajectory, or posterior state was constructed. The PDE model, BO problem, factor selection, scientific thresholds, E3, and superseded/unrun preregistration remain unchanged.",
        ]
    )
    (OUTPUT_DIRECTORY / "RESULTS.md").write_text("\n".join(lines) + "\n")


def run_resolution_gate() -> dict[str, Any]:
    started = time.perf_counter()
    config = _read_config()
    result_paths = [
        OUTPUT_DIRECTORY / "schedule_metrics.csv",
        OUTPUT_DIRECTORY / "gap_mcse_metrics.csv",
        OUTPUT_DIRECTORY / "validation_metrics.csv",
        OUTPUT_DIRECTORY / "summary.json",
        OUTPUT_DIRECTORY / "RESULTS.md",
    ]
    if any(path.exists() for path in result_paths) or FROZEN_SCHEDULE_PATH.exists():
        raise RuntimeError("resolution-gate outputs already exist; refusing overwrite")

    calibration: list[dict[str, Any]] = []
    selected_schedule: dict[str, Any] | None = None
    for schedule in config["calibration_schedules"]:
        record = _run_schedule("calibration", schedule, config)
        calibration.append(record)
        if record["global_pass"]:
            selected_schedule = schedule
            break

    validation: list[dict[str, Any]] = []
    frozen = None
    if selected_schedule is not None:
        frozen = _freeze_schedule(selected_schedule, config)
        validation.append(_run_schedule("validation", selected_schedule, config))

    decision_schedule = validation[0] if validation else calibration[-1]
    for state in decision_schedule["states"]:
        state["failure_mechanism"] = _failure_mechanism(state, config)
    classification, strategy_case, recommendation = _strategy_decision(
        calibration, validation
    )

    schedule_rows = _flatten_schedule_rows(calibration, phase="CALIBRATION")
    pd.DataFrame(schedule_rows).to_csv(
        OUTPUT_DIRECTORY / "schedule_metrics.csv", index=False
    )
    gap_rows = _gap_rows(calibration, phase="CALIBRATION")
    if validation:
        validation_rows = _flatten_schedule_rows(validation, phase="VALIDATION")
        pd.DataFrame(validation_rows).to_csv(
            OUTPUT_DIRECTORY / "validation_metrics.csv", index=False
        )
        gap_rows.extend(_gap_rows(validation, phase="VALIDATION"))
    pd.DataFrame(gap_rows).to_csv(
        OUTPUT_DIRECTORY / "gap_mcse_metrics.csv", index=False
    )

    runtime = _runtime_projection(decision_schedule["states"], config)
    mechanisms = {
        state["state_key"]: state["failure_mechanism"]
        for state in decision_schedule["states"]
    }
    if classification == "STANDARD_ESS_FULL_REFERENCE_PASS":
        recommended_backend = "STANDARD_ELLIPTICAL_SLICE_FULL"
        reliability_rule = (
            "freeze the selected 8-chain/no-thinning schedule and require Rhat <= 1.01, "
            "every required leader/challenger gap ESS >= 1000, exact group-action "
            "agreement, reciprocal group regret <= 0.005, full-vector difference <= "
            "0.01, and the unchanged n=24 reference-consistency checks"
        )
    elif classification == "BACKEND_HEALTHY_REFERENCE_RULE_TOO_BRITTLE":
        recommended_backend = (
            "STANDARD_ELLIPTICAL_SLICE_FULL_PENDING_DECISION_ALIGNED_RULE_AUDIT"
        )
        reliability_rule = (
            "none from this diagnostic; retain the current strict rule historically "
            "and complete the recommended separate prospective rule/backend task"
        )
    else:
        recommended_backend = "NONE_YET"
        reliability_rule = (
            "none from this diagnostic; retain the current strict rule historically "
            "and complete the recommended separate prospective rule/backend task"
        )
    if classification == "BACKEND_HEALTHY_REFERENCE_RULE_TOO_BRITTLE":
        mcse_interpretation = (
            "At the decision schedule, n=40 Rhat and required gap ESS were healthy and "
            "all reciprocal regrets were at most 0.01; the remaining strict failures are "
            "therefore decision-near-tie failures rather than evidence "
            "of an unmixed standard-ESS chain."
        )
    elif classification == "STANDARD_ESS_FULL_REFERENCE_PASS":
        mcse_interpretation = (
            "All strict convergence, decision, vector, and n=24 reference checks passed "
            "on calibration and the untouched fresh validation source."
        )
    elif classification == "STANDARD_ESS_MATERIAL_DECISION_UNCERTAINTY":
        mcse_interpretation = (
            "Well-mixed chains retained reciprocal action regret above 0.01, so the "
            "remaining disagreement is material rather than a finite-MC near tie."
        )
    else:
        mcse_interpretation = (
            "The longest evaluated schedule retained an Rhat or required gap-ESS "
            "failure, so standard ESS itself did not establish convergence."
        )

    _verify_prior_outputs(config)
    summary = {
        "diagnostic_id": config["diagnostic_id"],
        "development_only": True,
        "starting_main_sha": config["starting_main_sha"],
        "diagnostic_config_sha256": EXPECTED_CONFIG_SHA256,
        "calibration_source_seed": CALIBRATION_SEED,
        "fresh_final_validation_seed": (
            frozen["final_validation"]["seed"] if frozen else None
        ),
        "fresh_final_validation_seed_classification": (
            "DEVELOPMENT ONLY" if frozen else "NOT DERIVED_OR_ACCESSED"
        ),
        "prospective_source_seeds_accessed": False,
        "prospective_source_fields_constructed": False,
        "scientific_preregistration_created_or_executed": False,
        "previous_backend_rescue_classification_preserved": (
            "FULL_REFERENCE_BACKEND_UNRESOLVED"
        ),
        "calibration_schedules_reached": [
            item["schedule_id"] for item in calibration
        ],
        "calibration_schedules": calibration,
        "frozen_schedule": frozen,
        "validation": validation,
        "decision_schedule_id": decision_schedule["schedule_id"],
        "longest_or_selected_state_mechanisms": mechanisms,
        "mcse_interpretation": mcse_interpretation,
        "runtime_projection": runtime,
        "terminal_classification": classification,
        "strategy_case": strategy_case,
        "recommended_next_action": recommendation,
        "recommended_backend": recommended_backend,
        "recommended_reliability_rule": reliability_rule,
        "wall_seconds": time.perf_counter() - started,
        "peak_rss_bytes": max(
            state["peak_rss_bytes"]
            for schedule in calibration + validation
            for state in schedule["states"]
        ),
        "prior_backend_rescue_output_hashes_unchanged": True,
        "shadow_only_contract": backend.shadow_only_contract(),
        "environment": _environment(),
    }
    (OUTPUT_DIRECTORY / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    _write_results_markdown(summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--source-role", choices=("calibration", "validation"))
    parser.add_argument("--grid-size", type=int, choices=GRID_SIZES)
    parser.add_argument("--checkpoint", choices=CHECKPOINTS)
    parser.add_argument("--schedule-id", choices=("S1", "S2", "S3"))
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    if arguments.worker:
        if not all(
            (
                arguments.source_role,
                arguments.grid_size,
                arguments.checkpoint,
                arguments.schedule_id,
            )
        ):
            raise RuntimeError("worker arguments are incomplete")
        result = _worker(
            arguments.source_role,
            arguments.grid_size,
            arguments.checkpoint,
            arguments.schedule_id,
        )
        print(json.dumps(result, sort_keys=True))
        return
    result = run_resolution_gate()
    print(
        json.dumps(
            {
                "terminal_classification": result["terminal_classification"],
                "calibration_schedules_reached": result[
                    "calibration_schedules_reached"
                ],
                "fresh_final_validation_seed": result[
                    "fresh_final_validation_seed"
                ],
                "wall_seconds": result["wall_seconds"],
                "peak_rss_gb": result["peak_rss_bytes"] / 1_000_000_000.0,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
