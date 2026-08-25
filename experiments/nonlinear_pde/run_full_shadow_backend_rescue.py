#!/usr/bin/env python3
"""Development-only rescue of the nonlinear-E2 FULL-shadow backend.

Only the two explicitly named development source seeds can reach
``build_problem``.  Calibration pilots choose a proposal for each frozen
``(n, checkpoint)`` case; the resulting mapping is saved before the held-out
validation workers start.  This script cannot run the scientific E2 driver.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any

import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from conditioned_bo import full_shadow_backend as backend  # noqa: E402
from conditioned_bo import nonlinear_pde_locality as locality  # noqa: E402


OUTPUT_DIRECTORY = (
    Path(__file__).resolve().parent / "outputs" / "full_shadow_backend_rescue"
)
CONFIG_PATH = OUTPUT_DIRECTORY / "backend_config.json"
MCMC_CONFIG_PATH = OUTPUT_DIRECTORY / "mcmc_config.json"
ELLIPTICAL_CONFIG_PATH = OUTPUT_DIRECTORY / "elliptical_slice_config.json"
EXPECTED_CONFIG_SHA256 = (
    "408614b2c67411a0ad0ac59e5dfb973767f11ae127e9eb96b7c3422aed396258"
)
EXPECTED_MCMC_CONFIG_SHA256 = (
    "0fff47831b6f0a59fe93a1cd7cfecb775e1c5cfed0a887cc6f2ce26e0bd85d13"
)
EXPECTED_PRE_MH_SNIS_SUMMARY_SHA256 = (
    "fd8eea3aebab932ae38af82563c34a128ea98a5d20b8f8d53cbbfe1ba01112f1"
)
EXPECTED_ELLIPTICAL_CONFIG_SHA256 = (
    "0a71fb13162259b7524e0309a7c806c4c4efa2f1fdf5b6ba6e5c4d1b07faf18c"
)
EXPECTED_PRE_ESS_MH_SUMMARY_SHA256 = (
    "757043402d63039bfec8cb056b11355d806d506be978a7328a3486c6e3c671d3"
)
PRIOR_DIRECTORY = (
    Path(__file__).resolve().parent
    / "outputs"
    / "full_shadow_reliability_diagnostic"
)
CALIBRATION_SEED = 2026082401
VALIDATION_LABEL = "E2_FULL_SHADOW_BACKEND_VALIDATE_V1"
VALIDATION_SEED = 3321078991
PROSPECTIVE_SOURCE_SEEDS = (4215109622, 1083605379, 4045758625)
ALLOWED_SOURCE_ROLES = {
    "calibration": CALIBRATION_SEED,
    "validation": VALIDATION_SEED,
}
GRID_SIZES = (24, 40)
CHECKPOINTS = ("early", "middle", "late")
CHECKPOINT_QUERIES = {"early": 4, "middle": 8, "late": 12}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def derive_validation_seed(label: str = VALIDATION_LABEL) -> int:
    raw = int.from_bytes(hashlib.sha256(label.encode("utf-8")).digest()[:8], "big")
    return raw % (2**32 - 1)


def _read_config() -> dict[str, Any]:
    if _sha256(CONFIG_PATH) != EXPECTED_CONFIG_SHA256:
        raise RuntimeError("development backend configuration changed after freeze")
    config = json.loads(CONFIG_PATH.read_text())
    if derive_validation_seed() != VALIDATION_SEED:
        raise RuntimeError("held-out development-validation seed derivation changed")
    configured = config["development_sources"]
    if configured["calibration"]["seed"] != CALIBRATION_SEED:
        raise RuntimeError("calibration source seed changed")
    if configured["validation"]["seed"] != VALIDATION_SEED:
        raise RuntimeError("held-out validation source seed changed")
    if tuple(configured["prospective_source_seeds_forbidden"]) != (
        PROSPECTIVE_SOURCE_SEEDS
    ):
        raise RuntimeError("prospective source-seed denylist changed")
    if set(ALLOWED_SOURCE_ROLES.values()) & set(PROSPECTIVE_SOURCE_SEEDS):
        raise RuntimeError("a development source seed overlaps a prospective seed")
    for filename, expected in config["prior_reference"]["committed_sha256"].items():
        if _sha256(PRIOR_DIRECTORY / filename) != expected:
            raise RuntimeError(f"prior diagnostic was modified: {filename}")
    return config


def _read_mcmc_config() -> dict[str, Any]:
    if _sha256(MCMC_CONFIG_PATH) != EXPECTED_MCMC_CONFIG_SHA256:
        raise RuntimeError("independence-MH configuration changed after freeze")
    config = json.loads(MCMC_CONFIG_PATH.read_text())
    if config["parent_backend_config_sha256"] != EXPECTED_CONFIG_SHA256:
        raise RuntimeError("independence-MH parent configuration mismatch")
    if _sha256(OUTPUT_DIRECTORY / "selected_proposals.json") != config[
        "selected_proposals_sha256"
    ]:
        raise RuntimeError("calibrated proposal mapping changed before MCMC")
    return config


def _read_elliptical_config() -> dict[str, Any]:
    if _sha256(ELLIPTICAL_CONFIG_PATH) != EXPECTED_ELLIPTICAL_CONFIG_SHA256:
        raise RuntimeError("elliptical-slice configuration changed after freeze")
    config = json.loads(ELLIPTICAL_CONFIG_PATH.read_text())
    if config["parent_backend_config_sha256"] != EXPECTED_CONFIG_SHA256:
        raise RuntimeError("elliptical-slice parent configuration mismatch")
    if _sha256(OUTPUT_DIRECTORY / "selected_proposals.json") != config[
        "selected_proposals_sha256"
    ]:
        raise RuntimeError("proposal mapping changed before elliptical slice")
    return config


def _proposal_specs(config: dict[str, Any]) -> tuple[backend.ProposalSpec, ...]:
    return tuple(
        backend.ProposalSpec(
            name=item["name"],
            kind=item["kind"],
            curvature_lambda=item["curvature_lambda"],
            covariance_inflation=item["covariance_inflation"],
            tie_break_rank=rank,
        )
        for rank, item in enumerate(
            config["proposal_candidates_in_tie_break_order"]
        )
    )


def _source_seed_for_role(source_role: str) -> int:
    """Resolve only a permanent development role; arbitrary seeds are forbidden."""

    if source_role not in ALLOWED_SOURCE_ROLES:
        raise ValueError("source role must be calibration or validation")
    source_seed = ALLOWED_SOURCE_ROLES[source_role]
    if source_seed in PROSPECTIVE_SOURCE_SEEDS:
        raise RuntimeError("prospective E2 source seed access is forbidden")
    return source_seed


def _trajectory_seed(grid_size: int) -> int:
    label = f"E2_LOCALITY_TRAJECTORY_V1:{grid_size}:-2"
    return int(hashlib.sha256(label.encode("utf-8")).hexdigest()[:8], 16)


def _build_development_state(
    config: dict[str, Any],
    source_role: str,
    grid_size: int,
    checkpoint: str,
) -> tuple[locality.Problem, locality.BOState]:
    source_seed = _source_seed_for_role(source_role)
    design = config["state_design_unchanged"]
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


def _state_key(grid_size: int, checkpoint: str) -> str:
    return f"n{grid_size}_{checkpoint}"


def _state_identifier(source_role: str, grid_size: int, checkpoint: str) -> str:
    return (
        f"{source_role}_n{grid_size}_r-2_{checkpoint}_"
        f"q{CHECKPOINT_QUERIES[checkpoint]}"
    )


def _stream_seed(
    source_role: str,
    grid_size: int,
    checkpoint: str,
    phase: str,
    stream: str,
) -> int:
    label = (
        "E2_FULL_SHADOW_BACKEND_RESCUE_V1:"
        f"{source_role}:{grid_size}:{checkpoint}:{phase}:{stream}"
    )
    value = int.from_bytes(hashlib.sha256(label.encode()).digest()[:8], "big")
    seed = value % (2**32 - 1)
    if seed in PROSPECTIVE_SOURCE_SEEDS:
        raise RuntimeError("an RNG stream collided with a prospective source seed")
    return seed


def _prior_batch_seeds(grid_size: int, checkpoint: str) -> tuple[int, int]:
    identifier = (
        f"n{grid_size}_r-2_{checkpoint}_q{CHECKPOINT_QUERIES[checkpoint]}"
    )
    state_seed = int(hashlib.sha256(identifier.encode()).hexdigest()[:8], 16)
    base = state_seed + 900_000 + 100_003 * 2
    return base, base + 1


def _serialize_batch(
    batch: backend.SNISBatch, state: locality.BOState
) -> dict[str, Any]:
    summary = backend.batch_public_summary(batch, state)
    order = batch.top_five_local
    summary.update(
        {
            "rank1_rank2_gap": float(
                batch.acquisition[int(order[0])] - batch.acquisition[int(order[1])]
            ),
            "acquisition_vector": batch.acquisition.tolist(),
        }
    )
    return summary


def _laplace_context(
    config: dict[str, Any], state: locality.BOState, problem: locality.Problem
) -> backend.LaplaceContext:
    return backend.build_full_laplace_context(
        state,
        problem,
        gradient_tolerance=config["laplace"]["gradient_tolerance"],
        maximum_iterations=config["laplace"]["maximum_iterations"],
    )


def _prepare(
    spec: backend.ProposalSpec,
    state: locality.BOState,
    context: backend.LaplaceContext,
) -> backend.GaussianPrecisionProposal:
    return backend.prepare_gaussian_proposal(
        spec,
        context.mode,
        state.reference_precision,
        context.target_hessian,
    )


def _run_production_pair(
    config: dict[str, Any],
    state: locality.BOState,
    problem: locality.Problem,
    context: backend.LaplaceContext,
    spec: backend.ProposalSpec,
    *,
    source_role: str,
) -> tuple[backend.SNISBatch, backend.SNISBatch]:
    proposal = _prepare(spec, state, context)
    sample_count = config["calibration"]["production_samples_per_batch"]
    seeds = [
        _stream_seed(
            source_role,
            problem.grid_size,
            state.checkpoint_label,
            "production",
            f"batch_{batch}",
        )
        for batch in (0, 1)
    ]
    incumbent = locality.state_incumbent(state)
    batches = tuple(
        backend.run_snis_batch(
            state,
            problem,
            proposal,
            incumbent=incumbent,
            sample_count=sample_count,
            proposal_seed=seed,
        )
        for seed in seeds
    )
    return batches[0], batches[1]


def _verify_prior_n24_reference(
    state: locality.BOState,
    problem: locality.Problem,
    context: backend.LaplaceContext,
    specs: tuple[backend.ProposalSpec, ...],
) -> tuple[backend.SNISBatch, list[dict[str, Any]]]:
    baseline = next(
        spec for spec in specs if spec.kind == "BASELINE_INFLATED_LAPLACE"
    )
    proposal = _prepare(baseline, state, context)
    seeds = _prior_batch_seeds(24, state.checkpoint_label)
    incumbent = locality.state_incumbent(state)
    batches = [
        backend.run_snis_batch(
            state,
            problem,
            proposal,
            incumbent=incumbent,
            sample_count=32768,
            proposal_seed=seed,
        )
        for seed in seeds
    ]
    prior_json = json.loads((PRIOR_DIRECTORY / "diagnostic.json").read_text())
    prior_case = next(
        case
        for case in prior_json["cases"]
        if case["grid_size"] == 24
        and case["checkpoint"] == state.checkpoint_label
        and case["sample_count_per_batch"] == 32768
    )
    observed_actions = [batch.action_index for batch in batches]
    observed_ess = [batch.ess_fraction for batch in batches]
    if observed_actions != prior_case["batch_actions"]:
        raise RuntimeError("prior n=24 action reference did not reproduce")
    expected_ess = [
        prior_case["ess_fraction_batch_a"],
        prior_case["ess_fraction_batch_b"],
    ]
    if not np.allclose(observed_ess, expected_ess, rtol=2e-13, atol=2e-13):
        raise RuntimeError("prior n=24 ESS reference did not reproduce")
    pooled = backend.pooled_batch(batches[0], batches[1], state)
    verification = [
        {
            "batch": label,
            "action_index": batch.action_index,
            "ess_fraction": batch.ess_fraction,
            "expected_action_index": prior_case["batch_actions"][index],
            "expected_ess_fraction": expected_ess[index],
        }
        for index, (label, batch) in enumerate(zip(("A", "B"), batches))
    ]
    return pooled, verification


def _calibration_worker(
    grid_size: int, checkpoint: str
) -> dict[str, Any]:
    started = time.perf_counter()
    config = _read_config()
    specs = _proposal_specs(config)
    problem, state = _build_development_state(
        config, "calibration", grid_size, checkpoint
    )
    context = _laplace_context(config, state, problem)
    incumbent = locality.state_incumbent(state)
    pilots: list[dict[str, Any]] = []
    pilot_count = config["calibration"]["pilot_samples_per_candidate"]
    for spec in specs:
        proposal = _prepare(spec, state, context)
        seed = _stream_seed(
            "calibration", grid_size, checkpoint, "pilot", spec.name
        )
        batch = backend.run_snis_batch(
            state,
            problem,
            proposal,
            incumbent=incumbent,
            sample_count=pilot_count,
            proposal_seed=seed,
        )
        record = backend.batch_public_summary(batch, state)
        record.update(
            {
                "kind": spec.kind,
                "curvature_lambda": spec.curvature_lambda,
                "covariance_inflation": spec.covariance_inflation,
                "tie_break_rank": spec.tie_break_rank,
                "spd_check_passed": proposal.spd_check_passed,
                "minimum_cholesky_diagonal": proposal.minimum_cholesky_diagonal,
            }
        )
        pilots.append(record)
        del batch, proposal
    selected = backend.select_proposal_from_pilots(pilots, specs)
    batch_a, batch_b = _run_production_pair(
        config,
        state,
        problem,
        context,
        selected,
        source_role="calibration",
    )
    pair = backend.compare_independent_batches(batch_a, batch_b, state)
    prior_comparisons = None
    prior_verification = None
    prior_pooled = None
    if grid_size == 24:
        prior_pooled, prior_verification = _verify_prior_n24_reference(
            state, problem, context, specs
        )
        prior_comparisons = [
            backend.compare_batch_to_reference(batch, prior_pooled)
            for batch in (batch_a, batch_b)
        ]
    return {
        "source_role": "calibration",
        "source_seed": CALIBRATION_SEED,
        "state_key": _state_key(grid_size, checkpoint),
        "state_id": _state_identifier("calibration", grid_size, checkpoint),
        "grid_size": grid_size,
        "checkpoint": checkpoint,
        "checkpoint_queries": state.checkpoint_queries,
        "state_fingerprint": state.fingerprint(),
        "state_specific_incumbent": incumbent,
        "laplace_diagnostics": context.diagnostics,
        "laplace_work": context.work,
        "pilots": pilots,
        "selected_proposal": selected.name,
        "production_batches": [
            _serialize_batch(batch_a, state),
            _serialize_batch(batch_b, state),
        ],
        "pair_comparison": pair,
        "prior_n24_pooled_reference": (
            _serialize_batch(prior_pooled, state) if prior_pooled is not None else None
        ),
        "prior_n24_reproduction": prior_verification,
        "prior_n24_comparisons": prior_comparisons,
        "worker_wall_seconds": time.perf_counter() - started,
        "peak_rss_bytes": locality.peak_rss_bytes(),
    }


def _validation_worker(
    grid_size: int, checkpoint: str, proposal_name: str
) -> dict[str, Any]:
    started = time.perf_counter()
    config = _read_config()
    specs = _proposal_specs(config)
    matches = [spec for spec in specs if spec.name == proposal_name]
    if len(matches) != 1:
        raise RuntimeError("held-out worker received an unknown frozen proposal")
    problem, state = _build_development_state(
        config, "validation", grid_size, checkpoint
    )
    context = _laplace_context(config, state, problem)
    batch_a, batch_b = _run_production_pair(
        config,
        state,
        problem,
        context,
        matches[0],
        source_role="validation",
    )
    return {
        "source_role": "validation",
        "source_seed": VALIDATION_SEED,
        "state_key": _state_key(grid_size, checkpoint),
        "state_id": _state_identifier("validation", grid_size, checkpoint),
        "grid_size": grid_size,
        "checkpoint": checkpoint,
        "checkpoint_queries": state.checkpoint_queries,
        "state_fingerprint": state.fingerprint(),
        "state_specific_incumbent": locality.state_incumbent(state),
        "laplace_diagnostics": context.diagnostics,
        "laplace_work": context.work,
        "validation_pilot_run": False,
        "selected_proposal": proposal_name,
        "production_batches": [
            _serialize_batch(batch_a, state),
            _serialize_batch(batch_b, state),
        ],
        "pair_comparison": backend.compare_independent_batches(
            batch_a, batch_b, state
        ),
        "worker_wall_seconds": time.perf_counter() - started,
        "peak_rss_bytes": locality.peak_rss_bytes(),
    }


def _worker_subprocess(arguments: list[str]) -> dict[str, Any]:
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), *arguments],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def _pair_gate(
    result: dict[str, Any], *, require_action_agreement: bool
) -> dict[str, bool]:
    config = _read_config()
    if result["grid_size"] == 40:
        gate = config["held_out_validation"]["n40_gate"]
        max_vector = gate["maximum_acquisition_vector_difference"]
        max_regret = gate["maximum_reciprocal_cross_batch_action_regret"]
    else:
        gate = config["held_out_validation"]["n24_health_gate"]
        max_vector = gate["maximum_acquisition_vector_difference_between_batches"]
        max_regret = gate["maximum_reciprocal_cross_batch_action_regret"]
    pair = result["pair_comparison"]
    production = result["production_batches"]
    components = {
        "ess_pass": min(batch["ess_fraction"] for batch in production)
        >= gate["minimum_ess_fraction_each_batch"],
        "action_agreement_pass": (
            pair["action_agreement"] if require_action_agreement else True
        ),
        "vector_difference_pass": pair["maximum_acquisition_vector_difference"]
        <= max_vector,
        "reciprocal_regret_pass": pair["maximum_reciprocal_action_regret"]
        <= max_regret,
    }
    return {**components, "pass": all(components.values())}


def _calibration_n24_prior_gate(result: dict[str, Any]) -> dict[str, bool]:
    config = _read_config()
    gate = config["held_out_validation"]["n24_health_gate"]
    comparisons = result["prior_n24_comparisons"]
    components = {
        "prior_reproduction_pass": result["prior_n24_reproduction"] is not None,
        "prior_vector_difference_pass": max(
            comparison["maximum_acquisition_vector_difference"]
            for comparison in comparisons
        )
        <= gate["maximum_difference_from_prior_pooled_acquisition"],
        "prior_reciprocal_regret_pass": max(
            comparison["maximum_reciprocal_action_regret"]
            for comparison in comparisons
        )
        <= gate[
            "maximum_reciprocal_action_regret_against_prior_pooled_reference"
        ],
    }
    return {**components, "pass": all(components.values())}


def _write_tables(
    calibration: list[dict[str, Any]], validation: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    pilot_rows: list[dict[str, Any]] = []
    for state in calibration:
        for pilot in state["pilots"]:
            pilot_rows.append(
                {
                    "state_id": state["state_id"],
                    "grid_size": state["grid_size"],
                    "checkpoint": state["checkpoint"],
                    "source_seed": state["source_seed"],
                    "state_specific_incumbent": state["state_specific_incumbent"],
                    **{
                        key: value
                        for key, value in pilot.items()
                        if key not in {"top_five", "work"}
                    },
                    "top_five": json.dumps(pilot["top_five"], sort_keys=True),
                    "work": json.dumps(pilot["work"], sort_keys=True),
                    "selected": pilot["proposal_name"]
                    == state["selected_proposal"],
                }
            )
    pd.DataFrame(pilot_rows).to_csv(
        OUTPUT_DIRECTORY / "proposal_pilot.csv", index=False
    )

    production_rows: list[dict[str, Any]] = []
    for state in calibration + validation:
        for batch_label, batch in zip(("A", "B"), state["production_batches"]):
            production_rows.append(
                {
                    "state_id": state["state_id"],
                    "source_role": state["source_role"],
                    "source_seed": state["source_seed"],
                    "grid_size": state["grid_size"],
                    "checkpoint": state["checkpoint"],
                    "state_specific_incumbent": state["state_specific_incumbent"],
                    "selected_proposal": state["selected_proposal"],
                    "batch": batch_label,
                    **{
                        key: value
                        for key, value in batch.items()
                        if key not in {"top_five", "work", "acquisition_vector"}
                    },
                    "top_five": json.dumps(batch["top_five"], sort_keys=True),
                    "acquisition_vector": json.dumps(batch["acquisition_vector"]),
                    "work": json.dumps(batch["work"], sort_keys=True),
                    "validation_pilot_run": state.get("validation_pilot_run"),
                }
            )
    pd.DataFrame(production_rows).to_csv(
        OUTPUT_DIRECTORY / "production_snis.csv", index=False
    )

    validation_rows: list[dict[str, Any]] = []
    for state in calibration + validation:
        require_action = state["source_role"] == "validation" and state[
            "grid_size"
        ] == 40
        pair_gate = _pair_gate(state, require_action_agreement=require_action)
        prior_gate = (
            _calibration_n24_prior_gate(state)
            if state["source_role"] == "calibration" and state["grid_size"] == 24
            else None
        )
        pair = state["pair_comparison"]
        production = state["production_batches"]
        row = {
            "state_id": state["state_id"],
            "source_role": state["source_role"],
            "source_seed": state["source_seed"],
            "grid_size": state["grid_size"],
            "checkpoint": state["checkpoint"],
            "selected_proposal": state["selected_proposal"],
            "action_a": production[0]["action_index"],
            "action_b": production[1]["action_index"],
            "ess_fraction_a": production[0]["ess_fraction"],
            "ess_fraction_b": production[1]["ess_fraction"],
            "maximum_acquisition_vector_difference": pair[
                "maximum_acquisition_vector_difference"
            ],
            "maximum_top_five_union_difference": pair[
                "maximum_top_five_union_difference"
            ],
            "maximum_reciprocal_action_regret": pair[
                "maximum_reciprocal_action_regret"
            ],
            **pair_gate,
            "prior_n24_pass": prior_gate["pass"] if prior_gate else None,
            "laplace_converged": state["laplace_diagnostics"]["converged"],
            "laplace_iterations": state["laplace_diagnostics"]["iterations"],
            "laplace_gradient_inf_norm": state["laplace_diagnostics"][
                "gradient_inf_norm"
            ],
            "worker_wall_seconds": state["worker_wall_seconds"],
            "peak_rss_bytes": state["peak_rss_bytes"],
        }
        if prior_gate:
            row.update(
                {key: value for key, value in prior_gate.items() if key != "pass"}
            )
        validation_rows.append(row)
    pd.DataFrame(validation_rows).to_csv(
        OUTPUT_DIRECTORY / "backend_validation.csv", index=False
    )
    return validation_rows


def _environment() -> dict[str, str]:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }


def run_calibrated_snis_rescue() -> dict[str, Any]:
    started = time.perf_counter()
    config = _read_config()
    specs = _proposal_specs(config)
    calibration: list[dict[str, Any]] = []
    for grid_size in GRID_SIZES:
        for checkpoint in CHECKPOINTS:
            result = _worker_subprocess(
                [
                    "--worker-mode",
                    "calibration",
                    "--grid-size",
                    str(grid_size),
                    "--checkpoint",
                    checkpoint,
                ]
            )
            calibration.append(result)
            baseline_ess = next(
                pilot["ess_fraction"]
                for pilot in result["pilots"]
                if pilot["kind"] == "BASELINE_INFLATED_LAPLACE"
            )
            selected_ess = max(pilot["ess_fraction"] for pilot in result["pilots"])
            print(
                "[backend calibration] "
                f"n={grid_size} {checkpoint}: {result['selected_proposal']} "
                f"pilot ESS {selected_ess:.3f} vs baseline {baseline_ess:.3f}",
                flush=True,
            )

    selection = {
        "diagnostic_id": config["diagnostic_id"],
        "development_only": True,
        "calibration_source_seed": CALIBRATION_SEED,
        "config_sha256": EXPECTED_CONFIG_SHA256,
        "selection_rule": config["calibration"]["selection_rule"],
        "selected_proposals": {
            result["state_key"]: result["selected_proposal"]
            for result in calibration
        },
        "held_out_validation_started": False,
    }
    selection_path = OUTPUT_DIRECTORY / "selected_proposals.json"
    selection_path.write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n")
    selection_sha = _sha256(selection_path)
    frozen_selection = json.loads(selection_path.read_text())
    if _sha256(selection_path) != selection_sha:
        raise RuntimeError("proposal mapping changed before held-out validation")

    validation: list[dict[str, Any]] = []
    for grid_size in GRID_SIZES:
        for checkpoint in CHECKPOINTS:
            proposal_name = frozen_selection["selected_proposals"][
                _state_key(grid_size, checkpoint)
            ]
            result = _worker_subprocess(
                [
                    "--worker-mode",
                    "validation",
                    "--grid-size",
                    str(grid_size),
                    "--checkpoint",
                    checkpoint,
                    "--proposal-name",
                    proposal_name,
                ]
            )
            validation.append(result)
            pair = result["pair_comparison"]
            print(
                "[held-out validation] "
                f"n={grid_size} {checkpoint}: actions={pair['batch_actions']} "
                f"ESS=({result['production_batches'][0]['ess_fraction']:.3f},"
                f"{result['production_batches'][1]['ess_fraction']:.3f}) "
                f"maxdiff={pair['maximum_acquisition_vector_difference']:.4f} "
                f"regret={pair['maximum_reciprocal_action_regret']:.4f}",
                flush=True,
            )

    validation_rows = _write_tables(calibration, validation)
    heldout_n40 = [
        row
        for row in validation_rows
        if row["source_role"] == "validation" and row["grid_size"] == 40
    ]
    heldout_n24 = [
        row
        for row in validation_rows
        if row["source_role"] == "validation" and row["grid_size"] == 24
    ]
    calibration_n24 = [
        row
        for row in validation_rows
        if row["source_role"] == "calibration" and row["grid_size"] == 24
    ]
    n40_pass = len(heldout_n40) == 3 and all(row["pass"] for row in heldout_n40)
    n24_pass = (
        len(heldout_n24) == 3
        and all(row["pass"] for row in heldout_n24)
        and len(calibration_n24) == 3
        and all(row["pass"] and row["prior_n24_pass"] for row in calibration_n24)
    )
    passed = n40_pass and n24_pass
    classification = (
        "CALIBRATED_SNIS_BACKEND_PASS"
        if passed
        else "CALIBRATED_SNIS_BACKEND_FAIL_REQUIRES_INDEPENDENCE_MH"
    )
    pilot_improvements = []
    for result in calibration:
        baseline = next(
            pilot for pilot in result["pilots"] if pilot["kind"] == "BASELINE_INFLATED_LAPLACE"
        )
        selected = next(
            pilot
            for pilot in result["pilots"]
            if pilot["proposal_name"] == result["selected_proposal"]
        )
        pilot_improvements.append(
            {
                "state_key": result["state_key"],
                "baseline_ess_fraction": baseline["ess_fraction"],
                "selected_ess_fraction": selected["ess_fraction"],
                "absolute_improvement": selected["ess_fraction"]
                - baseline["ess_fraction"],
                "ratio": selected["ess_fraction"] / baseline["ess_fraction"],
            }
        )
    summary = {
        "diagnostic_id": config["diagnostic_id"],
        "development_only": True,
        "starting_main_sha": config["starting_main_sha"],
        "backend_config_sha256": EXPECTED_CONFIG_SHA256,
        "selected_proposals_sha256": selection_sha,
        "calibration_source_seed": CALIBRATION_SEED,
        "held_out_validation_source_seed": VALIDATION_SEED,
        "prospective_source_seeds_accessed": False,
        "scientific_preregistration_created_or_executed": False,
        "shadow_only_contract": backend.shadow_only_contract(),
        "calibration": calibration,
        "validation": validation,
        "pilot_ess_improvements": pilot_improvements,
        "calibrated_snis_gate": {
            "held_out_n40_pass": n40_pass,
            "n24_health_and_prior_agreement_pass": n24_pass,
            "pass": passed,
        },
        "terminal_backend_classification": classification,
        "sampler_escalation_required": not passed,
        "wall_seconds": time.perf_counter() - started,
        "peak_rss_bytes": max(
            result["peak_rss_bytes"] for result in calibration + validation
        ),
        "environment": _environment(),
    }
    (OUTPUT_DIRECTORY / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


def _mcmc_worker(
    source_role: str,
    grid_size: int,
    checkpoint: str,
    proposal_name: str,
    *,
    attempt: int,
    burn_in: int,
    retained_per_chain: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    config = _read_config()
    mcmc_config = _read_mcmc_config()
    specs = _proposal_specs(config)
    matches = [spec for spec in specs if spec.name == proposal_name]
    if len(matches) != 1:
        raise RuntimeError("MCMC worker received an unknown calibrated proposal")
    problem, state = _build_development_state(
        config, source_role, grid_size, checkpoint
    )
    context = _laplace_context(config, state, problem)
    proposal = _prepare(matches[0], state, context)
    chain_count = mcmc_config["chain_count"]
    initial_states = [context.mode.copy()]
    initializations = ["LAPLACE_MODE"]
    for chain_index in range(1, chain_count):
        seed = _stream_seed(
            source_role,
            grid_size,
            checkpoint,
            f"mh_attempt_{attempt}",
            f"chain_{chain_index}_initial",
        )
        initial_states.append(
            backend.sample_gaussian_precision(
                proposal, 1, np.random.default_rng(seed)
            )[0]
        )
        initializations.append("INDEPENDENT_SELECTED_PROPOSAL_DRAW")
    chains: list[backend.IndependenceMHChain] = []
    for chain_index in range(chain_count):
        proposal_seed = _stream_seed(
            source_role,
            grid_size,
            checkpoint,
            f"mh_attempt_{attempt}",
            f"chain_{chain_index}_proposal",
        )
        uniform_seed = _stream_seed(
            source_role,
            grid_size,
            checkpoint,
            f"mh_attempt_{attempt}",
            f"chain_{chain_index}_uniform",
        )
        chains.append(
            backend.run_independence_mh_chain(
                state,
                problem,
                proposal,
                incumbent=locality.state_incumbent(state),
                chain_index=chain_index,
                initial_state=initial_states[chain_index],
                initialization=initializations[chain_index],
                burn_in=burn_in,
                retained_count=retained_per_chain,
                proposal_seed=proposal_seed,
                uniform_seed=uniform_seed,
                proposal_evaluation_block_size=mcmc_config[
                    "proposal_evaluation_block_size"
                ],
            )
        )
    aggregate = backend.aggregate_independence_mh_chains(
        chains,
        state,
        group_a_chains=mcmc_config["independent_group_a_chains"],
        group_b_chains=mcmc_config["independent_group_b_chains"],
    )
    return {
        "backend": "LAPLACE_INDEPENDENCE_MH",
        "source_role": source_role,
        "source_seed": _source_seed_for_role(source_role),
        "state_key": _state_key(grid_size, checkpoint),
        "state_id": _state_identifier(source_role, grid_size, checkpoint),
        "grid_size": grid_size,
        "checkpoint": checkpoint,
        "checkpoint_queries": state.checkpoint_queries,
        "state_fingerprint": state.fingerprint(),
        "state_specific_incumbent": locality.state_incumbent(state),
        "selected_proposal": proposal_name,
        "proposal_spd_check_passed": proposal.spd_check_passed,
        "attempt": attempt,
        "burn_in": burn_in,
        "retained_per_chain": retained_per_chain,
        "laplace_diagnostics": context.diagnostics,
        "laplace_work": context.work,
        "aggregate": aggregate,
        "worker_wall_seconds": time.perf_counter() - started,
        "peak_rss_bytes": locality.peak_rss_bytes(),
    }


def _compare_acquisition_vectors(
    first: list[float], second: list[float], action_indices: list[int]
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


def _mcmc_state_gate(result: dict[str, Any]) -> dict[str, bool]:
    thresholds = _read_mcmc_config()["gate"]
    aggregate = result["aggregate"]
    components = {
        "acceptance_pass": aggregate["median_chain_acceptance"]
        >= thresholds["minimum_median_chain_acceptance"],
        "rhat_pass": aggregate["maximum_split_rhat"]
        <= thresholds["maximum_split_rhat_all_required_observables"],
        "gap_ess_pass": aggregate["minimum_leader_challenger_gap_ess"]
        >= thresholds["minimum_pooled_ess_each_leader_challenger_gap"],
        "group_action_pass": (
            aggregate["group_action_agreement"]
            if thresholds["require_independent_group_action_agreement"]
            else True
        ),
        "group_regret_pass": aggregate[
            "maximum_reciprocal_group_action_regret"
        ]
        <= thresholds["maximum_reciprocal_group_action_regret"],
        "group_vector_pass": aggregate[
            "maximum_group_acquisition_vector_difference"
        ]
        <= thresholds["maximum_group_acquisition_vector_difference"],
    }
    return {**components, "pass": all(components.values())}


def _mcmc_prior_n24_gate(
    result: dict[str, Any], snis_summary: dict[str, Any]
) -> dict[str, Any]:
    prior_state = next(
        state
        for state in snis_summary["calibration"]
        if state["state_key"] == result["state_key"]
    )
    prior = prior_state["prior_n24_pooled_reference"]
    action_indices = [
        int(value)
        for value in _build_development_state(
            _read_config(), "calibration", 24, result["checkpoint"]
        )[1].action_indices
    ]
    comparison = _compare_acquisition_vectors(
        result["aggregate"]["pooled_acquisition"],
        prior["acquisition_vector"],
        action_indices,
    )
    thresholds = _read_mcmc_config()["n24_prior_reference_gate"]
    components = {
        "prior_vector_pass": comparison["maximum_acquisition_vector_difference"]
        <= thresholds["maximum_pooled_acquisition_vector_difference"],
        "prior_regret_pass": comparison["maximum_reciprocal_action_regret"]
        <= thresholds["maximum_reciprocal_action_regret"],
    }
    return {**comparison, **components, "pass": all(components.values())}


def _mcmc_worker_subprocess(
    *,
    source_role: str,
    grid_size: int,
    checkpoint: str,
    proposal_name: str,
    attempt: int,
    burn_in: int,
    retained_per_chain: int,
) -> dict[str, Any]:
    return _worker_subprocess(
        [
            "--worker-mode",
            "independence-mh",
            "--source-role",
            source_role,
            "--grid-size",
            str(grid_size),
            "--checkpoint",
            checkpoint,
            "--proposal-name",
            proposal_name,
            "--attempt",
            str(attempt),
            "--burn-in",
            str(burn_in),
            "--retained-per-chain",
            str(retained_per_chain),
        ]
    )


def _write_mcmc_tables(
    calibration_attempts: list[dict[str, Any]],
    validation: list[dict[str, Any]],
) -> None:
    chain_rows: list[dict[str, Any]] = []
    scalar_rows: list[dict[str, Any]] = []
    for attempt_record in calibration_attempts:
        states = attempt_record["states"]
        attempt_status = "SELECTED" if attempt_record["selected"] else "NOT_SELECTED"
        for state in states:
            aggregate = state["aggregate"]
            for chain in aggregate["chain_diagnostics"]:
                chain_rows.append(
                    {
                        "state_id": state["state_id"],
                        "source_role": state["source_role"],
                        "source_seed": state["source_seed"],
                        "grid_size": state["grid_size"],
                        "checkpoint": state["checkpoint"],
                        "attempt": state["attempt"],
                        "attempt_status": attempt_status,
                        "selected_proposal": state["selected_proposal"],
                        **{key: value for key, value in chain.items() if key != "work"},
                        "work": json.dumps(chain["work"], sort_keys=True),
                        "maximum_split_rhat": aggregate["maximum_split_rhat"],
                        "minimum_gap_ess": aggregate[
                            "minimum_leader_challenger_gap_ess"
                        ],
                    }
                )
            for scalar in aggregate["scalar_diagnostics"]:
                scalar_rows.append(
                    {
                        "state_id": state["state_id"],
                        "source_role": state["source_role"],
                        "source_seed": state["source_seed"],
                        "grid_size": state["grid_size"],
                        "checkpoint": state["checkpoint"],
                        "attempt": state["attempt"],
                        "attempt_status": attempt_status,
                        **scalar,
                    }
                )
    for state in validation:
        aggregate = state["aggregate"]
        for chain in aggregate["chain_diagnostics"]:
            chain_rows.append(
                {
                    "state_id": state["state_id"],
                    "source_role": state["source_role"],
                    "source_seed": state["source_seed"],
                    "grid_size": state["grid_size"],
                    "checkpoint": state["checkpoint"],
                    "attempt": state["attempt"],
                    "attempt_status": "HELD_OUT_VALIDATION",
                    "selected_proposal": state["selected_proposal"],
                    **{key: value for key, value in chain.items() if key != "work"},
                    "work": json.dumps(chain["work"], sort_keys=True),
                    "maximum_split_rhat": aggregate["maximum_split_rhat"],
                    "minimum_gap_ess": aggregate[
                        "minimum_leader_challenger_gap_ess"
                    ],
                }
            )
        for scalar in aggregate["scalar_diagnostics"]:
            scalar_rows.append(
                {
                    "state_id": state["state_id"],
                    "source_role": state["source_role"],
                    "source_seed": state["source_seed"],
                    "grid_size": state["grid_size"],
                    "checkpoint": state["checkpoint"],
                    "attempt": state["attempt"],
                    "attempt_status": "HELD_OUT_VALIDATION",
                    **scalar,
                }
            )
    pd.DataFrame(chain_rows).to_csv(
        OUTPUT_DIRECTORY / "mcmc_chain_diagnostics.csv", index=False
    )
    pd.DataFrame(scalar_rows).to_csv(
        OUTPUT_DIRECTORY / "mcmc_scalar_diagnostics.csv", index=False
    )


def _append_mcmc_validation_rows(
    calibration_attempts: list[dict[str, Any]],
    validation: list[dict[str, Any]],
) -> None:
    path = OUTPUT_DIRECTORY / "backend_validation.csv"
    existing = pd.read_csv(path)
    existing.insert(0, "backend", "CALIBRATED_LAPLACE_SNIS")
    rows: list[dict[str, Any]] = []
    for attempt_record in calibration_attempts:
        for state in attempt_record["states"]:
            aggregate = state["aggregate"]
            gate = state["gate"]
            rows.append(
                {
                    "backend": "LAPLACE_INDEPENDENCE_MH",
                    "state_id": state["state_id"],
                    "source_role": state["source_role"],
                    "source_seed": state["source_seed"],
                    "grid_size": state["grid_size"],
                    "checkpoint": state["checkpoint"],
                    "selected_proposal": state["selected_proposal"],
                    "attempt": state["attempt"],
                    "attempt_status": (
                        "SELECTED" if attempt_record["selected"] else "NOT_SELECTED"
                    ),
                    "action_a": aggregate["group_a_action"],
                    "action_b": aggregate["group_b_action"],
                    "median_chain_acceptance": aggregate[
                        "median_chain_acceptance"
                    ],
                    "maximum_split_rhat": aggregate["maximum_split_rhat"],
                    "minimum_gap_ess": aggregate[
                        "minimum_leader_challenger_gap_ess"
                    ],
                    "maximum_acquisition_vector_difference": aggregate[
                        "maximum_group_acquisition_vector_difference"
                    ],
                    "maximum_reciprocal_action_regret": aggregate[
                        "maximum_reciprocal_group_action_regret"
                    ],
                    **gate,
                    "prior_n24_pass": state.get("prior_n24_gate", {}).get("pass"),
                    "worker_wall_seconds": state["worker_wall_seconds"],
                    "peak_rss_bytes": state["peak_rss_bytes"],
                }
            )
    for state in validation:
        aggregate = state["aggregate"]
        rows.append(
            {
                "backend": "LAPLACE_INDEPENDENCE_MH",
                "state_id": state["state_id"],
                "source_role": state["source_role"],
                "source_seed": state["source_seed"],
                "grid_size": state["grid_size"],
                "checkpoint": state["checkpoint"],
                "selected_proposal": state["selected_proposal"],
                "attempt": state["attempt"],
                "attempt_status": "HELD_OUT_VALIDATION",
                "action_a": aggregate["group_a_action"],
                "action_b": aggregate["group_b_action"],
                "median_chain_acceptance": aggregate["median_chain_acceptance"],
                "maximum_split_rhat": aggregate["maximum_split_rhat"],
                "minimum_gap_ess": aggregate[
                    "minimum_leader_challenger_gap_ess"
                ],
                "maximum_acquisition_vector_difference": aggregate[
                    "maximum_group_acquisition_vector_difference"
                ],
                "maximum_reciprocal_action_regret": aggregate[
                    "maximum_reciprocal_group_action_regret"
                ],
                **state["gate"],
                "worker_wall_seconds": state["worker_wall_seconds"],
                "peak_rss_bytes": state["peak_rss_bytes"],
            }
        )
    pd.concat((existing, pd.DataFrame(rows)), ignore_index=True, sort=False).to_csv(
        path, index=False
    )


def run_independence_mh_rescue() -> dict[str, Any]:
    started = time.perf_counter()
    _read_config()
    mcmc_config = _read_mcmc_config()
    summary_path = OUTPUT_DIRECTORY / "summary.json"
    if _sha256(summary_path) != EXPECTED_PRE_MH_SNIS_SUMMARY_SHA256:
        raise RuntimeError("calibrated-SNIS result changed before MCMC escalation")
    snis_summary = json.loads(summary_path.read_text())
    if snis_summary["terminal_backend_classification"] != (
        "CALIBRATED_SNIS_BACKEND_FAIL_REQUIRES_INDEPENDENCE_MH"
    ):
        raise RuntimeError("independence MH may run only after calibrated SNIS fails")
    (OUTPUT_DIRECTORY / "calibrated_snis_summary.json").write_text(
        json.dumps(snis_summary, indent=2, sort_keys=True) + "\n"
    )
    selection = json.loads(
        (OUTPUT_DIRECTORY / "selected_proposals.json").read_text()
    )["selected_proposals"]
    calibration_attempts: list[dict[str, Any]] = []
    selected_attempt: dict[str, Any] | None = None
    for schedule in mcmc_config["calibration_schedule"]:
        states: list[dict[str, Any]] = []
        for grid_size in GRID_SIZES:
            for checkpoint in CHECKPOINTS:
                state = _mcmc_worker_subprocess(
                    source_role="calibration",
                    grid_size=grid_size,
                    checkpoint=checkpoint,
                    proposal_name=selection[_state_key(grid_size, checkpoint)],
                    attempt=schedule["attempt"],
                    burn_in=schedule["burn_in"],
                    retained_per_chain=schedule["retained_per_chain"],
                )
                state["gate"] = _mcmc_state_gate(state)
                if grid_size == 24:
                    state["prior_n24_gate"] = _mcmc_prior_n24_gate(
                        state, snis_summary
                    )
                states.append(state)
                aggregate = state["aggregate"]
                print(
                    "[MH calibration] "
                    f"attempt={schedule['attempt']} n={grid_size} {checkpoint}: "
                    f"accept={aggregate['median_chain_acceptance']:.3f} "
                    f"Rhat={aggregate['maximum_split_rhat']:.4f} "
                    f"gapESS={aggregate['minimum_leader_challenger_gap_ess']:.0f} "
                    f"groups=({aggregate['group_a_action']},"
                    f"{aggregate['group_b_action']}) pass={state['gate']['pass']}",
                    flush=True,
                )
        passed = all(state["gate"]["pass"] for state in states) and all(
            state.get("prior_n24_gate", {"pass": True})["pass"] for state in states
        )
        record = {**schedule, "states": states, "pass": passed, "selected": False}
        calibration_attempts.append(record)
        if passed:
            record["selected"] = True
            selected_attempt = record
            break

    validation: list[dict[str, Any]] = []
    if selected_attempt is not None:
        selected_settings = {
            "diagnostic_id": mcmc_config["diagnostic_id"],
            "development_only": True,
            "selected_from_calibration_seed": CALIBRATION_SEED,
            "attempt": selected_attempt["attempt"],
            "burn_in": selected_attempt["burn_in"],
            "retained_per_chain": selected_attempt["retained_per_chain"],
            "chain_count": mcmc_config["chain_count"],
            "proposal_mapping_sha256": mcmc_config["selected_proposals_sha256"],
            "held_out_validation_started": False,
        }
        selected_path = OUTPUT_DIRECTORY / "mcmc_selected_settings.json"
        selected_path.write_text(
            json.dumps(selected_settings, indent=2, sort_keys=True) + "\n"
        )
        selected_sha = _sha256(selected_path)
        frozen_settings = json.loads(selected_path.read_text())
        if _sha256(selected_path) != selected_sha:
            raise RuntimeError("MCMC settings changed before held-out validation")
        for grid_size in GRID_SIZES:
            for checkpoint in CHECKPOINTS:
                state = _mcmc_worker_subprocess(
                    source_role="validation",
                    grid_size=grid_size,
                    checkpoint=checkpoint,
                    proposal_name=selection[_state_key(grid_size, checkpoint)],
                    attempt=frozen_settings["attempt"],
                    burn_in=frozen_settings["burn_in"],
                    retained_per_chain=frozen_settings["retained_per_chain"],
                )
                state["gate"] = _mcmc_state_gate(state)
                validation.append(state)
                aggregate = state["aggregate"]
                print(
                    "[MH held-out validation] "
                    f"n={grid_size} {checkpoint}: "
                    f"accept={aggregate['median_chain_acceptance']:.3f} "
                    f"Rhat={aggregate['maximum_split_rhat']:.4f} "
                    f"gapESS={aggregate['minimum_leader_challenger_gap_ess']:.0f} "
                    f"groups=({aggregate['group_a_action']},"
                    f"{aggregate['group_b_action']}) pass={state['gate']['pass']}",
                    flush=True,
                )
    else:
        selected_sha = None

    calibration_pass = selected_attempt is not None
    heldout_n40 = [state for state in validation if state["grid_size"] == 40]
    heldout_n24 = [state for state in validation if state["grid_size"] == 24]
    validation_pass = (
        calibration_pass
        and len(heldout_n40) == 3
        and all(state["gate"]["pass"] for state in heldout_n40)
        and len(heldout_n24) == 3
        and all(state["gate"]["pass"] for state in heldout_n24)
    )
    passed = calibration_pass and validation_pass
    classification = (
        "INDEPENDENCE_MH_BACKEND_PASS"
        if passed
        else "INDEPENDENCE_MH_BACKEND_FAIL_REQUIRES_ELLIPTICAL_SLICE"
    )
    _write_mcmc_tables(calibration_attempts, validation)
    _append_mcmc_validation_rows(calibration_attempts, validation)
    result = {
        "diagnostic_id": mcmc_config["diagnostic_id"],
        "development_only": True,
        "mcmc_config_sha256": EXPECTED_MCMC_CONFIG_SHA256,
        "calibration_source_seed": CALIBRATION_SEED,
        "held_out_validation_source_seed": VALIDATION_SEED,
        "prospective_source_seeds_accessed": False,
        "scientific_preregistration_created_or_executed": False,
        "calibration_attempts": calibration_attempts,
        "selected_settings_sha256": selected_sha,
        "validation": validation,
        "gate": {
            "calibration_pass": calibration_pass,
            "held_out_validation_pass": validation_pass,
            "pass": passed,
        },
        "terminal_backend_classification": classification,
        "sampler_escalation_required": not passed,
        "wall_seconds": time.perf_counter() - started,
        "peak_rss_bytes": max(
            [
                state["peak_rss_bytes"]
                for attempt in calibration_attempts
                for state in attempt["states"]
            ]
            + [state["peak_rss_bytes"] for state in validation]
        ),
        "shadow_only_contract": backend.shadow_only_contract(),
    }
    combined = {
        "diagnostic_id": "E2_FULL_SHADOW_BACKEND_RESCUE_DEVELOPMENT_ONLY_V1",
        "development_only": True,
        "starting_main_sha": snis_summary["starting_main_sha"],
        "backend_config_sha256": EXPECTED_CONFIG_SHA256,
        "mcmc_config_sha256": EXPECTED_MCMC_CONFIG_SHA256,
        "calibration_source_seed": CALIBRATION_SEED,
        "held_out_validation_source_seed": VALIDATION_SEED,
        "prospective_source_seeds_accessed": False,
        "scientific_preregistration_created_or_executed": False,
        "calibrated_snis": snis_summary,
        "independence_mh": result,
        "terminal_backend_classification": classification,
        "recommended_backend": (
            "LAPLACE_INDEPENDENCE_MH" if passed else None
        ),
        "sampler_escalation_required": not passed,
        "wall_seconds": snis_summary["wall_seconds"] + result["wall_seconds"],
        "peak_rss_bytes": max(snis_summary["peak_rss_bytes"], result["peak_rss_bytes"]),
        "shadow_only_contract": backend.shadow_only_contract(),
        "environment": _environment(),
    }
    summary_path.write_text(json.dumps(combined, indent=2, sort_keys=True) + "\n")
    return combined


def _elliptical_worker(
    source_role: str,
    grid_size: int,
    checkpoint: str,
    proposal_name: str,
    *,
    attempt: int,
    burn_in: int,
    retained_per_chain: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    config = _read_config()
    elliptical_config = _read_elliptical_config()
    specs = _proposal_specs(config)
    matches = [spec for spec in specs if spec.name == proposal_name]
    if len(matches) != 1:
        raise RuntimeError("elliptical worker received an unknown proposal")
    problem, state = _build_development_state(
        config, source_role, grid_size, checkpoint
    )
    context = _laplace_context(config, state, problem)
    initialization_proposal = _prepare(matches[0], state, context)
    reference_sampler = backend.prepare_reference_direction_sampler(
        state,
        problem,
        observation_noise_variance=config["state_design_unchanged"][
            "observation_noise_variance"
        ],
    )
    chain_count = elliptical_config["chain_count"]
    initial_states = [context.mode.copy()]
    initializations = ["LAPLACE_MODE"]
    for chain_index in range(1, chain_count):
        seed = _stream_seed(
            source_role,
            grid_size,
            checkpoint,
            f"ess_attempt_{attempt}",
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
        direction_seed = _stream_seed(
            source_role,
            grid_size,
            checkpoint,
            f"ess_attempt_{attempt}",
            f"chain_{chain_index}_reference_direction",
        )
        slice_seed = _stream_seed(
            source_role,
            grid_size,
            checkpoint,
            f"ess_attempt_{attempt}",
            f"chain_{chain_index}_slice",
        )
        chains.append(
            backend.run_elliptical_slice_chain(
                state,
                problem,
                reference_sampler,
                incumbent=locality.state_incumbent(state),
                chain_index=chain_index,
                initial_state=initial_states[chain_index],
                initialization=initializations[chain_index],
                burn_in=burn_in,
                retained_count=retained_per_chain,
                direction_seed=direction_seed,
                slice_seed=slice_seed,
                maximum_bracket_evaluations=elliptical_config[
                    "maximum_bracket_evaluations_per_transition"
                ],
            )
        )
    aggregate = backend.aggregate_independence_mh_chains(
        chains,
        state,
        group_a_chains=elliptical_config["independent_group_a_chains"],
        group_b_chains=elliptical_config["independent_group_b_chains"],
        backend_name="ELLIPTICAL_SLICE_FULL",
    )
    aggregate["transition_acceptance_interpretation"] = (
        "one accepted slice state per transition by construction"
    )
    aggregate["reference_direction_backend"] = (
        "EXACT_GAUSSIAN_REFERENCE_MATHERON_SAMPLER"
    )
    aggregate["mean_likelihood_evaluations_per_transition"] = float(
        sum(chain.likelihood_evaluations for chain in chains)
        / sum(chain.total_transitions for chain in chains)
    )
    aggregate["maximum_likelihood_evaluations_one_transition"] = int(
        max(chain.maximum_likelihood_evaluations_one_transition for chain in chains)
    )
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
        "initialization_proposal": proposal_name,
        "ellipse_reference": "EXACT_CURRENT_GAUSSIAN_BO_REFERENCE",
        "randomized_initial_angular_bracket": True,
        "attempt": attempt,
        "burn_in": burn_in,
        "retained_per_chain": retained_per_chain,
        "laplace_diagnostics": context.diagnostics,
        "laplace_work": context.work,
        "aggregate": aggregate,
        "worker_wall_seconds": time.perf_counter() - started,
        "peak_rss_bytes": locality.peak_rss_bytes(),
    }


def _elliptical_state_gate(result: dict[str, Any]) -> dict[str, bool]:
    thresholds = _read_elliptical_config()["gate"]
    aggregate = result["aggregate"]
    components = {
        "transition_acceptance_pass": aggregate["median_chain_acceptance"] == 1.0,
        "rhat_pass": aggregate["maximum_split_rhat"]
        <= thresholds["maximum_split_rhat_all_required_observables"],
        "gap_ess_pass": aggregate["minimum_leader_challenger_gap_ess"]
        >= thresholds["minimum_pooled_ess_each_leader_challenger_gap"],
        "group_action_pass": (
            aggregate["group_action_agreement"]
            if thresholds["require_independent_group_action_agreement"]
            else True
        ),
        "group_regret_pass": aggregate[
            "maximum_reciprocal_group_action_regret"
        ]
        <= thresholds["maximum_reciprocal_group_action_regret"],
        "group_vector_pass": aggregate[
            "maximum_group_acquisition_vector_difference"
        ]
        <= thresholds["maximum_group_acquisition_vector_difference"],
    }
    return {**components, "pass": all(components.values())}


def _elliptical_prior_n24_gate(
    result: dict[str, Any], snis_summary: dict[str, Any]
) -> dict[str, Any]:
    prior_state = next(
        state
        for state in snis_summary["calibration"]
        if state["state_key"] == result["state_key"]
    )
    prior = prior_state["prior_n24_pooled_reference"]
    action_indices = [
        int(value)
        for value in _build_development_state(
            _read_config(), "calibration", 24, result["checkpoint"]
        )[1].action_indices
    ]
    comparison = _compare_acquisition_vectors(
        result["aggregate"]["pooled_acquisition"],
        prior["acquisition_vector"],
        action_indices,
    )
    thresholds = _read_elliptical_config()["n24_prior_reference_gate"]
    components = {
        "prior_vector_pass": comparison["maximum_acquisition_vector_difference"]
        <= thresholds["maximum_pooled_acquisition_vector_difference"],
        "prior_regret_pass": comparison["maximum_reciprocal_action_regret"]
        <= thresholds["maximum_reciprocal_action_regret"],
    }
    return {**comparison, **components, "pass": all(components.values())}


def _elliptical_worker_subprocess(
    *,
    source_role: str,
    grid_size: int,
    checkpoint: str,
    proposal_name: str,
    attempt: int,
    burn_in: int,
    retained_per_chain: int,
) -> dict[str, Any]:
    return _worker_subprocess(
        [
            "--worker-mode",
            "elliptical-slice",
            "--source-role",
            source_role,
            "--grid-size",
            str(grid_size),
            "--checkpoint",
            checkpoint,
            "--proposal-name",
            proposal_name,
            "--attempt",
            str(attempt),
            "--burn-in",
            str(burn_in),
            "--retained-per-chain",
            str(retained_per_chain),
        ]
    )


def _append_elliptical_tables(
    calibration_attempts: list[dict[str, Any]],
    validation: list[dict[str, Any]],
) -> None:
    chain_path = OUTPUT_DIRECTORY / "mcmc_chain_diagnostics.csv"
    scalar_path = OUTPUT_DIRECTORY / "mcmc_scalar_diagnostics.csv"
    existing_chains = pd.read_csv(chain_path)
    existing_scalars = pd.read_csv(scalar_path)
    chain_rows: list[dict[str, Any]] = []
    scalar_rows: list[dict[str, Any]] = []
    all_groups = [
        (attempt["states"], "SELECTED" if attempt["selected"] else "NOT_SELECTED")
        for attempt in calibration_attempts
    ] + [(validation, "HELD_OUT_VALIDATION")]
    for states, status in all_groups:
        for state in states:
            aggregate = state["aggregate"]
            for chain in aggregate["chain_diagnostics"]:
                chain_rows.append(
                    {
                        "backend": "ELLIPTICAL_SLICE_FULL",
                        "state_id": state["state_id"],
                        "source_role": state["source_role"],
                        "source_seed": state["source_seed"],
                        "grid_size": state["grid_size"],
                        "checkpoint": state["checkpoint"],
                        "attempt": state["attempt"],
                        "attempt_status": status,
                        "selected_proposal": state["initialization_proposal"],
                        **{key: value for key, value in chain.items() if key != "work"},
                        "work": json.dumps(chain["work"], sort_keys=True),
                        "maximum_split_rhat": aggregate["maximum_split_rhat"],
                        "minimum_gap_ess": aggregate[
                            "minimum_leader_challenger_gap_ess"
                        ],
                        "mean_likelihood_evaluations_per_transition": aggregate[
                            "mean_likelihood_evaluations_per_transition"
                        ],
                    }
                )
            for scalar in aggregate["scalar_diagnostics"]:
                scalar_rows.append(
                    {
                        "backend": "ELLIPTICAL_SLICE_FULL",
                        "state_id": state["state_id"],
                        "source_role": state["source_role"],
                        "source_seed": state["source_seed"],
                        "grid_size": state["grid_size"],
                        "checkpoint": state["checkpoint"],
                        "attempt": state["attempt"],
                        "attempt_status": status,
                        **scalar,
                    }
                )
    if "backend" not in existing_chains.columns:
        existing_chains.insert(0, "backend", "LAPLACE_INDEPENDENCE_MH")
    if "backend" not in existing_scalars.columns:
        existing_scalars.insert(0, "backend", "LAPLACE_INDEPENDENCE_MH")
    pd.concat((existing_chains, pd.DataFrame(chain_rows)), ignore_index=True, sort=False).to_csv(
        chain_path, index=False
    )
    pd.concat((existing_scalars, pd.DataFrame(scalar_rows)), ignore_index=True, sort=False).to_csv(
        scalar_path, index=False
    )

    validation_path = OUTPUT_DIRECTORY / "backend_validation.csv"
    existing_validation = pd.read_csv(validation_path)
    rows: list[dict[str, Any]] = []
    all_validation_groups = [
        (attempt["states"], "SELECTED" if attempt["selected"] else "NOT_SELECTED")
        for attempt in calibration_attempts
    ] + [(validation, "HELD_OUT_VALIDATION")]
    for states, status in all_validation_groups:
        for state in states:
            aggregate = state["aggregate"]
            rows.append(
                {
                    "backend": "ELLIPTICAL_SLICE_FULL",
                    "state_id": state["state_id"],
                    "source_role": state["source_role"],
                    "source_seed": state["source_seed"],
                    "grid_size": state["grid_size"],
                    "checkpoint": state["checkpoint"],
                    "selected_proposal": state["initialization_proposal"],
                    "attempt": state["attempt"],
                    "attempt_status": status,
                    "action_a": aggregate["group_a_action"],
                    "action_b": aggregate["group_b_action"],
                    "median_chain_acceptance": aggregate[
                        "median_chain_acceptance"
                    ],
                    "maximum_split_rhat": aggregate["maximum_split_rhat"],
                    "minimum_gap_ess": aggregate[
                        "minimum_leader_challenger_gap_ess"
                    ],
                    "maximum_acquisition_vector_difference": aggregate[
                        "maximum_group_acquisition_vector_difference"
                    ],
                    "maximum_reciprocal_action_regret": aggregate[
                        "maximum_reciprocal_group_action_regret"
                    ],
                    **state["gate"],
                    "prior_n24_pass": state.get("prior_n24_gate", {}).get("pass"),
                    "worker_wall_seconds": state["worker_wall_seconds"],
                    "peak_rss_bytes": state["peak_rss_bytes"],
                    "mean_likelihood_evaluations_per_transition": aggregate[
                        "mean_likelihood_evaluations_per_transition"
                    ],
                }
            )
    pd.concat(
        (existing_validation, pd.DataFrame(rows)), ignore_index=True, sort=False
    ).to_csv(validation_path, index=False)


def rebuild_backend_validation_table() -> None:
    """Mechanically rebuild the combined gate table from structured results."""

    summary = json.loads((OUTPUT_DIRECTORY / "summary.json").read_text())
    snis = summary["calibrated_snis"]
    _write_tables(snis["calibration"], snis["validation"])
    mh = summary["independence_mh"]
    _append_mcmc_validation_rows(mh["calibration_attempts"], mh["validation"])
    elliptical = summary["elliptical_slice"]
    validation_path = OUTPUT_DIRECTORY / "backend_validation.csv"
    existing_validation = pd.read_csv(validation_path)
    rows: list[dict[str, Any]] = []
    groups = [
        (attempt["states"], "SELECTED" if attempt["selected"] else "NOT_SELECTED")
        for attempt in elliptical["calibration_attempts"]
    ] + [(elliptical["validation"], "HELD_OUT_VALIDATION")]
    for states, status in groups:
        for state in states:
            aggregate = state["aggregate"]
            rows.append(
                {
                    "backend": "ELLIPTICAL_SLICE_FULL",
                    "state_id": state["state_id"],
                    "source_role": state["source_role"],
                    "source_seed": state["source_seed"],
                    "grid_size": state["grid_size"],
                    "checkpoint": state["checkpoint"],
                    "selected_proposal": state["initialization_proposal"],
                    "attempt": state["attempt"],
                    "attempt_status": status,
                    "action_a": aggregate["group_a_action"],
                    "action_b": aggregate["group_b_action"],
                    "median_chain_acceptance": aggregate[
                        "median_chain_acceptance"
                    ],
                    "maximum_split_rhat": aggregate["maximum_split_rhat"],
                    "minimum_gap_ess": aggregate[
                        "minimum_leader_challenger_gap_ess"
                    ],
                    "maximum_acquisition_vector_difference": aggregate[
                        "maximum_group_acquisition_vector_difference"
                    ],
                    "maximum_reciprocal_action_regret": aggregate[
                        "maximum_reciprocal_group_action_regret"
                    ],
                    **state["gate"],
                    "prior_n24_pass": state.get("prior_n24_gate", {}).get("pass"),
                    "worker_wall_seconds": state["worker_wall_seconds"],
                    "peak_rss_bytes": state["peak_rss_bytes"],
                    "mean_likelihood_evaluations_per_transition": aggregate[
                        "mean_likelihood_evaluations_per_transition"
                    ],
                }
            )
    pd.concat(
        (existing_validation, pd.DataFrame(rows)), ignore_index=True, sort=False
    ).to_csv(validation_path, index=False)


def run_elliptical_slice_rescue() -> dict[str, Any]:
    started = time.perf_counter()
    _read_config()
    elliptical_config = _read_elliptical_config()
    summary_path = OUTPUT_DIRECTORY / "summary.json"
    if _sha256(summary_path) != EXPECTED_PRE_ESS_MH_SUMMARY_SHA256:
        raise RuntimeError("independence-MH result changed before ESS escalation")
    previous_summary = json.loads(summary_path.read_text())
    if previous_summary["terminal_backend_classification"] != (
        "INDEPENDENCE_MH_BACKEND_FAIL_REQUIRES_ELLIPTICAL_SLICE"
    ):
        raise RuntimeError("elliptical slice may run only after independence MH fails")
    snis_summary = previous_summary["calibrated_snis"]
    selection = json.loads(
        (OUTPUT_DIRECTORY / "selected_proposals.json").read_text()
    )["selected_proposals"]
    calibration_attempts: list[dict[str, Any]] = []
    selected_attempt: dict[str, Any] | None = None
    for schedule in elliptical_config["calibration_schedule"]:
        states: list[dict[str, Any]] = []
        for grid_size in GRID_SIZES:
            for checkpoint in CHECKPOINTS:
                state = _elliptical_worker_subprocess(
                    source_role="calibration",
                    grid_size=grid_size,
                    checkpoint=checkpoint,
                    proposal_name=selection[_state_key(grid_size, checkpoint)],
                    attempt=schedule["attempt"],
                    burn_in=schedule["burn_in"],
                    retained_per_chain=schedule["retained_per_chain"],
                )
                state["gate"] = _elliptical_state_gate(state)
                if grid_size == 24:
                    state["prior_n24_gate"] = _elliptical_prior_n24_gate(
                        state, snis_summary
                    )
                states.append(state)
                aggregate = state["aggregate"]
                print(
                    "[ESS calibration] "
                    f"attempt={schedule['attempt']} n={grid_size} {checkpoint}: "
                    f"Rhat={aggregate['maximum_split_rhat']:.4f} "
                    f"gapESS={aggregate['minimum_leader_challenger_gap_ess']:.0f} "
                    f"groups=({aggregate['group_a_action']},"
                    f"{aggregate['group_b_action']}) "
                    f"eval/transition={aggregate['mean_likelihood_evaluations_per_transition']:.2f} "
                    f"pass={state['gate']['pass']}",
                    flush=True,
                )
        passed = all(state["gate"]["pass"] for state in states) and all(
            state.get("prior_n24_gate", {"pass": True})["pass"] for state in states
        )
        record = {**schedule, "states": states, "pass": passed, "selected": False}
        calibration_attempts.append(record)
        if passed:
            record["selected"] = True
            selected_attempt = record
            break

    validation: list[dict[str, Any]] = []
    if selected_attempt is not None:
        selected_settings = {
            "diagnostic_id": elliptical_config["diagnostic_id"],
            "development_only": True,
            "selected_from_calibration_seed": CALIBRATION_SEED,
            "attempt": selected_attempt["attempt"],
            "burn_in": selected_attempt["burn_in"],
            "retained_per_chain": selected_attempt["retained_per_chain"],
            "chain_count": elliptical_config["chain_count"],
            "held_out_validation_started": False,
        }
        selected_path = OUTPUT_DIRECTORY / "elliptical_slice_selected_settings.json"
        selected_path.write_text(
            json.dumps(selected_settings, indent=2, sort_keys=True) + "\n"
        )
        selected_sha = _sha256(selected_path)
        frozen_settings = json.loads(selected_path.read_text())
        if _sha256(selected_path) != selected_sha:
            raise RuntimeError("ESS settings changed before held-out validation")
        for grid_size in GRID_SIZES:
            for checkpoint in CHECKPOINTS:
                state = _elliptical_worker_subprocess(
                    source_role="validation",
                    grid_size=grid_size,
                    checkpoint=checkpoint,
                    proposal_name=selection[_state_key(grid_size, checkpoint)],
                    attempt=frozen_settings["attempt"],
                    burn_in=frozen_settings["burn_in"],
                    retained_per_chain=frozen_settings["retained_per_chain"],
                )
                state["gate"] = _elliptical_state_gate(state)
                validation.append(state)
                aggregate = state["aggregate"]
                print(
                    "[ESS held-out validation] "
                    f"n={grid_size} {checkpoint}: "
                    f"Rhat={aggregate['maximum_split_rhat']:.4f} "
                    f"gapESS={aggregate['minimum_leader_challenger_gap_ess']:.0f} "
                    f"groups=({aggregate['group_a_action']},"
                    f"{aggregate['group_b_action']}) "
                    f"diff={aggregate['maximum_group_acquisition_vector_difference']:.4f} "
                    f"regret={aggregate['maximum_reciprocal_group_action_regret']:.4f} "
                    f"pass={state['gate']['pass']}",
                    flush=True,
                )
    else:
        selected_sha = None

    calibration_pass = selected_attempt is not None
    heldout_n40 = [state for state in validation if state["grid_size"] == 40]
    heldout_n24 = [state for state in validation if state["grid_size"] == 24]
    validation_pass = (
        calibration_pass
        and len(heldout_n40) == 3
        and all(state["gate"]["pass"] for state in heldout_n40)
        and len(heldout_n24) == 3
        and all(state["gate"]["pass"] for state in heldout_n24)
    )
    passed = calibration_pass and validation_pass
    classification = (
        "ELLIPTICAL_SLICE_BACKEND_PASS"
        if passed
        else "FULL_REFERENCE_BACKEND_UNRESOLVED"
    )
    _append_elliptical_tables(calibration_attempts, validation)
    result = {
        "diagnostic_id": elliptical_config["diagnostic_id"],
        "development_only": True,
        "elliptical_slice_config_sha256": EXPECTED_ELLIPTICAL_CONFIG_SHA256,
        "calibration_source_seed": CALIBRATION_SEED,
        "held_out_validation_source_seed": VALIDATION_SEED,
        "prospective_source_seeds_accessed": False,
        "scientific_preregistration_created_or_executed": False,
        "calibration_attempts": calibration_attempts,
        "selected_settings_sha256": selected_sha,
        "validation": validation,
        "gate": {
            "calibration_pass": calibration_pass,
            "held_out_validation_pass": validation_pass,
            "pass": passed,
        },
        "terminal_backend_classification": classification,
        "wall_seconds": time.perf_counter() - started,
        "peak_rss_bytes": max(
            [
                state["peak_rss_bytes"]
                for attempt in calibration_attempts
                for state in attempt["states"]
            ]
            + [state["peak_rss_bytes"] for state in validation]
        ),
        "shadow_only_contract": backend.shadow_only_contract(),
    }
    combined = {
        **{
            key: value
            for key, value in previous_summary.items()
            if key
            not in {
                "terminal_backend_classification",
                "recommended_backend",
                "sampler_escalation_required",
                "wall_seconds",
                "peak_rss_bytes",
            }
        },
        "elliptical_slice": result,
        "terminal_backend_classification": classification,
        "recommended_backend": "ELLIPTICAL_SLICE_FULL" if passed else None,
        "sampler_escalation_required": False,
        "wall_seconds": previous_summary["wall_seconds"] + result["wall_seconds"],
        "peak_rss_bytes": max(previous_summary["peak_rss_bytes"], result["peak_rss_bytes"]),
    }
    summary_path.write_text(json.dumps(combined, indent=2, sort_keys=True) + "\n")
    return combined


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        choices=("calibrated-snis", "independence-mh", "elliptical-slice"),
        default="calibrated-snis",
    )
    parser.add_argument(
        "--worker-mode",
        choices=(
            "calibration",
            "validation",
            "independence-mh",
            "elliptical-slice",
        ),
    )
    parser.add_argument("--source-role", choices=("calibration", "validation"))
    parser.add_argument("--grid-size", type=int, choices=GRID_SIZES)
    parser.add_argument("--checkpoint", choices=CHECKPOINTS)
    parser.add_argument("--proposal-name")
    parser.add_argument("--attempt", type=int)
    parser.add_argument("--burn-in", type=int)
    parser.add_argument("--retained-per-chain", type=int)
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    if arguments.worker_mode:
        if arguments.grid_size is None or arguments.checkpoint is None:
            raise RuntimeError("worker mode requires grid size and checkpoint")
        if arguments.worker_mode == "calibration":
            result = _calibration_worker(arguments.grid_size, arguments.checkpoint)
        elif arguments.worker_mode == "validation":
            if not arguments.proposal_name:
                raise RuntimeError("validation worker requires a frozen proposal name")
            result = _validation_worker(
                arguments.grid_size, arguments.checkpoint, arguments.proposal_name
            )
        elif arguments.worker_mode == "independence-mh":
            if (
                not arguments.source_role
                or not arguments.proposal_name
                or arguments.attempt is None
                or arguments.burn_in is None
                or arguments.retained_per_chain is None
            ):
                raise RuntimeError("independence-MH worker arguments are incomplete")
            result = _mcmc_worker(
                arguments.source_role,
                arguments.grid_size,
                arguments.checkpoint,
                arguments.proposal_name,
                attempt=arguments.attempt,
                burn_in=arguments.burn_in,
                retained_per_chain=arguments.retained_per_chain,
            )
        else:
            if (
                not arguments.source_role
                or not arguments.proposal_name
                or arguments.attempt is None
                or arguments.burn_in is None
                or arguments.retained_per_chain is None
            ):
                raise RuntimeError("elliptical-slice worker arguments are incomplete")
            result = _elliptical_worker(
                arguments.source_role,
                arguments.grid_size,
                arguments.checkpoint,
                arguments.proposal_name,
                attempt=arguments.attempt,
                burn_in=arguments.burn_in,
                retained_per_chain=arguments.retained_per_chain,
            )
        print(json.dumps(result, sort_keys=True))
        return
    if arguments.backend == "elliptical-slice":
        result = run_elliptical_slice_rescue()
        gate = result["elliptical_slice"]["gate"]
    elif arguments.backend == "independence-mh":
        result = run_independence_mh_rescue()
        gate = result["independence_mh"]["gate"]
    else:
        result = run_calibrated_snis_rescue()
        gate = result["calibrated_snis_gate"]
    print(
        json.dumps(
            {
                "terminal_backend_classification": result[
                    "terminal_backend_classification"
                ],
                "backend_gate": gate,
                "wall_seconds": result["wall_seconds"],
                "peak_rss_gb": result["peak_rss_bytes"] / 1_000_000_000.0,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
