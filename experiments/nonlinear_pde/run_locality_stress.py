#!/usr/bin/env python3
"""Run the preregistered E2 locality stress test.

Modes are deliberately separated:

``regression``
    Read-only replay of the accepted 24x24 structural quantity.
``smoke``
    Development-seed-only resource and mechanism test.
``scientific``
    Protected prospective execution from a clean preregistration commit.

The historical 0.060/0.075 tolerance discrepancy is resolved prospectively in
the replacement frozen config: scientific execution uses the primary 0.060.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
import platform
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy import stats


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from conditioned_bo.nonlinear_pde_influence import (  # noqa: E402
    DEFAULT_PARAMETERS,
    build_nonlinear_pde_comparison,
    structural_screening_bound,
)
from conditioned_bo.nonlinear_pde_locality import (  # noqa: E402
    BOState,
    FactorWork,
    MethodResult,
    Problem,
    build_common_bo_states,
    build_problem,
    derive_prospective_seed,
    factor_energy_sum,
    laplace_snis_inference,
    oracle_geometric_prefix,
    peak_rss_bytes,
    random_matched_subsets,
    run_selective_method,
    state_incumbent,
)


OUTPUT_DIRECTORY = Path(__file__).resolve().parent / "outputs" / "locality_stress_v1"
CONFIG_PATH = OUTPUT_DIRECTORY / "frozen_config.json"
PREREGISTRATION_PATH = OUTPUT_DIRECTORY / "PREREGISTRATION.md"
LOCKED_STRUCTURAL_VALUE = 0.03874403301354687
LOCKED_ACTIVE_INDICES = np.asarray(
    [
        179, 202, 203, 204, 205, 225, 226, 227, 228, 229,
        250, 251, 252, 253, 275, 298, 299, 300, 301, 322,
        323, 324, 325, 326, 345, 346, 347, 348, 349, 350,
        369, 370, 371, 372, 373, 374, 394, 395, 396, 397,
    ],
    dtype=np.int64,
)
METHODS = (
    "FULL",
    "ADAPTIVE_INFLUENCE",
    "DYNAMIC_GEOMETRIC_SHELL",
    "STATIC_INFLUENCE",
    "FIXED_CHALLENGER",
)


def _read_config() -> dict[str, Any]:
    config = json.loads(CONFIG_PATH.read_text())
    expected = [derive_prospective_seed(index) for index in range(3)]
    if config["prospective_source_seeds"] != expected:
        raise RuntimeError("literal prospective seeds do not match SHA-256 derivation")
    if config["development"]["source_seed"] in expected:
        raise RuntimeError("development seed overlaps prospective source seeds")
    return config


def _config_sha256() -> str:
    return hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest()


def _git_state() -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return commit, bool(status)


def _environment() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pandas": pd.__version__,
        "cpu_count": os.cpu_count(),
    }


def run_locked_regression() -> dict[str, Any]:
    _, derivative_bounds, _, _, matrix = build_nonlinear_pde_comparison(24)
    active = np.zeros(24 * 24, dtype=bool)
    active[LOCKED_ACTIVE_INDICES] = True
    value = structural_screening_bound(
        matrix,
        derivative_bounds,
        ~active,
        9 * 24 + 12,
        14 * 24 + 12,
        DEFAULT_PARAMETERS.gamma,
        DEFAULT_PARAMETERS.tau,
    )
    passed = bool(np.isclose(value, LOCKED_STRUCTURAL_VALUE, rtol=2e-15, atol=2e-14))
    result = {
        "passed": passed,
        "computed_structural_value": value,
        "locked_structural_value": LOCKED_STRUCTURAL_VALUE,
        "absolute_error": abs(value - LOCKED_STRUCTURAL_VALUE),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if not passed:
        raise RuntimeError("accepted 24x24 structural regression failed")
    return result


def _state_id(grid_size: int, replicate: int, state: BOState) -> str:
    return f"n{grid_size}_r{replicate}_{state.checkpoint_label}_q{state.checkpoint_queries}"


def _method_arguments(
    config: dict[str, Any], state: BOState, *, proposal_seed: int, smoke: bool
) -> dict[str, Any]:
    numerical = config["development"] if smoke else config["numerical"]
    selection = config["selection"]
    return {
        "epsilon": selection["stopping_tolerance"],
        "batch_size": selection["batch_size"],
        "maximum_refinement_stages": (
            config["development"]["maximum_refinement_stages"]
            if smoke
            else selection["maximum_refinement_stages"]
        ),
        "incumbent": state_incumbent(state, config["model"]["incumbent"]),
        "delta_mc": config["inference"]["delta_mc"],
        "minimum_reference_ess_fraction": config["inference"][
            "routine_reference_ess_escalation_threshold"
        ],
        "laplace_sample_count": numerical["laplace_sample_count"],
        "proposal_seed": proposal_seed,
        "proposal_inflation": config["inference"]["laplace_proposal_inflation"],
        "gradient_tolerance": config["inference"]["laplace_gradient_tolerance"],
        "maximum_laplace_iterations": config["inference"][
            "laplace_maximum_iterations"
        ],
    }


def _build_states(
    config: dict[str, Any], grid_size: int, source_seed: int, replicate: int, *, smoke: bool
) -> tuple[Problem, tuple[BOState, ...], float]:
    setup_start = time.perf_counter()
    problem = build_problem(
        grid_size,
        source_seed,
        source_perturbation_scale=config["source_field"]["perturbation_scale"],
    )
    numerical = config["development"] if smoke else config["numerical"]
    trajectory_seed = int(
        hashlib.sha256(
            f"E2_LOCALITY_TRAJECTORY_V1:{grid_size}:{replicate}".encode("utf-8")
        ).hexdigest()[:8],
        16,
    )
    states = build_common_bo_states(
        problem,
        initialization_size=config["bo_trajectory"]["initialization_size"],
        total_queries=config["bo_trajectory"]["total_queries"],
        checkpoint_queries=config["bo_trajectory"]["checkpoint_queries"],
        observation_noise_variance=config["bo_trajectory"][
            "observation_noise_variance"
        ],
        reference_sample_count=numerical["reference_sample_count"],
        trajectory_seed=trajectory_seed,
        incumbent=config["model"]["incumbent"],
    )
    return problem, states, time.perf_counter() - setup_start


def _shadow_batch(
    config: dict[str, Any],
    state: BOState,
    problem: Problem,
    sample_count: int,
    seed: int,
) -> tuple[Any, dict[str, int]]:
    active = np.ones(problem.grid_size**2, dtype=bool)
    work = FactorWork()
    inference = laplace_snis_inference(
        state,
        problem,
        active,
        incumbent=state_incumbent(state, config["model"]["incumbent"]),
        delta_mc=config["inference"]["delta_mc"],
        sample_count=sample_count,
        proposal_seed=seed,
        proposal_inflation=config["inference"]["laplace_proposal_inflation"],
        work=work,
        gradient_tolerance=config["inference"]["laplace_gradient_tolerance"],
        maximum_iterations=config["inference"]["laplace_maximum_iterations"],
    )
    return inference, work.to_dict()


def _full_shadow(
    config: dict[str, Any],
    state: BOState,
    problem: Problem,
    *,
    state_seed: int,
    smoke: bool,
) -> dict[str, Any]:
    shadow = config["full_shadow"]
    incumbent = state_incumbent(state, config["model"]["incumbent"])
    if smoke:
        initial = config["development"]["shadow_sample_count_per_batch"]
        escalated = initial
    else:
        initial = shadow["initial_sample_count_per_batch"]
        escalated = shadow["escalated_sample_count_per_batch"]
    attempts: list[dict[str, Any]] = []
    final_batches = None
    for attempt, sample_count in enumerate(dict.fromkeys((initial, escalated))):
        batches = [
            _shadow_batch(
                config,
                state,
                problem,
                sample_count,
                state_seed + 100_003 * attempt + batch,
            )
            for batch in range(2)
        ]
        inferences = [item[0] for item in batches]
        actions = [item.leader_index for item in inferences]
        ess = [item.ess_fraction for item in inferences]
        max_difference = float(
            np.max(np.abs(inferences[0].acquisition - inferences[1].acquisition))
        )
        reliable = (
            len(set(actions)) == 1
            and min(ess) >= shadow["minimum_ess_fraction"]
            and max_difference <= shadow["maximum_batch_acquisition_difference"]
        )
        attempts.append(
            {
                "attempt": attempt,
                "sample_count_per_batch": sample_count,
                "actions": actions,
                "ess_fractions": ess,
                "maximum_batch_acquisition_difference": max_difference,
                "reliable": reliable,
                "work": [item[1] for item in batches],
                "inference_seconds": [item.inference_seconds for item in inferences],
            }
        )
        final_batches = inferences
        if reliable:
            break
    assert final_batches is not None
    acquisition = 0.5 * (
        final_batches[0].acquisition + final_batches[1].acquisition
    )
    order = np.argsort(-acquisition, kind="stable")
    return {
        "state_specific_incumbent": incumbent,
        "reliable": attempts[-1]["reliable"],
        "action_index": int(state.action_indices[int(order[0])]),
        "challenger_index": int(state.action_indices[int(order[1])]),
        "acquisition": acquisition,
        "ess_fraction": float(min(attempts[-1]["ess_fractions"])),
        "attempts": attempts,
        "escalated": len(attempts) > 1,
    }


def _reference_ess(
    state: BOState, problem: Problem, factor_indices: Iterable[int]
) -> float:
    energies = factor_energy_sum(
        state.reference_samples,
        problem,
        np.asarray(list(factor_indices), dtype=np.int64),
    )
    weights = np.exp(-energies + np.min(energies))
    return float((weights.sum() ** 2 / np.dot(weights, weights)) / weights.size)


def _method_rows(
    state_id: str,
    grid_size: int,
    replicate: int,
    state: BOState,
    setup_seconds: float,
    methods: dict[str, MethodResult],
    shadow: dict[str, Any],
    problem: Problem,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    state_rows: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []
    full_best_local = int(np.argmax(shadow["acquisition"]))
    full_best = float(shadow["acquisition"][full_best_local])
    full_reference_ess = _reference_ess(state, problem, range(grid_size**2))
    incumbent_values = {
        float(result.audit["state_specific_incumbent"])
        for result in methods.values()
    }
    if len(incumbent_values) != 1:
        raise RuntimeError("method rows received inconsistent incumbents")
    incumbent = incumbent_values.pop()
    for method_name, result in methods.items():
        selected_local = int(np.flatnonzero(state.action_indices == result.action_index)[0])
        regret = (
            max(0.0, full_best - float(shadow["acquisition"][selected_local]))
            if shadow["reliable"]
            else np.nan
        )
        active_reference_ess = _reference_ess(
            state, problem, result.final_active_indices
        )
        row = {
            "state_id": state_id,
            "grid_size": grid_size,
            "N": grid_size**2,
            "source_replicate": replicate,
            "checkpoint": state.checkpoint_label,
            "checkpoint_queries": state.checkpoint_queries,
            "state_fingerprint": state.fingerprint(),
            "state_specific_incumbent": incumbent,
            "method": method_name,
            "action_index": result.action_index,
            "full_action_index": shadow["action_index"],
            "full_challenger_index": shadow["challenger_index"],
            "full_action_agreement": (
                result.action_index == shadow["action_index"]
                if shadow["reliable"]
                else np.nan
            ),
            "full_acquisition_regret": regret,
            "full_reference_reliable": shadow["reliable"],
            "full_reference_escalated": shadow["escalated"],
            "M": len(result.final_active_indices),
            "active_fraction": len(result.final_active_indices) / (grid_size**2),
            "full_fallback": result.full_fallback,
            "refinement_stages": len(result.stages),
            "factor_indices": json.dumps(result.final_active_indices),
            "factor_energy_evaluations": result.work["factor_energy_evaluations"],
            "factor_gradient_elements": result.work["factor_gradient_elements"],
            "factor_hessian_elements": result.work["factor_hessian_elements"],
            "sparse_comparison_solves": result.work["sparse_comparison_solves"],
            "active_target_inference_seconds": result.inference_seconds,
            "challenger_seconds": result.challenger_seconds,
            "total_method_seconds": result.total_seconds,
            "shared_setup_seconds": setup_seconds,
            "peak_rss_bytes": result.peak_rss_bytes,
            "active_reference_is_ess_fraction": active_reference_ess,
            "full_reference_is_ess_fraction": full_reference_ess,
            "routine_final_ess_fraction": result.final_ess_fraction,
            "routine_final_proposal": result.final_proposal,
            "shadow_laplace_is_ess_fraction": shadow["ess_fraction"],
            "audit": json.dumps(result.audit, sort_keys=True),
        }
        state_rows.append(row)
        for stage in result.stages:
            stage_rows.append(
                {
                    "state_id": state_id,
                    "grid_size": grid_size,
                    "source_replicate": replicate,
                    "checkpoint": state.checkpoint_label,
                    "method": method_name,
                    **asdict(stage),
                    "active_indices": json.dumps(stage.active_indices),
                    "activated_indices": json.dumps(stage.activated_indices),
                    "work": json.dumps(stage.work, sort_keys=True),
                }
            )
    return state_rows, stage_rows


def _run_one_state(
    config: dict[str, Any],
    problem: Problem,
    state: BOState,
    *,
    grid_size: int,
    replicate: int,
    setup_seconds: float,
    smoke: bool,
    methods_to_run: tuple[str, ...] = METHODS,
) -> dict[str, Any]:
    state_start = time.perf_counter()
    identifier = _state_id(grid_size, replicate, state)
    state_seed = int(hashlib.sha256(identifier.encode()).hexdigest()[:8], 16)
    incumbent = state_incumbent(state, config["model"]["incumbent"])
    methods: dict[str, MethodResult] = {}
    for offset, method in enumerate(methods_to_run):
        methods[method] = run_selective_method(
            method,
            state,
            problem,
            **_method_arguments(
                config,
                state,
                proposal_seed=state_seed + 10_000 * offset,
                smoke=smoke,
            ),
        )
    fingerprints = {result.audit["state_fingerprint"] for result in methods.values()}
    if fingerprints != {state.fingerprint()}:
        raise RuntimeError("paired methods did not receive byte-identical BO state")
    incumbents = {
        float(result.audit["state_specific_incumbent"]) for result in methods.values()
    }
    if incumbents != {incumbent}:
        raise RuntimeError("paired methods did not receive the same state-specific incumbent")
    shadow_start = time.perf_counter()
    shadow = _full_shadow(
        config, state, problem, state_seed=state_seed + 900_000, smoke=smoke
    )
    shadow_seconds = time.perf_counter() - shadow_start
    if float(shadow["state_specific_incumbent"]) != incumbent:
        raise RuntimeError("FULL shadow did not receive the state-specific incumbent")
    state_rows, stage_rows = _method_rows(
        identifier,
        grid_size,
        replicate,
        state,
        setup_seconds,
        methods,
        shadow,
        problem,
    )
    oracle = None
    random_rows: list[dict[str, Any]] = []
    oracle_seconds = 0.0
    random_seconds = 0.0
    if not smoke and shadow["reliable"]:
        oracle_start = time.perf_counter()
        oracle = oracle_geometric_prefix(
            state,
            problem,
            full_action_index=shadow["action_index"],
            full_challenger_index=shadow["challenger_index"],
            full_acquisition=shadow["acquisition"],
            batch_size=config["selection"]["batch_size"],
            regret_threshold=config["diagnostics"]["oracle_regret_threshold"],
            incumbent=incumbent,
            delta_mc=config["inference"]["delta_mc"],
        )
        oracle["state_specific_incumbent"] = incumbent
        oracle_seconds = time.perf_counter() - oracle_start
        random_spec = config["random_baseline"]
        if (
            grid_size in random_spec["grid_sizes"]
            and state.checkpoint_label in random_spec["checkpoints"]
        ):
            random_start = time.perf_counter()
            matched = len(methods["ADAPTIVE_INFLUENCE"].final_active_indices)
            random_seed = int(
                hashlib.sha256(f"E2_RANDOM_MATCHED:{identifier}".encode()).hexdigest()[:8],
                16,
            )
            random_rows = random_matched_subsets(
                state,
                problem,
                matched_count=matched,
                subset_count=random_spec["subsets_per_state"],
                random_seed=random_seed,
                incumbent=incumbent,
                delta_mc=config["inference"]["delta_mc"],
            )
            for row in random_rows:
                row["state_specific_incumbent"] = incumbent
                row["state_id"] = identifier
                selected_local = int(
                    np.flatnonzero(state.action_indices == row["action_index"])[0]
                )
                row["full_action_agreement"] = row["action_index"] == shadow["action_index"]
                row["full_acquisition_regret"] = float(
                    shadow["acquisition"].max() - shadow["acquisition"][selected_local]
                )
            random_seconds = time.perf_counter() - random_start
    elif not smoke:
        oracle = {
            "method": "ORACLE_GEOMETRIC_PREFIX",
            "evaluated": False,
            "reason": "FULL_REFERENCE_UNRELIABLE",
            "deployable": False,
            "feeds_deployable_methods": False,
            "state_specific_incumbent": incumbent,
        }
    return {
        "state_rows": state_rows,
        "stage_rows": stage_rows,
        "shadow": {key: value for key, value in shadow.items() if key != "acquisition"},
        "oracle": oracle,
        "random_rows": random_rows,
        "operational": {
            "state_specific_incumbent": incumbent,
            "core_method_seconds": float(
                sum(result.total_seconds for result in methods.values())
            ),
            "shadow_seconds": shadow_seconds,
            "oracle_seconds": oracle_seconds,
            "random_seconds": random_seconds,
            "total_state_seconds": time.perf_counter() - state_start,
            "peak_rss_bytes": peak_rss_bytes(),
        },
    }


def run_smoke(config: dict[str, Any]) -> dict[str, Any]:
    run_locked_regression()
    rows: list[dict[str, Any]] = []
    cases = config["development"]["grid_sizes"]
    started = time.perf_counter()
    for grid_size in cases:
        problem, states, setup_seconds = _build_states(
            config,
            grid_size,
            config["development"]["source_seed"],
            -1,
            smoke=True,
        )
        result = _run_one_state(
            config,
            problem,
            states[0],
            grid_size=grid_size,
            replicate=-1,
            setup_seconds=setup_seconds,
            smoke=True,
            methods_to_run=("FULL", "ADAPTIVE_INFLUENCE", "DYNAMIC_GEOMETRIC_SHELL"),
        )
        rows.extend(result["state_rows"])
        print(
            f"[smoke] n={grid_size} completed; peak_rss={peak_rss_bytes()/1e9:.3f} GB",
            flush=True,
        )
    elapsed = time.perf_counter() - started
    table = pd.DataFrame(rows)
    result = {
        "mode": "development_smoke",
        "development_seed_only": True,
        "prospective_seeds_evaluated": False,
        "grid_sizes": cases,
        "wall_seconds": elapsed,
        "peak_rss_bytes": peak_rss_bytes(),
        "peak_rss_gb": peak_rss_bytes() / 1_000_000_000.0,
        "routing_authoritative": False,
        "routing_note": "superseded by development_full_fidelity_profile.json",
        "environment": _environment(),
        "config_sha256": _config_sha256(),
        "rows": rows,
    }
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIRECTORY / "development_smoke.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    pd.DataFrame(rows).to_csv(
        OUTPUT_DIRECTORY / "development_resource_metrics.csv", index=False
    )
    print(json.dumps({key: result[key] for key in (
        "wall_seconds", "peak_rss_gb", "routing_authoritative")}, indent=2))
    return result


def run_full_fidelity_development_profile(config: dict[str, Any]) -> dict[str, Any]:
    """Profile the frozen n=40 early/late workload without prospective seeds."""

    run_locked_regression()
    development_seed = int(config["development"]["source_seed"])
    prospective = set(int(value) for value in config["prospective_source_seeds"])
    if development_seed in prospective:
        raise RuntimeError("development profile seed overlaps prospective seeds")
    profile_start = time.perf_counter()
    problem, states, setup_seconds = _build_states(
        config,
        40,
        development_seed,
        -2,
        smoke=False,
    )
    selected_states = [
        state for state in states if state.checkpoint_label in {"early", "late"}
    ]
    if [state.checkpoint_label for state in selected_states] != ["early", "late"]:
        raise RuntimeError("development profile did not select frozen early/late states")
    method_rows: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []
    state_profiles: list[dict[str, Any]] = []
    for state in selected_states:
        result = _run_one_state(
            config,
            problem,
            state,
            grid_size=40,
            replicate=-2,
            setup_seconds=setup_seconds,
            smoke=False,
            methods_to_run=METHODS,
        )
        method_rows.extend(result["state_rows"])
        stage_rows.extend(result["stage_rows"])
        state_profiles.append(
            {
                "state_id": _state_id(40, -2, state),
                "checkpoint": state.checkpoint_label,
                "checkpoint_queries": state.checkpoint_queries,
                "state_specific_incumbent": state_incumbent(
                    state, config["model"]["incumbent"]
                ),
                "operational": result["operational"],
                "full_shadow": result["shadow"],
                "oracle": result["oracle"],
                "random_subset_count": len(result["random_rows"]),
            }
        )
        print(
            "[full-fidelity profile] "
            f"{state.checkpoint_label} completed in "
            f"{result['operational']['total_state_seconds']:.1f}s; "
            f"peak_rss={result['operational']['peak_rss_bytes']/1e9:.3f} GB",
            flush=True,
        )
    stage_table = pd.DataFrame(stage_rows)
    method_profiles: list[dict[str, Any]] = []
    for row in method_rows:
        stages = stage_table[
            (stage_table.state_id == row["state_id"])
            & (stage_table.method == row["method"])
        ]
        method_profiles.append(
            {
                "state_id": row["state_id"],
                "checkpoint": row["checkpoint"],
                "method": row["method"],
                "state_specific_incumbent": row["state_specific_incumbent"],
                "actual_stages": int(len(stages)),
                "laplace_escalation_occurred": bool(
                    np.any(stages.proposal == "LAPLACE_SNIS")
                ),
                "stage_proposals": stages.proposal.tolist(),
                "full_fallback": bool(row["full_fallback"]),
                "M": int(row["M"]),
                "factor_energy_evaluations": int(row["factor_energy_evaluations"]),
                "factor_gradient_elements": int(row["factor_gradient_elements"]),
                "factor_hessian_elements": int(row["factor_hessian_elements"]),
                "sparse_comparison_solves": int(row["sparse_comparison_solves"]),
                "inference_seconds": float(row["active_target_inference_seconds"]),
                "challenger_seconds": float(row["challenger_seconds"]),
                "total_method_seconds": float(row["total_method_seconds"]),
                "peak_rss_bytes": int(row["peak_rss_bytes"]),
            }
        )
    observed_state_seconds = np.asarray(
        [profile["operational"]["total_state_seconds"] for profile in state_profiles],
        dtype=float,
    )
    projected_mean_seconds = float(45.0 * observed_state_seconds.mean() + 15.0 * setup_seconds)
    projected_conservative_seconds = float(
        45.0 * observed_state_seconds.max() + 15.0 * setup_seconds
    )
    peak_bytes = int(
        max(profile["operational"]["peak_rss_bytes"] for profile in state_profiles)
    )
    profile = {
        "mode": "development_full_fidelity_profile",
        "operational_only": True,
        "development_source_seed": development_seed,
        "trajectory_replicate_identifier": -2,
        "prospective_seeds_evaluated": False,
        "superseded_preregistration_sha": "c976ab186a730e5ddfd2270ccf1d60577ee0d6b6",
        "scientific_budgets_used": {
            "reference_sample_count": config["numerical"]["reference_sample_count"],
            "laplace_sample_count": config["numerical"]["laplace_sample_count"],
            "maximum_refinement_stages": config["selection"][
                "maximum_refinement_stages"
            ],
            "initial_full_shadow_samples_per_batch": config["full_shadow"][
                "initial_sample_count_per_batch"
            ],
            "full_shadow_batches": 2,
            "full_shadow_escalation_rule_active": True,
        },
        "shared_setup_seconds": setup_seconds,
        "profile_wall_seconds": time.perf_counter() - profile_start,
        "peak_rss_bytes": peak_bytes,
        "peak_rss_gb": peak_bytes / 1_000_000_000.0,
        "state_profiles": state_profiles,
        "method_profiles": method_profiles,
        "projection": {
            "basis": "45 times observed n=40 early/late mean or maximum state wall time, plus 15 observed domain-replicate setup costs; no synthetic fidelity/stage multiplier",
            "observed_state_seconds": observed_state_seconds.tolist(),
            "projected_mean_total_seconds": projected_mean_seconds,
            "projected_mean_total_minutes": projected_mean_seconds / 60.0,
            "projected_conservative_total_seconds": projected_conservative_seconds,
            "projected_conservative_total_minutes": projected_conservative_seconds
            / 60.0,
        },
        "local_route_allowed": (
            peak_bytes / 1_000_000_000.0
            < config["resource_routing"]["maximum_local_peak_rss_gb"]
            and projected_conservative_seconds / 60.0
            < config["resource_routing"]["maximum_local_runtime_minutes"]
        ),
        "environment": _environment(),
        "config_sha256_at_profile_time": _config_sha256(),
    }
    (OUTPUT_DIRECTORY / "development_full_fidelity_profile.json").write_text(
        json.dumps(profile, indent=2, sort_keys=True) + "\n"
    )
    pd.DataFrame(method_profiles).to_csv(
        OUTPUT_DIRECTORY / "development_full_fidelity_method_metrics.csv",
        index=False,
    )
    stage_table.to_csv(
        OUTPUT_DIRECTORY / "development_full_fidelity_stage_metrics.csv",
        index=False,
    )
    print(
        json.dumps(
            {
                "profile_wall_seconds": profile["profile_wall_seconds"],
                "peak_rss_gb": profile["peak_rss_gb"],
                "projected_mean_total_minutes": profile["projection"][
                    "projected_mean_total_minutes"
                ],
                "projected_conservative_total_minutes": profile["projection"][
                    "projected_conservative_total_minutes"
                ],
                "local_route_allowed": profile["local_route_allowed"],
            },
            indent=2,
        )
    )
    return profile


def _bootstrap_log_ratio(
    adaptive: FloatArray, baseline: FloatArray, config: dict[str, Any]
) -> dict[str, float]:
    logs = np.log(adaptive / baseline)
    rng = np.random.default_rng(config["statistics"]["bootstrap_seed"])
    count = config["statistics"]["bootstrap_resamples"]
    indices = rng.integers(0, logs.size, size=(count, logs.size))
    means = logs[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return {
        "mean_log_ratio": float(logs.mean()),
        "lower_log": float(low),
        "upper_log": float(high),
        "lower_ratio": float(np.exp(low)),
        "upper_ratio": float(np.exp(high)),
    }


def _paired_comparison(
    table: pd.DataFrame, baseline: str, config: dict[str, Any]
) -> dict[str, Any]:
    adaptive = table[table.method == "ADAPTIVE_INFLUENCE"].set_index("state_id")
    other = table[table.method == baseline].set_index("state_id")
    counts_a = adaptive.loc[other.index, "M"].to_numpy(dtype=float)
    counts_b = other["M"].to_numpy(dtype=float)
    ratio = counts_a / counts_b
    reliable_ids = adaptive.index[
        adaptive["full_reference_reliable"].astype(bool)
        & other.loc[adaptive.index, "full_reference_reliable"].astype(bool)
    ]
    reliable_adaptive = adaptive.loc[reliable_ids]
    reliable_other = other.loc[reliable_ids]
    return {
        "baseline": baseline,
        "median_ratio": float(np.median(ratio)),
        "geometric_mean_ratio": float(np.exp(np.mean(np.log(ratio)))),
        "wins": int(np.sum(counts_a < counts_b)),
        "ties": int(np.sum(counts_a == counts_b)),
        "losses": int(np.sum(counts_a > counts_b)),
        "win_fraction": float(np.mean(counts_a < counts_b)),
        "bootstrap": _bootstrap_log_ratio(counts_a, counts_b, config),
        "full_reference_reliable_quality_states": int(len(reliable_ids)),
        "adaptive_mean_regret": float(
            reliable_adaptive["full_acquisition_regret"].mean()
        ),
        "baseline_mean_regret": float(
            reliable_other["full_acquisition_regret"].mean()
        ),
    }


def _matched_quality_disadvantage(
    table: pd.DataFrame,
    baseline: str,
    regret_margin: float,
) -> dict[str, Any]:
    """Evaluate A4's quality/fallback composite on reliable FULL states only."""

    adaptive = table[table.method == "ADAPTIVE_INFLUENCE"].set_index("state_id")
    other = table[table.method == baseline].set_index("state_id").loc[
        adaptive.index
    ]
    reliable_ids = adaptive.index[
        adaptive.full_reference_reliable.astype(bool)
        & other.full_reference_reliable.astype(bool)
    ]
    reliable_adaptive = adaptive.loc[reliable_ids]
    reliable_other = other.loc[reliable_ids]
    worse = (
        reliable_other.full_acquisition_regret.to_numpy()
        > reliable_adaptive.full_acquisition_regret.to_numpy() + regret_margin
    ) | (
        reliable_other.full_fallback.to_numpy()
        & ~reliable_adaptive.full_fallback.to_numpy()
    )
    return {
        "full_reference_reliable_quality_states": int(len(reliable_ids)),
        "worse_quality_or_fallback_fraction": (
            float(np.mean(worse)) if len(reliable_ids) else float("nan")
        ),
    }


def _aggregate_and_verdict(
    table: pd.DataFrame, config: dict[str, Any]
) -> tuple[dict[str, Any], pd.DataFrame]:
    reliable_ids = set(table.loc[table.full_reference_reliable, "state_id"])
    adaptive = table[table.method == "ADAPTIVE_INFLUENCE"].copy()
    reliable = adaptive[adaptive.state_id.isin(reliable_ids)]
    medians = adaptive.groupby("N", as_index=False)["M"].median()
    slope = float(stats.theilslopes(np.log(medians.M), np.log(medians.N))[0])
    n40 = adaptive[adaptive.grid_size == 40]
    geometry = _paired_comparison(table, "DYNAMIC_GEOMETRIC_SHELL", config)
    static = _paired_comparison(table, "STATIC_INFLUENCE", config)
    fixed = _paired_comparison(table, "FIXED_CHALLENGER", config)
    criteria = config["verdict_thresholds"]
    if reliable.empty:
        agreement_fraction = float("nan")
        regret_p95 = float("nan")
        regret_max = float("nan")
        fallback_fraction = float("nan")
    else:
        agreement_fraction = float(reliable.full_action_agreement.mean())
        regret_p95 = float(np.quantile(reliable.full_acquisition_regret, 0.95))
        regret_max = float(reliable.full_acquisition_regret.max())
        fallback_fraction = float(reliable.full_fallback.mean())
    a1 = {
        "reliable_fraction": len(reliable_ids) / 45.0,
        "agreement_fraction": agreement_fraction,
        "regret_p95": regret_p95,
        "regret_max": regret_max,
        "fallback_fraction": fallback_fraction,
    }
    a1["pass"] = (
        a1["reliable_fraction"] >= criteria["a1"]["minimum_reliable_fraction"]
        and a1["agreement_fraction"] >= criteria["a1"]["minimum_action_agreement"]
        and a1["regret_p95"] <= criteria["a1"]["maximum_regret_p95"]
        and a1["regret_max"] <= criteria["a1"]["maximum_regret"]
        and a1["fallback_fraction"] <= criteria["a1"]["maximum_fallback_fraction"]
    )
    a2 = {
        "alpha": slope,
        "n40_median_active_fraction": float(n40.active_fraction.median()),
        "n40_p80_active_fraction": float(np.quantile(n40.active_fraction, 0.8)),
        "median_m_ratio_n40_n18": float(
            adaptive[adaptive.grid_size == 40].M.median()
            / adaptive[adaptive.grid_size == 18].M.median()
        ),
        "total_factor_ratio_n40_n18": 1600 / 324,
        "five_points": medians.to_dict("records"),
    }
    a2["pass"] = (
        a2["alpha"] <= criteria["a2"]["maximum_alpha"]
        and a2["n40_median_active_fraction"]
        <= criteria["a2"]["maximum_n40_median_active_fraction"]
        and a2["n40_p80_active_fraction"]
        <= criteria["a2"]["maximum_n40_p80_active_fraction"]
    )
    a3 = {**geometry}
    a3["pass"] = (
        geometry["geometric_mean_ratio"]
        <= criteria["a3"]["maximum_geometric_mean_ratio"]
        and geometry["win_fraction"] >= criteria["a3"]["minimum_win_fraction"]
        and geometry["bootstrap"]["upper_ratio"] < 1.0
        and geometry["adaptive_mean_regret"]
        <= geometry["baseline_mean_regret"] + criteria["matched_quality_regret_margin"]
    )
    ablation_quality = []
    for baseline, comparison in (("STATIC_INFLUENCE", static), ("FIXED_CHALLENGER", fixed)):
        quality = _matched_quality_disadvantage(
            table,
            baseline,
            criteria["matched_quality_regret_margin"],
        )
        ablation_quality.append(
            {
                "baseline": baseline,
                "geometric_mean_ratio": comparison["geometric_mean_ratio"],
                **quality,
            }
        )
    a4 = {
        "comparisons": ablation_quality,
        "pass": any(
            item["geometric_mean_ratio"] <= criteria["a4"]["maximum_count_ratio"]
            or item["worse_quality_or_fallback_fraction"]
            >= criteria["a4"]["minimum_quality_disadvantage_fraction"]
            for item in ablation_quality
        ),
    }
    full = table[table.method == "FULL"].set_index("state_id")
    ratios: dict[int, dict[str, float]] = {}
    for grid_size in (36, 40):
        a = adaptive[adaptive.grid_size == grid_size].set_index("state_id")
        f = full.loc[a.index]
        gradient_denominator = f.factor_gradient_elements.replace(0, np.nan)
        ratios[grid_size] = {
            "wall_ratio": float(np.median(a.total_method_seconds / f.total_method_seconds)),
            "gradient_work_ratio": float(
                np.nanmedian(a.factor_gradient_elements / gradient_denominator)
            ),
        }
    a5 = {"ratios": ratios}
    a5["pass"] = (
        ratios[40]["gradient_work_ratio"]
        <= criteria["a5"]["maximum_n40_gradient_work_ratio"]
        and ratios[40]["wall_ratio"] <= criteria["a5"]["maximum_n40_wall_ratio"]
        and ratios[36]["wall_ratio"] <= criteria["a5"]["maximum_n36_wall_ratio"]
    )
    a6 = {
        "n40_median_active_ess": float(n40.active_reference_is_ess_fraction.median()),
        "n40_median_full_ess": float(n40.full_reference_is_ess_fraction.median()),
        "n40_median_active_full_ess_ratio": float(
            np.median(
                n40.active_reference_is_ess_fraction
                / n40.full_reference_is_ess_fraction
            )
        ),
    }
    a6["pass"] = (
        a6["n40_median_active_ess"] >= criteria["a6"]["minimum_active_ess_fraction"]
        and a6["n40_median_active_full_ess_ratio"]
        >= criteria["a6"]["minimum_active_full_ess_ratio"]
    )
    gates = {"A1": a1, "A2": a2, "A3": a3, "A4": a4, "A5": a5, "A6": a6}
    if a1["reliable_fraction"] < criteria["a1"]["minimum_reliable_fraction"]:
        verdict = "INCONCLUSIVE_FULL_REFERENCE"
    elif all(gate["pass"] for gate in gates.values()):
        verdict = "PASS_STRONG_E2"
    elif a1["pass"] and a2["pass"] and a5["pass"] and a6["pass"] and not a3["pass"]:
        verdict = "PASS_SCALING_NOT_ADAPTIVITY"
    else:
        verdict = "FAIL_E2_MECHANISM"
    summary_source = table.copy()
    summary_source.loc[
        ~summary_source.full_reference_reliable.astype(bool),
        ["full_action_agreement", "full_acquisition_regret"],
    ] = np.nan
    summary_table = (
        summary_source.groupby(["method", "grid_size", "checkpoint"], as_index=False)
        .agg(
            M_median=("M", "median"),
            active_fraction_median=("active_fraction", "median"),
            full_agreement=("full_action_agreement", "mean"),
            full_regret_p95=(
                "full_acquisition_regret",
                lambda value: (
                    np.quantile(value.dropna(), 0.95)
                    if not value.dropna().empty
                    else np.nan
                ),
            ),
            fallback_fraction=("full_fallback", "mean"),
            stages_median=("refinement_stages", "median"),
            gradient_work_median=("factor_gradient_elements", "median"),
            wall_seconds_median=("total_method_seconds", "median"),
            active_ess_median=("active_reference_is_ess_fraction", "median"),
            full_ess_median=("full_reference_is_ess_fraction", "median"),
        )
    )
    return {
        "verdict": verdict,
        "gates": gates,
        "locality_comparison": geometry,
        "static_comparison": static,
        "fixed_challenger_comparison": fixed,
    }, summary_table


def _make_figures(table: pd.DataFrame, output: Path) -> None:
    colors = {
        "ADAPTIVE_INFLUENCE": "#0b7285",
        "DYNAMIC_GEOMETRIC_SHELL": "#e67700",
        "STATIC_INFLUENCE": "#7048e8",
        "FULL": "#343a40",
    }
    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    for method in colors:
        subset = table[table.method == method]
        axis.scatter(subset.N, subset.M, alpha=0.45, s=18, color=colors[method])
        trend = subset.groupby("N").M.median()
        axis.plot(trend.index, trend.values, marker="o", color=colors[method], label=method)
    axis.set(xlabel="Total residual factors N", ylabel="Final active factors M", title="E2 locality stress: scaling")
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output / "figure1_scaling.png", dpi=180)
    plt.close(figure)

    adaptive = table[table.method == "ADAPTIVE_INFLUENCE"].set_index("state_id")
    geometry = table[table.method == "DYNAMIC_GEOMETRIC_SHELL"].set_index("state_id")
    ratios = adaptive.loc[geometry.index, "M"] / geometry.M
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 4.2))
    axes[0].scatter(geometry.M, adaptive.loc[geometry.index, "M"], alpha=0.7)
    maximum = max(float(geometry.M.max()), float(adaptive.M.max()))
    axes[0].plot([0, maximum], [0, maximum], "k--", lw=1)
    axes[0].set(xlabel="Dynamic geometry M", ylabel="Adaptive influence M")
    axes[1].hist(ratios, bins=15, color=colors["ADAPTIVE_INFLUENCE"])
    axes[1].axvline(1.0, color="black", ls="--")
    axes[1].set(xlabel="M_adaptive / M_geometry", ylabel="States")
    figure.suptitle("Primary locality discrimination")
    figure.tight_layout()
    figure.savefig(output / "figure2_locality_stress.png", dpi=180)
    plt.close(figure)

    full = table[table.method == "FULL"].set_index("state_id")
    adaptive = adaptive.loc[full.index]
    computational = pd.DataFrame(
        {
            "N": adaptive.N,
            "wall": adaptive.total_method_seconds / full.total_method_seconds,
            "work": adaptive.factor_gradient_elements
            / full.factor_gradient_elements.replace(0, np.nan),
            "ess": adaptive.active_reference_is_ess_fraction
            / adaptive.full_reference_is_ess_fraction,
        }
    ).groupby("N").median()
    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    for column, label in (("wall", "wall time"), ("work", "gradient work"), ("ess", "active/full ESS")):
        axis.plot(computational.index, computational[column], marker="o", label=label)
    axis.axhline(1.0, color="black", ls="--", lw=1)
    axis.set(xlabel="Total residual factors N", ylabel="Ratio", title="Computational consequence")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "figure3_computational_consequence.png", dpi=180)
    plt.close(figure)

    spatial = table[
        (table.grid_size == 40)
        & (table.source_replicate == 0)
        & table.method.isin(["ADAPTIVE_INFLUENCE", "DYNAMIC_GEOMETRIC_SHELL"])
    ]
    figure, axes = plt.subplots(2, 3, figsize=(10.0, 6.5))
    for row_index, method in enumerate(("ADAPTIVE_INFLUENCE", "DYNAMIC_GEOMETRIC_SHELL")):
        for column_index, checkpoint in enumerate(("early", "middle", "late")):
            row = spatial[(spatial.method == method) & (spatial.checkpoint == checkpoint)].iloc[0]
            mask = np.zeros((40, 40), dtype=float)
            indices = np.asarray(json.loads(row.factor_indices), dtype=int)
            mask.ravel()[indices] = 1.0
            axes[row_index, column_index].imshow(mask, origin="lower", cmap="Blues", vmin=0, vmax=1)
            axes[row_index, column_index].set_title(f"{method}\n{checkpoint}; M={indices.size}", fontsize=8)
            axes[row_index, column_index].set_xticks([])
            axes[row_index, column_index].set_yticks([])
    figure.suptitle("Frozen n=40 replicate-0 spatial examples")
    figure.tight_layout()
    figure.savefig(output / "figure4_spatial_examples.png", dpi=180)
    plt.close(figure)


def _results_markdown(summary: dict[str, Any], provenance: dict[str, Any]) -> str:
    gates = summary["gates"]
    lines = [
        "# E2 Locality Stress V1 Results",
        "",
        f"Terminal verdict: **{summary['verdict']}**",
        "",
        "## Provenance",
        "",
        f"- Preregistration SHA: `{provenance['preregistration_sha']}`",
        f"- Implementation/run SHA: `{provenance['run_sha']}`",
        f"- Frozen config SHA-256: `{provenance['config_sha256']}`",
        f"- Machine/environment: `{json.dumps(provenance['environment'], sort_keys=True)}`",
        "",
        "## Frozen criteria",
        "",
        "| Criterion | Result |",
        "|---|---:|",
    ]
    for name, gate in gates.items():
        lines.append(f"| {name} | {'PASS' if gate['pass'] else 'FAIL'} |")
    lines.extend(
        [
            "",
            "## Primary results",
            "",
            f"- Scaling alpha: `{gates['A2']['alpha']}`; five points: `{json.dumps(gates['A2']['five_points'])}`.",
            f"- Adaptive versus geometry: `{json.dumps(summary['locality_comparison'], sort_keys=True)}`.",
            f"- Adaptive versus static: `{json.dumps(summary['static_comparison'], sort_keys=True)}`.",
            f"- Adaptive versus fixed challenger: `{json.dumps(summary['fixed_challenger_comparison'], sort_keys=True)}`.",
            f"- Decision validity: `{json.dumps(gates['A1'], sort_keys=True)}`.",
            f"- Active/FULL inference: `{json.dumps(gates['A6'], sort_keys=True)}`.",
            f"- End-to-end compute: `{json.dumps(gates['A5'], sort_keys=True)}`.",
            "",
            "## Scientific interpretation",
            "",
            (
                "The terminal verdict above is applied mechanically to the frozen thresholds. "
                "It is evidence about BO decision conditioning in this fixed nonlinear-PDE family, "
                "not sampler novelty or a rigorous finite-sample E2 certificate."
            ),
            "",
            "## Strongest skeptical interpretation",
            "",
            (
                "The dynamic geometric-shell and hindsight oracle-geometric diagnostics test whether "
                "the observed factor sparsity is adequately explained by ordinary PDE locality. "
                "A failure of A3 must be reported as locality/scaling evidence rather than an "
                "adaptivity advantage."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def run_scientific(config: dict[str, Any], preregistration_sha: str) -> dict[str, Any]:
    commit, dirty = _git_state()
    if dirty:
        raise RuntimeError("scientific mode requires a clean worktree")
    if commit != preregistration_sha:
        raise RuntimeError("HEAD does not match the supplied preregistration SHA")
    run_locked_regression()
    all_state_rows: list[dict[str, Any]] = []
    all_stage_rows: list[dict[str, Any]] = []
    resource_rows: list[dict[str, Any]] = []
    shadows: dict[str, Any] = {}
    oracles: dict[str, Any] = {}
    random_rows: list[dict[str, Any]] = []
    run_start = time.perf_counter()
    for grid_size in config["domain_sizes"]:
        for replicate, source_seed in enumerate(config["prospective_source_seeds"]):
            problem, states, setup_seconds = _build_states(
                config, grid_size, source_seed, replicate, smoke=False
            )
            for state in states:
                identifier = _state_id(grid_size, replicate, state)
                state_start = time.perf_counter()
                result = _run_one_state(
                    config,
                    problem,
                    state,
                    grid_size=grid_size,
                    replicate=replicate,
                    setup_seconds=setup_seconds,
                    smoke=False,
                )
                all_state_rows.extend(result["state_rows"])
                all_stage_rows.extend(result["stage_rows"])
                shadows[identifier] = result["shadow"]
                oracles[identifier] = result["oracle"]
                random_rows.extend(result["random_rows"])
                resource_rows.append(
                    {
                        "state_id": identifier,
                        "wall_seconds": time.perf_counter() - state_start,
                        "shared_setup_seconds": setup_seconds,
                        "peak_rss_bytes": peak_rss_bytes(),
                    }
                )
                # Save deterministic structured checkpoints after every state so
                # a long Colab run never exists only in process memory.
                pd.DataFrame(all_state_rows).to_csv(
                    OUTPUT_DIRECTORY / "state_metrics.csv", index=False
                )
                pd.DataFrame(all_stage_rows).to_csv(
                    OUTPUT_DIRECTORY / "stage_metrics.csv", index=False
                )
                pd.DataFrame(resource_rows).to_csv(
                    OUTPUT_DIRECTORY / "resource_metrics.csv", index=False
                )
                pd.DataFrame(random_rows).to_csv(
                    OUTPUT_DIRECTORY / "random_baseline_metrics.csv", index=False
                )
                (OUTPUT_DIRECTORY / "full_shadow.json").write_text(
                    json.dumps(shadows, indent=2, sort_keys=True) + "\n"
                )
                (OUTPUT_DIRECTORY / "oracle_geometric_prefix.json").write_text(
                    json.dumps(oracles, indent=2, sort_keys=True) + "\n"
                )
                print(
                    f"[scientific] {identifier} completed in {resource_rows[-1]['wall_seconds']:.1f}s",
                    flush=True,
                )
    table = pd.DataFrame(all_state_rows)
    if table.state_id.nunique() != 45:
        raise RuntimeError("scientific design did not produce exactly 45 paired states")
    summary, method_summary = _aggregate_and_verdict(table, config)
    run_sha, _ = _git_state()
    provenance = {
        "preregistration_sha": preregistration_sha,
        "run_sha": run_sha,
        "config_sha256": _config_sha256(),
        "environment": _environment(),
        "wall_seconds": time.perf_counter() - run_start,
    }
    summary = {**summary, "provenance": provenance}
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT_DIRECTORY / "state_metrics.csv", index=False)
    pd.DataFrame(all_stage_rows).to_csv(OUTPUT_DIRECTORY / "stage_metrics.csv", index=False)
    method_summary.to_csv(OUTPUT_DIRECTORY / "method_summary.csv", index=False)
    pd.DataFrame(resource_rows).to_csv(OUTPUT_DIRECTORY / "resource_metrics.csv", index=False)
    pd.DataFrame(random_rows).to_csv(OUTPUT_DIRECTORY / "random_baseline_metrics.csv", index=False)
    (OUTPUT_DIRECTORY / "full_shadow.json").write_text(
        json.dumps(shadows, indent=2, sort_keys=True) + "\n"
    )
    (OUTPUT_DIRECTORY / "oracle_geometric_prefix.json").write_text(
        json.dumps(oracles, indent=2, sort_keys=True) + "\n"
    )
    (OUTPUT_DIRECTORY / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    (OUTPUT_DIRECTORY / "RESULTS.md").write_text(
        _results_markdown(summary, provenance)
    )
    _make_figures(table, OUTPUT_DIRECTORY)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("regression", "smoke", "full-fidelity-profile", "scientific"),
        required=True,
    )
    parser.add_argument("--preregistration-sha")
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    if arguments.mode == "regression":
        run_locked_regression()
        return
    config = _read_config()
    if arguments.mode == "smoke":
        run_smoke(config)
        return
    if arguments.mode == "full-fidelity-profile":
        run_full_fidelity_development_profile(config)
        return
    if not arguments.preregistration_sha:
        raise RuntimeError("scientific mode requires --preregistration-sha")
    summary = run_scientific(config, arguments.preregistration_sha)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
