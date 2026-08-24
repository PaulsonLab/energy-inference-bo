#!/usr/bin/env python3
"""Development-only diagnosis of the frozen E2 FULL-shadow reliability rule.

This file is deliberately separate from ``run_locality_stress.py``.  It reads
the replacement frozen config but cannot dispatch prospective source seeds or
change the scientific FULL-reference rule.
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
from scipy import linalg


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

import conditioned_bo.nonlinear_pde_locality as locality  # noqa: E402


FROZEN_PREREGISTRATION_SHA = "0dcb76f5f2d053e098b472ac9984182b837295b5"
FROZEN_CONFIG_SHA256 = (
    "2717ece2e5581a7224e1a7a5cb5f69c8291c14ad0ea5183aff54154072b4748b"
)
DEVELOPMENT_SOURCE_SEED = 2026082401
TRAJECTORY_REPLICATE_IDENTIFIER = -2
GRID_SIZES = (24, 40)
CHECKPOINTS = ("early", "middle", "late")
PRIMARY_SAMPLE_COUNTS = (8192, 16384, 32768)
OPTIONAL_SAMPLE_COUNT = 65536
OUTPUT_DIRECTORY = (
    Path(__file__).resolve().parent
    / "outputs"
    / "full_shadow_reliability_diagnostic"
)
CONFIG_PATH = (
    Path(__file__).resolve().parent
    / "outputs"
    / "locality_stress_v1"
    / "frozen_config.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_frozen_config() -> dict[str, Any]:
    config = json.loads(CONFIG_PATH.read_text())
    if _sha256(CONFIG_PATH) != FROZEN_CONFIG_SHA256:
        raise RuntimeError("frozen config changed after replacement preregistration")
    if config["development"]["source_seed"] != DEVELOPMENT_SOURCE_SEED:
        raise RuntimeError("development source seed does not match the frozen config")
    if DEVELOPMENT_SOURCE_SEED in config["prospective_source_seeds"]:
        raise RuntimeError("development source seed overlaps a prospective seed")
    return config


def _assert_frozen_head() -> None:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if head != FROZEN_PREREGISTRATION_SHA:
        raise RuntimeError(
            "diagnostic must run with the unexecuted replacement preregistration at HEAD"
        )


def _trajectory_seed(grid_size: int) -> int:
    label = (
        "E2_LOCALITY_TRAJECTORY_V1:"
        f"{grid_size}:{TRAJECTORY_REPLICATE_IDENTIFIER}"
    )
    return int(hashlib.sha256(label.encode("utf-8")).hexdigest()[:8], 16)


def _build_development_states(
    config: dict[str, Any], grid_size: int
) -> tuple[locality.Problem, tuple[locality.BOState, ...]]:
    problem = locality.build_problem(
        grid_size,
        DEVELOPMENT_SOURCE_SEED,
        source_perturbation_scale=config["source_field"]["perturbation_scale"],
    )
    states = locality.build_common_bo_states(
        problem,
        initialization_size=config["bo_trajectory"]["initialization_size"],
        total_queries=config["bo_trajectory"]["total_queries"],
        checkpoint_queries=config["bo_trajectory"]["checkpoint_queries"],
        observation_noise_variance=config["bo_trajectory"][
            "observation_noise_variance"
        ],
        reference_sample_count=config["numerical"]["reference_sample_count"],
        trajectory_seed=_trajectory_seed(grid_size),
        incumbent=config["model"]["incumbent"],
    )
    return problem, states


def _state_identifier(grid_size: int, state: locality.BOState) -> str:
    return (
        f"n{grid_size}_r{TRAJECTORY_REPLICATE_IDENTIFIER}_"
        f"{state.checkpoint_label}_q{state.checkpoint_queries}"
    )


def _batch_seeds(identifier: str, sample_count: int) -> tuple[int, int]:
    counts = PRIMARY_SAMPLE_COUNTS + (OPTIONAL_SAMPLE_COUNT,)
    if sample_count not in counts:
        raise ValueError(f"unsupported diagnostic sample count: {sample_count}")
    attempt = counts.index(sample_count)
    state_seed = int(hashlib.sha256(identifier.encode()).hexdigest()[:8], 16)
    base = state_seed + 900_000 + 100_003 * attempt
    return base, base + 1


def _diagnostic_laplace_snis(
    config: dict[str, Any],
    state: locality.BOState,
    problem: locality.Problem,
    *,
    sample_count: int,
    proposal_seed: int,
) -> dict[str, Any]:
    """Run the frozen Laplace-SNIS calculation while retaining pooling terms."""

    started = time.perf_counter()
    active = np.ones(problem.grid_size**2, dtype=bool)
    work = locality.FactorWork()
    mode, hessian, diagnostics = locality.laplace_mode(
        state,
        problem,
        active,
        work=work,
        gradient_tolerance=config["inference"]["laplace_gradient_tolerance"],
        maximum_iterations=config["inference"]["laplace_maximum_iterations"],
    )
    inflation = config["inference"]["laplace_proposal_inflation"]
    dense_hessian = hessian.toarray()
    factor = np.linalg.cholesky(dense_hessian / inflation)
    rng = np.random.default_rng(proposal_seed)
    white = rng.standard_normal((sample_count, mode.size))
    samples = mode + linalg.solve_triangular(
        factor.T, white.T, lower=False, check_finite=False
    ).T
    energies = locality.factor_energy_sum(
        samples,
        problem,
        np.arange(problem.grid_size**2, dtype=np.int64),
        work,
    )
    reference_delta = samples - state.reference_mean
    log_target = -0.5 * np.einsum(
        "bi,bi->b",
        reference_delta,
        (state.reference_precision @ reference_delta.T).T,
        optimize=True,
    ) - energies
    proposal_delta = samples - mode
    log_proposal = -0.5 * np.einsum(
        "bi,bi->b",
        proposal_delta,
        (hessian @ proposal_delta.T).T / inflation,
        optimize=True,
    )
    log_weights = log_target - log_proposal
    incumbent = locality.state_incumbent(state, config["model"]["incumbent"])
    utility = np.maximum(samples[:, state.action_indices] - incumbent, 0.0)
    shifted = log_weights - np.max(log_weights)
    weights = np.exp(shifted)
    normalized = weights / weights.sum()
    acquisition = normalized @ utility
    order = np.argsort(-acquisition, kind="stable")
    ess_absolute = float(weights.sum() ** 2 / np.dot(weights, weights))
    diagnostics = {
        **diagnostics,
        "dense_hessian_bytes": int(dense_hessian.nbytes),
        "proposal_inflation": inflation,
        "sample_count": sample_count,
    }
    return {
        "proposal_seed": proposal_seed,
        "action_index": int(state.action_indices[int(order[0])]),
        "action_local_index": int(order[0]),
        "top_five_local_indices": order[:5],
        "acquisition": acquisition,
        "rank1_rank2_gap": float(acquisition[order[0]] - acquisition[order[1]]),
        "ess_fraction": ess_absolute / sample_count,
        "ess_absolute": ess_absolute,
        "laplace_diagnostics": diagnostics,
        "wall_seconds": time.perf_counter() - started,
        "work": work.to_dict(),
        "pool_log_weights": log_weights,
        "pool_utility": utility,
    }


def _top_five(batch: dict[str, Any], state: locality.BOState) -> list[dict[str, Any]]:
    return [
        {
            "rank": rank,
            "action_index": int(state.action_indices[int(local_index)]),
            "acquisition": float(batch["acquisition"][int(local_index)]),
        }
        for rank, local_index in enumerate(batch["top_five_local_indices"], start=1)
    ]


def _pooled_diagnostic(
    first: dict[str, Any],
    second: dict[str, Any],
    state: locality.BOState,
) -> dict[str, Any]:
    log_weights = np.concatenate(
        (first["pool_log_weights"], second["pool_log_weights"])
    )
    utility = np.concatenate((first["pool_utility"], second["pool_utility"]), axis=0)
    shifted = log_weights - np.max(log_weights)
    weights = np.exp(shifted)
    acquisition = weights @ utility / weights.sum()
    order = np.argsort(-acquisition, kind="stable")
    return {
        "diagnostic_only": True,
        "action_index": int(state.action_indices[int(order[0])]),
        "top_two_gap": float(acquisition[order[0]] - acquisition[order[1]]),
        "top_five": [
            {
                "rank": rank,
                "action_index": int(state.action_indices[int(local_index)]),
                "acquisition": float(acquisition[int(local_index)]),
            }
            for rank, local_index in enumerate(order[:5], start=1)
        ],
        "ess_absolute": float(weights.sum() ** 2 / np.dot(weights, weights)),
        "ess_fraction": float(
            (weights.sum() ** 2 / np.dot(weights, weights)) / log_weights.size
        ),
    }


def classify_failure(
    *,
    converged: bool,
    finite: bool,
    ess_pass: bool,
    action_agreement: bool,
    vector_pass: bool,
    maximum_cross_regret: float,
    pooled_top_two_gap: float,
    materiality_scale: float,
) -> str | None:
    """Classify failure without changing any component of the frozen rule."""

    if not converged or not finite:
        return "OTHER_NUMERICAL_FAILURE"
    if ess_pass and action_agreement and vector_pass:
        return None
    if not ess_pass:
        return "LOW_ESS"
    if not action_agreement:
        if (
            maximum_cross_regret <= materiality_scale
            and pooled_top_two_gap <= materiality_scale
        ):
            return "NEAR_TIE_ACTION_INSTABILITY"
        return "ACTION_INSTABILITY_WITH_MATERIAL_GAP"
    if not vector_pass:
        return "GLOBAL_VECTOR_DISAGREEMENT_ONLY"
    return "OTHER_NUMERICAL_FAILURE"


def _case_metrics(
    config: dict[str, Any],
    state: locality.BOState,
    first: dict[str, Any],
    second: dict[str, Any],
    *,
    grid_size: int,
    sample_count: int,
    pair_wall_seconds: float,
    worker_wall_seconds: float,
    peak_rss: int,
) -> dict[str, Any]:
    acquisition_a = first["acquisition"]
    acquisition_b = second["acquisition"]
    local_a = first["action_local_index"]
    local_b = second["action_local_index"]
    union = np.unique(
        np.concatenate(
            (first["top_five_local_indices"], second["top_five_local_indices"])
        )
    )
    maximum_difference = float(np.max(np.abs(acquisition_a - acquisition_b)))
    union_difference = float(
        np.max(np.abs(acquisition_a[union] - acquisition_b[union]))
    )
    regret_a_under_b = float(max(0.0, acquisition_b[local_b] - acquisition_b[local_a]))
    regret_b_under_a = float(max(0.0, acquisition_a[local_a] - acquisition_a[local_b]))
    pooled = _pooled_diagnostic(first, second, state)
    shadow = config["full_shadow"]
    action_agreement = first["action_index"] == second["action_index"]
    ess_pass = min(first["ess_fraction"], second["ess_fraction"]) >= shadow[
        "minimum_ess_fraction"
    ]
    vector_pass = maximum_difference <= shadow[
        "maximum_batch_acquisition_difference"
    ]
    converged = bool(
        first["laplace_diagnostics"].get("converged", False)
        and second["laplace_diagnostics"].get("converged", False)
    )
    finite = bool(
        np.all(np.isfinite(acquisition_a))
        and np.all(np.isfinite(acquisition_b))
        and np.isfinite(first["ess_fraction"])
        and np.isfinite(second["ess_fraction"])
    )
    materiality_scale = config["verdict_thresholds"][
        "matched_quality_regret_margin"
    ]
    classification = classify_failure(
        converged=converged,
        finite=finite,
        ess_pass=ess_pass,
        action_agreement=action_agreement,
        vector_pass=vector_pass,
        maximum_cross_regret=max(regret_a_under_b, regret_b_under_a),
        pooled_top_two_gap=pooled["top_two_gap"],
        materiality_scale=materiality_scale,
    )
    result = {
        "state_id": _state_identifier(grid_size, state),
        "grid_size": grid_size,
        "checkpoint": state.checkpoint_label,
        "checkpoint_queries": state.checkpoint_queries,
        "state_fingerprint": state.fingerprint(),
        "state_specific_incumbent": locality.state_incumbent(
            state, config["model"]["incumbent"]
        ),
        "sample_count_per_batch": sample_count,
        "batch_actions": [first["action_index"], second["action_index"]],
        "action_agreement": action_agreement,
        "top_five_batch_a": _top_five(first, state),
        "top_five_batch_b": _top_five(second, state),
        "rank1_rank2_gap_batch_a": first["rank1_rank2_gap"],
        "rank1_rank2_gap_batch_b": second["rank1_rank2_gap"],
        "ess_fraction_batch_a": first["ess_fraction"],
        "ess_fraction_batch_b": second["ess_fraction"],
        "ess_absolute_batch_a": first["ess_absolute"],
        "ess_absolute_batch_b": second["ess_absolute"],
        "maximum_acquisition_vector_difference": maximum_difference,
        "maximum_top_five_union_difference": union_difference,
        "batch_a_action_regret_under_batch_b": regret_a_under_b,
        "batch_b_action_regret_under_batch_a": regret_b_under_a,
        "maximum_cross_batch_action_regret": max(
            regret_a_under_b, regret_b_under_a
        ),
        "pooled_batch_diagnostic": pooled,
        "laplace_mode_batch_a": first["laplace_diagnostics"],
        "laplace_mode_batch_b": second["laplace_diagnostics"],
        "batch_wall_seconds": [first["wall_seconds"], second["wall_seconds"]],
        "pair_wall_seconds": pair_wall_seconds,
        "worker_wall_seconds_including_state_construction": worker_wall_seconds,
        "peak_rss_bytes": peak_rss,
        "peak_rss_gb": peak_rss / 1_000_000_000.0,
        "frozen_rule_components": {
            "minimum_ess_fraction": shadow["minimum_ess_fraction"],
            "maximum_batch_acquisition_difference": shadow[
                "maximum_batch_acquisition_difference"
            ],
            "ess_pass": ess_pass,
            "action_agreement_pass": action_agreement,
            "vector_difference_pass": vector_pass,
        },
        "frozen_rule_reliable": bool(
            converged and finite and ess_pass and action_agreement and vector_pass
        ),
        "failure_classification": classification,
        "classification_materiality_scale": materiality_scale,
    }
    return result


def _worker_case(
    grid_size: int, checkpoint: str, sample_count: int
) -> dict[str, Any]:
    worker_start = time.perf_counter()
    config = _read_frozen_config()
    problem, states = _build_development_states(config, grid_size)
    matches = [state for state in states if state.checkpoint_label == checkpoint]
    if len(matches) != 1:
        raise RuntimeError("failed to resolve the requested frozen checkpoint")
    state = matches[0]
    identifier = _state_identifier(grid_size, state)
    seeds = _batch_seeds(identifier, sample_count)
    pair_start = time.perf_counter()
    first = _diagnostic_laplace_snis(
        config,
        state,
        problem,
        sample_count=sample_count,
        proposal_seed=seeds[0],
    )
    second = _diagnostic_laplace_snis(
        config,
        state,
        problem,
        sample_count=sample_count,
        proposal_seed=seeds[1],
    )
    pair_wall = time.perf_counter() - pair_start
    result = _case_metrics(
        config,
        state,
        first,
        second,
        grid_size=grid_size,
        sample_count=sample_count,
        pair_wall_seconds=pair_wall,
        worker_wall_seconds=time.perf_counter() - worker_start,
        peak_rss=locality.peak_rss_bytes(),
    )
    result["batch_seeds"] = list(seeds)
    return result


def _summary_row(case: dict[str, Any]) -> dict[str, Any]:
    pooled = case["pooled_batch_diagnostic"]
    components = case["frozen_rule_components"]
    return {
        "state_id": case["state_id"],
        "grid_size": case["grid_size"],
        "checkpoint": case["checkpoint"],
        "sample_count_per_batch": case["sample_count_per_batch"],
        "action_a": case["batch_actions"][0],
        "action_b": case["batch_actions"][1],
        "action_agreement": case["action_agreement"],
        "rank1_rank2_gap_a": case["rank1_rank2_gap_batch_a"],
        "rank1_rank2_gap_b": case["rank1_rank2_gap_batch_b"],
        "ess_fraction_a": case["ess_fraction_batch_a"],
        "ess_fraction_b": case["ess_fraction_batch_b"],
        "ess_absolute_a": case["ess_absolute_batch_a"],
        "ess_absolute_b": case["ess_absolute_batch_b"],
        "maximum_vector_difference": case[
            "maximum_acquisition_vector_difference"
        ],
        "maximum_top_five_union_difference": case[
            "maximum_top_five_union_difference"
        ],
        "cross_regret_a_under_b": case[
            "batch_a_action_regret_under_batch_b"
        ],
        "cross_regret_b_under_a": case[
            "batch_b_action_regret_under_batch_a"
        ],
        "pooled_action": pooled["action_index"],
        "pooled_top_two_gap": pooled["top_two_gap"],
        "ess_pass": components["ess_pass"],
        "action_pass": components["action_agreement_pass"],
        "vector_pass": components["vector_difference_pass"],
        "frozen_rule_reliable": case["frozen_rule_reliable"],
        "failure_classification": case["failure_classification"] or "PASS",
        "pair_wall_seconds": case["pair_wall_seconds"],
        "peak_rss_gb": case["peak_rss_gb"],
    }


def _environment() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "numpy": np.__version__,
    }


def run_diagnostic(sample_counts: tuple[int, ...]) -> dict[str, Any]:
    _assert_frozen_head()
    config = _read_frozen_config()
    del config
    cases: list[dict[str, Any]] = []
    started = time.perf_counter()
    for grid_size in GRID_SIZES:
        for checkpoint in CHECKPOINTS:
            for sample_count in sample_counts:
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker",
                    "--grid-size",
                    str(grid_size),
                    "--checkpoint",
                    checkpoint,
                    "--sample-count",
                    str(sample_count),
                ]
                completed = subprocess.run(
                    command,
                    cwd=REPOSITORY_ROOT,
                    check=True,
                    capture_output=True,
                    text=True,
                )
                case = json.loads(completed.stdout)
                cases.append(case)
                print(
                    "[FULL-shadow diagnostic] "
                    f"n={grid_size} {checkpoint} S={sample_count}: "
                    f"actions={case['batch_actions']} "
                    f"ESS=({case['ess_fraction_batch_a']:.3f},"
                    f"{case['ess_fraction_batch_b']:.3f}) "
                    f"class={case['failure_classification'] or 'PASS'}",
                    flush=True,
                )
    result = {
        "diagnostic_id": "E2_FULL_SHADOW_RELIABILITY_DEVELOPMENT_ONLY_V1",
        "operational_development_only": True,
        "claim_mapping": "E2 FULL-reference reliability and action-quality diagnostics",
        "frozen_preregistration_sha": FROZEN_PREREGISTRATION_SHA,
        "frozen_preregistration_executed": False,
        "frozen_config_sha256": FROZEN_CONFIG_SHA256,
        "development_source_seed": DEVELOPMENT_SOURCE_SEED,
        "prospective_source_seeds_evaluated": False,
        "trajectory_replicate_identifier": TRAJECTORY_REPLICATE_IDENTIFIER,
        "grid_sizes": list(GRID_SIZES),
        "checkpoints": list(CHECKPOINTS),
        "sample_counts_per_batch": list(sample_counts),
        "optional_65536_used": OPTIONAL_SAMPLE_COUNT in sample_counts,
        "frozen_rule_unchanged": {
            "minimum_ess_fraction": 0.2,
            "require_action_agreement": True,
            "maximum_batch_acquisition_difference": 0.01,
        },
        "classification_precedence": [
            "OTHER_NUMERICAL_FAILURE for nonconvergence/nonfinite output",
            "LOW_ESS when either frozen ESS-fraction check fails",
            "NEAR_TIE_ACTION_INSTABILITY when actions differ but reciprocal regret and pooled gap are each at most the pre-existing 0.01 decision-quality scale",
            "ACTION_INSTABILITY_WITH_MATERIAL_GAP for remaining action disagreement",
            "GLOBAL_VECTOR_DISAGREEMENT_ONLY when only the uniform acquisition-vector check fails",
        ],
        "cases": cases,
        "wall_seconds": time.perf_counter() - started,
        "environment": _environment(),
    }
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIRECTORY / "diagnostic.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    import pandas as pd

    pd.DataFrame([_summary_row(case) for case in cases]).to_csv(
        OUTPUT_DIRECTORY / "summary.csv", index=False
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--grid-size", type=int, choices=GRID_SIZES)
    parser.add_argument("--checkpoint", choices=CHECKPOINTS)
    parser.add_argument(
        "--sample-count", type=int, choices=PRIMARY_SAMPLE_COUNTS + (OPTIONAL_SAMPLE_COUNT,)
    )
    parser.add_argument(
        "--sample-counts",
        type=int,
        nargs="+",
        default=list(PRIMARY_SAMPLE_COUNTS),
        choices=PRIMARY_SAMPLE_COUNTS + (OPTIONAL_SAMPLE_COUNT,),
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    if arguments.worker:
        if (
            arguments.grid_size is None
            or arguments.checkpoint is None
            or arguments.sample_count is None
        ):
            raise RuntimeError("worker mode requires one complete diagnostic case")
        result = _worker_case(
            arguments.grid_size,
            arguments.checkpoint,
            arguments.sample_count,
        )
        print(json.dumps(result, sort_keys=True))
        return
    sample_counts = tuple(dict.fromkeys(arguments.sample_counts))
    if any(value not in PRIMARY_SAMPLE_COUNTS + (OPTIONAL_SAMPLE_COUNT,) for value in sample_counts):
        raise RuntimeError("unsupported sample count")
    result = run_diagnostic(sample_counts)
    print(
        json.dumps(
            {
                "cases": len(result["cases"]),
                "wall_seconds": result["wall_seconds"],
                "optional_65536_used": result["optional_65536_used"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
