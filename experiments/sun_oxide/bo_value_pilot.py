#!/usr/bin/env python3
"""Run the preregistered Sun-oxide NO_PBE versus FULL_PBE value pilot.

The ``smoke`` command is oracle-isolated.  The ``run`` command repeats the
smoke checks, opens the retrospective oracle only after they pass, and keeps
all unqueried target values outside both acquisition implementations.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from typing import Any, Sequence
import zipfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from conditioned_bo.bo_value import (
    NumericalFailure,
    RetrospectiveOracle,
    TimedFactorBank,
    aurc,
    construct_gaussian_reference,
    draw_laplace_samples,
    exact_support_marginal,
    fit_laplace_approximation,
    fixed_initial_positions,
    freeze_target_scale,
    gaussian_expected_improvement,
    laplace_log_importance_weights,
    schur_complement_precision,
    select_unobserved_action,
    selected_gaussian_marginals,
    simple_regret_trajectory,
    snis_expected_improvement,
    snis_pairwise_gap_standard_error,
    stable_self_normalized_weights,
)


DEFAULT_CONFIG = ROOT / "experiments/sun_oxide/configs/bo_value_pilot.json"
EXPECTED_INPUTS = {
    "action_node_mapping",
    "benchmark_manifest",
    "gw_oracle",
    "nlr_data_use_notice",
    "normalized_model_manifest",
    "normalized_pbe_factor_bank",
    "pbe_support",
    "q0",
}
ORACLE_INPUT = "gw_oracle"
VALIDATION_NAMES = (
    "seed_0_initial",
    "seed_0_after_6_queries",
    "seed_0_after_12_queries",
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_canonical_json(value), encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _current_sha(repository_root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("Unsupported BO value config schema")
    if config.get("benchmark_name") != "CURRENT_NLR_PBE_GW_V1":
        raise ValueError("Frozen benchmark name changed")
    if config.get("action_count") != 191:
        raise ValueError("Frozen action count changed")
    if config["bo"]["methods"] != ["NO_PBE", "FULL_PBE"]:
        raise ValueError("Only NO_PBE and FULL_PBE are permitted")
    if config["bo"]["seeds"] != list(range(12)):
        raise ValueError("Frozen seed list changed")
    if config["bo"]["initial_action_count"] != 8:
        raise ValueError("Frozen initialization count changed")
    if config["bo"]["sequential_query_count"] != 12:
        raise ValueError("Frozen sequential-query count changed")
    if config["gaussian_reference"]["observation_noise_standardized"] != 0.05:
        raise ValueError("Frozen standardized observation noise changed")
    if config["gaussian_reference"]["hyperparameter_optimization_during_bo"]:
        raise ValueError("Online hyperparameter fitting is forbidden")
    if config["factor_bank"]["weight_exact"] != "1/499":
        raise ValueError("Frozen PBE factor weight changed")
    if config["importance_validation"]["sample_count"] != 4096:
        raise ValueError("Frozen IS sample count changed")
    if config["tuning_after_gw_results"]:
        raise ValueError("Post-target tuning is forbidden")
    if set(config.get("inputs", {})) != EXPECTED_INPUTS:
        raise ValueError(f"Input interface must be exactly {sorted(EXPECTED_INPUTS)}")
    return config


def _verified_inputs(
    repository_root: Path, config: dict[str, Any], *, include_oracle: bool
) -> dict[str, Path]:
    root = repository_root.resolve()
    result: dict[str, Path] = {}
    for name in sorted(EXPECTED_INPUTS):
        if name == ORACLE_INPUT and not include_oracle:
            continue
        specification = config["inputs"][name]
        if set(specification) != {"path", "sha256"}:
            raise ValueError(f"Invalid frozen input declaration for {name}")
        path = (root / specification["path"]).resolve()
        if path != root and root not in path.parents:
            raise ValueError(f"Input escapes repository root: {path}")
        observed = _sha256(path)
        if observed != specification["sha256"]:
            raise ValueError(f"Frozen input hash mismatch for {name}: {observed}")
        result[name] = path
    return result


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader)


def _load_static_inputs(paths: dict[str, Path]) -> dict[str, Any]:
    actions = _read_csv_rows(paths["action_node_mapping"])
    support_rows = _read_csv_rows(paths["pbe_support"])
    factor_rows = _read_csv_rows(paths["normalized_pbe_factor_bank"])
    q0 = sparse.load_npz(paths["q0"]).tocsc().astype(np.float64)
    if q0.shape != (2142, 2142) or len(actions) != 191 or len(support_rows) != 500:
        raise ValueError("Frozen graph/support/action dimensions changed")
    if len(factor_rows) != 124718:
        raise ValueError("Frozen normalized factor count changed")
    support_nodes = np.asarray(
        [int(row["node_index"]) for row in support_rows], dtype=np.int64
    )
    if not np.all(np.diff(support_nodes) > 0):
        raise ValueError("Support nodes are not in frozen increasing-node order")
    support_by_node = {int(node): position for position, node in enumerate(support_nodes)}
    action_nodes = np.asarray([int(row["node_index"]) for row in actions], dtype=np.int64)
    if len(set(action_nodes.tolist())) != 191:
        raise ValueError("Action nodes are not unique")
    try:
        action_support_positions = np.asarray(
            [support_by_node[int(node)] for node in action_nodes], dtype=np.int64
        )
    except KeyError as exc:
        raise ValueError("An action lies outside PBE_SUPPORT_500_V1") from exc
    endpoint_pairs = np.asarray(
        [[int(row["support_i"]), int(row["support_j"])] for row in factor_rows],
        dtype=np.int64,
    )
    signs = np.asarray([int(row["sign_s_ij"]) for row in factor_rows], dtype=np.int8)
    weights = np.asarray([float(row["weight"]) for row in factor_rows])
    if not np.all(weights == 1.0 / 499.0) or set(signs.tolist()) != {-1, 1}:
        raise ValueError("Frozen factor weight or signs changed")
    if endpoint_pairs.min() < 0 or endpoint_pairs.max() >= 500:
        raise ValueError("A PBE factor endpoint lies outside support")
    pbe_by_support = np.asarray(
        [float(row["pbe_band_gap_ev"]) for row in support_rows], dtype=np.float64
    )
    return {
        "actions": actions,
        "action_keys": [row["action_key"] for row in actions],
        "action_nodes": action_nodes,
        "action_support_positions": action_support_positions,
        "support_nodes": support_nodes,
        "endpoint_pairs": endpoint_pairs,
        "signs": signs,
        "pbe_action_values": pbe_by_support[action_support_positions],
        "q0": q0,
    }


def scientific_smoke(
    repository_root: Path, config_path: Path, run_sha: str
) -> dict[str, Any]:
    """Oracle-isolated environment, artifact, and mathematical smoke test."""

    if _current_sha(repository_root) != run_sha:
        raise ValueError("RUN_SHA does not equal the detached repository HEAD")
    config = _load_config(config_path)
    paths = _verified_inputs(repository_root, config, include_oracle=False)
    benchmark = json.loads(paths["benchmark_manifest"].read_text(encoding="utf-8"))
    model_manifest = json.loads(
        paths["normalized_model_manifest"].read_text(encoding="utf-8")
    )
    if benchmark["counts"]["legacy_compositions"] != 2142:
        raise ValueError("Benchmark latent-node count changed")
    if benchmark["counts"]["strict_gw_actions"] != 191:
        raise ValueError("Benchmark action count changed")
    if model_manifest["verdict"] != "PASS_NORMALIZED_PBE_MODEL":
        raise ValueError("Normalized PBE model is not a committed PASS")
    static = _load_static_inputs(paths)

    rhs = np.zeros(2142, dtype=np.float64)
    rhs[static["action_nodes"][0]] = 1.0
    factorization = splu(static["q0"])
    solution = factorization.solve(rhs)
    sparse_residual = float(
        np.linalg.norm(static["q0"] @ solution - rhs) / np.linalg.norm(rhs)
    )
    if sparse_residual > 1e-9:
        raise NumericalFailure("Actual Q0 sparse smoke solve failed")

    toy_q = sparse.csr_matrix(
        [[2.0, -0.4, -0.2], [-0.4, 1.8, -0.3], [-0.2, -0.3, 1.7]]
    )
    toy_state = construct_gaussian_reference(toy_q, [0], [0.25], sigma_obs=0.5)
    toy_support = np.asarray([0, 2], dtype=np.int64)
    toy_marginal = exact_support_marginal(toy_state, toy_support)
    toy_schur = schur_complement_precision(toy_state.precision.toarray(), toy_support)
    if not np.allclose(toy_marginal.precision, toy_schur, rtol=1e-12, atol=1e-12):
        raise NumericalFailure("Toy exact marginal does not match Schur complement")
    if np.allclose(
        toy_marginal.precision,
        toy_state.precision.toarray()[np.ix_(toy_support, toy_support)],
        rtol=1e-12,
        atol=1e-12,
    ):
        raise NumericalFailure("Coupled toy failed to reject principal precision shortcut")
    toy_bank = TimedFactorBank([[0, 1]], [1], dimension=2, chunk_size=1)
    toy_laplace = fit_laplace_approximation(
        toy_marginal.mean,
        toy_marginal.precision,
        toy_bank,
        np.zeros(2),
    )
    toy_samples = draw_laplace_samples(toy_laplace, 8, np.random.default_rng(7103))
    if toy_samples.shape != (8, 2) or not np.all(np.isfinite(toy_samples)):
        raise NumericalFailure("Toy Laplace sample smoke failed")

    versions = {
        name: importlib.metadata.version(name)
        for name in ("numpy", "pandas", "scipy", "matplotlib")
    }
    report = {
        "status": "SMOKE_PASS",
        "run_sha": run_sha,
        "config_sha256": _sha256(config_path),
        "python": platform.python_version(),
        "packages": versions,
        "oracle_file_opened": False,
        "safe_input_hashes_verified": sorted(paths),
        "q0_shape": list(static["q0"].shape),
        "factor_count": int(static["endpoint_pairs"].shape[0]),
        "support_count": int(static["support_nodes"].size),
        "action_count": int(static["action_nodes"].size),
        "actual_q0_sparse_solve_relative_residual": sparse_residual,
        "toy_exact_marginal_schur_match": True,
        "toy_principal_submatrix_shortcut_rejected": True,
        "toy_map_gradient_infinity_norm": toy_laplace.diagnostics[
            "gradient_infinity_norm"
        ],
        "toy_laplace_cholesky_success": True,
        "toy_laplace_sample_shape": list(toy_samples.shape),
    }
    return report


def _load_oracle_after_smoke(
    paths: dict[str, Path], action_keys: Sequence[str]
) -> RetrospectiveOracle:
    rows = _read_csv_rows(paths[ORACLE_INPUT])
    if [row["action_key"] for row in rows] != list(action_keys):
        raise ValueError("Oracle action order does not match the frozen action map")
    return RetrospectiveOracle(
        action_keys,
        [float(row["gw_band_gap_ev"]) for row in rows],
    )


def _no_pbe_decision(
    static: dict[str, Any],
    observed_positions: Sequence[int],
    observed_z: Sequence[float],
    config: dict[str, Any],
) -> tuple[int, dict[str, Any]]:
    started = time.perf_counter()
    residual_tolerance = config["gaussian_reference"][
        "solve_relative_residual_maximum"
    ]
    state = construct_gaussian_reference(
        static["q0"],
        static["action_nodes"][np.asarray(observed_positions, dtype=np.int64)],
        observed_z,
        sigma_obs=config["gaussian_reference"]["observation_noise_standardized"],
        residual_tolerance=residual_tolerance,
    )
    means, variances, variance_diagnostics = selected_gaussian_marginals(
        state,
        static["action_nodes"],
        residual_tolerance=residual_tolerance,
    )
    acquisition_started = time.perf_counter()
    ei = gaussian_expected_improvement(means, variances, max(observed_z))
    selected = select_unobserved_action(
        ei, observed_positions, static["action_keys"]
    )
    acquisition_seconds = time.perf_counter() - acquisition_started
    diagnostics = {
        "method": "NO_PBE",
        **state.diagnostics,
        **variance_diagnostics,
        "ei_seconds": acquisition_seconds,
        "selected_action_position": selected,
        "selected_action_key": static["action_keys"][selected],
        "selected_ei_standardized": float(ei[selected]),
        "total_decision_seconds": time.perf_counter() - started,
        "hyperparameters_fit_or_updated": False,
    }
    return selected, diagnostics


def _full_pbe_context(
    static: dict[str, Any],
    observed_positions: Sequence[int],
    observed_z: Sequence[float],
    previous_map: np.ndarray | None,
    config: dict[str, Any],
    *,
    decision_kind: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    residual_tolerance = config["gaussian_reference"][
        "solve_relative_residual_maximum"
    ]
    state = construct_gaussian_reference(
        static["q0"],
        static["action_nodes"][np.asarray(observed_positions, dtype=np.int64)],
        observed_z,
        sigma_obs=config["gaussian_reference"]["observation_noise_standardized"],
        residual_tolerance=residual_tolerance,
    )
    marginal = exact_support_marginal(
        state,
        static["support_nodes"],
        residual_tolerance=residual_tolerance,
    )
    factor_bank = TimedFactorBank(
        static["endpoint_pairs"],
        static["signs"],
        dimension=500,
        weight=config["factor_bank"]["weight"],
        chunk_size=config["factor_bank"]["chunk_size"],
    )
    if previous_map is None:
        initial_map = np.zeros(500, dtype=np.float64)
        retry_map = None
        warm_status = "zero_fallback_first_state"
    else:
        initial_map = np.asarray(previous_map, dtype=np.float64)
        retry_map = np.zeros(500, dtype=np.float64)
        warm_status = "preceding_FULL_PBE_MAP"
    laplace_config = config["full_pbe_laplace"]
    laplace = fit_laplace_approximation(
        marginal.mean,
        marginal.precision,
        factor_bank,
        initial_map,
        retry_map=retry_map,
        gradient_tolerance=laplace_config[
            "map_gradient_infinity_norm_maximum"
        ],
        optimizer_gradient_tolerance=laplace_config[
            "optimizer_gradient_tolerance"
        ],
        function_tolerance=laplace_config["function_tolerance"],
        maximum_iterations=laplace_config["maximum_iterations"],
        residual_tolerance=laplace_config["solve_relative_residual_maximum"],
    )
    acquisition_started = time.perf_counter()
    action_support = static["action_support_positions"]
    means = laplace.map[action_support]
    variances = np.maximum(np.diag(laplace.covariance)[action_support], 0.0)
    ei = gaussian_expected_improvement(means, variances, max(observed_z))
    selected = select_unobserved_action(
        ei, observed_positions, static["action_keys"]
    )
    acquisition_seconds = time.perf_counter() - acquisition_started
    diagnostics = {
        "method": "FULL_PBE",
        "decision_kind": decision_kind,
        **state.diagnostics,
        **marginal.diagnostics,
        **laplace.diagnostics,
        "ei_seconds": acquisition_seconds,
        "selected_action_position": selected,
        "selected_action_key": static["action_keys"][selected],
        "selected_ei_standardized": float(ei[selected]),
        "warm_start_status": warm_status,
        "total_decision_seconds": time.perf_counter() - started,
        "importance_validation_seconds_in_total": 0.0,
        "hyperparameters_fit_or_updated": False,
    }
    return {
        "gaussian": state,
        "marginal": marginal,
        "laplace": laplace,
        "ei": ei,
        "selected": selected,
        "diagnostics": diagnostics,
    }


def _importance_validation(
    name: str,
    context: dict[str, Any],
    static: dict[str, Any],
    observed_positions: Sequence[int],
    observed_z: Sequence[float],
    config: dict[str, Any],
) -> dict[str, Any]:
    validation = config["importance_validation"]
    validation_only_inference_seconds = (
        float(context["diagnostics"]["total_decision_seconds"])
        if context["diagnostics"].get("decision_kind", "").startswith(
            "validation_only"
        )
        else 0.0
    )
    started = time.perf_counter()
    rng_seed = int(validation["rng_seeds"][name])
    samples = draw_laplace_samples(
        context["laplace"], validation["sample_count"], np.random.default_rng(rng_seed)
    )
    log_weights, weight_diagnostics = laplace_log_importance_weights(
        samples,
        context["marginal"].mean,
        context["marginal"].precision,
        context["laplace"],
        static["endpoint_pairs"],
        static["signs"],
        weight=config["factor_bank"]["weight"],
        sample_chunk_size=validation["sample_chunk_size"],
        factor_chunk_size=validation["factor_chunk_size"],
    )
    weights, ess = stable_self_normalized_weights(log_weights)
    action_samples = samples[:, static["action_support_positions"]]
    is_ei = snis_expected_improvement(action_samples, weights, max(observed_z))
    is_selected = select_unobserved_action(
        is_ei, observed_positions, static["action_keys"]
    )
    laplace_selected = int(context["selected"])
    improvement = np.maximum(action_samples - max(observed_z), 0.0)
    gap_samples = improvement[:, is_selected] - improvement[:, laplace_selected]
    gap_estimate, gap_se = snis_pairwise_gap_standard_error(gap_samples, weights)
    regret = float(is_ei[is_selected] - is_ei[laplace_selected])
    threshold = max(0.02, 2.0 * gap_se)
    ess_fraction = ess / validation["sample_count"]
    passed = bool(
        ess_fraction >= validation["ess_fraction_minimum"]
        and regret <= threshold
    )
    observed_set = set(int(position) for position in observed_positions)
    unobserved_ei = [
        {
            "action_position": position,
            "action_key": static["action_keys"][position],
            "snis_ei_standardized": float(is_ei[position]),
        }
        for position in range(len(static["action_keys"]))
        if position not in observed_set
    ]
    importance_seconds = time.perf_counter() - started
    return {
        "schema_version": 1,
        "state": name,
        "sample_count": validation["sample_count"],
        "rng_seed": rng_seed,
        "ess": ess,
        "ess_fraction": ess_fraction,
        "ess_fraction_minimum": validation["ess_fraction_minimum"],
        "laplace_selected_action_position": laplace_selected,
        "laplace_selected_action_key": static["action_keys"][laplace_selected],
        "is_selected_action_position": is_selected,
        "is_selected_action_key": static["action_keys"][is_selected],
        "laplace_action_snis_ei_standardized": float(is_ei[laplace_selected]),
        "is_selected_snis_ei_standardized": float(is_ei[is_selected]),
        "is_estimated_regret_standardized": regret,
        "pairwise_gap_snis_estimate_standardized": gap_estimate,
        "pairwise_gap_mc_se_standardized": gap_se,
        "decision_regret_threshold_standardized": threshold,
        "passed": passed,
        "validation_alters_bo_decision": False,
        "laplace_is_exact_conditioned_posterior": False,
        "weight_diagnostics": weight_diagnostics,
        "importance_sampling_seconds": importance_seconds,
        "validation_only_state_inference_seconds": validation_only_inference_seconds,
        "total_validation_seconds": (
            importance_seconds + validation_only_inference_seconds
        ),
        "importance_validation_in_routine_decision_time": False,
        "unobserved_action_snis_ei": unobserved_ei,
    }


def _append_initial_rows(
    rows: list[dict[str, Any]],
    seed: int,
    method: str,
    positions: np.ndarray,
    values_ev: np.ndarray,
    standardized: np.ndarray,
    static: dict[str, Any],
) -> None:
    for order, (position, value, z_value) in enumerate(
        zip(positions.tolist(), values_ev.tolist(), standardized.tolist(), strict=True),
        start=1,
    ):
        rows.append(
            {
                "seed": seed,
                "method": method,
                "phase": "initial",
                "initial_order": order,
                "query_index": 0,
                "action_position": position,
                "action_key": static["action_keys"][position],
                "node_index": int(static["action_nodes"][position]),
                "gw_band_gap_ev": value,
                "standardized_observation": z_value,
                "best_observed_gw_ev": "",
                "simple_regret_ev": "",
            }
        )


def _run_no_pbe(
    seed: int,
    initial_positions: np.ndarray,
    initial_values: np.ndarray,
    scale: Any,
    oracle: RetrospectiveOracle,
    static: dict[str, Any],
    config: dict[str, Any],
    trajectory_rows: list[dict[str, Any]],
    inference_rows: list[dict[str, Any]],
    timing_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    observed_positions = initial_positions.tolist()
    observed_values = initial_values.tolist()
    observed_z = scale.standardize(initial_values).tolist()
    sequential: list[dict[str, Any]] = []
    for query_index in range(1, 13):
        print(f"PROGRESS seed={seed} method=NO_PBE iteration={query_index}/12", flush=True)
        selected, diagnostics = _no_pbe_decision(
            static, observed_positions, observed_z, config
        )
        diagnostics.update({"seed": seed, "query_index": query_index})
        inference_rows.append(diagnostics)
        timing_rows.append(dict(diagnostics))
        value = oracle.query(
            selected,
            seed=seed,
            method="NO_PBE",
            stage=f"sequential_{query_index}",
        )
        z_value = float(scale.standardize([value])[0])
        observed_positions.append(selected)
        observed_values.append(value)
        observed_z.append(z_value)
        row = {
            "seed": seed,
            "method": "NO_PBE",
            "phase": "sequential",
            "initial_order": "",
            "query_index": query_index,
            "action_position": selected,
            "action_key": static["action_keys"][selected],
            "node_index": int(static["action_nodes"][selected]),
            "gw_band_gap_ev": value,
            "standardized_observation": z_value,
            "best_observed_gw_ev": "",
            "simple_regret_ev": "",
        }
        trajectory_rows.append(row)
        sequential.append(row)
    return sequential


def _run_full_pbe(
    seed: int,
    initial_positions: np.ndarray,
    initial_values: np.ndarray,
    scale: Any,
    oracle: RetrospectiveOracle,
    static: dict[str, Any],
    config: dict[str, Any],
    trajectory_rows: list[dict[str, Any]],
    inference_rows: list[dict[str, Any]],
    timing_rows: list[dict[str, Any]],
    validations: dict[str, dict[str, Any]],
    *,
    initial_context: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    observed_positions = initial_positions.tolist()
    observed_values = initial_values.tolist()
    observed_z = scale.standardize(initial_values).tolist()
    previous_map: np.ndarray | None = None
    context = initial_context
    sequential: list[dict[str, Any]] = []
    for query_index in range(1, 13):
        print(f"PROGRESS seed={seed} method=FULL_PBE iteration={query_index}/12", flush=True)
        if context is None:
            context = _full_pbe_context(
                static,
                observed_positions,
                observed_z,
                previous_map,
                config,
                decision_kind="routine_bo_decision",
            )
        diagnostics = dict(context["diagnostics"])
        diagnostics.update({"seed": seed, "query_index": query_index})
        inference_rows.append(diagnostics)
        timing_rows.append(dict(diagnostics))
        if seed == 0 and query_index == 7:
            validations["seed_0_after_6_queries"] = _importance_validation(
                "seed_0_after_6_queries",
                context,
                static,
                observed_positions,
                observed_z,
                config,
            )
            print(
                "IS_VALIDATION seed_0_after_6_queries "
                f"passed={validations['seed_0_after_6_queries']['passed']}",
                flush=True,
            )
        selected = int(context["selected"])
        previous_map = np.asarray(context["laplace"].map, dtype=np.float64)
        value = oracle.query(
            selected,
            seed=seed,
            method="FULL_PBE",
            stage=f"sequential_{query_index}",
        )
        z_value = float(scale.standardize([value])[0])
        observed_positions.append(selected)
        observed_values.append(value)
        observed_z.append(z_value)
        row = {
            "seed": seed,
            "method": "FULL_PBE",
            "phase": "sequential",
            "initial_order": "",
            "query_index": query_index,
            "action_position": selected,
            "action_key": static["action_keys"][selected],
            "node_index": int(static["action_nodes"][selected]),
            "gw_band_gap_ev": value,
            "standardized_observation": z_value,
            "best_observed_gw_ev": "",
            "simple_regret_ev": "",
        }
        trajectory_rows.append(row)
        sequential.append(row)
        context = None

    if seed == 0:
        validation_context = _full_pbe_context(
            static,
            observed_positions,
            observed_z,
            previous_map,
            config,
            decision_kind="validation_only_after_12_queries",
        )
        validation_diagnostics = dict(validation_context["diagnostics"])
        validation_diagnostics.update({"seed": seed, "query_index": 13})
        inference_rows.append(validation_diagnostics)
        timing_rows.append(dict(validation_diagnostics))
        validations["seed_0_after_12_queries"] = _importance_validation(
            "seed_0_after_12_queries",
            validation_context,
            static,
            observed_positions,
            observed_z,
            config,
        )
        print(
            "IS_VALIDATION seed_0_after_12_queries "
            f"passed={validations['seed_0_after_12_queries']['passed']}",
            flush=True,
        )
    return sequential


def _checkpoint_seed(
    output_dir: Path,
    seed: int,
    trajectory_rows: Sequence[dict[str, Any]],
    inference_rows: Sequence[dict[str, Any]],
) -> None:
    _write_json(
        output_dir / "checkpoints" / f"seed_{seed:02d}.json",
        {
            "seed": seed,
            "completed": True,
            "trajectories": [row for row in trajectory_rows if row["seed"] == seed],
            "inference_diagnostics": [
                row for row in inference_rows if row["seed"] == seed
            ],
        },
    )
    print(f"CHECKPOINT seed={seed} complete", flush=True)


def _post_run_metrics(
    trajectory_rows: list[dict[str, Any]],
    seed_scales: dict[int, dict[str, Any]],
    oracle_values: np.ndarray,
    static: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    oracle_maximum = float(np.max(oracle_values))
    optimum_positions = set(np.flatnonzero(oracle_values == oracle_maximum).tolist())
    top_ten_positions = set(
        sorted(range(oracle_values.size), key=lambda position: (-oracle_values[position], static["action_keys"][position]))[:10]
    )
    seed_summaries: list[dict[str, Any]] = []
    sequential_lookup: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for seed in range(12):
        for method in ("NO_PBE", "FULL_PBE"):
            initial_rows = [
                row
                for row in trajectory_rows
                if row["seed"] == seed and row["method"] == method and row["phase"] == "initial"
            ]
            sequential_rows = sorted(
                [
                    row
                    for row in trajectory_rows
                    if row["seed"] == seed
                    and row["method"] == method
                    and row["phase"] == "sequential"
                ],
                key=lambda row: int(row["query_index"]),
            )
            initial_values = np.asarray([row["gw_band_gap_ev"] for row in initial_rows])
            sequential_values = np.asarray(
                [row["gw_band_gap_ev"] for row in sequential_rows]
            )
            regrets = simple_regret_trajectory(
                sequential_values, initial_values, oracle_maximum
            )
            running_best = np.maximum.accumulate(
                np.concatenate(([float(np.max(initial_values))], sequential_values))
            )[1:]
            for row, best, regret in zip(
                sequential_rows, running_best.tolist(), regrets.tolist(), strict=True
            ):
                row["best_observed_gw_ev"] = best
                row["simple_regret_ev"] = regret
            observed_positions = {
                int(row["action_position"]) for row in initial_rows + sequential_rows
            }
            initial_positions = {int(row["action_position"]) for row in initial_rows}
            if initial_positions & top_ten_positions:
                time_top_ten: int | None = 0
            else:
                matches = [
                    int(row["query_index"])
                    for row in sequential_rows
                    if int(row["action_position"]) in top_ten_positions
                ]
                time_top_ten = min(matches) if matches else None
            seed_summaries.append(
                {
                    "seed": seed,
                    "method": method,
                    "mu_seed_ev": seed_scales[seed]["mean_ev"],
                    "scale_seed_ev": seed_scales[seed]["scale_ev"],
                    "initial_action_positions": json.dumps(
                        [int(row["action_position"]) for row in initial_rows]
                    ),
                    "initial_action_keys": json.dumps(
                        [row["action_key"] for row in initial_rows]
                    ),
                    "aurc_ev": aurc(regrets),
                    "final_simple_regret_ev": float(regrets[-1]),
                    "time_to_first_oracle_top_10": time_top_ten,
                    "global_optimum_discovered": bool(
                        observed_positions & optimum_positions
                    ),
                }
            )
            sequential_lookup[(seed, method)] = sequential_rows

    by_method = {
        method: [row for row in seed_summaries if row["method"] == method]
        for method in ("NO_PBE", "FULL_PBE")
    }
    no_aurc = np.asarray([row["aurc_ev"] for row in by_method["NO_PBE"]])
    full_aurc = np.asarray([row["aurc_ev"] for row in by_method["FULL_PBE"]])
    no_final = np.asarray(
        [row["final_simple_regret_ev"] for row in by_method["NO_PBE"]]
    )
    full_final = np.asarray(
        [row["final_simple_regret_ev"] for row in by_method["FULL_PBE"]]
    )
    median_trajectories = {
        method: np.median(
            np.asarray(
                [
                    [float(row["simple_regret_ev"]) for row in sequential_lookup[(seed, method)]]
                    for seed in range(12)
                ]
            ),
            axis=0,
        ).tolist()
        for method in ("NO_PBE", "FULL_PBE")
    }
    paired = (full_aurc - no_aurc).tolist()
    metrics = {
        "oracle_maximum_ev": oracle_maximum,
        "median_aurc_ev": {
            "NO_PBE": float(np.median(no_aurc)),
            "FULL_PBE": float(np.median(full_aurc)),
        },
        "median_final_simple_regret_ev": {
            "NO_PBE": float(np.median(no_final)),
            "FULL_PBE": float(np.median(full_final)),
        },
        "paired_aurc_difference_full_minus_no_ev": paired,
        "fraction_seeds_full_pbe_wins": float(np.mean(full_aurc < no_aurc)),
        "time_to_first_oracle_top_10": {
            method: [row["time_to_first_oracle_top_10"] for row in by_method[method]]
            for method in ("NO_PBE", "FULL_PBE")
        },
        "global_optimum_discovery_count": {
            method: int(sum(row["global_optimum_discovered"] for row in by_method[method]))
            for method in ("NO_PBE", "FULL_PBE")
        },
        "median_simple_regret_trajectory_ev": median_trajectories,
        "pbe_vs_gw_spearman_191_actions_diagnostic": float(
            spearmanr(static["pbe_action_values"], oracle_values).statistic
        ),
    }
    return seed_summaries, metrics


def _render_figure(path: Path, metrics: dict[str, Any] | None) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 4.5))
    if metrics is None:
        axis.text(
            0.5,
            0.5,
            "Scientific rollout not completed",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
        axis.set_axis_off()
    else:
        x = np.arange(1, 13)
        for method, color in (("NO_PBE", "#3B6FB6"), ("FULL_PBE", "#D95F59")):
            axis.plot(
                x,
                metrics["median_simple_regret_trajectory_ev"][method],
                marker="o",
                label=method,
                color=color,
            )
        axis.set_xlabel("Sequential GW query")
        axis.set_ylabel("Median simple regret (eV)")
        axis.set_xticks(x)
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _render_results(
    terminal_state: str,
    run_sha: str,
    config_sha256: str,
    metrics: dict[str, Any] | None,
    validations: dict[str, dict[str, Any]],
    numerical_error: str | None,
) -> str:
    lines = [
        "# Sun oxide GW BO value pilot",
        "",
        f"Terminal state: `{terminal_state}`.",
        "",
        f"- RUN_SHA: `{run_sha}`",
        f"- Frozen config SHA-256: `{config_sha256}`",
        "- Methods: `NO_PBE`, `FULL_PBE`",
        "- FULL_PBE routine posterior: Laplace approximation, not the exact conditioned posterior.",
        "- Oracle values were isolated from both acquisitions and used globally only after all rollouts for evaluation.",
    ]
    if numerical_error is not None:
        lines.extend(["", f"Numerical error: `{numerical_error}`"])
    lines.extend(["", "## Laplace importance validation", ""])
    for name in VALIDATION_NAMES:
        report = validations.get(name, {"status": "NOT_RUN"})
        if "passed" in report:
            lines.append(
                f"- `{name}`: passed={report['passed']}, ESS/N={report['ess_fraction']:.6g}, "
                f"IS regret={report['is_estimated_regret_standardized']:.6g}, "
                f"gap MC SE={report['pairwise_gap_mc_se_standardized']:.6g}."
            )
        else:
            lines.append(f"- `{name}`: {report.get('status', 'NOT_RUN')}.")
    if metrics is not None:
        lines.extend(
            [
                "",
                "## Frozen scientific metrics",
                "",
                f"- Median AURC, NO_PBE / FULL_PBE: "
                f"`{metrics['median_aurc_ev']['NO_PBE']}` / `{metrics['median_aurc_ev']['FULL_PBE']}` eV.",
                f"- Median final regret, NO_PBE / FULL_PBE: "
                f"`{metrics['median_final_simple_regret_ev']['NO_PBE']}` / "
                f"`{metrics['median_final_simple_regret_ev']['FULL_PBE']}` eV.",
                f"- Fraction of seeds FULL_PBE wins: `{metrics['fraction_seeds_full_pbe_wins']}`.",
                f"- Global-optimum discovery count, NO_PBE / FULL_PBE: "
                f"`{metrics['global_optimum_discovery_count']['NO_PBE']}` / "
                f"`{metrics['global_optimum_discovery_count']['FULL_PBE']}`.",
                f"- PBE-vs-GW Spearman diagnostic: "
                f"`{metrics['pbe_vs_gw_spearman_191_actions_diagnostic']}`.",
            ]
        )
    return "\n".join(lines) + "\n"


def _finalize_outputs(
    output_dir: Path,
    zip_path: Path,
    config_path: Path,
    run_sha: str,
    smoke_report: dict[str, Any],
    terminal_state: str,
    trajectory_rows: list[dict[str, Any]],
    seed_summaries: list[dict[str, Any]],
    inference_rows: list[dict[str, Any]],
    timing_rows: list[dict[str, Any]],
    validations: dict[str, dict[str, Any]],
    metrics: dict[str, Any] | None,
    oracle: RetrospectiveOracle | None,
    numerical_error: str | None,
    started_at_utc: str,
) -> str:
    config_sha256 = _sha256(config_path)
    shutil.copyfile(config_path, output_dir / "frozen_config.json")
    _write_json(output_dir / "environment_report.json", smoke_report)
    _write_csv(output_dir / "trajectories.csv", trajectory_rows)
    _write_csv(output_dir / "seed_summaries.csv", seed_summaries)
    _write_json(output_dir / "inference_diagnostics.json", inference_rows)
    _write_csv(output_dir / "timing_diagnostics.csv", timing_rows)
    for name in VALIDATION_NAMES:
        report = validations.get(name, {"state": name, "status": "NOT_RUN"})
        _write_json(output_dir / f"is_validation_{name}.json", report)
    _write_json(
        output_dir / "oracle_access_log.json",
        [] if oracle is None else oracle.access_log,
    )
    provenance = {
        "schema_version": 1,
        "run_sha": run_sha,
        "config_sha256": config_sha256,
        "started_at_utc": started_at_utc,
        "terminal_state": terminal_state,
        "smoke_passed_before_oracle_open": smoke_report.get("status") == "SMOKE_PASS",
        "smoke_report_records_oracle_file_opened": smoke_report.get(
            "oracle_file_opened"
        ),
        "input_sha256": {
            name: specification["sha256"]
            for name, specification in _load_config(config_path)["inputs"].items()
        },
    }
    _write_json(output_dir / "provenance.json", provenance)
    run_summary = {
        "schema_version": 1,
        "terminal_state": terminal_state,
        "scientific_verdict": (
            "PASS_PBE_VALUE"
            if terminal_state == "PASS_PBE_VALUE_COLAB"
            else "FAIL_PBE_VALUE"
            if terminal_state == "FAIL_PBE_VALUE_COLAB"
            else None
        ),
        "run_sha": run_sha,
        "config_sha256": config_sha256,
        "completed_seed_count": len({row["seed"] for row in seed_summaries}),
        "metrics": metrics,
        "validation_passed": {
            name: validations.get(name, {}).get("passed") for name in VALIDATION_NAMES
        },
        "numerical_error": numerical_error,
    }
    _write_json(output_dir / "run_summary.json", run_summary)
    (output_dir / "RESULTS.md").write_text(
        _render_results(
            terminal_state,
            run_sha,
            config_sha256,
            metrics,
            validations,
            numerical_error,
        ),
        encoding="utf-8",
    )
    shutil.copyfile(
        ROOT / "experiments/sun_oxide/benchmark/NLR_DATA_USE_NOTICE.txt",
        output_dir / "NLR_DATA_USE_NOTICE.txt",
    )
    _render_figure(output_dir / "simple_regret_trajectories.png", metrics)

    artifact_paths = sorted(
        path.relative_to(output_dir).as_posix()
        for path in output_dir.rglob("*")
        if path.is_file() and path.name != "artifact_manifest.json"
    )
    manifest = {
        "schema_version": 1,
        "run_sha": run_sha,
        "config_sha256": config_sha256,
        "terminal_state": terminal_state,
        "files": [
            {
                "path": relative,
                "size_bytes": (output_dir / relative).stat().st_size,
                "sha256": _sha256(output_dir / relative),
            }
            for relative in artifact_paths
        ],
    }
    _write_json(output_dir / "artifact_manifest.json", manifest)
    if zip_path.exists():
        raise FileExistsError(f"Refusing to overwrite ZIP: {zip_path}")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for entry in manifest["files"]:
            archive.write(output_dir / entry["path"], entry["path"])
        archive.write(output_dir / "artifact_manifest.json", "artifact_manifest.json")
    zip_sha256 = _sha256(zip_path)
    print("ZIP_PATH", zip_path, flush=True)
    print("ZIP_SHA256", zip_sha256, flush=True)
    print(terminal_state, flush=True)
    return zip_sha256


def run_pilot(
    repository_root: Path,
    config_path: Path,
    output_dir: Path,
    zip_path: Path,
    run_sha: str,
) -> str:
    from datetime import datetime, timezone

    started_at_utc = datetime.now(timezone.utc).isoformat()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    smoke_report = scientific_smoke(repository_root, config_path, run_sha)
    print("SMOKE_PASS oracle_opened=false", flush=True)
    config = _load_config(config_path)
    all_paths = _verified_inputs(repository_root, config, include_oracle=True)
    static = _load_static_inputs(all_paths)
    oracle = _load_oracle_after_smoke(all_paths, static["action_keys"])
    print("ORACLE_OPENED_AFTER_SMOKE", flush=True)

    trajectory_rows: list[dict[str, Any]] = []
    seed_summaries: list[dict[str, Any]] = []
    inference_rows: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []
    validations: dict[str, dict[str, Any]] = {}
    seed_scales: dict[int, dict[str, float]] = {}
    metrics: dict[str, Any] | None = None
    numerical_error: str | None = None
    terminal_state = "NUMERICAL_FAILURE_COLAB"

    try:
        for seed in config["bo"]["seeds"]:
            initial_positions = fixed_initial_positions(191, 8, seed)
            initial_values = np.asarray(
                [
                    oracle.query(
                        int(position),
                        seed=seed,
                        method="SHARED_INITIAL",
                        stage=f"initial_{order}",
                    )
                    for order, position in enumerate(initial_positions, start=1)
                ],
                dtype=np.float64,
            )
            scale = freeze_target_scale(initial_values, scale_floor_ev=0.25)
            seed_scales[seed] = {
                "mean_ev": scale.mean_ev,
                "scale_ev": scale.scale_ev,
            }
            standardized_initial = scale.standardize(initial_values)
            for method in ("NO_PBE", "FULL_PBE"):
                _append_initial_rows(
                    trajectory_rows,
                    seed,
                    method,
                    initial_positions,
                    initial_values,
                    standardized_initial,
                    static,
                )

            initial_context = None
            if seed == 0:
                initial_context = _full_pbe_context(
                    static,
                    initial_positions.tolist(),
                    standardized_initial.tolist(),
                    None,
                    config,
                    decision_kind="routine_bo_decision",
                )
                validations["seed_0_initial"] = _importance_validation(
                    "seed_0_initial",
                    initial_context,
                    static,
                    initial_positions.tolist(),
                    standardized_initial.tolist(),
                    config,
                )
                print(
                    "IS_PRECHECK seed_0_initial "
                    f"passed={validations['seed_0_initial']['passed']}",
                    flush=True,
                )
                if not validations["seed_0_initial"]["passed"]:
                    terminal_state = "LAPLACE_VALIDATION_BLOCKED"
                    for later in VALIDATION_NAMES[1:]:
                        validations[later] = {
                            "state": later,
                            "status": "NOT_RUN_PRECHECK_BLOCKED",
                        }
                    break

            _run_no_pbe(
                seed,
                initial_positions,
                initial_values,
                scale,
                oracle,
                static,
                config,
                trajectory_rows,
                inference_rows,
                timing_rows,
            )
            _run_full_pbe(
                seed,
                initial_positions,
                initial_values,
                scale,
                oracle,
                static,
                config,
                trajectory_rows,
                inference_rows,
                timing_rows,
                validations,
                initial_context=initial_context,
            )
            _checkpoint_seed(output_dir, seed, trajectory_rows, inference_rows)
        else:
            oracle.unlock_post_run_evaluation()
            oracle_values = oracle.evaluation_values()
            seed_summaries, metrics = _post_run_metrics(
                trajectory_rows, seed_scales, oracle_values, static
            )
            all_valid = all(validations[name]["passed"] for name in VALIDATION_NAMES)
            if not all_valid:
                terminal_state = "LAPLACE_VALIDATION_FAILED"
            else:
                aurc_condition = (
                    metrics["median_aurc_ev"]["FULL_PBE"]
                    <= 0.90 * metrics["median_aurc_ev"]["NO_PBE"]
                )
                final_condition = (
                    metrics["median_final_simple_regret_ev"]["FULL_PBE"]
                    <= metrics["median_final_simple_regret_ev"]["NO_PBE"]
                )
                terminal_state = (
                    "PASS_PBE_VALUE_COLAB"
                    if aurc_condition and final_condition
                    else "FAIL_PBE_VALUE_COLAB"
                )
    except Exception as exc:
        numerical_error = f"{type(exc).__name__}: {exc}"
        terminal_state = "NUMERICAL_FAILURE_COLAB"

    return _finalize_outputs(
        output_dir,
        zip_path,
        config_path,
        run_sha,
        smoke_report,
        terminal_state,
        trajectory_rows,
        seed_summaries,
        inference_rows,
        timing_rows,
        validations,
        metrics,
        oracle,
        numerical_error,
        started_at_utc,
    )


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("smoke", "run"):
        command = subparsers.add_parser(name)
        command.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
        command.add_argument("--repository-root", type=Path, default=ROOT)
        command.add_argument("--run-sha", required=True)
        if name == "run":
            command.add_argument("--output-dir", type=Path, required=True)
            command.add_argument("--zip-path", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    arguments = _parse_arguments()
    try:
        if arguments.command == "smoke":
            report = scientific_smoke(
                arguments.repository_root.resolve(),
                arguments.config.resolve(),
                arguments.run_sha,
            )
            print(_canonical_json(report), end="")
            print("SMOKE_PASS")
            return 0
        run_pilot(
            arguments.repository_root.resolve(),
            arguments.config.resolve(),
            arguments.output_dir.resolve(),
            arguments.zip_path.resolve(),
            arguments.run_sha,
        )
        return 0
    except Exception as exc:
        if arguments.command == "smoke":
            print(f"INSTALLATION_BLOCKED {type(exc).__name__}: {exc}")
        else:
            print(f"NUMERICAL_FAILURE_COLAB {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
