#!/usr/bin/env python3
"""Sun-oxide adaptive E3 engineering smoke and frozen fresh validation.

The ``smoke`` command is target-free.  ``engineering-smoke`` uses only the
already-consumed seeds 0--2.  ``run`` is the preregistered fresh seeds 12--31
execution and must only be launched from the frozen Colab handoff.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from typing import Any, Sequence
import zipfile

os.environ["MPLBACKEND"] = "Agg"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/energy_inference_bo_matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/energy_inference_bo_cache")
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import linalg, sparse
from scipy.sparse.linalg import splu

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from conditioned_bo.adaptive_pbe import (
    ActiveFactorState,
    AdaptiveDecision,
    AdaptiveSettings,
    adaptive_pbe_decision,
    construct_menz_support_reference,
    full_pbe_context,
    sherman_morrison_observation_update,
)
from conditioned_bo.bo_value import (
    ExactSupportMarginal,
    LaplaceState,
    NumericalFailure,
    RetrospectiveOracle,
    aurc,
    construct_gaussian_reference,
    draw_laplace_samples,
    exact_support_marginal,
    fixed_initial_positions,
    freeze_target_scale,
    gaussian_expected_improvement,
    laplace_log_importance_weights,
    precompute_exact_support_reference,
    select_unobserved_action,
    simple_regret_trajectory,
    snis_expected_improvement,
    snis_pairwise_gap_standard_error,
    stable_self_normalized_weights,
    update_exact_support_reference,
)


DEFAULT_CONFIG = ROOT / "experiments/sun_oxide/configs/adaptive_e3_validation.json"
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
METHODS = ("NO_PBE", "FULL_PBE_OPT", "ADAPTIVE_PBE")
VALIDATION_NAMES = (
    "seed_12_initial",
    "seed_12_after_6_queries",
    "seed_12_after_12_queries",
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


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader)


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
        raise ValueError("Unsupported adaptive E3 config schema")
    if config.get("benchmark_name") != "CURRENT_NLR_PBE_GW_V1":
        raise ValueError("Frozen benchmark changed")
    if config.get("action_count") != 191:
        raise ValueError("Frozen action count changed")
    if config["support"] != {
        "action_count": 191,
        "all_actions_in_support": True,
        "name": "PBE_SUPPORT_500_V1",
        "support_count": 500,
    }:
        raise ValueError("Frozen support changed")
    if config["bo"]["methods"] != list(METHODS):
        raise ValueError("Frozen validation methods changed")
    if config["bo"]["seeds"] != list(range(12, 32)):
        raise ValueError("Fresh validation seeds must be exactly 12--31")
    if config["engineering_smoke"]["seeds"] != [0, 1, 2]:
        raise ValueError("Engineering smoke seeds changed")
    if config["bo"]["initial_action_count"] != 8:
        raise ValueError("Frozen initialization count changed")
    if config["bo"]["sequential_query_count"] != 12:
        raise ValueError("Frozen sequential query count changed")
    if config["factor_bank"]["factor_count"] != 124718:
        raise ValueError("Frozen factor count changed")
    if config["factor_bank"]["weight_exact"] != "1/499":
        raise ValueError("Frozen factor weight changed")
    if config["gaussian_reference"]["observation_noise_standardized"] != 0.05:
        raise ValueError("Frozen observation noise changed")
    if config["gaussian_reference"]["hyperparameter_optimization_during_bo"]:
        raise ValueError("Online reference hyperparameter optimization is forbidden")
    if config["importance_validation"]["sample_count"] != 4096:
        raise ValueError("Frozen SNIS sample count changed")
    if config["bootstrap"]["resamples"] != 10000:
        raise ValueError("Frozen paired bootstrap count changed")
    if config["tuning_after_gw_results"]:
        raise ValueError("Post-target tuning is forbidden")
    if set(config.get("inputs", {})) != EXPECTED_INPUTS:
        raise ValueError(f"Input interface must be exactly {sorted(EXPECTED_INPUTS)}")
    AdaptiveSettings(
        epsilon_struct=config["adaptive"]["epsilon_struct"],
        rho=config["adaptive"]["rho"],
        max_stages=config["adaptive"]["max_stages"],
        weight=config["factor_bank"]["weight"],
    ).validate()
    return config


def _settings(config: dict[str, Any]) -> AdaptiveSettings:
    laplace = config["full_pbe_laplace"]
    return AdaptiveSettings(
        epsilon_struct=config["adaptive"]["epsilon_struct"],
        rho=config["adaptive"]["rho"],
        max_stages=config["adaptive"]["max_stages"],
        weight=config["factor_bank"]["weight"],
        chunk_size=config["factor_bank"]["chunk_size"],
        map_gradient_tolerance=laplace["map_gradient_infinity_norm_maximum"],
        optimizer_gradient_tolerance=laplace["optimizer_gradient_tolerance"],
        function_tolerance=laplace["function_tolerance"],
        maximum_iterations=laplace["maximum_iterations"],
        solve_residual_tolerance=laplace["solve_relative_residual_maximum"],
    )


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
        raise ValueError("Support nodes are not in increasing frozen order")
    support_by_node = {int(node): position for position, node in enumerate(support_nodes)}
    action_nodes = np.asarray([int(row["node_index"]) for row in actions], dtype=np.int64)
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
    return {
        "actions": actions,
        "action_keys": [row["action_key"] for row in actions],
        "action_nodes": action_nodes,
        "action_support_positions": action_support_positions,
        "support_nodes": support_nodes,
        "endpoint_pairs": endpoint_pairs,
        "signs": signs,
        "q0": q0,
    }


def _build_a0(static: dict[str, Any], weight: float) -> sparse.csc_matrix:
    full_pairs = static["support_nodes"][static["endpoint_pairs"]]
    row = np.concatenate((full_pairs[:, 0], full_pairs[:, 1]))
    col = np.concatenate((full_pairs[:, 1], full_pairs[:, 0]))
    adjacency = sparse.coo_matrix(
        (np.full(row.size, weight), (row, col)), shape=static["q0"].shape
    ).tocsc()
    adjacency.sum_duplicates()
    return sparse.csc_matrix(static["q0"] - 0.25 * adjacency)


def _direct_menz_block(
    a0: sparse.csc_matrix, support_nodes: np.ndarray, observed_support: np.ndarray
) -> np.ndarray:
    diagonal = np.zeros(a0.shape[0], dtype=np.float64)
    diagonal[support_nodes[observed_support]] = 400.0
    matrix = sparse.csc_matrix(a0 + sparse.diags(diagonal))
    factorization = splu(matrix, permc_spec="COLAMD")
    rhs = np.zeros((matrix.shape[0], support_nodes.size), dtype=np.float64)
    rhs[support_nodes, np.arange(support_nodes.size)] = 1.0
    return np.asarray(factorization.solve(rhs)[support_nodes], dtype=np.float64)


def scientific_smoke(
    repository_root: Path, config_path: Path, run_sha: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Target-free full-size regression of both one-time exact precomputes."""

    if _current_sha(repository_root) != run_sha:
        raise ValueError("RUN_SHA does not equal repository HEAD")
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
    tolerance = config["gaussian_reference"]["solve_relative_residual_maximum"]
    support_reference = precompute_exact_support_reference(
        static["q0"], static["support_nodes"], residual_tolerance=tolerance
    )
    menz_reference = construct_menz_support_reference(
        static["q0"],
        static["support_nodes"],
        static["endpoint_pairs"],
        weight=config["factor_bank"]["weight"],
        residual_tolerance=tolerance,
    )

    reference_errors: list[dict[str, Any]] = []
    patterns = (
        np.asarray([], dtype=np.int64),
        np.asarray([0], dtype=np.int64),
        np.asarray([1, 80, 233, 499], dtype=np.int64),
    )
    for pattern_index, positions in enumerate(patterns):
        values = np.linspace(-0.75, 0.85, positions.size, dtype=np.float64)
        optimized = update_exact_support_reference(
            support_reference, positions, values, residual_tolerance=tolerance
        )
        old_state = construct_gaussian_reference(
            static["q0"],
            static["support_nodes"][positions],
            values,
            residual_tolerance=tolerance,
        )
        old = exact_support_marginal(
            old_state, static["support_nodes"], residual_tolerance=tolerance
        )
        record = {
            "pattern": pattern_index,
            "observation_count": int(positions.size),
            "mean_max_abs_error": float(np.max(np.abs(optimized.mean - old.mean))),
            "covariance_max_abs_error": float(
                np.max(np.abs(optimized.covariance - old.covariance))
            ),
            "precision_max_abs_error": float(
                np.max(np.abs(optimized.precision - old.precision))
            ),
        }
        if record["mean_max_abs_error"] > 1e-10:
            raise NumericalFailure(f"Optimized support mean regression failed: {record}")
        if record["covariance_max_abs_error"] > 1e-10:
            raise NumericalFailure(f"Optimized support covariance regression failed: {record}")
        if record["precision_max_abs_error"] > 1e-8:
            raise NumericalFailure(f"Optimized support precision regression failed: {record}")
        reference_errors.append(record)

    a0 = _build_a0(static, config["factor_bank"]["weight"])
    menz_errors: list[dict[str, Any]] = []
    for pattern_index, positions in enumerate(patterns):
        updated = np.asarray(menz_reference.covariance, dtype=np.float64).copy()
        for position in positions.tolist():
            updated = sherman_morrison_observation_update(updated, position)
        direct = _direct_menz_block(a0, static["support_nodes"], positions)
        error = float(np.max(np.abs(updated - direct)))
        if error > 1e-10:
            raise NumericalFailure(f"Menz state update regression failed: {error}")
        menz_errors.append(
            {
                "pattern": pattern_index,
                "observation_count": int(positions.size),
                "maximum_absolute_error": error,
            }
        )

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
        "scientific_state": "PASS_PBE_VALUE",
        "dimensions": {
            "graph_nodes": 2142,
            "support_nodes": 500,
            "gw_actions": 191,
            "normalized_pbe_factors": 124718,
            "omega": config["factor_bank"]["weight"],
        },
        "support_reference_precompute": support_reference.diagnostics,
        "menz_reference_precompute": menz_reference.diagnostics,
        "support_reference_old_routine_regressions": reference_errors,
        "menz_direct_sparse_solve_regressions": menz_errors,
    }
    precomputed = {
        "support_reference": support_reference,
        "menz_reference": menz_reference,
        "static": static,
    }
    return report, precomputed


def _load_oracle_after_smoke(
    paths: dict[str, Path], action_keys: Sequence[str]
) -> RetrospectiveOracle:
    rows = _read_csv_rows(paths[ORACLE_INPUT])
    if [row["action_key"] for row in rows] != list(action_keys):
        raise ValueError("Oracle action order does not match the frozen action map")
    return RetrospectiveOracle(
        action_keys, [float(row["gw_band_gap_ev"]) for row in rows]
    )


def _support_marginal(
    support_reference: Any,
    static: dict[str, Any],
    observed_action_positions: Sequence[int],
    observed_z: Sequence[float],
    config: dict[str, Any],
) -> ExactSupportMarginal:
    actions = np.asarray(observed_action_positions, dtype=np.int64)
    return update_exact_support_reference(
        support_reference,
        static["action_support_positions"][actions],
        observed_z,
        sigma_obs=config["gaussian_reference"]["observation_noise_standardized"],
        residual_tolerance=config["gaussian_reference"][
            "solve_relative_residual_maximum"
        ],
    )


def _no_pbe_decision(
    support_reference: Any,
    static: dict[str, Any],
    observed_positions: Sequence[int],
    observed_z: Sequence[float],
    config: dict[str, Any],
) -> tuple[int, dict[str, Any], ExactSupportMarginal]:
    started = time.perf_counter()
    marginal = _support_marginal(
        support_reference, static, observed_positions, observed_z, config
    )
    action_support = static["action_support_positions"]
    ei = gaussian_expected_improvement(
        marginal.mean[action_support],
        np.maximum(np.diag(marginal.covariance)[action_support], 0.0),
        max(observed_z),
    )
    selected = select_unobserved_action(
        ei, observed_positions, static["action_keys"]
    )
    diagnostics = {
        "method": "NO_PBE",
        **marginal.diagnostics,
        "selected_action_position": selected,
        "selected_action_key": static["action_keys"][selected],
        "selected_ei_standardized": float(ei[selected]),
        "pbe_conditioning_seconds": 0.0,
        "factor_energy_gradient_calls": 0,
        "factor_energy_gradient_element_work": 0,
        "factor_hessian_calls": 0,
        "factor_hessian_element_work": 0,
        "active_factor_count": 0,
        "active_factor_fraction": 0.0,
        "adaptive_stages": 0,
        "total_decision_seconds": time.perf_counter() - started,
        "hyperparameters_fit_or_updated": False,
    }
    return selected, diagnostics, marginal


def _full_decision(
    support_reference: Any,
    static: dict[str, Any],
    observed_positions: Sequence[int],
    observed_z: Sequence[float],
    previous_map: np.ndarray | None,
    config: dict[str, Any],
) -> tuple[Any, dict[str, Any], ExactSupportMarginal]:
    started = time.perf_counter()
    marginal = _support_marginal(
        support_reference, static, observed_positions, observed_z, config
    )
    context = full_pbe_context(
        marginal,
        static["endpoint_pairs"],
        static["signs"],
        static["action_support_positions"],
        observed_positions,
        static["action_keys"],
        max(observed_z),
        previous_map,
        _settings(config),
    )
    diagnostics = {
        **marginal.diagnostics,
        **context.diagnostics,
        "selected_action_position": int(context.selected),
        "selected_action_key": static["action_keys"][context.selected],
        "selected_ei_standardized": float(context.ei[context.selected]),
        "adaptive_stages": 0,
        "structural_envelope": 0.0,
        "full_bank_fallback": False,
        "structurally_certified": True,
        "total_decision_seconds": time.perf_counter() - started,
        "hyperparameters_fit_or_updated": False,
    }
    return context, diagnostics, marginal


def _adaptive_decision(
    support_reference: Any,
    menz_covariance: np.ndarray,
    static: dict[str, Any],
    observed_positions: Sequence[int],
    observed_z: Sequence[float],
    factor_state: ActiveFactorState,
    previous_map: np.ndarray | None,
    config: dict[str, Any],
    *,
    pending_menz_update_seconds: float,
) -> tuple[AdaptiveDecision, dict[str, Any], ExactSupportMarginal]:
    started = time.perf_counter()
    marginal = _support_marginal(
        support_reference, static, observed_positions, observed_z, config
    )
    decision = adaptive_pbe_decision(
        marginal,
        menz_covariance,
        static["endpoint_pairs"],
        static["signs"],
        static["action_support_positions"],
        observed_positions,
        static["action_keys"],
        max(observed_z),
        factor_state,
        previous_map,
        _settings(config),
    )
    diagnostics = {
        **marginal.diagnostics,
        **decision.diagnostics,
        "selected_action_position": int(decision.selected),
        "selected_action_key": static["action_keys"][decision.selected],
        "selected_ei_standardized": float(decision.ei[decision.selected]),
        "active_factor_count": decision.active_count,
        "active_factor_fraction": decision.active_fraction,
        "adaptive_stages": decision.adaptive_stages,
        "structural_envelope": decision.structural_envelope,
        "worst_challenger_action_position": decision.worst_challenger,
        "full_bank_fallback": decision.full_bank_fallback,
        "structurally_certified": decision.structurally_certified,
        "menz_incremental_update_seconds": pending_menz_update_seconds,
        "pbe_conditioning_seconds": (
            decision.diagnostics["pbe_conditioning_seconds"]
            + pending_menz_update_seconds
        ),
        "total_decision_seconds": (
            time.perf_counter() - started + pending_menz_update_seconds
        ),
        "hyperparameters_fit_or_updated": False,
    }
    return decision, diagnostics, marginal


def _append_initial_rows(
    rows: list[dict[str, Any]],
    seed: int,
    methods: Sequence[str],
    positions: np.ndarray,
    values_ev: np.ndarray,
    standardized: np.ndarray,
    static: dict[str, Any],
) -> None:
    for method in methods:
        for order, (position, value, z_value) in enumerate(
            zip(
                positions.tolist(),
                values_ev.tolist(),
                standardized.tolist(),
                strict=True,
            ),
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


def _trajectory_row(
    seed: int,
    method: str,
    query_index: int,
    selected: int,
    value: float,
    z_value: float,
    static: dict[str, Any],
) -> dict[str, Any]:
    return {
        "seed": seed,
        "method": method,
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


def _run_no_pbe_trajectory(
    seed: int,
    initial_positions: np.ndarray,
    initial_values: np.ndarray,
    scale: Any,
    oracle: RetrospectiveOracle,
    precomputed: dict[str, Any],
    config: dict[str, Any],
    trajectory_rows: list[dict[str, Any]],
    inference_rows: list[dict[str, Any]],
) -> None:
    static = precomputed["static"]
    observed_positions = initial_positions.tolist()
    observed_z = scale.standardize(initial_values).tolist()
    for query_index in range(1, 13):
        print(f"PROGRESS seed={seed} method=NO_PBE iteration={query_index}/12", flush=True)
        selected, diagnostics, _ = _no_pbe_decision(
            precomputed["support_reference"],
            static,
            observed_positions,
            observed_z,
            config,
        )
        diagnostics.update({"seed": seed, "query_index": query_index})
        inference_rows.append(diagnostics)
        value = oracle.query(
            selected, seed=seed, method="NO_PBE", stage=f"sequential_{query_index}"
        )
        z_value = float(scale.standardize([value])[0])
        observed_positions.append(selected)
        observed_z.append(z_value)
        trajectory_rows.append(
            _trajectory_row(
                seed, "NO_PBE", query_index, selected, value, z_value, static
            )
        )


def _run_full_trajectory(
    seed: int,
    initial_positions: np.ndarray,
    initial_values: np.ndarray,
    scale: Any,
    oracle: RetrospectiveOracle,
    precomputed: dict[str, Any],
    config: dict[str, Any],
    trajectory_rows: list[dict[str, Any]],
    inference_rows: list[dict[str, Any]],
) -> None:
    static = precomputed["static"]
    observed_positions = initial_positions.tolist()
    observed_z = scale.standardize(initial_values).tolist()
    previous_map: np.ndarray | None = None
    for query_index in range(1, 13):
        print(
            f"PROGRESS seed={seed} method=FULL_PBE_OPT iteration={query_index}/12",
            flush=True,
        )
        context, diagnostics, _ = _full_decision(
            precomputed["support_reference"],
            static,
            observed_positions,
            observed_z,
            previous_map,
            config,
        )
        diagnostics.update({"seed": seed, "query_index": query_index})
        inference_rows.append(diagnostics)
        selected = int(context.selected)
        previous_map = np.asarray(context.map, dtype=np.float64)
        value = oracle.query(
            selected,
            seed=seed,
            method="FULL_PBE_OPT",
            stage=f"sequential_{query_index}",
        )
        z_value = float(scale.standardize([value])[0])
        observed_positions.append(selected)
        observed_z.append(z_value)
        trajectory_rows.append(
            _trajectory_row(
                seed,
                "FULL_PBE_OPT",
                query_index,
                selected,
                value,
                z_value,
                static,
            )
        )


def _gaussian_laplace_state(reference: ExactSupportMarginal) -> LaplaceState:
    cholesky = linalg.cholesky(reference.precision, lower=True, check_finite=True)
    return LaplaceState(
        map=np.asarray(reference.mean, dtype=np.float64),
        precision=np.asarray(reference.precision, dtype=np.float64),
        cholesky=cholesky,
        covariance=np.asarray(reference.covariance, dtype=np.float64),
        diagnostics={"approximation": "exact Gaussian reference"},
    )


def _importance_validation(
    name: str,
    decision: AdaptiveDecision,
    reference: ExactSupportMarginal,
    factor_state: ActiveFactorState,
    static: dict[str, Any],
    observed_positions: Sequence[int],
    observed_z: Sequence[float],
    config: dict[str, Any],
) -> dict[str, Any]:
    validation = config["importance_validation"]
    started = time.perf_counter()
    laplace = (
        decision.laplace
        if decision.laplace is not None
        else _gaussian_laplace_state(reference)
    )
    rng_seed = int(validation["rng_seeds"][name])
    samples = draw_laplace_samples(
        laplace, validation["sample_count"], np.random.default_rng(rng_seed)
    )
    active_indices = factor_state.active_indices()
    log_weights, weight_diagnostics = laplace_log_importance_weights(
        samples,
        reference.mean,
        reference.precision,
        laplace,
        static["endpoint_pairs"][active_indices],
        static["signs"][active_indices],
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
    laplace_selected = int(decision.selected)
    improvement = np.maximum(action_samples - max(observed_z), 0.0)
    gap_samples = improvement[:, is_selected] - improvement[:, laplace_selected]
    gap_estimate, gap_se = snis_pairwise_gap_standard_error(gap_samples, weights)
    regret = float(is_ei[is_selected] - is_ei[laplace_selected])
    threshold = max(0.02, 2.0 * gap_se)
    ess_fraction = ess / validation["sample_count"]
    return {
        "schema_version": 1,
        "state": name,
        "sample_count": validation["sample_count"],
        "rng_seed": rng_seed,
        "active_factor_count": factor_state.active_count,
        "active_factor_fraction": factor_state.active_count / factor_state.factor_count,
        "ess": ess,
        "ess_fraction": ess_fraction,
        "ess_fraction_minimum": validation["ess_fraction_minimum"],
        "laplace_selected_action_position": laplace_selected,
        "laplace_selected_action_key": static["action_keys"][laplace_selected],
        "is_selected_action_position": is_selected,
        "is_selected_action_key": static["action_keys"][is_selected],
        "is_estimated_regret_standardized": regret,
        "pairwise_gap_snis_estimate_standardized": gap_estimate,
        "pairwise_gap_mc_se_standardized": gap_se,
        "decision_regret_threshold_standardized": threshold,
        "passed": bool(
            ess_fraction >= validation["ess_fraction_minimum"]
            and regret <= threshold
        ),
        "validation_alters_adaptive_policy": False,
        "importance_validation_in_routine_timing": False,
        "weight_diagnostics": weight_diagnostics,
        "importance_sampling_seconds": time.perf_counter() - started,
    }


def _run_adaptive_trajectory(
    seed: int,
    initial_positions: np.ndarray,
    initial_values: np.ndarray,
    scale: Any,
    oracle: RetrospectiveOracle,
    precomputed: dict[str, Any],
    config: dict[str, Any],
    trajectory_rows: list[dict[str, Any]],
    inference_rows: list[dict[str, Any]],
    shadow_rows: list[dict[str, Any]],
    stage_rows: list[dict[str, Any]],
    validations: dict[str, dict[str, Any]],
    *,
    run_importance_validation: bool,
) -> None:
    static = precomputed["static"]
    observed_positions = initial_positions.tolist()
    observed_z = scale.standardize(initial_values).tolist()
    factor_state = ActiveFactorState.empty(static["endpoint_pairs"], 500)
    previous_map: np.ndarray | None = None
    shadow_previous_map: np.ndarray | None = None
    menz_covariance = np.asarray(
        precomputed["menz_reference"].covariance, dtype=np.float64
    ).copy()
    update_started = time.perf_counter()
    for action_position in observed_positions:
        support_position = int(static["action_support_positions"][action_position])
        menz_covariance = sherman_morrison_observation_update(
            menz_covariance, support_position
        )
    pending_menz_seconds = time.perf_counter() - update_started

    for query_index in range(1, 13):
        print(
            f"PROGRESS seed={seed} method=ADAPTIVE_PBE iteration={query_index}/12",
            flush=True,
        )
        decision, diagnostics, marginal = _adaptive_decision(
            precomputed["support_reference"],
            menz_covariance,
            static,
            observed_positions,
            observed_z,
            factor_state,
            previous_map,
            config,
            pending_menz_update_seconds=pending_menz_seconds,
        )
        diagnostics.update({"seed": seed, "query_index": query_index})
        inference_rows.append(diagnostics)
        for stage in diagnostics["stage_records"]:
            stage_rows.append(
                {
                    "seed": seed,
                    "query_index": query_index,
                    **stage,
                }
            )

        shadow_started = time.perf_counter()
        shadow = full_pbe_context(
            marginal,
            static["endpoint_pairs"],
            static["signs"],
            static["action_support_positions"],
            observed_positions,
            static["action_keys"],
            max(observed_z),
            shadow_previous_map,
            _settings(config),
        )
        shadow_seconds = time.perf_counter() - shadow_started
        shadow_previous_map = np.asarray(shadow.map, dtype=np.float64)
        shadow_regret = float(
            shadow.ei[shadow.selected] - shadow.ei[decision.selected]
        )
        if shadow_regret < -1e-12:
            raise NumericalFailure("Shadow FULL EI regret became negative")
        shadow_rows.append(
            {
                "seed": seed,
                "query_index": query_index,
                "adaptive_action_position": int(decision.selected),
                "adaptive_action_key": static["action_keys"][decision.selected],
                "shadow_full_action_position": int(shadow.selected),
                "shadow_full_action_key": static["action_keys"][shadow.selected],
                "action_agreement": bool(decision.selected == shadow.selected),
                "shadow_full_laplace_ei_regret_standardized": max(0.0, shadow_regret),
                "active_factor_count": decision.active_count,
                "active_factor_fraction": decision.active_fraction,
                "structural_envelope": decision.structural_envelope,
                "shadow_full_seconds_excluded_from_adaptive_runtime": shadow_seconds,
                "shadow_affected_adaptive_policy": False,
            }
        )

        validation_name = None
        if seed == 12 and query_index == 1:
            validation_name = "seed_12_initial"
        elif seed == 12 and query_index == 7:
            validation_name = "seed_12_after_6_queries"
        if run_importance_validation and validation_name is not None:
            validations[validation_name] = _importance_validation(
                validation_name,
                decision,
                marginal,
                factor_state,
                static,
                observed_positions,
                observed_z,
                config,
            )
            print(
                f"IS_VALIDATION {validation_name} "
                f"passed={validations[validation_name]['passed']}",
                flush=True,
            )

        selected = int(decision.selected)
        previous_map = np.asarray(decision.map, dtype=np.float64)
        value = oracle.query(
            selected,
            seed=seed,
            method="ADAPTIVE_PBE",
            stage=f"sequential_{query_index}",
        )
        z_value = float(scale.standardize([value])[0])
        observed_positions.append(selected)
        observed_z.append(z_value)
        trajectory_rows.append(
            _trajectory_row(
                seed,
                "ADAPTIVE_PBE",
                query_index,
                selected,
                value,
                z_value,
                static,
            )
        )
        update_started = time.perf_counter()
        menz_covariance = sherman_morrison_observation_update(
            menz_covariance,
            int(static["action_support_positions"][selected]),
        )
        pending_menz_seconds = time.perf_counter() - update_started

    if seed == 12 and run_importance_validation:
        final_decision, _, final_marginal = _adaptive_decision(
            precomputed["support_reference"],
            menz_covariance,
            static,
            observed_positions,
            observed_z,
            factor_state,
            previous_map,
            config,
            pending_menz_update_seconds=pending_menz_seconds,
        )
        validations["seed_12_after_12_queries"] = _importance_validation(
            "seed_12_after_12_queries",
            final_decision,
            final_marginal,
            factor_state,
            static,
            observed_positions,
            observed_z,
            config,
        )
        print(
            "IS_VALIDATION seed_12_after_12_queries "
            f"passed={validations['seed_12_after_12_queries']['passed']}",
            flush=True,
        )


def _old_full_actions(config: dict[str, Any], repository_root: Path) -> dict[int, list[int]]:
    path = repository_root / config["engineering_smoke"]["old_trajectory_path"]
    rows = _read_csv_rows(path)
    result: dict[int, list[int]] = {}
    for seed in config["engineering_smoke"]["seeds"]:
        selected = sorted(
            (
                row
                for row in rows
                if int(row["seed"]) == seed
                and row["method"] == config["engineering_smoke"]["old_full_method"]
                and row["phase"] == "sequential"
            ),
            key=lambda row: int(row["query_index"]),
        )
        result[seed] = [int(row["action_position"]) for row in selected]
    return result


def _selected_actions(
    trajectory_rows: Sequence[dict[str, Any]], seed: int, method: str
) -> list[int]:
    return [
        int(row["action_position"])
        for row in sorted(
            (
                row
                for row in trajectory_rows
                if row["seed"] == seed
                and row["method"] == method
                and row["phase"] == "sequential"
            ),
            key=lambda row: int(row["query_index"]),
        )
    ]


def run_engineering_smoke(
    repository_root: Path,
    config_path: Path,
    output_dir: Path,
    run_sha: str,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    smoke_report, precomputed = scientific_smoke(repository_root, config_path, run_sha)
    print("TARGET_FREE_SMOKE_PASS oracle_opened=false", flush=True)
    config = _load_config(config_path)
    paths = _verified_inputs(repository_root, config, include_oracle=True)
    oracle = _load_oracle_after_smoke(paths, precomputed["static"]["action_keys"])
    trajectory_rows: list[dict[str, Any]] = []
    inference_rows: list[dict[str, Any]] = []
    shadow_rows: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []
    validations: dict[str, dict[str, Any]] = {}
    old_actions = _old_full_actions(config, repository_root)

    for seed in config["engineering_smoke"]["seeds"]:
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
        _append_initial_rows(
            trajectory_rows,
            seed,
            ("FULL_PBE_OPT", "ADAPTIVE_PBE"),
            initial_positions,
            initial_values,
            scale.standardize(initial_values),
            precomputed["static"],
        )
        _run_full_trajectory(
            seed,
            initial_positions,
            initial_values,
            scale,
            oracle,
            precomputed,
            config,
            trajectory_rows,
            inference_rows,
        )
        _run_adaptive_trajectory(
            seed,
            initial_positions,
            initial_values,
            scale,
            oracle,
            precomputed,
            config,
            trajectory_rows,
            inference_rows,
            shadow_rows,
            stage_rows,
            validations,
            run_importance_validation=False,
        )
        _write_json(
            output_dir / "checkpoints" / f"seed_{seed:02d}.json",
            {
                "seed": seed,
                "completed": True,
                "trajectories": [row for row in trajectory_rows if row["seed"] == seed],
                "inference_diagnostics": [
                    row for row in inference_rows if row["seed"] == seed
                ],
                "shadow_full": [row for row in shadow_rows if row["seed"] == seed],
            },
        )
        print(f"CHECKPOINT seed={seed} complete", flush=True)

    full_matches = {
        str(seed): {
            "old": old_actions[seed],
            "optimized": _selected_actions(trajectory_rows, seed, "FULL_PBE_OPT"),
            "exact_match": old_actions[seed]
            == _selected_actions(trajectory_rows, seed, "FULL_PBE_OPT"),
        }
        for seed in config["engineering_smoke"]["seeds"]
    }
    full_rows = [row for row in inference_rows if row["method"] == "FULL_PBE_OPT"]
    adaptive_rows = [row for row in inference_rows if row["method"] == "ADAPTIVE_PBE"]
    adaptive_fractions = np.asarray(
        [row["active_factor_fraction"] for row in adaptive_rows], dtype=np.float64
    )
    full_times = np.asarray(
        [row["pbe_conditioning_seconds"] for row in full_rows], dtype=np.float64
    )
    adaptive_times = np.asarray(
        [row["pbe_conditioning_seconds"] for row in adaptive_rows], dtype=np.float64
    )
    median_fraction = float(np.median(adaptive_fractions))
    median_full_time = float(np.median(full_times))
    median_adaptive_time = float(np.median(adaptive_times))
    pathological = bool(
        median_fraction
        > config["engineering_smoke"][
            "pathological_active_fraction_strictly_greater_than"
        ]
        and median_adaptive_time
        > config["engineering_smoke"][
            "pathological_conditioning_time_ratio_strictly_greater_than"
        ]
        * median_full_time
    )
    mechanical = bool(
        all(item["exact_match"] for item in full_matches.values())
        and all(
            row["structurally_certified"] or row["full_bank_fallback"]
            for row in adaptive_rows
        )
        and all(row["factor_energy_gradient_element_work"] >= 0 for row in inference_rows)
        and all(row["factor_hessian_element_work"] >= 0 for row in inference_rows)
        and all(row["pbe_conditioning_seconds"] >= 0.0 for row in inference_rows)
    )
    terminal_state = (
        "ADAPTIVE_ENGINEERING_PATHOLOGICAL"
        if mechanical and pathological
        else "ENGINEERING_SMOKE_PASS"
        if mechanical
        else "ENGINEERING_SMOKE_FAILED"
    )
    summary = {
        "schema_version": 1,
        "terminal_state": terminal_state,
        "scientific_evidence": False,
        "implementation_sha": run_sha,
        "config_sha256": _sha256(config_path),
        "seeds": config["engineering_smoke"]["seeds"],
        "optimized_full_old_decision_regression": full_matches,
        "mechanically_sound": mechanical,
        "all_adaptive_certified_or_fallback": all(
            row["structurally_certified"] or row["full_bank_fallback"]
            for row in adaptive_rows
        ),
        "adaptive_fallback_decision_count": int(
            sum(bool(row["full_bank_fallback"]) for row in adaptive_rows)
        ),
        "adaptive_shadow_full_action_agreement_fraction": float(
            np.mean([row["action_agreement"] for row in shadow_rows])
        ),
        "median_adaptive_active_factor_fraction": median_fraction,
        "median_pbe_conditioning_seconds": {
            "FULL_PBE_OPT": median_full_time,
            "ADAPTIVE_PBE": median_adaptive_time,
        },
        "median_adaptive_to_full_conditioning_time_ratio": (
            median_adaptive_time / median_full_time
        ),
        "pathological_stop": pathological,
        "reference_precomputes": {
            "shared_500d": smoke_report["support_reference_precompute"],
            "adaptive_menz": smoke_report["menz_reference_precompute"],
        },
    }
    _write_json(output_dir / "target_free_smoke.json", smoke_report)
    _write_json(output_dir / "run_summary.json", summary)
    _write_csv(output_dir / "trajectories.csv", trajectory_rows)
    serializable_inference = []
    for row in inference_rows:
        serializable_inference.append({key: value for key, value in row.items() if key != "stage_records"})
    _write_csv(output_dir / "timing_and_work.csv", serializable_inference)
    _write_csv(output_dir / "shadow_full.csv", shadow_rows)
    _write_json(output_dir / "adaptive_stage_diagnostics.json", stage_rows)
    _write_json(output_dir / "oracle_access_log.json", oracle.access_log)
    print(terminal_state, flush=True)
    return summary


def _paired_bootstrap(
    no_aurc: np.ndarray,
    full_aurc: np.ndarray,
    adaptive_aurc: np.ndarray,
    full_time: np.ndarray,
    adaptive_time: np.ndarray,
    config: dict[str, Any],
) -> dict[str, Any]:
    bootstrap = config["bootstrap"]
    rng = np.random.default_rng(bootstrap["rng_seed"])
    count = int(bootstrap["resamples"])
    sample_size = no_aurc.size
    aurc_difference = np.empty(count, dtype=np.float64)
    retained = np.empty(count, dtype=np.float64)
    time_ratio = np.empty(count, dtype=np.float64)
    for index in range(count):
        selected = rng.integers(0, sample_size, size=sample_size)
        no_median = float(np.median(no_aurc[selected]))
        full_median = float(np.median(full_aurc[selected]))
        adaptive_median = float(np.median(adaptive_aurc[selected]))
        aurc_difference[index] = float(
            np.median(adaptive_aurc[selected] - full_aurc[selected])
        )
        denominator = no_median - full_median
        retained[index] = (
            (no_median - adaptive_median) / denominator
            if denominator != 0.0
            else np.nan
        )
        time_ratio[index] = float(
            np.median(adaptive_time[selected]) / np.median(full_time[selected])
        )

    def interval(values: np.ndarray) -> dict[str, float]:
        finite = values[np.isfinite(values)]
        return {
            "lower_95": float(np.percentile(finite, 2.5)),
            "upper_95": float(np.percentile(finite, 97.5)),
        }

    return {
        "paired": True,
        "resamples": count,
        "rng_seed": bootstrap["rng_seed"],
        "median_aurc_difference_adaptive_minus_full_ev": interval(aurc_difference),
        "retained_benefit_ratio": interval(retained),
        "median_conditioning_time_ratio_adaptive_over_full": interval(time_ratio),
    }


def _post_run_metrics(
    trajectory_rows: list[dict[str, Any]],
    inference_rows: list[dict[str, Any]],
    shadow_rows: list[dict[str, Any]],
    stage_rows: list[dict[str, Any]],
    validations: dict[str, dict[str, Any]],
    seed_scales: dict[int, dict[str, float]],
    oracle_values: np.ndarray,
    static: dict[str, Any],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    oracle_maximum = float(np.max(oracle_values))
    optimum_positions = set(np.flatnonzero(oracle_values == oracle_maximum).tolist())
    top_ten_positions = set(
        sorted(
            range(oracle_values.size),
            key=lambda position: (-oracle_values[position], static["action_keys"][position]),
        )[:10]
    )
    timing_lookup = {
        (int(row["seed"]), row["method"], int(row["query_index"])): row
        for row in inference_rows
    }
    seed_summaries: list[dict[str, Any]] = []
    sequential_lookup: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for seed in config["bo"]["seeds"]:
        for method in METHODS:
            selected_rows = [
                row
                for row in trajectory_rows
                if int(row["seed"]) == seed and row["method"] == method
            ]
            initial = sorted(
                (row for row in selected_rows if row["phase"] == "initial"),
                key=lambda row: int(row["initial_order"]),
            )
            sequential = sorted(
                (row for row in selected_rows if row["phase"] == "sequential"),
                key=lambda row: int(row["query_index"]),
            )
            initial_values = np.asarray(
                [float(row["gw_band_gap_ev"]) for row in initial], dtype=np.float64
            )
            sequential_values = np.asarray(
                [float(row["gw_band_gap_ev"]) for row in sequential], dtype=np.float64
            )
            regrets = simple_regret_trajectory(
                sequential_values, initial_values, oracle_maximum
            )
            running_best = np.maximum.accumulate(
                np.concatenate(([float(np.max(initial_values))], sequential_values))
            )[1:]
            cumulative_time = 0.0
            energy_work = 0
            hessian_work = 0
            for query_index, (row, best, regret) in enumerate(
                zip(sequential, running_best.tolist(), regrets.tolist(), strict=True),
                start=1,
            ):
                timing = timing_lookup[(seed, method, query_index)]
                cumulative_time += float(timing["pbe_conditioning_seconds"])
                energy_work += int(timing["factor_energy_gradient_element_work"])
                hessian_work += int(timing["factor_hessian_element_work"])
                row["best_observed_gw_ev"] = best
                row["simple_regret_ev"] = regret
                row["cumulative_pbe_conditioning_seconds"] = cumulative_time
            observed = {int(row["action_position"]) for row in selected_rows}
            initial_set = {int(row["action_position"]) for row in initial}
            if initial_set & top_ten_positions:
                time_top_ten: int | None = 0
            else:
                matches = [
                    int(row["query_index"])
                    for row in sequential
                    if int(row["action_position"]) in top_ten_positions
                ]
                time_top_ten = min(matches) if matches else None
            seed_summaries.append(
                {
                    "seed": seed,
                    "method": method,
                    "mu_seed_ev": seed_scales[seed]["mean_ev"],
                    "scale_seed_ev": seed_scales[seed]["scale_ev"],
                    "aurc_ev": aurc(regrets),
                    "final_simple_regret_ev": float(regrets[-1]),
                    "cumulative_pbe_conditioning_seconds": cumulative_time,
                    "factor_energy_gradient_element_work": energy_work,
                    "factor_hessian_element_work": hessian_work,
                    "time_to_first_oracle_top_10": time_top_ten,
                    "global_optimum_discovered": bool(observed & optimum_positions),
                }
            )
            sequential_lookup[(seed, method)] = sequential

    by_method = {
        method: [row for row in seed_summaries if row["method"] == method]
        for method in METHODS
    }
    arrays = {
        method: {
            "aurc": np.asarray([row["aurc_ev"] for row in by_method[method]]),
            "final": np.asarray(
                [row["final_simple_regret_ev"] for row in by_method[method]]
            ),
            "time": np.asarray(
                [row["cumulative_pbe_conditioning_seconds"] for row in by_method[method]]
            ),
            "energy_work": np.asarray(
                [row["factor_energy_gradient_element_work"] for row in by_method[method]]
            ),
            "hessian_work": np.asarray(
                [row["factor_hessian_element_work"] for row in by_method[method]]
            ),
        }
        for method in METHODS
    }
    median_aurc = {
        method: float(np.median(arrays[method]["aurc"])) for method in METHODS
    }
    denominator = median_aurc["NO_PBE"] - median_aurc["FULL_PBE_OPT"]
    retained_benefit = (
        (median_aurc["NO_PBE"] - median_aurc["ADAPTIVE_PBE"]) / denominator
        if denominator != 0.0
        else float("nan")
    )
    median_time = {
        method: float(np.median(arrays[method]["time"])) for method in METHODS
    }
    time_ratio = median_time["ADAPTIVE_PBE"] / median_time["FULL_PBE_OPT"]
    adaptive_inference = [
        row for row in inference_rows if row["method"] == "ADAPTIVE_PBE"
    ]
    criteria = {
        "fresh_full_value": bool(
            median_aurc["FULL_PBE_OPT"] <= 0.90 * median_aurc["NO_PBE"]
        ),
        "retained_benefit": bool(retained_benefit >= 0.90),
        "final_regret": bool(
            np.median(arrays["ADAPTIVE_PBE"]["final"])
            <= np.median(arrays["FULL_PBE_OPT"]["final"]) + 0.10
        ),
        "conditioning_wall_clock": bool(time_ratio <= 0.80),
        "adaptive_termination": all(
            row["structurally_certified"] or row["full_bank_fallback"]
            for row in adaptive_inference
        ),
        "seed_12_snis": all(
            validations.get(name, {}).get("passed", False)
            for name in VALIDATION_NAMES
        ),
        "numerical_and_oracle_isolation": True,
    }
    terminal_state = (
        "PASS_ADAPTIVE_E3" if all(criteria.values()) else "FAIL_ADAPTIVE_E3"
    )
    median_regret_trajectory = {
        method: np.median(
            np.asarray(
                [
                    [float(row["simple_regret_ev"]) for row in sequential_lookup[(seed, method)]]
                    for seed in config["bo"]["seeds"]
                ]
            ),
            axis=0,
        ).tolist()
        for method in METHODS
    }
    bootstrap = _paired_bootstrap(
        arrays["NO_PBE"]["aurc"],
        arrays["FULL_PBE_OPT"]["aurc"],
        arrays["ADAPTIVE_PBE"]["aurc"],
        arrays["FULL_PBE_OPT"]["time"],
        arrays["ADAPTIVE_PBE"]["time"],
        config,
    )
    shadow_regrets = np.asarray(
        [row["shadow_full_laplace_ei_regret_standardized"] for row in shadow_rows]
    )
    active_fractions = np.asarray(
        [row["active_factor_fraction"] for row in adaptive_inference]
    )
    envelopes = np.asarray([row["structural_envelope"] for row in adaptive_inference])
    stages = np.asarray([row["adaptive_stages"] for row in adaptive_inference])
    metrics = {
        "terminal_state": terminal_state,
        "criteria": criteria,
        "oracle_maximum_ev": oracle_maximum,
        "median_aurc_ev": median_aurc,
        "retained_benefit_ratio": float(retained_benefit),
        "median_final_simple_regret_ev": {
            method: float(np.median(arrays[method]["final"])) for method in METHODS
        },
        "median_cumulative_pbe_conditioning_seconds": median_time,
        "median_conditioning_time_ratio_adaptive_over_full": float(time_ratio),
        "median_factor_energy_gradient_element_work": {
            method: float(np.median(arrays[method]["energy_work"])) for method in METHODS
        },
        "median_factor_hessian_element_work": {
            method: float(np.median(arrays[method]["hessian_work"])) for method in METHODS
        },
        "median_simple_regret_trajectory_ev": median_regret_trajectory,
        "global_optimum_discovery_count": {
            method: int(sum(row["global_optimum_discovered"] for row in by_method[method]))
            for method in METHODS
        },
        "shadow_full": {
            "action_agreement_fraction": float(
                np.mean([row["action_agreement"] for row in shadow_rows])
            ),
            "median_laplace_ei_regret_standardized": float(np.median(shadow_regrets)),
            "maximum_laplace_ei_regret_standardized": float(np.max(shadow_regrets)),
        },
        "adaptive_structure": {
            "median_active_factor_fraction": float(np.median(active_fractions)),
            "iqr_active_factor_fraction": np.percentile(active_fractions, [25, 75]).tolist(),
            "median_structural_envelope": float(np.median(envelopes)),
            "maximum_structural_envelope": float(np.max(envelopes)),
            "median_stage_count": float(np.median(stages)),
            "maximum_stage_count": int(np.max(stages)),
            "fallback_decision_count": int(
                sum(row["full_bank_fallback"] for row in adaptive_inference)
            ),
        },
        "bootstrap_95_ci": bootstrap,
    }
    primary_table = [
        {
            "method": method,
            "median_aurc_ev": median_aurc[method],
            "median_final_simple_regret_ev": float(
                np.median(arrays[method]["final"])
            ),
            "global_optimum_discoveries": metrics["global_optimum_discovery_count"][method],
            "median_cumulative_pbe_conditioning_seconds": median_time[method],
            "median_factor_energy_gradient_element_work": metrics[
                "median_factor_energy_gradient_element_work"
            ][method],
            "median_factor_hessian_element_work": metrics[
                "median_factor_hessian_element_work"
            ][method],
        }
        for method in METHODS
    ]
    return seed_summaries, metrics, bootstrap, primary_table


def _render_figures(
    output_dir: Path,
    trajectory_rows: list[dict[str, Any]],
    inference_rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> None:
    sequential = [row for row in trajectory_rows if row["phase"] == "sequential"]
    colors = {
        "NO_PBE": "#3B6FB6",
        "FULL_PBE_OPT": "#D95F59",
        "ADAPTIVE_PBE": "#2B8C6B",
    }
    x = np.arange(1, 13)
    figure, axis = plt.subplots(figsize=(7.4, 4.6))
    for method in METHODS:
        values = np.asarray(
            [
                [
                    float(row["simple_regret_ev"])
                    for row in sorted(
                        (
                            item
                            for item in sequential
                            if item["seed"] == seed and item["method"] == method
                        ),
                        key=lambda item: int(item["query_index"]),
                    )
                ]
                for seed in config["bo"]["seeds"]
            ]
        )
        median = np.median(values, axis=0)
        lower, upper = np.percentile(values, [25, 75], axis=0)
        axis.plot(x, median, marker="o", label=method, color=colors[method])
        axis.fill_between(x, lower, upper, alpha=0.18, color=colors[method])
    axis.set_xlabel("Sequential GW query")
    axis.set_ylabel("Simple regret (eV), median + IQR")
    axis.set_xticks(x)
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_dir / "simple_regret_vs_query.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.4, 4.6))
    for method in METHODS:
        regrets = []
        times = []
        for query_index in range(1, 13):
            regrets.append(
                np.median(
                    [
                        row["simple_regret_ev"]
                        for row in sequential
                        if row["method"] == method and row["query_index"] == query_index
                    ]
                )
            )
            times.append(
                np.median(
                    [
                        row["cumulative_pbe_conditioning_seconds"]
                        for row in sequential
                        if row["method"] == method and row["query_index"] == query_index
                    ]
                )
            )
        axis.plot(times, regrets, marker="o", label=method, color=colors[method])
    axis.set_xlabel("Median cumulative PBE-conditioning wall-clock (s)")
    axis.set_ylabel("Median simple regret (eV)")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_dir / "simple_regret_vs_conditioning_time.png", dpi=180)
    plt.close(figure)

    adaptive = [row for row in inference_rows if row["method"] == "ADAPTIVE_PBE"]
    fraction_matrix = np.asarray(
        [
            [
                row["active_factor_fraction"]
                for row in sorted(
                    (item for item in adaptive if item["seed"] == seed),
                    key=lambda item: item["query_index"],
                )
            ]
            for seed in config["bo"]["seeds"]
        ]
    )
    figure, axis = plt.subplots(figsize=(7.4, 4.6))
    median = np.median(fraction_matrix, axis=0)
    lower, upper = np.percentile(fraction_matrix, [25, 75], axis=0)
    axis.plot(x, median, marker="o", color=colors["ADAPTIVE_PBE"])
    axis.fill_between(x, lower, upper, alpha=0.18, color=colors["ADAPTIVE_PBE"])
    axis.set_xlabel("Sequential GW query")
    axis.set_ylabel("Active PBE factor fraction, median + IQR")
    axis.set_xticks(x)
    axis.set_ylim(0.0, 1.02)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "active_factor_fraction_vs_query.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.4, 4.6))
    for method in ("FULL_PBE_OPT", "ADAPTIVE_PBE"):
        matrix = []
        for seed in config["bo"]["seeds"]:
            rows = sorted(
                (
                    item
                    for item in inference_rows
                    if item["seed"] == seed and item["method"] == method
                ),
                key=lambda item: item["query_index"],
            )
            matrix.append(np.cumsum([row["pbe_conditioning_seconds"] for row in rows]))
        values = np.asarray(matrix)
        median = np.median(values, axis=0)
        lower, upper = np.percentile(values, [25, 75], axis=0)
        axis.plot(x, median, marker="o", label=method, color=colors[method])
        axis.fill_between(x, lower, upper, alpha=0.18, color=colors[method])
    axis.set_xlabel("Sequential GW query")
    axis.set_ylabel("Cumulative PBE-conditioning time (s), median + IQR")
    axis.set_xticks(x)
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_dir / "cumulative_conditioning_time.png", dpi=180)
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
        "# Sun oxide adaptive E3 fresh validation",
        "",
        f"Terminal state: `{terminal_state}`.",
        "",
        f"- RUN_SHA: `{run_sha}`",
        f"- Frozen config SHA-256: `{config_sha256}`",
        "- Methods: `NO_PBE`, `FULL_PBE_OPT`, `ADAPTIVE_PBE`.",
        "- Routine FULL and ADAPTIVE inference use Laplace approximations.",
        "- Shadow FULL and SNIS are excluded from routine ADAPTIVE timing.",
    ]
    if numerical_error is not None:
        lines.extend(["", f"Numerical error: `{numerical_error}`"])
    lines.extend(["", "## Seed-12 active-target SNIS", ""])
    for name in VALIDATION_NAMES:
        report = validations.get(name, {"status": "NOT_RUN"})
        if "passed" in report:
            lines.append(
                f"- `{name}`: passed={report['passed']}, "
                f"ESS/N={report['ess_fraction']:.6g}, "
                f"IS regret={report['is_estimated_regret_standardized']:.6g}, "
                f"gap MC SE={report['pairwise_gap_mc_se_standardized']:.6g}."
            )
        else:
            lines.append(f"- `{name}`: {report.get('status', 'NOT_RUN')}.")
    if metrics is not None:
        lines.extend(["", "## Frozen primary metrics", ""])
        for method in METHODS:
            lines.append(
                f"- `{method}` median AURC / final regret / conditioning time: "
                f"`{metrics['median_aurc_ev'][method]}` eV / "
                f"`{metrics['median_final_simple_regret_ev'][method]}` eV / "
                f"`{metrics['median_cumulative_pbe_conditioning_seconds'][method]}` s."
            )
        lines.extend(
            [
                f"- Retained-benefit ratio: `{metrics['retained_benefit_ratio']}`.",
                "- Median ADAPTIVE/FULL conditioning-time ratio: "
                f"`{metrics['median_conditioning_time_ratio_adaptive_over_full']}`.",
                f"- Frozen criteria: `{metrics['criteria']}`.",
                "",
                "Absolute seconds are hardware-specific. Factor energy-gradient and "
                "Hessian element work are reported in `primary_metric_table.csv` and "
                "the complete timing record.",
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
    shadow_rows: list[dict[str, Any]],
    stage_rows: list[dict[str, Any]],
    validations: dict[str, dict[str, Any]],
    metrics: dict[str, Any] | None,
    bootstrap: dict[str, Any] | None,
    primary_table: list[dict[str, Any]],
    oracle: RetrospectiveOracle | None,
    numerical_error: str | None,
    started_at_utc: str,
) -> str:
    config_sha256 = _sha256(config_path)
    shutil.copyfile(config_path, output_dir / "frozen_config.json")
    _write_json(output_dir / "environment_and_target_free_smoke.json", smoke_report)
    _write_csv(output_dir / "trajectories.csv", trajectory_rows)
    _write_csv(output_dir / "seed_summaries.csv", seed_summaries)
    serializable_inference = [
        {key: value for key, value in row.items() if key != "stage_records"}
        for row in inference_rows
    ]
    _write_csv(output_dir / "timings_and_work.csv", serializable_inference)
    _write_csv(output_dir / "shadow_full.csv", shadow_rows)
    _write_json(output_dir / "adaptive_stage_diagnostics.json", stage_rows)
    _write_csv(output_dir / "primary_metric_table.csv", primary_table)
    _write_json(output_dir / "paired_bootstrap.json", bootstrap or {})
    for name in VALIDATION_NAMES:
        _write_json(
            output_dir / f"is_validation_{name}.json",
            validations.get(name, {"state": name, "status": "NOT_RUN"}),
        )
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
        "target_free_smoke_passed_before_oracle_open": smoke_report.get("status")
        == "SMOKE_PASS",
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
            "PASS_ADAPTIVE_E3" if terminal_state == "PASS_ADAPTIVE_E3" else None
        ),
        "run_sha": run_sha,
        "config_sha256": config_sha256,
        "completed_seed_count": len({row["seed"] for row in seed_summaries}),
        "metrics": metrics,
        "validation_passed": {
            name: validations.get(name, {}).get("passed") for name in VALIDATION_NAMES
        },
        "numerical_error": numerical_error,
        "precompute_timing": {
            "shared_500d_reference": smoke_report.get(
                "support_reference_precompute"
            ),
            "adaptive_c_h0": smoke_report.get("menz_reference_precompute"),
            "amortization_decisions": 20 * 12,
            "shared_500d_reference_amortized_seconds_per_method_decision": (
                smoke_report.get("support_reference_precompute", {}).get(
                    "total_seconds", 0.0
                )
                / (20 * 12 * 3)
            ),
            "adaptive_c_h0_amortized_seconds_per_adaptive_decision": (
                smoke_report.get("menz_reference_precompute", {}).get(
                    "total_seconds", 0.0
                )
                / (20 * 12)
            ),
        },
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
    if metrics is not None:
        _render_figures(output_dir, trajectory_rows, inference_rows, _load_config(config_path))

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


def run_validation(
    repository_root: Path,
    config_path: Path,
    output_dir: Path,
    zip_path: Path,
    run_sha: str,
) -> str:
    started_at_utc = datetime.now(timezone.utc).isoformat()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    smoke_report, precomputed = scientific_smoke(repository_root, config_path, run_sha)
    print("SMOKE_PASS oracle_opened=false", flush=True)
    config = _load_config(config_path)
    paths = _verified_inputs(repository_root, config, include_oracle=True)
    oracle = _load_oracle_after_smoke(paths, precomputed["static"]["action_keys"])
    print("ORACLE_OPENED_AFTER_SMOKE", flush=True)
    trajectory_rows: list[dict[str, Any]] = []
    inference_rows: list[dict[str, Any]] = []
    shadow_rows: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []
    validations: dict[str, dict[str, Any]] = {}
    seed_scales: dict[int, dict[str, float]] = {}
    seed_summaries: list[dict[str, Any]] = []
    primary_table: list[dict[str, Any]] = []
    metrics: dict[str, Any] | None = None
    bootstrap: dict[str, Any] | None = None
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
            _append_initial_rows(
                trajectory_rows,
                seed,
                METHODS,
                initial_positions,
                initial_values,
                scale.standardize(initial_values),
                precomputed["static"],
            )
            _run_no_pbe_trajectory(
                seed,
                initial_positions,
                initial_values,
                scale,
                oracle,
                precomputed,
                config,
                trajectory_rows,
                inference_rows,
            )
            _run_full_trajectory(
                seed,
                initial_positions,
                initial_values,
                scale,
                oracle,
                precomputed,
                config,
                trajectory_rows,
                inference_rows,
            )
            _run_adaptive_trajectory(
                seed,
                initial_positions,
                initial_values,
                scale,
                oracle,
                precomputed,
                config,
                trajectory_rows,
                inference_rows,
                shadow_rows,
                stage_rows,
                validations,
                run_importance_validation=True,
            )
            _write_json(
                output_dir / "checkpoints" / f"seed_{seed:02d}.json",
                {
                    "seed": seed,
                    "completed": True,
                    "trajectories": [
                        row for row in trajectory_rows if row["seed"] == seed
                    ],
                    "inference_diagnostics": [
                        {key: value for key, value in row.items() if key != "stage_records"}
                        for row in inference_rows
                        if row["seed"] == seed
                    ],
                    "shadow_full": [row for row in shadow_rows if row["seed"] == seed],
                },
            )
            print(f"CHECKPOINT seed={seed} complete", flush=True)
        oracle.unlock_post_run_evaluation()
        oracle_values = oracle.evaluation_values()
        seed_summaries, metrics, bootstrap, primary_table = _post_run_metrics(
            trajectory_rows,
            inference_rows,
            shadow_rows,
            stage_rows,
            validations,
            seed_scales,
            oracle_values,
            precomputed["static"],
            config,
        )
        terminal_state = metrics["terminal_state"]
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
        shadow_rows,
        stage_rows,
        validations,
        metrics,
        bootstrap,
        primary_table,
        oracle,
        numerical_error,
        started_at_utc,
    )


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("smoke", "engineering-smoke", "run"):
        command = subparsers.add_parser(name)
        command.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
        command.add_argument("--repository-root", type=Path, default=ROOT)
        command.add_argument("--run-sha", required=True)
        if name in ("engineering-smoke", "run"):
            command.add_argument("--output-dir", type=Path, required=True)
        if name == "run":
            command.add_argument("--zip-path", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    arguments = _parse_arguments()
    try:
        if arguments.command == "smoke":
            report, _ = scientific_smoke(
                arguments.repository_root.resolve(),
                arguments.config.resolve(),
                arguments.run_sha,
            )
            print(_canonical_json(report), end="")
            print("SMOKE_PASS")
            return 0
        if arguments.command == "engineering-smoke":
            summary = run_engineering_smoke(
                arguments.repository_root.resolve(),
                arguments.config.resolve(),
                arguments.output_dir.resolve(),
                arguments.run_sha,
            )
            return 0 if summary["mechanically_sound"] else 1
        run_validation(
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
        elif arguments.command == "engineering-smoke":
            print(f"ENGINEERING_SMOKE_FAILED {type(exc).__name__}: {exc}")
        else:
            print(f"NUMERICAL_FAILURE_COLAB {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
