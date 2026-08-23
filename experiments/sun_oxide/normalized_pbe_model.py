#!/usr/bin/env python3
"""Run the frozen target-blind normalized PBE replacement-model gate."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Sequence

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from conditioned_bo.normalized_pbe_model import (
    ACTION_COUNT,
    ADDITIONAL_COUNT,
    FACTOR_BANK_NAME,
    SUPPORT_COUNT,
    SUPPORT_NAME,
    TEMPERATURE,
    WEIGHT,
    build_strict_pair_bank,
    chunked_influence_diagnostics,
    descriptor_graph_action_pairs,
    farthest_point_support,
    preference_objective_and_gradient,
    signal_diagnostics,
    solve_preference_map,
    standardize_descriptor_space,
    support_coverage_diagnostics,
    weighted_adjacency,
    weighted_graph_diagnostics,
    weighted_logistic_energy,
    weighted_logistic_gradient,
    weighted_logistic_hessian,
)
from conditioned_bo.pbe_factor_theory import (
    build_comparison_matrix,
    factorize_sparse,
    load_action_mapping,
    load_legacy_nodes,
    observation_pattern_diagnostics,
    selected_inverse_rows,
    sha256_file,
    validate_comparison_matrix,
)


DEFAULT_CONFIG = ROOT / "experiments/sun_oxide/configs/normalized_pbe_model.json"
DEFAULT_OUTPUT = ROOT / "experiments/sun_oxide/outputs/normalized_pbe_model"
EXPECTED_INPUT_NAMES = {
    "action_node_mapping",
    "adjacent_factor_bank",
    "descriptor_matrix",
    "legacy_pbe",
    "nlr_data_use_notice",
    "q0",
}
SCIENTIFIC_OUTPUTS = [
    "RESULTS.md",
    "pbe_support_500.csv",
    "normalized_pbe_factor_bank.csv",
    "model_summary.json",
    "theory_summary.json",
    "pbe_signal_summary.json",
    "influence_summary.json",
    "influence_pair_summary.csv",
    "NLR_DATA_USE_NOTICE.txt",
]


def _canonical_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def _write_json(path: Path, value: Any) -> None:
    path.write_text(_canonical_json(value), encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[dict[str, Any]], columns: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(columns), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _current_sha(repository_root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _verified_inputs(repository_root: Path, config: dict[str, Any]) -> dict[str, Path]:
    declared = config.get("inputs")
    if not isinstance(declared, dict) or set(declared) != EXPECTED_INPUT_NAMES:
        raise ValueError(f"Input interface must contain exactly {sorted(EXPECTED_INPUT_NAMES)}")
    inputs: dict[str, Path] = {}
    root = repository_root.resolve()
    for name in sorted(EXPECTED_INPUT_NAMES):
        specification = declared[name]
        if set(specification) != {"path", "sha256"}:
            raise ValueError(f"Invalid frozen input declaration for {name}")
        path = (repository_root / specification["path"]).resolve()
        if path != root and root not in path.parents:
            raise ValueError(f"Input escapes repository root: {path}")
        observed = sha256_file(path)
        if observed != specification["sha256"]:
            raise ValueError(f"Frozen input hash mismatch for {name}: {observed}")
        inputs[name] = path
    return inputs


def _q0_diagnostics(q0: sparse.csr_matrix) -> dict[str, Any]:
    difference = (q0 - q0.T).tocsr()
    symmetry_error = float(np.max(np.abs(difference.data))) if difference.nnz else 0.0
    off_diagonal = q0.copy()
    off_diagonal.setdiag(0.0)
    off_diagonal.eliminate_zeros()
    maximum_off_diagonal = float(np.max(off_diagonal.data)) if off_diagonal.nnz else 0.0
    minimum_eigenvalue = float(
        eigsh(q0, k=1, which="SA", return_eigenvectors=False, tol=1e-12)[0]
    )
    return {
        "shape": list(q0.shape),
        "nnz": int(q0.nnz),
        "symmetry_max_abs_error": symmetry_error,
        "maximum_off_diagonal": maximum_off_diagonal,
        "smallest_eigenvalue_numerical": minimum_eigenvalue,
    }


def _factor_calculus_diagnostics() -> dict[str, Any]:
    margins = (-4.0, -0.7, 0.0, 1.3, 5.0)
    finite_gradient_error = 0.0
    finite_hessian_error = 0.0
    minimum_hessian_eigenvalue = math.inf
    maximum_gradient = 0.0
    maximum_mixed = 0.0
    gradient_step = 1e-6
    hessian_step = 2e-5
    for margin in margins:
        local = np.asarray([margin, -0.2], dtype=np.float64)
        sign = -1 if margin < 0.0 else 1
        gradient = weighted_logistic_gradient(local, (0, 1), sign)
        hessian = weighted_logistic_hessian(local, (0, 1), sign)
        finite_gradient = np.empty(2, dtype=np.float64)
        finite_hessian = np.empty((2, 2), dtype=np.float64)
        for coordinate in range(2):
            delta = np.zeros(2, dtype=np.float64)
            delta[coordinate] = gradient_step
            finite_gradient[coordinate] = (
                weighted_logistic_energy(local + delta, (0, 1), sign)
                - weighted_logistic_energy(local - delta, (0, 1), sign)
            ) / (2.0 * gradient_step)
            delta[coordinate] = hessian_step
            finite_hessian[:, coordinate] = (
                weighted_logistic_gradient(local + delta, (0, 1), sign)
                - weighted_logistic_gradient(local - delta, (0, 1), sign)
            ) / (2.0 * hessian_step)
        finite_gradient_error = max(
            finite_gradient_error, float(np.max(np.abs(gradient - finite_gradient)))
        )
        finite_hessian_error = max(
            finite_hessian_error, float(np.max(np.abs(hessian - finite_hessian)))
        )
        minimum_hessian_eigenvalue = min(
            minimum_hessian_eigenvalue, float(np.linalg.eigvalsh(hessian).min())
        )
        maximum_gradient = max(maximum_gradient, float(np.max(np.abs(gradient))))
        maximum_mixed = max(maximum_mixed, float(abs(hessian[0, 1])))
    zero_hessian = weighted_logistic_hessian([0.0, 0.0], (0, 1), 1)
    exact_mixed_at_zero = float(abs(zero_hessian[0, 1]))
    return {
        "energy": "omega * softplus(-s_ij * (Y_i-Y_j))",
        "endpoint_gradient_absolute_bound": WEIGHT,
        "mixed_hessian_absolute_bound": WEIGHT / 4.0,
        "maximum_endpoint_gradient_on_regression_grid": maximum_gradient,
        "maximum_mixed_curvature_on_regression_grid": maximum_mixed,
        "mixed_curvature_at_zero": exact_mixed_at_zero,
        "minimum_hessian_eigenvalue_on_regression_grid": minimum_hessian_eigenvalue,
        "hessian_positive_semidefinite": minimum_hessian_eigenvalue >= -1e-15,
        "finite_difference_gradient_max_abs_error": finite_gradient_error,
        "finite_difference_hessian_max_abs_error": finite_hessian_error,
        "stable_softplus_and_sigmoid": bool(
            np.isfinite(weighted_logistic_energy([1e300, -1e300], (0, 1), 1))
        ),
    }


def _read_adjacent_endpoints(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not {"node_a", "node_b"}.issubset(reader.fieldnames):
            raise ValueError("Adjacent factor bank lacks endpoint columns")
        rows = [(int(row["node_a"]), int(row["node_b"])) for row in reader]
    endpoints = np.asarray(rows, dtype=np.int64)
    if endpoints.shape != (1681, 2):
        raise ValueError("Adjacent baseline factor bank changed")
    return endpoints


def _expanded_observation_patterns(
    config: dict[str, Any], action_nodes: np.ndarray, a0_smallest_eigenvalue: float
) -> list[dict[str, Any]]:
    patterns: list[dict[str, Any]] = []
    for specification in config["observation_uniformity"]["patterns"]:
        positions = specification["action_positions"]
        action_positions = (
            np.arange(action_nodes.size, dtype=np.int64)
            if positions == "all"
            else np.asarray(positions, dtype=np.int64)
        )
        patterns.append(
            {
                "name": specification["name"],
                "node_indices": action_nodes[action_positions].tolist(),
                "diagonal_precision": specification["diagonal_precision"],
                "a0_smallest_eigenvalue": a0_smallest_eigenvalue,
            }
        )
    return patterns


def _support_rows(nodes, actions, support, fps) -> list[dict[str, Any]]:
    action_by_node = {int(action["node_index"]): action for action in actions}
    fps_step = {
        int(node): (step + 1, float(distance))
        for step, (node, distance) in enumerate(
            zip(fps.additional_indices, fps.additional_selection_distances, strict=True)
        )
    }
    rows: list[dict[str, Any]] = []
    for support_position, node_index in enumerate(support.tolist()):
        node = nodes[int(node_index)]
        action = action_by_node.get(int(node_index))
        step, distance = fps_step.get(int(node_index), (0, 0.0))
        rows.append(
            {
                "support_position": support_position,
                "node_index": int(node_index),
                "role": "action" if action else "pbe_only_fps",
                "fps_addition_step": step,
                "fps_selection_distance": distance,
                "action_key": "" if action is None else action["action_key"],
                "composition_key": node.composition_key,
                "normalized_formula": node.normalized_formula,
                "pbe_band_gap_ev": node.pbe_band_gap_text,
            }
        )
    return rows


def _factor_rows(bank) -> list[dict[str, Any]]:
    return [
        {
            "factor_index": index,
            "support_i": int(support_i),
            "support_j": int(support_j),
            "node_i": int(node_i),
            "node_j": int(node_j),
            "sign_s_ij": int(sign),
            "weight": WEIGHT,
        }
        for index, ((support_i, support_j), (node_i, node_j), sign) in enumerate(
            zip(
                bank.support_endpoint_pairs.tolist(),
                bank.node_endpoint_pairs.tolist(),
                bank.signs.tolist(),
                strict=True,
            )
        )
    ]


def _results_markdown(
    starting_sha: str,
    model: dict[str, Any],
    theory: dict[str, Any],
    signal: dict[str, Any],
    influence: dict[str, Any],
) -> str:
    coverage = model["support_selection"]["coverage"]
    graph = model["weighted_factor_graph"]
    comparison = theory["comparison"]
    dense = signal["normalized_dense"]
    adjacent = signal["adjacent_chain_baseline"]
    fractions = influence["pair_diagnostics"]["factor_fractions_required"]
    fraction_lines = "\n".join(
        f"- {label.replace('_', ' ')}: 25th / median / 75th / 90th percentile "
        f"`{values['q25']}` / `{values['median']}` / `{values['q75']}` / `{values['p90']}`."
        for label, values in fractions.items()
    )
    return f"""# Normalized PBE replacement-model gate

Terminal verdict: `PASS_NORMALIZED_PBE_MODEL`.

This target-blind E3 gate ran no Bayesian optimization, posterior inference,
or GW-target analysis.

## Frozen model

- Starting SHA: `{starting_sha}`
- Support / actions / added PBE-only nodes: `{model['support_count']}` / `{model['action_count']}` / `{model['additional_pbe_only_count']}`
- Strict factors / omitted exact-tie pairs: `{model['strict_factor_count']}` / `{model['omitted_exact_tie_pair_count']}`
- Exact-tie groups and size histogram: `{model['exact_tie_group_count']}` / `{model['exact_tie_group_size_histogram']}`
- Initial / final descriptor covering radius: `{coverage['initial_covering_radius']}` / `{coverage['final_covering_radius']}`
- Maximum weighted row sum / `||W_pbe||_2`: `{graph['maximum_weighted_row_sum']}` / `{graph['spectral_norm_numerical']}`

The bank is a normalized composite/generalized-Bayes ranking energy, not an
independent all-pairs likelihood. Support selection uses descriptors and action
membership only; exact PBE ties create no factor.

## Existing Menz compatibility

- Definition: `A0 = Q0 - 0.25 W_pbe`
- Analytic / numerical `lambda_min(A0)`: `{comparison['analytic_smallest_eigenvalue_lower_bound']}` / `{comparison['smallest_eigenvalue_numerical']}`
- Symmetric / nonpositive off-diagonals / SPD: `{comparison['symmetric']}` / `{comparison['off_diagonals_nonpositive']}` / `{comparison['positive_definite']}`
- Factorization / 191-RHS solve time: `{influence['sparse_a0']['factorization_seconds']}` / `{influence['sparse_a0']['action_multiple_rhs_solve_seconds']}` seconds
- Solve relative residual: `{influence['sparse_a0']['action_solve_relative_frobenius_residual']}`

The weighted logistic Hessian is PSD and has mixed-curvature magnitude at most
`omega/4`. Later nonnegative diagonal observation precision retains the
committed M-matrix/resolvent monotonicity, so `C=A0^-1` is conservative. No
dense inverse was formed and no theory-ledger change is proposed.

## PBE-only MAP signal

Normalized dense support/action Spearman: `{dense['support_500']['spearman_pbe_rank_vs_map_rank']}` / `{dense['actions_191']['spearman_pbe_rank_vs_map_rank']}`.
Adjacent baseline support/action Spearman: `{adjacent['support_500']['spearman_pbe_rank_vs_map_rank']}` / `{adjacent['actions_191']['spearman_pbe_rank_vs_map_rank']}`.

The complete MAP standard deviations, ranges, strict-pair accuracies, and
top-minus-bottom-decile contrasts are recorded in `pbe_signal_summary.json`.

## Target-blind influence diagnostic

The deterministic diagnostic has 256 action pairs, 64 from each
descriptor-graph-distance rank quartile. Fractions of strict factors needed to
reach structural influence totals are:

{fraction_lines}

Total structural influence is summarized in `influence_summary.json`. These
quantities have no sparsity pass/fail threshold.

## Isolation

GW oracle read: `False`. GW target statistics computed: `False`. The input
interface is an exact allowlist of hashes and contains no target table.
"""


def run(
    config_path: Path,
    repository_root: Path,
    output_dir: Path,
    starting_sha: str,
) -> dict[str, Any]:
    if _current_sha(repository_root) != starting_sha:
        raise ValueError("starting_sha must equal the checked-out HEAD")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["benchmark_name"] != "CURRENT_NLR_PBE_GW_V1":
        raise ValueError("Unexpected benchmark")
    expected_factor = {
        "energy": "omega * log(1 + exp(-s_ij * (Y_i - Y_j)))",
        "name": FACTOR_BANK_NAME,
        "temperature": TEMPERATURE,
        "weight": WEIGHT,
    }
    expected_support = {
        "action_count": ACTION_COUNT,
        "additional_pbe_only_count": ADDITIONAL_COUNT,
        "name": SUPPORT_NAME,
        "selection": "deterministic farthest-point sampling in standardized Magpie descriptor space",
        "support_count": SUPPORT_COUNT,
    }
    if config["factor_bank"] != expected_factor or config["support"] != expected_support:
        raise ValueError("Frozen normalized model specification changed")
    inputs = _verified_inputs(repository_root, config)

    nodes = load_legacy_nodes(inputs["legacy_pbe"], config["legacy_node_count"])
    actions = load_action_mapping(inputs["action_node_mapping"], ACTION_COUNT, len(nodes))
    action_nodes = np.asarray([action["node_index"] for action in actions], dtype=np.int64)
    for action in actions:
        node = nodes[int(action["node_index"])]
        if node.composition_key != action["composition_key"] or node.normalized_formula != action["normalized_formula"]:
            raise ValueError("Action mapping does not match legacy node row")

    with np.load(inputs["descriptor_matrix"], allow_pickle=False) as archive:
        descriptor_keys = archive["composition_keys"].astype(str)
        descriptor_formulas = archive["normalized_formulas"].astype(str)
        raw_descriptors = np.asarray(archive["raw_descriptors"], dtype=np.float64)
        feature_names = archive["feature_names"].astype(str)
    if raw_descriptors.shape != (len(nodes), 132) or feature_names.size != 132:
        raise ValueError("Frozen descriptor matrix shape changed")
    if descriptor_keys.tolist() != [node.composition_key for node in nodes]:
        raise ValueError("Descriptor rows do not align with legacy nodes")
    if descriptor_formulas.tolist() != [node.normalized_formula for node in nodes]:
        raise ValueError("Descriptor formulas do not align with legacy nodes")
    standardized, scaling = standardize_descriptor_space(raw_descriptors)
    if scaling["mean_sha256"] != config["descriptor_standardization"]["mean_sha256"]:
        raise ValueError("Descriptor mean hash changed")
    if scaling["scale_sha256"] != config["descriptor_standardization"]["scale_sha256"]:
        raise ValueError("Descriptor scale hash changed")

    keys = [node.composition_key for node in nodes]
    fps_started = time.perf_counter()
    fps = farthest_point_support(standardized, keys, action_nodes, SUPPORT_COUNT)
    fps_seconds = time.perf_counter() - fps_started
    fps_repeat = farthest_point_support(standardized, keys, action_nodes[::-1], SUPPORT_COUNT)
    if not np.array_equal(fps.selected_indices, fps_repeat.selected_indices):
        raise ValueError("Farthest-point support membership is not deterministic")
    if not np.array_equal(fps.additional_indices, fps_repeat.additional_indices):
        raise ValueError("Farthest-point support order is not deterministic")
    coverage = support_coverage_diagnostics(fps, action_nodes, len(nodes))
    if not coverage["all_actions_included"] or fps.additional_indices.size != ADDITIONAL_COUNT:
        raise ValueError("Frozen support membership constraints failed")

    bank = build_strict_pair_bank(fps.selected_indices, nodes)
    bank_repeat = build_strict_pair_bank(fps_repeat.selected_indices, nodes)
    if not (
        np.array_equal(bank.node_endpoint_pairs, bank_repeat.node_endpoint_pairs)
        and np.array_equal(bank.signs, bank_repeat.signs)
    ):
        raise ValueError("Strict pair-bank ordering is not deterministic")
    weighted_graph = weighted_adjacency(len(nodes), bank.node_endpoint_pairs, WEIGHT)
    weighted_graph_summary = weighted_graph_diagnostics(weighted_graph)
    tolerances = config["tolerances"]
    if weighted_graph_summary["maximum_weighted_row_sum"] > 1.0 + tolerances["weighted_row_sum"]:
        raise ValueError("Weighted row-sum bound failed")
    if weighted_graph_summary["spectral_norm_numerical"] > 1.0 + tolerances["eigenvalue"]:
        raise ValueError("Weighted adjacency spectral bound failed")

    q0 = sparse.load_npz(inputs["q0"]).tocsr()
    if q0.shape != (len(nodes), len(nodes)):
        raise ValueError("Q0 shape does not match legacy nodes")
    q0_summary = _q0_diagnostics(q0)
    a0 = build_comparison_matrix(q0, weighted_graph)
    comparison = validate_comparison_matrix(
        q0,
        weighted_graph,
        a0,
        q0_analytic_eigenvalue_floor=1.0,
        factor_graph_norm_upper_bound=1.0,
        symmetry_tolerance=tolerances["symmetry"],
        sign_tolerance=tolerances["sign"],
        eigenvalue_tolerance=tolerances["eigenvalue"],
    )
    if comparison["analytic_smallest_eigenvalue_lower_bound"] < 0.75:
        raise ValueError("Analytic A0 eigenvalue lower bound failed")
    calculus = _factor_calculus_diagnostics()
    if calculus["finite_difference_gradient_max_abs_error"] > tolerances["finite_difference"]:
        raise ValueError("Weighted logistic gradient finite difference failed")
    if calculus["finite_difference_hessian_max_abs_error"] > tolerances["finite_difference"]:
        raise ValueError("Weighted logistic Hessian finite difference failed")
    if not calculus["hessian_positive_semidefinite"]:
        raise ValueError("Weighted logistic Hessian PSD check failed")
    if calculus["mixed_curvature_at_zero"] > WEIGHT / 4.0:
        raise ValueError("Weighted logistic mixed-curvature bound failed")

    a0_factorization, factorization_seconds = factorize_sparse(a0)
    action_rows, action_solve = selected_inverse_rows(a0, a0_factorization, action_nodes)
    if action_solve["relative_frobenius_residual"] > tolerances["solve_relative_residual"]:
        raise ValueError("Sparse multiple-RHS solve residual failed")
    if action_solve["minimum_selected_inverse_entry"] < -tolerances["inverse_nonnegative"]:
        raise ValueError("Selected inverse rows violate M-matrix nonnegativity")

    patterns = _expanded_observation_patterns(
        config, action_nodes, comparison["smallest_eigenvalue_numerical"]
    )
    probe_positions = np.asarray(
        config["observation_uniformity"]["probe_action_positions"], dtype=np.int64
    )
    observation_started = time.perf_counter()
    observations = observation_pattern_diagnostics(
        a0,
        a0_factorization,
        patterns,
        action_nodes[probe_positions],
        eigenvalue_tolerance=tolerances["eigenvalue"],
        identity_tolerance=tolerances["observation_identity"],
        nonnegative_tolerance=tolerances["inverse_nonnegative"],
        factor_mixed_curvature_bound=WEIGHT / 4.0,
    )
    observation_seconds = time.perf_counter() - observation_started

    map_config = config["map"]
    dense_map, dense_optimization = solve_preference_map(
        q0,
        bank.node_endpoint_pairs,
        bank.signs,
        WEIGHT,
        chunk_size=map_config["chunk_size"],
        gradient_tolerance=map_config["gradient_tolerance"],
        function_tolerance=map_config["function_tolerance"],
        maximum_iterations=map_config["maximum_iterations"],
    )
    adjacent_endpoints = _read_adjacent_endpoints(inputs["adjacent_factor_bank"])
    adjacent_signs = np.asarray(
        [
            1 if nodes[int(left)].pbe_band_gap > nodes[int(right)].pbe_band_gap else -1
            for left, right in adjacent_endpoints
        ],
        dtype=np.int8,
    )
    adjacent_map, adjacent_optimization = solve_preference_map(
        q0,
        adjacent_endpoints,
        adjacent_signs,
        1.0,
        chunk_size=map_config["chunk_size"],
        gradient_tolerance=map_config["gradient_tolerance"],
        function_tolerance=map_config["function_tolerance"],
        maximum_iterations=map_config["maximum_iterations"],
    )
    for name, optimization in (
        ("normalized dense", dense_optimization),
        ("adjacent chain", adjacent_optimization),
    ):
        if not optimization["success"]:
            raise ValueError(f"{name} MAP optimizer failed: {optimization['message']}")
        if optimization["gradient_infinity_norm"] > map_config["required_gradient_infinity_norm"]:
            raise ValueError(f"{name} MAP gradient residual is too large")
    dense_signal = {
        "support_500": signal_diagnostics(nodes, dense_map, bank.support_indices),
        "actions_191": signal_diagnostics(nodes, dense_map, action_nodes),
    }
    adjacent_signal = {
        "support_500": signal_diagnostics(nodes, adjacent_map, bank.support_indices),
        "actions_191": signal_diagnostics(nodes, adjacent_map, action_nodes),
    }

    diagnostic_pairs, pair_selection = descriptor_graph_action_pairs(
        q0,
        actions,
        pair_count=config["influence"]["diagnostic_pair_count"],
        pairs_per_quartile=config["influence"]["pairs_per_distance_quartile"],
    )
    influence_started = time.perf_counter()
    pair_rows, pair_diagnostics = chunked_influence_diagnostics(
        action_rows,
        bank.node_endpoint_pairs,
        diagnostic_pairs,
        config["influence"]["thresholds"],
        weight=WEIGHT,
        chunk_size=config["influence"]["chunk_size"],
        nonnegative_tolerance=tolerances["inverse_nonnegative"],
    )
    influence_seconds = time.perf_counter() - influence_started

    model_summary = {
        "schema_version": 1,
        "benchmark_name": config["benchmark_name"],
        "support_name": SUPPORT_NAME,
        "factor_bank": FACTOR_BANK_NAME,
        "support_count": int(bank.support_indices.size),
        "action_count": int(action_nodes.size),
        "additional_pbe_only_count": int(fps.additional_indices.size),
        "strict_factor_count": bank.strict_factor_count,
        "omitted_exact_tie_pair_count": bank.omitted_exact_tie_pair_count,
        "exact_tie_group_count": bank.exact_tie_group_count,
        "exact_tie_group_size_histogram": bank.exact_tie_group_size_histogram,
        "complete_possible_pair_count": SUPPORT_COUNT * (SUPPORT_COUNT - 1) // 2,
        "global_weight": WEIGHT,
        "global_weight_exact": "1/499",
        "temperature": TEMPERATURE,
        "factor_ordering": "support endpoints ordered by increasing committed node index",
        "support_selection": {
            "deterministic": True,
            "fps_wall_seconds_first_run": fps_seconds,
            "descriptor_standardization": scaling,
            "coverage": coverage,
        },
        "weighted_factor_graph": weighted_graph_summary,
        "exact_ties_omitted": True,
        "composite_generalized_bayes_energy": True,
        "independent_all_pairs_likelihood": False,
        "input_sha256": {name: sha256_file(path) for name, path in sorted(inputs.items())},
    }
    theory_summary = {
        "schema_version": 1,
        "benchmark_name": config["benchmark_name"],
        "factor_bank": FACTOR_BANK_NAME,
        "factor_calculus": calculus,
        "q0": q0_summary,
        "comparison_definition": "A0 = Q0 - 0.25 W_pbe",
        "comparison": comparison,
        "uniformity_argument": (
            "Qt=Q0+Dt and At=A0+Dt for nonnegative diagonal Dt; symmetric "
            "nonsingular M-matrix inverse monotonicity makes C=A0^-1 conservative"
        ),
        "full_size_observation_patterns": observations,
        "full_size_observation_checks_seconds": observation_seconds,
        "dense_inverse_formed": False,
        "new_covariance_theorem_introduced": False,
    }
    pbe_signal_summary = {
        "schema_version": 1,
        "target_blind": True,
        "diagnostic_not_gate": True,
        "normalized_dense": {"optimization": dense_optimization, **dense_signal},
        "adjacent_chain_baseline": {"optimization": adjacent_optimization, **adjacent_signal},
    }
    influence_summary = {
        "schema_version": 1,
        "target_blind": True,
        "diagnostic_not_gate": True,
        "formula": "omega * (C[x,a] + C[x,b] + C[xhat,a] + C[xhat,b])",
        "action_pair_selection": pair_selection,
        "sparse_a0": {
            "factorization_count": 1,
            "factorization_seconds": factorization_seconds,
            "action_multiple_rhs_count": int(action_nodes.size),
            "action_multiple_rhs_solve_seconds": action_solve["multiple_rhs_solve_seconds"],
            "action_solve_relative_frobenius_residual": action_solve["relative_frobenius_residual"],
            "selected_submatrix_symmetry_max_abs_error": action_solve["selected_submatrix_symmetry_max_abs_error"],
            "minimum_selected_inverse_entry": action_solve["minimum_selected_inverse_entry"],
            "full_dense_inverse_formed": False,
        },
        "pair_diagnostics": pair_diagnostics,
        "pair_influence_computation_seconds": influence_seconds,
        "target_isolation": {
            "target_values_read": False,
            "target_statistics_computed": False,
            "allowed_scientific_input_paths": [
                str(inputs[name].relative_to(repository_root)) for name in sorted(inputs)
            ],
        },
    }

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    support_rows = _support_rows(nodes, actions, bank.support_indices, fps)
    factor_rows = _factor_rows(bank)
    _write_csv(output_dir / "pbe_support_500.csv", support_rows, list(support_rows[0]))
    _write_csv(output_dir / "normalized_pbe_factor_bank.csv", factor_rows, list(factor_rows[0]))
    model_summary["support_csv_sha256"] = sha256_file(output_dir / "pbe_support_500.csv")
    model_summary["factor_bank_csv_sha256"] = sha256_file(output_dir / "normalized_pbe_factor_bank.csv")
    _write_json(output_dir / "model_summary.json", model_summary)
    _write_json(output_dir / "theory_summary.json", theory_summary)
    _write_json(output_dir / "pbe_signal_summary.json", pbe_signal_summary)
    _write_json(output_dir / "influence_summary.json", influence_summary)
    _write_csv(output_dir / "influence_pair_summary.csv", pair_rows, list(pair_rows[0]))
    shutil.copyfile(inputs["nlr_data_use_notice"], output_dir / "NLR_DATA_USE_NOTICE.txt")
    (output_dir / "RESULTS.md").write_text(
        _results_markdown(starting_sha, model_summary, theory_summary, pbe_signal_summary, influence_summary),
        encoding="utf-8",
    )
    artifact_manifest = {
        "schema_version": 1,
        "starting_sha": starting_sha,
        "verdict": "PASS_NORMALIZED_PBE_MODEL",
        "files": [
            {
                "path": name,
                "sha256": sha256_file(output_dir / name),
                "size_bytes": (output_dir / name).stat().st_size,
            }
            for name in SCIENTIFIC_OUTPUTS
        ],
    }
    _write_json(output_dir / "artifact_manifest.json", artifact_manifest)
    return {
        "verdict": "PASS_NORMALIZED_PBE_MODEL",
        "support_count": int(bank.support_indices.size),
        "strict_factor_count": bank.strict_factor_count,
        "output_dir": str(output_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--starting-sha", required=True)
    arguments = parser.parse_args()
    result = run(
        arguments.config.resolve(),
        ROOT,
        arguments.output_dir.resolve(),
        arguments.starting_sha,
    )
    print(_canonical_json(result), end="")


if __name__ == "__main__":
    main()
