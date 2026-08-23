#!/usr/bin/env python3
"""Run the frozen, target-blind PBE factor/Menz compatibility diagnostic."""

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

from conditioned_bo.pbe_factor_theory import (
    FACTOR_BANK_NAME,
    MAX_MIXED_CURVATURE,
    TEMPERATURE,
    build_comparison_matrix,
    build_factor_bank,
    factor_adjacency,
    factor_endpoint_array,
    factor_graph_diagnostics,
    factor_rows_for_csv,
    factorize_sparse,
    load_action_mapping,
    load_legacy_nodes,
    logistic_order_energy,
    logistic_order_gradient,
    logistic_order_hessian,
    observation_pattern_diagnostics,
    reference_graph_distances,
    selected_inverse_rows,
    sha256_file,
    summarize_pair_influence,
    validate_comparison_matrix,
)


DEFAULT_CONFIG = ROOT / "experiments/sun_oxide/configs/pbe_factor_theory.json"
DEFAULT_OUTPUT = ROOT / "experiments/sun_oxide/outputs/pbe_factor_theory"
EXPECTED_INPUT_NAMES = {
    "action_node_mapping",
    "legacy_pbe",
    "nlr_data_use_notice",
    "q0",
}
SCIENTIFIC_OUTPUTS = [
    "RESULTS.md",
    "pbe_factor_bank.csv",
    "factor_summary.json",
    "theory_summary.json",
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
        raise ValueError(
            f"Input interface must contain exactly {sorted(EXPECTED_INPUT_NAMES)}"
        )
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
    margins = np.asarray([-1000.0, -50.0, -3.0, 0.0, 2.0, 50.0, 1000.0])
    maximum_endpoint_gradient = 0.0
    minimum_hessian_eigenvalue = math.inf
    maximum_mixed_curvature = 0.0
    energies: list[float] = []
    for margin in margins:
        local = np.asarray([margin, 0.0])
        energies.append(logistic_order_energy(local))
        gradient = logistic_order_gradient(local)
        hessian = logistic_order_hessian(local)
        maximum_endpoint_gradient = max(
            maximum_endpoint_gradient, float(np.max(np.abs(gradient)))
        )
        minimum_hessian_eigenvalue = min(
            minimum_hessian_eigenvalue, float(np.linalg.eigvalsh(hessian).min())
        )
        maximum_mixed_curvature = max(
            maximum_mixed_curvature, float(abs(hessian[0, 1]))
        )
    if not np.all(np.isfinite(energies)):
        raise ValueError("Stable softplus evaluation failed")
    if maximum_endpoint_gradient > 1.0:
        raise ValueError("Logistic endpoint-gradient bound failed")
    if minimum_hessian_eigenvalue < -1e-15:
        raise ValueError("Logistic Hessian PSD check failed")
    if maximum_mixed_curvature > MAX_MIXED_CURVATURE:
        raise ValueError("Logistic mixed-curvature bound failed")
    return {
        "energy": "softplus(z_a - z_b) = log(1 + exp(-(z_b - z_a)))",
        "gradient": "[sigmoid(z_a-z_b), -sigmoid(z_a-z_b)]",
        "hessian": "sigmoid(d)(1-sigmoid(d)) * [[1,-1],[-1,1]]",
        "endpoint_gradient_absolute_bound": 1.0,
        "mixed_hessian_absolute_bound": MAX_MIXED_CURVATURE,
        "hessian_positive_semidefinite": True,
        "maximum_endpoint_gradient_on_regression_grid": maximum_endpoint_gradient,
        "minimum_hessian_eigenvalue_on_regression_grid": minimum_hessian_eigenvalue,
        "maximum_mixed_curvature_on_regression_grid": maximum_mixed_curvature,
        "stable_softplus_and_sigmoid": True,
    }


def _expanded_observation_patterns(
    config: dict[str, Any], action_nodes: np.ndarray, a0_smallest_eigenvalue: float
) -> list[dict[str, Any]]:
    patterns: list[dict[str, Any]] = []
    for specification in config["observation_uniformity"]["patterns"]:
        positions = specification["action_positions"]
        if positions == "all":
            action_positions = np.arange(action_nodes.size, dtype=np.int64)
        elif isinstance(positions, list):
            action_positions = np.asarray(positions, dtype=np.int64)
        else:
            raise ValueError(f"Invalid action positions for {specification['name']}")
        if action_positions.size and (
            action_positions.min() < 0 or action_positions.max() >= action_nodes.size
        ):
            raise ValueError(f"Action position outside mapping for {specification['name']}")
        patterns.append(
            {
                "name": specification["name"],
                "node_indices": action_nodes[action_positions].tolist(),
                "diagonal_precision": specification["diagonal_precision"],
                "a0_smallest_eigenvalue": a0_smallest_eigenvalue,
            }
        )
    return patterns


def _tie_summary(tie_groups: Sequence[dict[str, Any]], node_count: int) -> dict[str, Any]:
    histogram: dict[str, int] = {}
    for group in tie_groups:
        label = str(group["size"])
        histogram[label] = histogram.get(label, 0) + 1
    skipped = sum(int(group["size"]) - 1 for group in tie_groups)
    return {
        "exact_pbe_tie_group_count": len(tie_groups),
        "exact_pbe_tie_group_sizes": [int(group["size"]) for group in tie_groups],
        "exact_pbe_tie_group_size_histogram": histogram,
        "nodes_in_exact_pbe_tie_groups": sum(int(group["size"]) for group in tie_groups),
        "fraction_of_nodes_in_exact_pbe_tie_groups": float(
            sum(int(group["size"]) for group in tie_groups) / node_count
        ),
        "skipped_adjacent_ties": skipped,
        "tie_groups": list(tie_groups),
    }


def _results_markdown(
    starting_sha: str,
    factor_summary: dict[str, Any],
    theory_summary: dict[str, Any],
    influence_summary: dict[str, Any],
) -> str:
    ties = factor_summary["ties"]
    graph = factor_summary["factor_graph"]
    comparison = theory_summary["comparison"]
    solve = influence_summary["sparse_a0"]
    fraction = influence_summary["pair_diagnostics"]["factor_fraction_required"]
    fraction_lines = "\n".join(
        f"- {label.replace('_', ' ')}: 25th / median / 75th / 90th percentile "
        f"`{values['q25']}` / `{values['median']}` / `{values['q75']}` / `{values['p90']}`."
        for label, values in fraction.items()
    )
    return f"""# Frozen PBE-order factor and existing-theory compatibility result

Terminal verdict: `PASS_PBE_FACTOR_THEORY`.

This is the model-specific E3 construction for `CURRENT_NLR_PBE_GW_V1`.
It constructs no posterior, runs no Bayesian optimization, and changes no
project-level covariance theorem.

## Frozen factor bank

- Starting SHA: `{starting_sha}`
- Factor model: `{FACTOR_BANK_NAME}`
- Temperature: `{TEMPERATURE}`
- Legacy nodes / strict adjacent factors: `{factor_summary['legacy_node_count']}` / `{factor_summary['factor_count']}`
- Exact PBE tie groups / skipped adjacent ties: `{ties['exact_pbe_tie_group_count']}` / `{ties['skipped_adjacent_ties']}`
- Tie-size histogram: `{ties['exact_pbe_tie_group_size_histogram']}`
- Incident nodes / fraction: `{graph['incident_node_count']}` / `{graph['incident_node_fraction']}`
- Endpoint degree distribution / maximum: `{graph['degree_distribution']}` / `{graph['maximum_degree']}`

This is not Sun et al.'s all-pairs likelihood. It is a sparse, transparent
ordinal legacy-information model. Every retained relation is a true strict
ordering in the frozen PBE data; no arbitrary preference is inserted inside
an exact PBE tie. No comparison across PBE and GW numerical values is made.

## Factor calculus and existing Menz construction

For `d = z_a-z_b`, `e=softplus(d)`. Therefore
`gradient(e)=[sigmoid(d),-sigmoid(d)]` and
`Hessian(e)=sigmoid(d)(1-sigmoid(d))[[1,-1],[-1,1]]`. The endpoint gradient
magnitudes are at most one, the Hessian is positive semidefinite, and the
mixed-Hessian magnitude is at most `1/4`. Stable softplus/sigmoid evaluations
and finite-difference regressions cover these identities.

With one scalar Menz block per node, Gaussian conditional curvature is
`Q0_ii`. Convex preference factors cannot reduce it, while a factor edge can
increase cross-coordinate Hessian magnitude by at most `1/4`. Thus the
existing Menz construction is bounded by `A0 = Q0 - 0.25 R`; this is a
model-specific use of the paper's existing theory, not a new theorem.

- `max degree(R)`: `{graph['maximum_degree']}`
- `||R||_2` (numerical): `{graph['spectral_norm_numerical']}`
- Analytic `lambda_min(A0)` lower bound: `{comparison['analytic_smallest_eigenvalue_lower_bound']}`
- Numerical `lambda_min(A0)`: `{comparison['smallest_eigenvalue_numerical']}`
- Symmetric / nonpositive off-diagonals / SPD: `{comparison['symmetric']}` / `{comparison['off_diagonals_nonpositive']}` / `{comparison['positive_definite']}`

Nonnegative diagonal observation precision gives `At=A0+Dt`. The committed
full-size synthetic patterns remain SPD and satisfy the sampled sparse-solve
resolvent identity `A0^-1-At^-1=A0^-1 Dt At^-1`; the factor Hessian bound is
unchanged. Hence the single fixed operator `C=A0^-1` is conservative at later
scalar-observation BO iterations. No dense inverse was formed.

## Target-blind influence diagnostic

- Sparse A0 factorization / 191-RHS solve time: `{solve['factorization_seconds']}` / `{solve['action_multiple_rhs_solve_seconds']}` seconds.
- Unordered action pairs: `{influence_summary['pair_diagnostics']['unordered_action_pair_count']}`

Fractions of factors required to account for structural influence:

{fraction_lines}

These quantities are diagnostics only and have no pass/fail sparsity
threshold. Factor-set variation and reference-graph distance summaries are in
`influence_summary.json`; every action-pair diagnostic is in
`influence_pair_summary.csv`.

## Isolation

GW oracle read: `False`. No GW target statistic was computed. The scientific
input interface admitted only the frozen legacy PBE table, sparse `Q0`, action
mapping, and NLR data-use notice.
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
    if config["factor_bank"] != {
        "construction": "sort by increasing exact-decimal pbe_band_gap_ev, then composition_key; retain strict consecutive relations only",
        "energy": "log(1 + exp(-(z_b - z_a)))",
        "name": FACTOR_BANK_NAME,
        "temperature": TEMPERATURE,
    }:
        raise ValueError("Factor-bank freeze changed")
    inputs = _verified_inputs(repository_root, config)

    nodes = load_legacy_nodes(inputs["legacy_pbe"], config["legacy_node_count"])
    actions = load_action_mapping(
        inputs["action_node_mapping"], config["action_count"], len(nodes)
    )
    for action in actions:
        node = nodes[int(action["node_index"])]
        if (
            node.composition_key != action["composition_key"]
            or node.normalized_formula != action["normalized_formula"]
        ):
            raise ValueError("Action mapping does not match the legacy node row")

    ordered, factors, tie_groups = build_factor_bank(nodes)
    second_order, second_factors, second_ties = build_factor_bank(tuple(reversed(nodes)))
    if ordered != second_order or factors != second_factors or tie_groups != second_ties:
        raise ValueError("Factor-bank construction is not deterministic")
    endpoints = factor_endpoint_array(factors)
    factor_graph = factor_adjacency(len(nodes), endpoints)
    graph_diagnostics = factor_graph_diagnostics(factor_graph)
    tolerances = config["tolerances"]
    if graph_diagnostics["maximum_degree"] > 2:
        raise ValueError("Factor graph is not a path subgraph")
    if (
        graph_diagnostics["spectral_norm_numerical"]
        > config["menz_comparison"]["factor_graph_spectral_norm_upper_bound"]
        + tolerances["eigenvalue"]
    ):
        raise ValueError("Factor graph spectral norm exceeds two")

    q0 = sparse.load_npz(inputs["q0"]).tocsr()
    if q0.shape != (len(nodes), len(nodes)):
        raise ValueError("Q0 shape does not match legacy nodes")
    q0_summary = _q0_diagnostics(q0)
    if q0_summary["symmetry_max_abs_error"] > tolerances["symmetry"]:
        raise ValueError("Q0 is not symmetric")
    if q0_summary["maximum_off_diagonal"] > tolerances["sign"]:
        raise ValueError("Q0 off-diagonals are not nonpositive")
    if (
        q0_summary["smallest_eigenvalue_numerical"]
        < config["menz_comparison"]["q0_smallest_eigenvalue_analytic_lower_bound"]
        - tolerances["eigenvalue"]
    ):
        raise ValueError("Q0 numerical eigenvalue violates its frozen analytic floor")

    a0 = build_comparison_matrix(q0, factor_graph)
    comparison = validate_comparison_matrix(
        q0,
        factor_graph,
        a0,
        q0_analytic_eigenvalue_floor=config["menz_comparison"][
            "q0_smallest_eigenvalue_analytic_lower_bound"
        ],
        factor_graph_norm_upper_bound=config["menz_comparison"][
            "factor_graph_spectral_norm_upper_bound"
        ],
        symmetry_tolerance=tolerances["symmetry"],
        sign_tolerance=tolerances["sign"],
        eigenvalue_tolerance=tolerances["eigenvalue"],
    )
    calculus = _factor_calculus_diagnostics()

    a0_factorization, factorization_seconds = factorize_sparse(a0)
    action_nodes = np.asarray([action["node_index"] for action in actions], dtype=np.int64)
    action_rows, action_solve = selected_inverse_rows(
        a0, a0_factorization, action_nodes
    )
    if action_solve["relative_frobenius_residual"] > tolerances["solve_relative_residual"]:
        raise ValueError("Action-row multiple-RHS solve residual is too large")
    if action_solve["minimum_selected_inverse_entry"] < -tolerances["inverse_nonnegative"]:
        raise ValueError("Selected A0 inverse rows have a negative entry")

    patterns = _expanded_observation_patterns(
        config, action_nodes, comparison["smallest_eigenvalue_numerical"]
    )
    probe_positions = np.asarray(
        config["observation_uniformity"]["probe_action_positions"], dtype=np.int64
    )
    observation_started = time.perf_counter()
    observation_checks = observation_pattern_diagnostics(
        a0,
        a0_factorization,
        patterns,
        action_nodes[probe_positions],
        eigenvalue_tolerance=tolerances["eigenvalue"],
        identity_tolerance=tolerances["observation_identity"],
        nonnegative_tolerance=tolerances["inverse_nonnegative"],
    )
    observation_seconds = time.perf_counter() - observation_started

    distances = reference_graph_distances(q0, action_nodes)
    influence_started = time.perf_counter()
    pair_rows, pair_diagnostics = summarize_pair_influence(
        action_rows,
        endpoints,
        actions,
        config["influence"]["thresholds"],
        distances,
        nonnegative_tolerance=tolerances["inverse_nonnegative"],
    )
    influence_seconds = time.perf_counter() - influence_started

    tie_summary = _tie_summary(tie_groups, len(nodes))
    if tie_summary["skipped_adjacent_ties"] + len(factors) != len(nodes) - 1:
        raise ValueError("Strict factors and skipped ties do not partition adjacencies")
    factor_summary = {
        "schema_version": 1,
        "benchmark_name": config["benchmark_name"],
        "factor_bank": FACTOR_BANK_NAME,
        "temperature": TEMPERATURE,
        "legacy_node_count": len(nodes),
        "factor_count": len(factors),
        "construction_deterministic": True,
        "ordering": ["pbe_band_gap_ev increasing (exact decimal)", "composition_key increasing"],
        "exact_ties_omitted": True,
        "ties": tie_summary,
        "factor_graph": graph_diagnostics,
        "input_sha256": {
            name: sha256_file(path) for name, path in sorted(inputs.items())
        },
    }
    theory_summary = {
        "schema_version": 1,
        "benchmark_name": config["benchmark_name"],
        "factor_bank": FACTOR_BANK_NAME,
        "factor_calculus": calculus,
        "q0": q0_summary,
        "comparison_definition": "A0 = Q0 - 0.25 R",
        "comparison": comparison,
        "menz_blocks": "one scalar coordinate per block",
        "uniformity_argument": (
            "Qt=Q0+Dt with Dt nonnegative diagonal gives At=A0+Dt; symmetric "
            "nonsingular M-matrix inverse monotonicity makes C=A0^-1 conservative"
        ),
        "full_size_observation_patterns": observation_checks,
        "full_size_observation_checks_seconds": observation_seconds,
        "dense_inverse_formed": False,
        "new_covariance_theorem_introduced": False,
    }
    influence_summary = {
        "schema_version": 1,
        "benchmark_name": config["benchmark_name"],
        "factor_bank": FACTOR_BANK_NAME,
        "sensitivity_bounds": {
            "factor": "L_i(e_j) <= 1{i=a} + 1{i=b}",
            "ei_gap": "L_i(F_x_xhat) <= 1{i=x} + 1{i=xhat}",
            "factor_contribution": (
                "C[x,a] + C[x,b] + C[xhat,a] + C[xhat,b]"
            ),
        },
        "sparse_a0": {
            "factorization_count": 1,
            "factorization_seconds": factorization_seconds,
            "action_multiple_rhs_count": len(actions),
            "action_multiple_rhs_solve_seconds": action_solve[
                "multiple_rhs_solve_seconds"
            ],
            "action_solve_relative_frobenius_residual": action_solve[
                "relative_frobenius_residual"
            ],
            "action_selected_submatrix_symmetry_max_abs_error": action_solve[
                "selected_submatrix_symmetry_max_abs_error"
            ],
            "minimum_selected_inverse_entry": action_solve[
                "minimum_selected_inverse_entry"
            ],
            "full_dense_inverse_formed": False,
        },
        "pair_diagnostics": pair_diagnostics,
        "pair_influence_computation_seconds": influence_seconds,
        "diagnostic_not_gate": True,
        "target_isolation": {
            "gw_values_read": False,
            "gw_target_statistics_computed": False,
            "allowed_scientific_input_paths": [
                str(inputs[name].relative_to(repository_root))
                for name in sorted(inputs)
            ],
        },
    }

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    factor_csv_rows = factor_rows_for_csv(factors)
    factor_columns = [
        "factor_index",
        "factor_bank",
        "temperature",
        "sorted_position_a",
        "sorted_position_b",
        "node_a",
        "node_b",
        "composition_key_a",
        "composition_key_b",
        "normalized_formula_a",
        "normalized_formula_b",
        "pbe_band_gap_a_ev",
        "pbe_band_gap_b_ev",
        "pbe_gap_difference_ev",
        "strict_relation",
        "energy",
    ]
    _write_csv(output_dir / "pbe_factor_bank.csv", factor_csv_rows, factor_columns)
    factor_summary["factor_bank_csv_sha256"] = sha256_file(
        output_dir / "pbe_factor_bank.csv"
    )
    _write_json(output_dir / "factor_summary.json", factor_summary)
    _write_json(output_dir / "theory_summary.json", theory_summary)
    _write_json(output_dir / "influence_summary.json", influence_summary)
    _write_csv(output_dir / "influence_pair_summary.csv", pair_rows, list(pair_rows[0]))
    shutil.copyfile(inputs["nlr_data_use_notice"], output_dir / "NLR_DATA_USE_NOTICE.txt")
    (output_dir / "RESULTS.md").write_text(
        _results_markdown(starting_sha, factor_summary, theory_summary, influence_summary),
        encoding="utf-8",
    )
    artifact_manifest = {
        "schema_version": 1,
        "starting_sha": starting_sha,
        "verdict": "PASS_PBE_FACTOR_THEORY",
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
        "verdict": "PASS_PBE_FACTOR_THEORY",
        "factor_count": len(factors),
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
