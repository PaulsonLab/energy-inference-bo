"""Validate the accepted nonlinear-PDE T2-B structural construction.

This runner is a proof/implementation regression check.  It evaluates no
factor energies, performs no posterior inference, and does not rerun the E2
campaign.  The empirical inference allowance and total stopping envelope are
recorded only as locked provenance from the archived notebook replay.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import scipy
from scipy import sparse

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from conditioned_bo.nonlinear_pde_influence import (
    DEFAULT_PARAMETERS,
    analytic_interior_row_margin,
    build_nonlinear_pde_comparison,
    diagonal_dominance_threshold,
    extreme_eigenvalues,
    maximum_nonzeros_per_row,
    omitted_factor_load,
    row_dominance_margins,
    solve_comparison,
    structural_screening_bound,
)


OUTPUT_DIRECTORY = (
    Path(__file__).resolve().parent / "outputs" / "t2b_structural_validation"
)
PROTOTYPE_PATH = (
    REPOSITORY_ROOT
    / "notebooks"
    / "prototypes"
    / "DEC_Nonlinear_PDE_BO_Demo.ipynb"
)

FROZEN_CONFIG: dict[str, Any] = {
    "provenance": {
        "audit_base_commit": "fe8d994119a47b0709651f26a418c946032e90f5",
        "prototype_path": "notebooks/prototypes/DEC_Nonlinear_PDE_BO_Demo.ipynb",
        "prototype_sha256": "73459edc0545ea2470a0cba5ab3cf60d18508b78b0acae08068790905b48e6fd",
        "replay_seed": 911,
    },
    "model": {
        "grid_size": 24,
        "q0": 3.5,
        "q_laplacian": 0.6,
        "coupling": 0.12,
        "nonlinearity": 0.25,
        "gamma": 0.08,
        "tau": 0.30,
        "incumbent": 0.55,
    },
    "locked_replay": {
        "active_factor_indices": [
            179,
            202,
            203,
            204,
            205,
            225,
            226,
            227,
            228,
            229,
            250,
            251,
            252,
            253,
            275,
            298,
            299,
            300,
            301,
            322,
            323,
            324,
            325,
            326,
            345,
            346,
            347,
            348,
            349,
            350,
            369,
            370,
            371,
            372,
            373,
            374,
            394,
            395,
            396,
            397,
        ],
        "leader_site": [14, 12],
        "challenger_site": [9, 12],
        "sparse_gap": -0.01275005361387142,
        "structural_bound": 0.03874403301354687,
        "empirical_inference_allowance": 0.023528500294517668,
        "empirical_total_envelope": 0.049522479694193114,
        "stopping_tolerance": 0.06,
        "active_reference_is_ess_fraction": 0.843854775843621,
        "full_reference_is_ess_fraction_rounded": 0.062,
        "full_laplace_is_ess_fraction_rounded": 0.589,
        "observed_grid_ei_regret": 0.0,
    },
    "regression_targets": {
        "minimum_rho": 4.633333333333334,
        "maximum_kappa": 0.8666666666666667,
        "minimum_row_margin": 2.2130666666666667,
        "lambda_min_A": 2.236679651060477,
        "lambda_max_A": 9.120457035761966,
        "condition_number_A": 4.077676940208081,
        "maximum_nonzeros_per_row": 13,
        "matrix_max_abs_tolerance": 2e-15,
        "structural_value_abs_tolerance": 2e-14,
    },
}


def literal_archived_notebook_construction() -> tuple[np.ndarray, np.ndarray]:
    """Reproduce the prototype's dense ``Q`` and ``A`` construction literally."""

    model = FROZEN_CONFIG["model"]
    n = int(model["grid_size"])
    n_sites = n * n
    q0 = float(model["q0"])
    q_laplacian = float(model["q_laplacian"])
    coupling = float(model["coupling"])
    nonlinearity = float(model["nonlinearity"])
    gamma = float(model["gamma"])
    tau = float(model["tau"])

    precision = np.zeros((n_sites, n_sites), dtype=float)
    degree = np.zeros(n_sites, dtype=float)
    index = lambda row, column: row * n + column

    for row in range(n):
        for column in range(n):
            site = index(row, column)
            for row_step, column_step in ((1, 0), (0, 1)):
                neighbor_row = row + row_step
                neighbor_column = column + column_step
                if neighbor_row < n and neighbor_column < n:
                    neighbor = index(neighbor_row, neighbor_column)
                    degree[site] += 1.0
                    degree[neighbor] += 1.0
    np.fill_diagonal(precision, q0 + q_laplacian * degree)

    for row in range(n):
        for column in range(n):
            site = index(row, column)
            for row_step, column_step in ((1, 0), (0, 1)):
                neighbor_row = row + row_step
                neighbor_column = column + column_step
                if neighbor_row < n and neighbor_column < n:
                    neighbor = index(neighbor_row, neighbor_column)
                    precision[site, neighbor] -= q_laplacian
                    precision[neighbor, site] -= q_laplacian

    supports: list[np.ndarray] = []
    for row in range(n):
        for column in range(n):
            center = index(row, column)
            neighbors = []
            for neighbor_row, neighbor_column in (
                (row - 1, column),
                (row + 1, column),
                (row, column - 1),
                (row, column + 1),
            ):
                if 0 <= neighbor_row < n and 0 <= neighbor_column < n:
                    neighbors.append(index(neighbor_row, neighbor_column))
            supports.append(np.asarray([center, *neighbors], dtype=int))

    rho = np.diag(precision).copy() - gamma * nonlinearity / tau
    kappa = np.abs(precision.copy())
    np.fill_diagonal(kappa, 0.0)
    for support in supports:
        derivative_bound = np.full(support.size, coupling, dtype=float)
        derivative_bound[0] = 1.0 + nonlinearity
        for local_row, site in enumerate(support):
            for local_column, neighbor in enumerate(support):
                if site != neighbor:
                    kappa[site, neighbor] += (
                        gamma
                        / tau**2
                        * derivative_bound[local_row]
                        * derivative_bound[local_column]
                    )
    return precision, np.diag(rho) - kappa


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


def run() -> dict[str, Any]:
    start = time.perf_counter()
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    model = FROZEN_CONFIG["model"]
    locked = FROZEN_CONFIG["locked_replay"]
    targets = FROZEN_CONFIG["regression_targets"]
    parameters = DEFAULT_PARAMETERS
    if {
        "q0": parameters.q0,
        "q_laplacian": parameters.q_laplacian,
        "coupling": parameters.coupling,
        "nonlinearity": parameters.nonlinearity,
        "gamma": parameters.gamma,
        "tau": parameters.tau,
    } != {key: model[key] for key in (
        "q0",
        "q_laplacian",
        "coupling",
        "nonlinearity",
        "gamma",
        "tau",
    )}:
        raise RuntimeError("frozen model and implementation defaults disagree")

    prototype_hash = hashlib.sha256(PROTOTYPE_PATH.read_bytes()).hexdigest()
    if prototype_hash != FROZEN_CONFIG["provenance"]["prototype_sha256"]:
        raise RuntimeError("archived nonlinear-PDE notebook hash changed")

    grid_size = int(model["grid_size"])
    precision, derivative_bounds, rho, kappa, matrix = (
        build_nonlinear_pde_comparison(grid_size, parameters)
    )
    archived_precision, archived_matrix = literal_archived_notebook_construction()
    clean_matrix = matrix.toarray()
    precision_difference = float(
        np.max(np.abs(precision.toarray() - archived_precision))
    )
    matrix_difference = float(np.max(np.abs(clean_matrix - archived_matrix)))

    minimum_eigenvalue, maximum_eigenvalue = extreme_eigenvalues(matrix)
    margins = row_dominance_margins(matrix)
    diagnostics = {
        "minimum_rho": float(rho.min()),
        "maximum_kappa": float(kappa.data.max()),
        "minimum_row_dominance_margin": float(margins.min()),
        "lambda_min_A": minimum_eigenvalue,
        "lambda_max_A": maximum_eigenvalue,
        "condition_number_A": maximum_eigenvalue / minimum_eigenvalue,
        "maximum_nonzeros_per_row": maximum_nonzeros_per_row(matrix),
        "diagonal_dominance_q0_threshold": diagonal_dominance_threshold(parameters),
        "analytic_interior_row_margin": analytic_interior_row_margin(parameters),
    }

    n_sites = grid_size * grid_size
    active = np.zeros(n_sites, dtype=bool)
    active[np.asarray(locked["active_factor_indices"], dtype=int)] = True
    omitted = ~active
    leader_row, leader_column = locked["leader_site"]
    challenger_row, challenger_column = locked["challenger_site"]
    leader = int(leader_row * grid_size + leader_column)
    challenger = int(challenger_row * grid_size + challenger_column)
    structural_value = structural_screening_bound(
        matrix,
        derivative_bounds,
        omitted,
        challenger,
        leader,
        parameters.gamma,
        parameters.tau,
    )
    load = omitted_factor_load(
        derivative_bounds, omitted, parameters.gamma, parameters.tau
    )
    archived_solution = np.linalg.solve(archived_matrix, load)
    archived_structural_value = float(
        archived_solution[challenger] + archived_solution[leader]
    )
    sparse_residual = float(
        np.linalg.norm(matrix @ solve_comparison(matrix, load) - load)
    )

    checks = {
        "prototype_hash_matches": True,
        "precision_matches_archived": precision_difference
        <= targets["matrix_max_abs_tolerance"],
        "comparison_matrix_matches_archived": matrix_difference
        <= targets["matrix_max_abs_tolerance"],
        "structural_value_matches_locked_replay": abs(
            structural_value - locked["structural_bound"]
        )
        <= targets["structural_value_abs_tolerance"],
        "archived_and_clean_structural_values_match": abs(
            structural_value - archived_structural_value
        )
        <= targets["structural_value_abs_tolerance"],
        "strict_row_diagonal_dominance": bool(np.all(margins > 0.0)),
        "positive_definite": minimum_eigenvalue > 0.0,
        "no_factor_energies_evaluated": True,
    }
    scalar_targets = {
        "minimum_rho": diagnostics["minimum_rho"],
        "maximum_kappa": diagnostics["maximum_kappa"],
        "minimum_row_margin": diagnostics["minimum_row_dominance_margin"],
        "lambda_min_A": diagnostics["lambda_min_A"],
        "lambda_max_A": diagnostics["lambda_max_A"],
        "condition_number_A": diagnostics["condition_number_A"],
        "maximum_nonzeros_per_row": diagnostics["maximum_nonzeros_per_row"],
    }
    for key, actual in scalar_targets.items():
        expected = targets[key]
        if isinstance(expected, int):
            if actual != expected:
                raise RuntimeError(f"regression mismatch for {key}: {actual} != {expected}")
        elif not np.isclose(actual, expected, rtol=2e-10, atol=2e-12):
            raise RuntimeError(f"regression mismatch for {key}: {actual} != {expected}")
    if not all(checks.values()):
        raise RuntimeError(f"structural validation failed: {checks}")

    commit, worktree_dirty = _git_state()
    config_text = json.dumps(FROZEN_CONFIG, indent=2, sort_keys=True) + "\n"
    config_path = OUTPUT_DIRECTORY / "frozen_config.json"
    config_path.write_text(config_text)
    summary = {
        "verdict": "PASS",
        "analytic_status": "NONLINEAR_PDE_T2B_PROVED",
        "t4_status": "PROVED_FOR_THIS_FAMILY",
        "end_to_end_finite_sample_certificate": False,
        "inference_error_certification": "OPEN_BLOCKER",
        "git": {"commit": commit, "worktree_dirty_during_run": worktree_dirty},
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "prototype_sha256": prototype_hash,
        "analytic_constants": {
            "center_derivative_bound": parameters.center_derivative_bound,
            "outer_curvature_scale_gamma_over_tau_squared": parameters.outer_curvature_scale,
            "factor_gradient_scale_gamma_over_tau": parameters.gradient_scale,
            "worst_case_negative_curvature_gamma_eta_over_tau": parameters.negative_curvature,
        },
        "matrix_diagnostics": diagnostics,
        "notebook_comparison": {
            "precision_max_abs_difference": precision_difference,
            "comparison_matrix_max_abs_difference": matrix_difference,
            "clean_equals_archived_to_tolerance": checks[
                "comparison_matrix_matches_archived"
            ],
            "rigorous_correction_factor": 1.0,
            "computed_floating_ratio": structural_value
            / archived_structural_value,
        },
        "structural_replay": {
            "active_factors": int(active.sum()),
            "total_factors": n_sites,
            "leader_site": locked["leader_site"],
            "challenger_site": locked["challenger_site"],
            "clean_structural_value": structural_value,
            "archived_structural_value": archived_structural_value,
            "locked_structural_value": locked["structural_bound"],
            "sparse_solve_residual_l2": sparse_residual,
        },
        "locked_empirical_provenance": {
            "sparse_gap": locked["sparse_gap"],
            "empirical_inference_allowance": locked[
                "empirical_inference_allowance"
            ],
            "empirical_total_envelope": locked["empirical_total_envelope"],
            "stopping_tolerance": locked["stopping_tolerance"],
            "classification": "EMPIRICAL_NOT_A_FINITE_SAMPLE_CERTIFICATE",
        },
        "checks": checks,
        "runtime": {
            "wall_seconds": time.perf_counter() - start,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
    }
    (OUTPUT_DIRECTORY / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    results = f"""# T2-B Nonlinear-PDE Structural Validation

Status: **PASS**

This artifact validates the accepted nonlinear-PDE comparison construction. It
does not evaluate factor energies, perform posterior inference, or establish a
finite-sample end-to-end action certificate.

## Matrix regression

- minimum `rho`: `{diagnostics['minimum_rho']:.10f}`;
- maximum `kappa`: `{diagnostics['maximum_kappa']:.10f}`;
- minimum row-dominance margin: `{diagnostics['minimum_row_dominance_margin']:.10f}`;
- `lambda_min(A)`: `{diagnostics['lambda_min_A']:.10f}`;
- `lambda_max(A)`: `{diagnostics['lambda_max_A']:.10f}`;
- `cond_2(A)`: `{diagnostics['condition_number_A']:.10f}`;
- maximum nonzeros per row: `{diagnostics['maximum_nonzeros_per_row']}`;
- maximum absolute clean/notebook matrix difference: `{matrix_difference:.3e}`.

## Locked structural replay

- active factors: `{int(active.sum())}/{n_sites}`;
- leader/challenger: `{tuple(locked['leader_site'])}` / `{tuple(locked['challenger_site'])}`;
- structural value: `{structural_value:.17f}`;
- literal notebook-construction value: `{archived_structural_value:.17f}`;
- locked replay value: `{locked['structural_bound']:.17f}`;
- rigorous correction factor: `{structural_value / archived_structural_value:.1f}`.

The theorem-backed structural term is separate from the archived empirical
inference allowance `{locked['empirical_inference_allowance']:.10f}` and total
envelope `{locked['empirical_total_envelope']:.10f}`. Inference-error
certification remains open.
"""
    (OUTPUT_DIRECTORY / "RESULTS.md").write_text(results)
    print(
        "[PDE T2-B] "
        f"min_rho={diagnostics['minimum_rho']:.10f} "
        f"max_kappa={diagnostics['maximum_kappa']:.10f}"
    )
    print(
        "[PDE T2-B] "
        f"row_margin={diagnostics['minimum_row_dominance_margin']:.10f} "
        f"lambda_min={diagnostics['lambda_min_A']:.10f}"
    )
    print(
        "[PDE T2-B] "
        f"A_clean_minus_A_notebook={matrix_difference:.3e} "
        f"structural={structural_value:.17f}"
    )
    print("[RESULT] nonlinear-PDE T2-B: PASS")
    return summary


if __name__ == "__main__":
    run()
