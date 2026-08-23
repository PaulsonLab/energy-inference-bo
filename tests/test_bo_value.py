from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
from scipy import sparse

from conditioned_bo.bo_value import (
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
    simple_regret_trajectory,
    snis_expected_improvement,
    snis_pairwise_gap_standard_error,
    stable_self_normalized_weights,
    weighted_logistic_energy_gradient,
    weighted_logistic_energy_samples,
    weighted_logistic_hessian_dense,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/sun_oxide/configs/bo_value_pilot.json"
DRIVER = ROOT / "experiments/sun_oxide/bo_value_pilot.py"
NOTEBOOK = ROOT / "experiments/sun_oxide/colab_bo_value_pilot.ipynb"


def _finite_gradient(function, value: np.ndarray, step: float = 1e-6) -> np.ndarray:
    result = np.empty_like(value)
    for coordinate in range(value.size):
        delta = np.zeros_like(value)
        delta[coordinate] = step
        result[coordinate] = (function(value + delta) - function(value - delta)) / (
            2.0 * step
        )
    return result


def _finite_hessian(function, value: np.ndarray, step: float = 2e-5) -> np.ndarray:
    result = np.empty((value.size, value.size), dtype=np.float64)
    for coordinate in range(value.size):
        delta = np.zeros_like(value)
        delta[coordinate] = step
        result[:, coordinate] = (function(value + delta) - function(value - delta)) / (
            2.0 * step
        )
    return result


def test_fixed_seed_initialization_matches_frozen_numpy_protocol() -> None:
    first = fixed_initial_positions(191, 8, 0)
    second = fixed_initial_positions(191, 8, 0)
    np.testing.assert_array_equal(first, [156, 14, 3, 95, 57, 50, 7, 117])
    np.testing.assert_array_equal(first, second)
    assert len(set(first.tolist())) == 8


def test_initial_target_scaling_is_frozen_and_has_quarter_ev_floor() -> None:
    initial = np.asarray([1.00, 1.10, 0.90, 1.05, 0.95, 1.00, 1.02, 0.98])
    scale = freeze_target_scale(initial)
    assert scale.mean_ev == np.mean(initial)
    assert scale.scale_ev == 0.25
    before = scale.standardize([1.5])
    _later_observation_that_must_not_refit = 100.0
    after = scale.standardize([1.5])
    np.testing.assert_array_equal(before, after)


def test_config_forbids_all_online_hyperparameter_fitting() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["gaussian_reference"]["hyperparameter_optimization_during_bo"] is False
    assert config["target_scaling"]["frozen_across_methods_and_seed"] is True
    assert config["factor_bank"]["weight_exact"] == "1/499"
    assert config["gaussian_reference"]["observation_noise_standardized"] == 0.05
    assert config["tuning_after_gw_results"] is False
    assert set(config["bo"]["methods"]) == {"NO_PBE", "FULL_PBE"}


def test_exact_gaussian_reference_update_matches_dense_formula() -> None:
    q0 = sparse.csr_matrix(
        [[2.0, -0.3, 0.0], [-0.3, 1.8, -0.2], [0.0, -0.2, 1.4]]
    )
    nodes = np.asarray([0, 2])
    z = np.asarray([0.4, -0.7])
    state = construct_gaussian_reference(q0, nodes, z, sigma_obs=0.5)
    expected_q = q0.toarray() + np.diag([4.0, 0.0, 4.0])
    expected_h = np.asarray([1.6, 0.0, -2.8])
    np.testing.assert_allclose(state.precision.toarray(), expected_q)
    np.testing.assert_allclose(state.information, expected_h)
    np.testing.assert_allclose(state.mean, np.linalg.solve(expected_q, expected_h))
    assert state.diagnostics["posterior_mean_solve_relative_residual"] <= 1e-9


def test_exact_marginal_equals_inverse_and_schur_complement() -> None:
    q = np.asarray(
        [
            [2.4, -0.6, -0.2, 0.0],
            [-0.6, 2.1, -0.4, -0.1],
            [-0.2, -0.4, 1.9, -0.5],
            [0.0, -0.1, -0.5, 1.7],
        ]
    )
    state = construct_gaussian_reference(sparse.csr_matrix(q), [], [], sigma_obs=0.05)
    support = np.asarray([0, 2])
    marginal = exact_support_marginal(state, support)
    expected_covariance = np.linalg.inv(q)[np.ix_(support, support)]
    expected_precision = np.linalg.inv(expected_covariance)
    np.testing.assert_allclose(marginal.covariance, expected_covariance, atol=1e-13)
    np.testing.assert_allclose(marginal.precision, expected_precision, atol=1e-13)
    np.testing.assert_allclose(
        marginal.precision, schur_complement_precision(q, support), atol=1e-13
    )
    assert marginal.diagnostics["principal_precision_submatrix_used"] is False


def test_principal_precision_shortcut_explicitly_fails_on_coupled_fixture() -> None:
    q = np.asarray(
        [[2.0, -0.8, 0.0], [-0.8, 1.7, -0.6], [0.0, -0.6, 1.5]]
    )
    support = np.asarray([0, 2])
    exact = schur_complement_precision(q, support)
    shortcut = q[np.ix_(support, support)]
    assert np.max(np.abs(exact - shortcut)) > 0.1


def test_weighted_logistic_energy_gradient_and_hessian() -> None:
    pairs = np.asarray([[0, 1], [0, 2], [1, 2]], dtype=np.int64)
    signs = np.asarray([1, -1, 1], dtype=np.int8)
    value = np.asarray([0.3, -0.8, 0.4])
    energy, gradient = weighted_logistic_energy_gradient(
        value, pairs, signs, chunk_size=2
    )
    hessian = weighted_logistic_hessian_dense(value, pairs, signs, chunk_size=2)
    finite_gradient = _finite_gradient(
        lambda point: weighted_logistic_energy_gradient(
            point, pairs, signs, chunk_size=2
        )[0],
        value,
    )
    finite_hessian = _finite_hessian(
        lambda point: weighted_logistic_energy_gradient(
            point, pairs, signs, chunk_size=2
        )[1],
        value,
    )
    assert np.isfinite(energy)
    np.testing.assert_allclose(gradient, finite_gradient, rtol=1e-8, atol=1e-11)
    np.testing.assert_allclose(hessian, finite_hessian, rtol=1e-8, atol=1e-11)
    assert np.linalg.eigvalsh(hessian).min() >= -1e-14


def test_strongly_convex_map_laplace_covariance_and_warm_start() -> None:
    precision = np.asarray(
        [[1.7, -0.2, 0.0], [-0.2, 1.6, -0.1], [0.0, -0.1, 1.5]]
    )
    mean = np.asarray([0.2, -0.1, 0.4])
    pairs = np.asarray([[0, 1], [1, 2]], dtype=np.int64)
    signs = np.asarray([1, -1], dtype=np.int8)
    first_bank = TimedFactorBank(pairs, signs, dimension=3, chunk_size=1)
    first = fit_laplace_approximation(mean, precision, first_bank, np.zeros(3))
    assert first.diagnostics["optimizer_success"]
    assert first.diagnostics["gradient_infinity_norm"] <= 1e-5
    np.testing.assert_allclose(
        first.covariance, np.linalg.inv(first.precision), rtol=1e-12, atol=1e-12
    )
    assert np.linalg.eigvalsh(first.precision).min() > 0.0

    second_bank = TimedFactorBank(pairs, signs, dimension=3, chunk_size=1)
    second = fit_laplace_approximation(mean + 0.01, precision, second_bank, first.map)
    assert second.diagnostics["optimizer_success"]
    assert len(second.diagnostics["optimizer_attempts"]) == 1
    samples = draw_laplace_samples(second, 12, np.random.default_rng(77))
    assert samples.shape == (12, 3)
    assert np.all(np.isfinite(samples))


def test_first_state_uses_one_deterministic_retry_without_model_change(
    monkeypatch,
) -> None:
    calls: list[np.ndarray] = []

    def fake_minimize(function, start, **kwargs):
        del function, kwargs
        calls.append(np.asarray(start, dtype=np.float64).copy())
        candidate = np.asarray([2e-5, 0.0]) if len(calls) == 1 else np.zeros(2)
        return SimpleNamespace(
            x=candidate,
            success=True,
            status=0,
            message="fixture",
            nit=1,
            nfev=1,
        )

    monkeypatch.setattr("conditioned_bo.bo_value.minimize", fake_minimize)
    bank = TimedFactorBank(
        np.empty((0, 2), dtype=np.int64),
        np.empty(0, dtype=np.int8),
        dimension=2,
    )
    state = fit_laplace_approximation(
        np.zeros(2), np.eye(2), bank, np.zeros(2), gradient_tolerance=1e-5
    )
    assert len(calls) == 2
    np.testing.assert_array_equal(calls[0], np.zeros(2))
    np.testing.assert_array_equal(calls[1], [2e-5, 0.0])
    assert [
        attempt["start_source"] for attempt in state.diagnostics["optimizer_attempts"]
    ] == ["initial_map", "first_attempt_candidate"]
    assert state.diagnostics["gradient_infinity_norm"] == 0.0


def test_gaussian_ei_matches_formula_and_zero_variance_limit() -> None:
    means = np.asarray([0.2, -0.3, 0.8])
    variances = np.asarray([0.0, 0.25, 1.0])
    incumbent = 0.1
    observed = gaussian_expected_improvement(means, variances, incumbent)
    assert observed[0] == 0.1
    gamma = (means[1:] - incumbent) / np.sqrt(variances[1:])
    from scipy.stats import norm

    expected = (means[1:] - incumbent) * norm.cdf(gamma) + np.sqrt(
        variances[1:]
    ) * norm.pdf(gamma)
    np.testing.assert_allclose(observed[1:], expected)


def test_importance_weight_formula_and_chunked_factor_energy() -> None:
    precision = np.asarray([[1.8, -0.2], [-0.2, 1.4]])
    mean = np.asarray([0.1, -0.2])
    pairs = np.asarray([[0, 1]], dtype=np.int64)
    signs = np.asarray([1], dtype=np.int8)
    bank = TimedFactorBank(pairs, signs, dimension=2)
    state = fit_laplace_approximation(mean, precision, bank, np.zeros(2))
    samples = np.asarray([[0.0, 0.0], [0.4, -0.7], [-0.2, 0.3]])
    log_weights, diagnostics = laplace_log_importance_weights(
        samples,
        mean,
        precision,
        state,
        pairs,
        signs,
        sample_chunk_size=2,
        factor_chunk_size=1,
    )
    factor_energy = weighted_logistic_energy_samples(
        samples, pairs, signs, sample_chunk_size=2, factor_chunk_size=1
    )
    target_delta = samples - mean
    proposal_delta = samples - state.map
    expected = (
        -0.5 * np.einsum("bi,ij,bj->b", target_delta, precision, target_delta)
        - factor_energy
        + 0.5
        * np.einsum(
            "bi,ij,bj->b", proposal_delta, state.precision, proposal_delta
        )
    )
    np.testing.assert_allclose(log_weights, expected)
    assert diagnostics["full_samples_by_factors_materialized"] is False
    assert diagnostics["maximum_materialized_factor_sample_shape"] == [2, 1]


def test_ess_snis_ei_and_pairwise_gap_standard_error() -> None:
    normalized, ess = stable_self_normalized_weights([0.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(normalized, np.full(4, 0.25))
    assert ess == 4.0
    action_samples = np.asarray(
        [[0.0, 0.5], [1.0, 0.0], [2.0, 1.0], [0.5, 2.0]]
    )
    ei = snis_expected_improvement(action_samples, normalized, incumbent=0.5)
    np.testing.assert_allclose(ei, [0.5, 0.5])
    gaps = np.maximum(action_samples[:, 1] - 0.5, 0.0) - np.maximum(
        action_samples[:, 0] - 0.5, 0.0
    )
    estimate, standard_error = snis_pairwise_gap_standard_error(gaps, normalized)
    assert estimate == 0.0
    assert standard_error == np.sqrt(np.sum((0.25**2) * gaps**2))


def test_observed_exclusion_and_exact_tie_breaking_by_stable_key() -> None:
    acquisition = np.asarray([10.0, 4.0, 4.0, 1.0])
    keys = ["observed", "key-z", "key-a", "key-b"]
    selected = select_unobserved_action(acquisition, [0], keys)
    assert selected == 2


def test_regret_and_aurc_metrics_use_raw_ev_post_query_values() -> None:
    regrets = simple_regret_trajectory(
        sequential_values_ev=[1.5, 2.4, 2.0],
        initial_values_ev=[0.5, 1.0],
        oracle_maximum_ev=2.5,
    )
    np.testing.assert_allclose(regrets, [1.0, 0.1, 0.1])
    np.testing.assert_allclose(aurc(regrets), 1.2)


def test_oracle_access_guard_blocks_full_values_until_post_run() -> None:
    oracle = RetrospectiveOracle(["a", "b", "c"], [1.0, 2.0, 3.0])
    assert oracle.query(1, seed=0, method="NO_PBE", stage="initial") == 2.0
    try:
        oracle.evaluation_values()
    except PermissionError:
        pass
    else:
        raise AssertionError("Full oracle evaluation was available during BO")
    oracle.unlock_post_run_evaluation()
    np.testing.assert_array_equal(oracle.evaluation_values(), [1.0, 2.0, 3.0])
    assert [entry["access"] for entry in oracle.access_log] == [
        "queried_action",
        "post_run_evaluation_unlocked",
        "full_oracle_evaluation",
    ]


def test_driver_smoke_is_explicitly_oracle_isolated_before_run() -> None:
    source = DRIVER.read_text(encoding="utf-8")
    smoke_start = source.index("def scientific_smoke(")
    oracle_loader_start = source.index("def _load_oracle_after_smoke(")
    smoke_source = source[smoke_start:oracle_loader_start]
    assert "include_oracle=False" in smoke_source
    assert "gw_band_gap_ev" not in smoke_source
    run_start = source.index("def run_pilot(")
    run_source = source[run_start:]
    assert run_source.index("scientific_smoke(") < run_source.index(
        "_load_oracle_after_smoke("
    )


def test_driver_overrides_colab_inline_backend_before_matplotlib_import() -> None:
    source = DRIVER.read_text(encoding="utf-8")
    assignment = 'os.environ["MPLBACKEND"] = "Agg"'
    assert source.index(assignment) < source.index("import matplotlib")
    environment = os.environ.copy()
    environment["MPLBACKEND"] = "module://matplotlib_inline.backend_inline"
    completed = subprocess.run(
        [sys.executable, str(DRIVER), "--help"],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr


def test_colab_notebook_reuses_frozen_cpu_bootstrap_and_terminal_contract() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    assert "git', 'ls-remote'" in source
    assert "git', 'checkout', '--detach', RUN_SHA" in source
    assert "UV_BOOTSTRAP_VERSION = '0.10.11'" in source
    assert "PYTHON_VERSION = '3.12.13'" in source
    assert "--require-hashes" in source and "--no-deps" in source
    assert "pip', 'check'" in source
    assert "'smoke'" in source and "'run'" in source
    assert "sun_oxide_bo_value_pilot_outputs.zip" in source
    for terminal in (
        "PASS_PBE_VALUE_COLAB",
        "FAIL_PBE_VALUE_COLAB",
        "LAPLACE_VALIDATION_BLOCKED",
        "LAPLACE_VALIDATION_FAILED",
        "NUMERICAL_FAILURE_COLAB",
        "INSTALLATION_BLOCKED",
    ):
        assert terminal in source
