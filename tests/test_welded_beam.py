import copy
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from botorch.test_functions.synthetic import WeldedBeamSO
from scipy.integrate import quad

from decision_tilt.welded_beam import (
    OutputTransform,
    bootstrap_mean_interval,
    build_candidate_set,
    build_state,
    classify_result,
    exact_constrained_ei,
    fit_independent_gps,
    gaussian_improvement_log_moments,
    load_config,
    pairwise_ranking_disagreement,
    posterior_marginals,
    qmc_candidate_metrics,
    qmc_standard_normals,
)
from decision_tilt.welded_beam_experiment import _validate_top_moments


CONFIG = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "welded_beam_shift"
    / "config.json"
)


@pytest.fixture(scope="module")
def protocol() -> dict:
    return load_config(CONFIG)[0]


def test_protocol_is_frozen() -> None:
    protocol, digest = load_config(CONFIG)
    assert digest == "eed30cb385f2a49aea06f235593e590e056e2c49655950376988d9fbe109b777"
    assert protocol["states"]["seeds"] == [4101, 4102, 4103]
    assert protocol["qmc"]["sample_counts"] == [64, 128, 256, 512, 1024]


def test_welded_beam_sign_and_feasibility_conventions(protocol: dict) -> None:
    problem = WeldedBeamSO(dtype=torch.double)
    anchor = torch.tensor(
        [protocol["states"]["fixed_feasible_anchor_raw"]], dtype=torch.double
    )
    cost = problem.evaluate_true(anchor)
    slacks = problem.evaluate_slack(anchor, noise=False)
    assert float(cost) > 0.0
    assert torch.all(slacks > 0.0)
    assert bool(problem.is_feasible(anchor, noise=False))
    assert float(-cost) == pytest.approx(-3.4222075)


def test_state_and_candidate_construction_are_deterministic(protocol: dict) -> None:
    first = build_state(protocol, 4101)
    second = build_state(protocol, 4101)
    other = build_state(protocol, 4102)
    torch.testing.assert_close(first.train_x, second.train_x, atol=0.0, rtol=0.0)
    assert not torch.equal(first.train_x[1:], other.train_x[1:])
    assert first.feasible_count >= 1
    assert len(first.train_x) == 64
    candidates_a = build_candidate_set(protocol)
    candidates_b = build_candidate_set(protocol)
    torch.testing.assert_close(candidates_a, candidates_b, atol=0.0, rtol=0.0)
    assert candidates_a.shape == (16384, 4)


@pytest.mark.parametrize("z", [-12.0, -8.0, -3.0, 0.0, 2.5, 8.0])
def test_improvement_moments_match_adaptive_quadrature(z: float) -> None:
    mean = z
    variance = 1.0
    log_first, log_second = gaussian_improvement_log_moments(
        np.array([mean]), np.array([variance]), 0.0
    )
    normalizer = math.sqrt(2.0 * math.pi)
    lower = -z
    first = quad(
        lambda excess: excess
        * math.exp(-0.5 * (lower + excess) ** 2)
        / normalizer,
        0.0,
        np.inf,
        epsabs=1e-300,
        epsrel=1e-13,
    )[0]
    second = quad(
        lambda excess: excess**2
        * math.exp(-0.5 * (lower + excess) ** 2)
        / normalizer,
        0.0,
        np.inf,
        epsabs=1e-300,
        epsrel=1e-13,
    )[0]
    assert float(log_first[0]) == pytest.approx(math.log(first), abs=3e-9)
    assert float(log_second[0]) == pytest.approx(math.log(second), abs=3e-9)


def test_exact_constrained_ei_and_shift_identity(protocol: dict) -> None:
    means = np.array([[0.4] + [0.25] * 6])
    variances = np.array([[0.7**2] + [0.5**2] * 6])
    result = exact_constrained_ei(means, variances, 0.1, protocol)
    first = float(result["ei"][0])
    feasibility = float(result["feasibility"][0])
    acquisition = float(result["acquisition"][0])
    assert acquisition == pytest.approx(first * feasibility, rel=1e-14)
    ratio = math.exp(float(result["d2"][0]))
    assert float(result["chi_square"][0]) == pytest.approx(ratio - 1.0)
    assert float(result["ess_fraction"][0]) == pytest.approx(1.0 / ratio)
    generator = np.random.default_rng(91)
    samples = means + np.sqrt(variances) * generator.standard_normal((1_000_000, 1, 7))
    utility = np.maximum(samples[..., 0] - 0.1, 0.0)
    utility *= np.all(samples[..., 1:] >= 0.0, axis=-1)
    assert float(utility.mean()) == pytest.approx(acquisition, rel=1.5e-2)
    empirical_ess = float(utility.mean() ** 2 / np.mean(utility**2))
    assert empirical_ess == pytest.approx(float(result["ess_fraction"][0]), rel=3e-2)


def test_independent_top_moment_audit_does_not_double_count_feasibility(
    protocol: dict,
) -> None:
    means = np.array([[0.4] + [-0.5] * 6, [0.7] + [0.1] * 6])
    variances = np.array([[0.7**2] + [0.8**2] * 6, [0.4**2] + [0.6**2] * 6])
    exact = exact_constrained_ei(means, variances, 0.1, protocol)
    assert _validate_top_moments(means, variances, 0.1, exact) < 1e-10


def test_output_transform_and_small_gp_posterior(protocol: dict) -> None:
    transform = OutputTransform(center=2.0, scale=3.0)
    standardized = transform.standardize(torch.tensor([2.0, 5.0], dtype=torch.double))
    torch.testing.assert_close(standardized, torch.tensor([0.0, 1.0], dtype=torch.double))
    mean, variance = transform.untransform_moments(
        torch.tensor([0.0], dtype=torch.double),
        torch.tensor([0.25], dtype=torch.double),
    )
    torch.testing.assert_close(mean, torch.tensor([2.0], dtype=torch.double))
    torch.testing.assert_close(variance, torch.tensor([2.25], dtype=torch.double))

    reduced = copy.deepcopy(protocol)
    reduced["gp"]["maximum_iterations"] = 40
    state = build_state(reduced, 4101)
    # One output is sufficient to verify train-only scaling and raw-space recovery.
    state = replace(state, train_outputs=state.train_outputs[:, :1])
    fitted = fit_independent_gps(state, reduced)
    posterior_mean, posterior_variance = posterior_marginals(
        fitted, state.train_x[:3], reduced
    )
    assert posterior_mean.shape == posterior_variance.shape == (3, 1)
    assert np.isfinite(posterior_mean).all()
    assert np.all(posterior_variance > 0.0)
    np.testing.assert_allclose(
        posterior_mean[:, 0], state.train_outputs[:3, 0].numpy(), atol=1e-2, rtol=1e-2
    )


def test_qmc_is_reproducible_and_nested() -> None:
    first = qmc_standard_normals(1024, 7, 77)
    second = qmc_standard_normals(1024, 7, 77)
    other = qmc_standard_normals(1024, 7, 78)
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, other)
    np.testing.assert_array_equal(first[:256], qmc_standard_normals(256, 7, 77))


def test_qmc_converges_on_controlled_gaussian_candidates(protocol: dict) -> None:
    reduced = copy.deepcopy(protocol)
    reduced["qmc"]["sample_counts"] = [1024]
    reduced["qmc"]["scramble_repetitions"] = 16
    reduced["candidates"]["qmc_chunk_size"] = 3
    means = np.array(
        [
            [0.8] + [3.0] * 6,
            [0.3] + [3.0] * 6,
            [-0.2] + [3.0] * 6,
        ]
    )
    variances = np.full_like(means, 0.2**2)
    exact = exact_constrained_ei(means, variances, 0.0, reduced)
    rows = qmc_candidate_metrics(means, variances, 0.0, exact, reduced, 4101)
    assert np.mean([row["exact_best_selected"] for row in rows]) == 1.0
    assert np.mean([row["normalized_regret"] for row in rows]) < 1e-3


def test_pairwise_disagreement_counts_reversals_and_ties() -> None:
    exact = np.array([3.0, 2.0, 1.0])
    assert pairwise_ranking_disagreement(exact, exact) == 0.0
    assert pairwise_ranking_disagreement(exact, -exact) == 1.0
    assert pairwise_ranking_disagreement(exact, np.ones(3)) == 0.5


def test_bootstrap_interval_is_deterministic() -> None:
    values = np.linspace(0.0, 1.0, 64)
    first = bootstrap_mean_interval(
        values, repetitions=1000, seed=12, confidence_level=0.95
    )
    second = bootstrap_mean_interval(
        values, repetitions=1000, seed=12, confidence_level=0.95
    )
    assert first == second
    assert first[1] < first[0] < first[2]


def _gate_inputs(kind: str) -> tuple[list[dict], list[dict]]:
    if kind == "positive":
        ess, regret_low, pair_low, regret_up, pair_up = 0.1, 0.02, 0.12, 0.03, 0.15
    elif kind == "negative":
        ess, regret_low, pair_low, regret_up, pair_up = 0.3, 0.0, 0.0, 0.001, 0.01
    else:
        ess, regret_low, pair_low, regret_up, pair_up = 0.18, 0.005, 0.07, 0.01, 0.08
    states = [
        {"state_seed": seed, "top32_median_ess_fraction": ess}
        for seed in [4101, 4102, 4103]
    ]
    qmc = []
    for seed in [4101, 4102, 4103]:
        for count in [64, 128, 256, 512, 1024]:
            qmc.append(
                {
                    "state_seed": seed,
                    "sample_count": count,
                    "regret_lower_95": regret_low,
                    "regret_upper_95": regret_up,
                    "pairwise_lower_95": pair_low,
                    "pairwise_upper_95": pair_up,
                }
            )
    return states, qmc


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        ("positive", "WELDED_BEAM_SHIFT_POSITIVE_REVIEW_REQUIRED"),
        ("negative", "WELDED_BEAM_SHIFT_NEGATIVE_REVIEW_REQUIRED"),
        ("intermediate", "WELDED_BEAM_SHIFT_INCONCLUSIVE_REVIEW_REQUIRED"),
    ],
)
def test_gate_classification(protocol: dict, kind: str, expected: str) -> None:
    states, qmc = _gate_inputs(kind)
    assert classify_result(states, qmc, protocol, valid=True)["status"] == expected
    assert (
        classify_result(states, qmc, protocol, valid=False)["status"]
        == "WELDED_BEAM_SHIFT_INCONCLUSIVE_REVIEW_REQUIRED"
    )
