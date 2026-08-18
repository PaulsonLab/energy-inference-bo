import json
from pathlib import Path

import numpy as np
import pytest
import torch
from botorch.utils.objective import compute_smoothed_feasibility_indicator
from botorch.utils.safe_math import fatmax, log_fatplus
from scipy.stats import multivariate_t

from decision_tilt.constrained import (
    BeliefPair,
    atomic_save_npz,
    atomic_write_json,
    decision_shift_from_log_utility,
    evaluate_batches,
    fit_conjugate_scale_process,
    load_protocol,
    log_qlogei_utility,
    multivariate_t_log_density,
    sample_belief_pair,
    scrambled_sobol_uniforms,
)
from decision_tilt.constrained_experiment import ExecutionProfile, _state_summary


CONFIG = Path(__file__).resolve().parents[1] / "experiments" / "constrained_batch_shift" / "config.json"


@pytest.fixture(scope="module")
def protocol() -> dict:
    return load_protocol(CONFIG)[0]


@pytest.fixture(scope="module")
def belief_pair(protocol: dict) -> BeliefPair:
    engine = torch.quasirandom.SobolEngine(6, scramble=True, seed=71)
    x = engine.draw(12).to(torch.double)
    objective = torch.sin(2.0 * torch.pi * x[:, 0]) + 0.2 * x[:, 1]
    constraint = x.square().sum(-1) - 1.0
    return BeliefPair(
        fit_conjugate_scale_process(x, objective, protocol, maximum_iterations=30),
        fit_conjugate_scale_process(x, constraint, protocol, maximum_iterations=30),
    )


def test_frozen_protocol_contract_and_hash(protocol: dict) -> None:
    _, digest = load_protocol(CONFIG)
    assert len(digest) == 64
    assert protocol["states"]["seeds"] == list(range(3101, 3109))
    assert protocol["practical_qmc"]["sample_counts"] == [64, 128, 256, 512, 1024, 2048]
    assert protocol["gate"]["primary_practical_sample_count"] == 512
    assert protocol["compute"]["implementation_authorized"] is False


def test_multivariate_t_log_density_matches_scipy() -> None:
    location = torch.tensor([0.3, -0.2], dtype=torch.double)
    scale = torch.tensor([[1.2, 0.35], [0.35, 0.8]], dtype=torch.double)
    value = torch.tensor([0.7, -1.1], dtype=torch.double)
    actual = float(multivariate_t_log_density(value, location, scale, 7.0))
    expected = multivariate_t.logpdf(value.numpy(), loc=location.numpy(), shape=scale.numpy(), df=7.0)
    assert actual == pytest.approx(expected, abs=2e-12, rel=2e-12)


def test_matched_gaussian_and_student_covariances(
    belief_pair: BeliefPair, protocol: dict
) -> None:
    batches = torch.rand(3, 4, 6, dtype=torch.double)
    record = belief_pair.validate_match(batches, protocol["beliefs"]["required_mean_covariance_match_tolerance"])
    assert record["maximum_covariance_error"] < 1e-12


def test_student_t_qmc_samples_reproduce_moments(belief_pair: BeliefPair) -> None:
    batch = torch.rand(1, 4, 6, dtype=torch.double)
    base = scrambled_sobol_uniforms(16384, 10, 901)
    samples = sample_belief_pair(belief_pair, batch, base, "student_t")[:, 0, :, 0]
    moments = belief_pair.objective.predict(batch)
    np.testing.assert_allclose(samples.mean(0), moments.location[0], atol=4e-3, rtol=4e-3)
    empirical = torch.cov(samples.T)
    np.testing.assert_allclose(empirical, moments.covariance[0], atol=8e-3, rtol=4e-2)


def test_frozen_log_utility_matches_trusted_botorch_primitives(protocol: dict) -> None:
    samples = torch.tensor(
        [[[[0.2, -0.1], [0.7, 0.3], [-0.4, -0.5], [0.1, 0.0]]]],
        dtype=torch.double,
    )
    actual = log_qlogei_utility(samples, 0.15, protocol["acquisition"])
    objective = samples[..., 0]
    improvement = log_fatplus(objective - 0.15, tau=1e-6)
    feasibility = compute_smoothed_feasibility_indicator(
        [lambda values: values[..., 1]], samples, eta=1e-3, log=True, fat=True
    )
    expected = fatmax(improvement + feasibility, dim=-1, tau=1e-2)
    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)
    assert torch.isfinite(actual).all()


def test_decision_shift_quantities_from_known_weights() -> None:
    utility = torch.tensor([0.1, 0.2, 0.7, 1.5], dtype=torch.double)
    actual = decision_shift_from_log_utility(torch.log(utility))
    first = utility.mean()
    second = utility.square().mean()
    chi = second / first.square() - 1.0
    assert float(actual["acquisition"]) == pytest.approx(float(first))
    assert float(actual["chi_square"]) == pytest.approx(float(chi))
    assert float(actual["d2"]) == pytest.approx(float(torch.log1p(chi)))
    assert float(actual["ess_fraction"]) == pytest.approx(float(first.square() / second))


def test_acquisition_gradient_matches_central_difference(
    belief_pair: BeliefPair, protocol: dict
) -> None:
    batch = torch.rand(1, 4, 6, dtype=torch.double) * 0.5
    base = scrambled_sobol_uniforms(256, 10, 1234)
    analytic = evaluate_batches(
        belief_pair, batch, base, "student_t", 0.0, protocol["acquisition"]
    )["gradient"][0, 0, 0]
    step = 2e-5
    plus = batch.clone()
    minus = batch.clone()
    plus[0, 0, 0] += step
    minus[0, 0, 0] -= step
    value_plus = evaluate_batches(
        belief_pair, plus, base, "student_t", 0.0, protocol["acquisition"], with_gradients=False
    )["log_acquisition"][0]
    value_minus = evaluate_batches(
        belief_pair, minus, base, "student_t", 0.0, protocol["acquisition"], with_gradients=False
    )["log_acquisition"][0]
    finite = (value_plus - value_minus) / (2.0 * step)
    assert float(analytic) == pytest.approx(float(finite), abs=2e-5, rel=2e-4)


def test_qmc_reproducibility_and_scramble_difference() -> None:
    first = scrambled_sobol_uniforms(128, 10, 44)
    second = scrambled_sobol_uniforms(128, 10, 44)
    other = scrambled_sobol_uniforms(128, 10, 45)
    torch.testing.assert_close(first, second, atol=0.0, rtol=0.0)
    assert not torch.equal(first, other)


def test_atomic_checkpoint_round_trip(tmp_path: Path) -> None:
    metadata = tmp_path / "metadata.json"
    arrays = tmp_path / "arrays.npz"
    atomic_write_json(metadata, {"protocol_hash": "abc", "complete": True})
    atomic_save_npz(arrays, x=np.arange(8), y=np.eye(2))
    assert json.loads(metadata.read_text())["complete"] is True
    loaded = np.load(arrays)
    np.testing.assert_array_equal(loaded["x"], np.arange(8))
    np.testing.assert_array_equal(loaded["y"], np.eye(2))


def test_full_state_summary_ignores_provenance_columns(protocol: dict) -> None:
    reference = {
        "acquisition": np.linspace(0.1, 1.0, 40),
        "ess_fraction": np.full(40, 0.5),
    }
    practical = [
        {
            "sample_count": 512,
            "repetition": repetition,
            "belief": "gaussian",
            "seed": 3101,
            "median_high_relative_value_error": 0.01,
            "mean_high_pairwise_ranking_disagreement": 0.02,
            "high_kendall_tau": 0.95,
            "top10_overlap": 1.0,
            "median_high_gradient_cosine": 0.99,
            "median_high_gradient_relative_error": 0.03,
            "selected_panel_reference_regret": 0.01,
        }
        for repetition in range(2)
    ]
    optimizer = [
        {
            "sample_count": 512,
            "repetition": 0,
            "belief": "gaussian",
            "seed": 3101,
            "reference_regret": 0.01,
        }
    ]
    profile = ExecutionProfile(
        "test", 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, (1,), (512,), 2, 1, 1, 1, 1, False,
    )
    summary = _state_summary(
        "gaussian", reference, practical, optimizer, protocol, profile
    )
    assert summary["qmc_512"]["median_high_relative_value_error"] == pytest.approx(0.01)
    assert "belief" not in summary["qmc_512"]
    assert "seed" not in summary["qmc_512"]
