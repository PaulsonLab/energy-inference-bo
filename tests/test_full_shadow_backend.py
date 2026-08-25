from __future__ import annotations

import hashlib
import json
from pathlib import Path
import runpy

import numpy as np
import pytest
from scipy import sparse, stats

from conditioned_bo import full_shadow_backend as backend
from conditioned_bo import nonlinear_pde_locality as locality


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIRECTORY = (
    REPOSITORY_ROOT
    / "experiments/nonlinear_pde/outputs/full_shadow_backend_rescue"
)
CONFIG_PATH = OUTPUT_DIRECTORY / "backend_config.json"
EXPECTED_CONFIG_SHA256 = (
    "408614b2c67411a0ad0ac59e5dfb973767f11ae127e9eb96b7c3422aed396258"
)


@pytest.fixture(scope="module")
def tiny_full_context():
    problem = locality.build_problem(5, 2026082401, source_perturbation_scale=0.04)
    state = locality.build_common_bo_states(
        problem,
        initialization_size=2,
        total_queries=4,
        checkpoint_queries=(2, 3, 4),
        observation_noise_variance=0.0025,
        reference_sample_count=64,
        trajectory_seed=731,
        incumbent=0.55,
    )[0]
    context = backend.build_full_laplace_context(
        state,
        problem,
        gradient_tolerance=1e-8,
        maximum_iterations=12,
    )
    return problem, state, context


def _specs() -> tuple[backend.ProposalSpec, ...]:
    config = json.loads(CONFIG_PATH.read_text())
    return tuple(
        backend.ProposalSpec(
            name=item["name"],
            kind=item["kind"],
            curvature_lambda=item["curvature_lambda"],
            covariance_inflation=item["covariance_inflation"],
            tie_break_rank=rank,
        )
        for rank, item in enumerate(
            config["proposal_candidates_in_tie_break_order"]
        )
    )


def test_curvature_tempered_precision_is_exact_formula() -> None:
    reference = sparse.csr_matrix([[4.0, 0.5], [0.5, 3.0]])
    hessian = sparse.csr_matrix([[7.0, -0.25], [-0.25, 5.0]])
    observed = backend.curvature_tempered_precision(reference, hessian, 0.4)
    expected = reference.toarray() + 0.4 * (
        hessian.toarray() - reference.toarray()
    )
    np.testing.assert_array_equal(observed.toarray(), expected)


def test_every_candidate_precision_is_spd_in_representative_fixture(
    tiny_full_context,
) -> None:
    _, state, context = tiny_full_context
    for spec in _specs():
        proposal = backend.prepare_gaussian_proposal(
            spec,
            context.mode,
            state.reference_precision,
            context.target_hessian,
        )
        assert proposal.spd_check_passed
        assert proposal.minimum_cholesky_diagonal > 0.0
        assert np.linalg.eigvalsh(proposal.precision.toarray()).min() > 0.0


def test_exact_gaussian_proposal_log_density() -> None:
    mean = np.asarray([0.3, -0.7])
    precision = np.asarray([[3.0, 0.4], [0.4, 1.8]])
    samples = np.asarray([[0.2, -0.1], [1.0, -2.0], [-0.5, 0.4]])
    observed = backend.gaussian_log_density(samples, mean, precision)
    expected = stats.multivariate_normal(
        mean=mean, cov=np.linalg.inv(precision)
    ).logpdf(samples)
    np.testing.assert_allclose(observed, expected, rtol=2e-14, atol=2e-14)


def test_baseline_proposal_matches_existing_laplace_snis(tiny_full_context) -> None:
    problem, state, context = tiny_full_context
    spec = next(spec for spec in _specs() if spec.kind == "BASELINE_INFLATED_LAPLACE")
    proposal = backend.prepare_gaussian_proposal(
        spec,
        context.mode,
        state.reference_precision,
        context.target_hessian,
    )
    observed = backend.run_snis_batch(
        state,
        problem,
        proposal,
        incumbent=locality.state_incumbent(state),
        sample_count=64,
        proposal_seed=991,
    )
    expected = locality.laplace_snis_inference(
        state,
        problem,
        np.ones(25, dtype=bool),
        incumbent=locality.state_incumbent(state),
        delta_mc=0.05,
        sample_count=64,
        proposal_seed=991,
        proposal_inflation=1.1,
        work=locality.FactorWork(),
        gradient_tolerance=1e-8,
        maximum_iterations=12,
    )
    np.testing.assert_allclose(
        observed.acquisition, expected.acquisition, rtol=2e-14, atol=2e-14
    )
    assert observed.action_index == expected.leader_index
    assert observed.ess_fraction == pytest.approx(expected.ess_fraction, rel=2e-14)


def test_pilot_summary_discards_all_estimation_terms(tiny_full_context) -> None:
    problem, state, context = tiny_full_context
    proposal = backend.prepare_gaussian_proposal(
        _specs()[0],
        context.mode,
        state.reference_precision,
        context.target_hessian,
    )
    pilot = backend.run_snis_batch(
        state,
        problem,
        proposal,
        incumbent=locality.state_incumbent(state),
        sample_count=16,
        proposal_seed=12,
    )
    summary = backend.batch_public_summary(pilot, state)
    assert {
        "samples",
        "log_weights",
        "utility",
        "acquisition",
        "acquisition_vector",
    }.isdisjoint(summary)


def test_proposal_selection_has_deterministic_baseline_first_tie_breaking() -> None:
    specs = _specs()
    records = [
        {"proposal_name": spec.name, "ess_fraction": 0.4} for spec in specs
    ]
    assert backend.select_proposal_from_pilots(records, specs).name == specs[0].name
    records[-1]["ess_fraction"] = 0.5
    assert backend.select_proposal_from_pilots(records, specs).name == specs[-1].name


def test_backend_config_and_development_seed_isolation() -> None:
    assert hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest() == EXPECTED_CONFIG_SHA256
    runner = runpy.run_path(
        str(
            REPOSITORY_ROOT
            / "experiments/nonlinear_pde/run_full_shadow_backend_rescue.py"
        )
    )
    config = runner["_read_config"]()
    expected_validation = int.from_bytes(
        hashlib.sha256(b"E2_FULL_SHADOW_BACKEND_VALIDATE_V1").digest()[:8],
        "big",
    ) % (2**32 - 1)
    assert expected_validation == 3321078991
    assert runner["derive_validation_seed"]() == expected_validation
    prospective = set(config["development_sources"]["prospective_source_seeds_forbidden"])
    assert {2026082401, 3321078991}.isdisjoint(prospective)
    assert runner["_source_seed_for_role"]("calibration") == 2026082401
    assert runner["_source_seed_for_role"]("validation") == 3321078991
    with pytest.raises(ValueError):
        runner["_source_seed_for_role"]("scientific")


def test_validation_path_has_no_pilot_and_output_cannot_feed_selection() -> None:
    config = json.loads(CONFIG_PATH.read_text())
    assert config["held_out_validation"]["validation_pilots_run"] is False
    assert config["shadow_only_fairness"]["feeds_deployable_factor_selection"] is False
    assert backend.shadow_only_contract()["feeds_deployable_factor_selection"] is False
    deployable_source = (
        REPOSITORY_ROOT / "src/conditioned_bo/nonlinear_pde_locality.py"
    ).read_text()
    assert "full_shadow_backend" not in deployable_source


def test_prior_diagnostic_is_hash_locked_and_unchanged() -> None:
    config = json.loads(CONFIG_PATH.read_text())
    prior = (
        REPOSITORY_ROOT
        / "experiments/nonlinear_pde/outputs/full_shadow_reliability_diagnostic"
    )
    for filename, expected in config["prior_reference"]["committed_sha256"].items():
        assert hashlib.sha256((prior / filename).read_bytes()).hexdigest() == expected


def test_independence_mh_acceptance_ratio_matches_analytic_one_dimensional_target() -> None:
    target = stats.norm(loc=1.0, scale=0.7)
    proposal = stats.norm(loc=-0.3, scale=1.4)
    current = -0.2
    proposed = 0.9
    current_log_weight = target.logpdf(current) - proposal.logpdf(current)
    proposed_log_weight = target.logpdf(proposed) - proposal.logpdf(proposed)
    expected = min(0.0, proposed_log_weight - current_log_weight)
    observed = backend.independence_mh_log_acceptance(
        current_log_weight, proposed_log_weight
    )
    assert observed == pytest.approx(expected, rel=0.0, abs=1e-15)
    assert backend.independence_mh_accept(
        current_log_weight, proposed_log_weight, np.exp(expected) * 0.9
    )


def test_independence_mh_gaussian_target_fixture_has_known_moments() -> None:
    rng = np.random.default_rng(4401)
    proposals = rng.normal(loc=0.4, scale=1.3, size=50000)
    uniforms = rng.random(proposals.size)
    # Target and proposal are identical, so every exact independence-MH ratio
    # is one and the retained chain is iid from the known Gaussian target.
    accepted = [
        backend.independence_mh_accept(0.0, 0.0, float(uniform))
        for uniform in uniforms
    ]
    assert all(accepted)
    assert np.mean(proposals) == pytest.approx(0.4, abs=0.015)
    assert np.var(proposals) == pytest.approx(1.3**2, rel=0.025)


def test_rhat_and_autocorrelation_ess_on_iid_gaussian_chains() -> None:
    chains = np.random.default_rng(17).normal(size=(4, 5000))
    assert backend.split_rhat(chains) <= 1.01
    assert backend.autocorrelation_effective_sample_size(chains) >= 0.7 * chains.size


def test_split_rhat_detects_deliberately_shifted_chains() -> None:
    rng = np.random.default_rng(18)
    chains = rng.normal(size=(4, 3000))
    chains[2:] += 1.0
    assert backend.split_rhat(chains) > 1.1


def test_autocorrelation_ess_matches_known_ar1_scale() -> None:
    rng = np.random.default_rng(19)
    rho = 0.8
    chains = np.empty((4, 12000))
    for chain in chains:
        innovations = rng.normal(scale=np.sqrt(1.0 - rho**2), size=chain.size)
        chain[0] = innovations[0]
        for index in range(1, chain.size):
            chain[index] = rho * chain[index - 1] + innovations[index]
    observed = backend.autocorrelation_effective_sample_size(chains[:, 2000:])
    expected = chains[:, 2000:].size * (1.0 - rho) / (1.0 + rho)
    assert 0.65 * expected <= observed <= 1.5 * expected


def test_vectorized_full_factor_energy_matches_existing_exact_routine(
    tiny_full_context,
) -> None:
    problem, state, _ = tiny_full_context
    samples = state.reference_samples[:11]
    expected = locality.factor_energy_sum(
        samples, problem, np.arange(problem.grid_size**2)
    )
    observed = backend.full_factor_energy_vectorized(samples, problem)
    np.testing.assert_allclose(observed, expected, rtol=2e-14, atol=2e-14)


def test_elliptical_slice_uses_randomized_initial_brackets() -> None:
    current = np.asarray([0.2, -0.4])
    mean = np.zeros(2)
    direction = np.asarray([1.0, 0.3])
    angles = []
    for seed in range(8):
        _, _, evaluations, initial_angle = backend.elliptical_slice_transition(
            current,
            mean,
            direction,
            0.0,
            lambda _value: 0.0,
            np.random.default_rng(seed),
            maximum_bracket_evaluations=10,
        )
        assert evaluations == 1
        assert 0.0 <= initial_angle < 2.0 * np.pi
        angles.append(initial_angle)
    assert len(set(angles)) == len(angles)


def test_reference_directions_match_exact_state_gaussian_covariance(
    tiny_full_context,
) -> None:
    problem, state, _ = tiny_full_context
    sampler = backend.prepare_reference_direction_sampler(
        state, problem, observation_noise_variance=0.0025
    )
    rng = np.random.default_rng(611)
    draws = np.asarray(
        [backend.sample_reference_direction(sampler, rng) for _ in range(6000)]
    )
    expected = np.linalg.inv(state.reference_precision.toarray())
    np.testing.assert_allclose(np.mean(draws, axis=0), 0.0, atol=0.025)
    np.testing.assert_allclose(
        np.diag(np.cov(draws, rowvar=False)),
        np.diag(expected),
        rtol=0.09,
        atol=0.008,
    )


def test_elliptical_slice_invariance_on_gaussian_gaussian_toy() -> None:
    rng_direction = np.random.default_rng(7001)
    rng_slice = np.random.default_rng(7002)
    reference_mean = np.zeros(2)
    extra_precision = np.diag([0.5, 1.5])

    def log_likelihood(value):
        return -0.5 * float(value @ extra_precision @ value)

    current = np.zeros(2)
    current_log_likelihood = log_likelihood(current)
    retained = []
    for iteration in range(14000):
        direction = rng_direction.normal(size=2)
        current, current_log_likelihood, _, _ = backend.elliptical_slice_transition(
            current,
            reference_mean,
            direction,
            current_log_likelihood,
            log_likelihood,
            rng_slice,
            maximum_bracket_evaluations=100,
        )
        if iteration >= 2000:
            retained.append(current.copy())
    samples = np.asarray(retained)
    expected_covariance = np.linalg.inv(np.eye(2) + extra_precision)
    np.testing.assert_allclose(np.mean(samples, axis=0), 0.0, atol=0.035)
    np.testing.assert_allclose(
        np.cov(samples, rowvar=False), expected_covariance, rtol=0.08, atol=0.025
    )
