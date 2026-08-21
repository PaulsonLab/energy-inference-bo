from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import runpy

import numpy as np
import pytest
from scipy import sparse
from scipy.special import ndtr

from conditioned_bo.inference_certification import (
    CertificationBatchRegistry,
    ProposalCapExceeded,
    active_set_sha256,
    certify_symmetry_grid_round,
    ei_gap_q_lipschitz,
    inference_confidence_radius,
    make_batch_id,
    q_inverse_norm,
    rejection_sample_symmetry_target,
    symmetry_active_energy_batch,
)
from conditioned_bo.symmetry_influence import (
    conditional_expected_improvement,
    ei_action_coefficients,
    ei_block_decision_footprint,
    omitted_factor_load,
    ou_symmetry_comparison,
    reflection_blocks,
    solve_comparison,
    symmetry_logcosh_energy,
)


def _standard_normal_draw(dimension: int):
    def draw(rng: np.random.Generator, count: int) -> np.ndarray:
        return rng.standard_normal((count, dimension))

    return draw


def test_energy_sign_envelope_and_empty_target_acceptance() -> None:
    samples = np.array(
        [[-2.0, 1.5, -0.4, 0.8], [0.0, 0.0, 1.0, -1.0]], dtype=float
    )
    blocks = np.array([[0, 1], [2, 3]])
    energy = symmetry_active_energy_batch(samples, blocks, [0, 1], 0.05, 0.5)
    assert np.all(energy >= 0.0)
    weights = np.exp(-energy)
    assert np.all((weights > 0.0) & (weights <= 1.0))
    np.testing.assert_array_equal(
        symmetry_active_energy_batch(samples, blocks, [], 0.05, 0.5),
        np.zeros(samples.shape[0]),
    )

    result = rejection_sample_symmetry_target(
        rng=np.random.default_rng(9001),
        reference_draw=_standard_normal_draw(4),
        n_accepted=100,
        proposal_chunk_size=25,
        proposal_cap=100,
        blocks=blocks,
        active_factors=[],
        gamma=0.05,
        tau=0.5,
    )
    assert result.samples.shape == (100, 4)
    assert result.proposals_generated == 100
    assert result.proposals_consumed == 100
    assert result.accepted_candidates == 100
    assert result.acceptance_rate == 1.0


def test_stable_vectorized_logcosh_matches_scalar_and_extreme_inputs() -> None:
    differences = np.array([-1e300, -20.0, -1e-8, 0.0, 1e-8, 20.0, 1e300])
    samples = np.column_stack((np.zeros(differences.size), differences))
    blocks = np.array([[0, 1]])
    vectorized = symmetry_active_energy_batch(samples, blocks, [0], 0.05, 1.0)
    scalar = np.array(
        [symmetry_logcosh_energy(value, 0.05, 1.0) for value in samples]
    )
    assert np.all(np.isfinite(vectorized))
    np.testing.assert_allclose(vectorized, scalar, rtol=2e-15, atol=1e-16)


def test_low_dimensional_rejection_sampler_matches_quadrature_moments() -> None:
    blocks = np.array([[0, 1]])
    gamma = 0.25
    tau = 1.1
    nodes, weights_1d = np.polynomial.hermite.hermgauss(50)
    grid_left, grid_right = np.meshgrid(
        np.sqrt(2.0) * nodes, np.sqrt(2.0) * nodes, indexing="ij"
    )
    points = np.column_stack((grid_left.ravel(), grid_right.ravel()))
    reference_weights = np.outer(weights_1d, weights_1d).ravel() / np.pi
    target_weights = reference_weights * np.exp(
        -symmetry_active_energy_batch(points, blocks, [0], gamma, tau)
    )
    target_weights /= target_weights.sum()
    exact_mean = target_weights @ points
    exact_second = target_weights @ np.square(points)

    result = rejection_sample_symmetry_target(
        rng=np.random.default_rng(24680),
        reference_draw=_standard_normal_draw(2),
        n_accepted=120_000,
        proposal_chunk_size=4_000,
        proposal_cap=250_000,
        blocks=blocks,
        active_factors=[0],
        gamma=gamma,
        tau=tau,
    )
    np.testing.assert_allclose(result.samples.mean(axis=0), exact_mean, atol=0.012)
    np.testing.assert_allclose(
        np.square(result.samples).mean(axis=0), exact_second, atol=0.025
    )


def test_proposal_order_and_chunk_reproducibility() -> None:
    kwargs = {
        "reference_draw": _standard_normal_draw(4),
        "n_accepted": 2_000,
        "proposal_chunk_size": 257,
        "proposal_cap": 10_000,
        "blocks": np.array([[0, 1], [2, 3]]),
        "active_factors": [0, 1],
        "gamma": 0.1,
        "tau": 0.8,
    }
    first = rejection_sample_symmetry_target(
        rng=np.random.default_rng(112358), **kwargs
    )
    second = rejection_sample_symmetry_target(
        rng=np.random.default_rng(112358), **kwargs
    )
    np.testing.assert_array_equal(first.samples, second.samples)
    assert first.proposals_generated == second.proposals_generated
    assert first.proposals_consumed == second.proposals_consumed
    assert first.chunks == second.chunks


def test_proposal_cap_is_enforced() -> None:
    with pytest.raises(ProposalCapExceeded) as captured:
        rejection_sample_symmetry_target(
            rng=np.random.default_rng(7),
            reference_draw=_standard_normal_draw(2),
            n_accepted=1_000,
            proposal_chunk_size=100,
            proposal_cap=200,
            blocks=np.array([[0, 1]]),
            active_factors=[0],
            gamma=100.0,
            tau=0.01,
        )
    assert captured.value.proposals_generated == 200
    assert captured.value.accepted_samples < 1_000


def test_fresh_batch_bookkeeping_and_no_reuse() -> None:
    seed_root = np.random.SeedSequence(8675309)
    children = seed_root.spawn(5)
    states = []
    batch_ids = []
    for round_index, child in enumerate(children):
        state = {
            "entropy": int(child.state["entropy"]),
            "spawn_key": list(child.state["spawn_key"]),
            "pool_size": int(child.state["pool_size"]),
        }
        states.append(tuple(state["spawn_key"]))
        batch_ids.append(
            make_batch_id(
                config_sha256="a" * 64,
                round_index=round_index,
                child_seed_state=state,
                active_factors=list(range(round_index)),
                leader_index=round_index,
            )
        )
    assert len(set(states)) == len(states)
    assert len(set(batch_ids)) == len(batch_ids)

    registry = CertificationBatchRegistry()
    first = registry.claim(batch_ids[0], [], 3)
    assert first.active_set_hash == active_set_sha256([])
    with pytest.raises(RuntimeError, match="reuse is forbidden"):
        registry.claim(batch_ids[0], [], 3)
    with pytest.raises(RuntimeError, match="reuse is forbidden"):
        registry.claim(batch_ids[0], [0], 4)


def test_q_inverse_norm_sparse_dense_orientation_without_inverse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    precision = np.array([[3.0, -0.4], [-0.4, 1.7]])
    vector = np.array([0.8, -1.1])
    expected = np.sqrt(vector @ np.linalg.solve(precision, vector))

    def forbidden_inverse(*args, **kwargs):
        raise AssertionError("explicit inverse is forbidden")

    monkeypatch.setattr(np.linalg, "inv", forbidden_inverse)
    np.testing.assert_allclose(q_inverse_norm(precision, vector), expected)
    np.testing.assert_allclose(
        q_inverse_norm(sparse.csr_matrix(precision), vector), expected
    )


def test_ei_gap_q_lipschitz_finite_difference_and_self_zero() -> None:
    precision = np.array([[2.2, 0.3], [0.3, 1.4]])
    action_coefficients = np.array([0.7, -0.2])
    leader_coefficients = np.array([-0.1, 0.55])
    action_variance = 0.31
    leader_variance = 0.18
    incumbent = 0.25
    latent = np.array([0.4, -0.7])

    def gap(value: np.ndarray) -> float:
        return float(
            conditional_expected_improvement(
                action_coefficients @ value, action_variance, incumbent
            )
            - conditional_expected_improvement(
                leader_coefficients @ value, leader_variance, incumbent
            )
        )

    step = 1e-6
    finite_gradient = np.empty(2)
    for coordinate in range(2):
        perturbation = np.zeros(2)
        perturbation[coordinate] = step
        finite_gradient[coordinate] = (
            gap(latent + perturbation) - gap(latent - perturbation)
        ) / (2.0 * step)
    action_slope = ndtr(
        (action_coefficients @ latent - incumbent) / np.sqrt(action_variance)
    )
    leader_slope = ndtr(
        (leader_coefficients @ latent - incumbent) / np.sqrt(leader_variance)
    )
    analytic_gradient = (
        action_slope * action_coefficients - leader_slope * leader_coefficients
    )
    np.testing.assert_allclose(finite_gradient, analytic_gradient, rtol=2e-9)

    lipschitz = ei_gap_q_lipschitz(
        precision, action_coefficients, leader_coefficients
    )
    assert q_inverse_norm(precision, analytic_gradient) <= lipschitz + 1e-14
    assert (
        ei_gap_q_lipschitz(
            precision,
            action_coefficients,
            action_coefficients,
            self_comparison=True,
        )
        == 0.0
    )


def test_confidence_constant_and_two_sided_factor() -> None:
    locked = np.log(2.0 * 401 * 15 / 0.05)
    np.testing.assert_allclose(locked, 12.390891082522716, rtol=0.0, atol=1e-15)
    one_sided = np.log(401 * 15 / 0.05)
    assert locked > one_sided
    np.testing.assert_allclose(locked - one_sided, np.log(2.0))
    radius = inference_confidence_radius(1.0, 1_500_000, 401, 15, 0.05)
    np.testing.assert_allclose(radius, np.sqrt(2.0 * locked / 1_500_000))


def test_structural_result_is_unchanged() -> None:
    n_factors = 40
    spacing = 0.05
    lengthscale = 0.125
    gamma = 0.05
    tau = 0.5
    precision, blocks, _, _, comparison = ou_symmetry_comparison(
        n_factors, spacing, lengthscale
    )
    radii = (np.arange(n_factors) + 0.5) * spacing
    points = np.concatenate((-radii[::-1], radii))
    actions = np.linspace(-0.58, -0.06, 401)
    leader_index = 286
    challenger_index = 282
    leader, _ = ei_action_coefficients(
        float(actions[leader_index]), points, lengthscale, precision
    )
    challenger, _ = ei_action_coefficients(
        float(actions[challenger_index]), points, lengthscale, precision
    )
    footprint = ei_block_decision_footprint(challenger, leader, blocks)
    active = np.zeros(n_factors, dtype=bool)
    active[:15] = True
    transported = solve_comparison(
        comparison, omitted_factor_load(active, gamma, tau)
    )
    np.testing.assert_allclose(
        footprint @ transported,
        0.005339222052900117,
        rtol=2e-13,
        atol=2e-15,
    )


def _quadrature_ei_curve(
    action_means: np.ndarray,
    coefficients: np.ndarray,
    variances: np.ndarray,
    incumbent: float,
    gamma: float,
    tau: float,
) -> np.ndarray:
    nodes, one_dimensional_weights = np.polynomial.hermite.hermgauss(45)
    left, right = np.meshgrid(
        np.sqrt(2.0) * nodes, np.sqrt(2.0) * nodes, indexing="ij"
    )
    latent = np.column_stack((left.ravel(), right.ravel()))
    weights = np.outer(one_dimensional_weights, one_dimensional_weights).ravel()
    weights /= np.pi
    weights *= np.exp(
        -symmetry_active_energy_batch(
            latent, np.array([[0, 1]]), [0], gamma, tau
        )
    )
    weights /= weights.sum()
    curve = np.empty(action_means.size)
    for index in range(action_means.size):
        means = action_means[index] + latent @ coefficients[index]
        curve[index] = weights @ conditional_expected_improvement(
            means, float(variances[index]), incumbent
        )
    return curve


def test_synthetic_end_to_end_certificate_upper_bounds_exact_full_regret() -> None:
    actions = np.array([-0.4, 0.0, 0.4])
    action_means = np.array([-0.02, 0.015, 0.035])
    coefficients = np.array([[0.22, -0.08], [0.26, 0.03], [0.30, 0.07]])
    variances = np.array([0.28, 0.25, 0.22])
    gamma = 0.2
    tau = 1.0
    incumbent = 0.1
    exact_curve = _quadrature_ei_curve(
        action_means, coefficients, variances, incumbent, gamma, tau
    )
    returned_leader = 0
    exact_regret = float(exact_curve.max() - exact_curve[returned_leader])

    sampled = rejection_sample_symmetry_target(
        rng=np.random.default_rng(424242),
        reference_draw=_standard_normal_draw(2),
        n_accepted=120_000,
        proposal_chunk_size=4_000,
        proposal_cap=250_000,
        blocks=np.array([[0, 1]]),
        active_factors=[0],
        gamma=gamma,
        tau=tau,
    )
    result = certify_symmetry_grid_round(
        samples=sampled.samples,
        batch_id="synthetic-exact-full",
        registry=CertificationBatchRegistry(),
        active_factors=[0],
        actions=actions,
        leader_index=returned_leader,
        latent_mean=np.zeros(2),
        action_means=action_means,
        action_coefficients=coefficients,
        conditional_variances=variances,
        incumbent=incumbent,
        precision_q=np.eye(2),
        reflection_blocks=np.array([[0, 1]]),
        comparison_matrix_a=np.eye(1),
        omitted_load=np.zeros(1),
        r_max=1,
        delta=0.05,
        sample_chunk_size=4_000,
    )
    assert result.u_cert <= 0.5
    assert exact_regret > 0.0
    assert exact_regret <= result.u_cert + 2e-5


def test_401_row_output_integrity(tmp_path: Path) -> None:
    actions = np.linspace(-1.0, 1.0, 401)
    coefficients = np.column_stack((0.15 + 0.02 * actions, -0.08 * actions))
    action_means = 0.03 - 0.02 * np.square(actions - 0.1)
    variances = np.full(actions.size, 0.4)
    samples = np.random.default_rng(10101).standard_normal((5_000, 2))
    result = certify_symmetry_grid_round(
        samples=samples,
        batch_id="output-integrity",
        registry=CertificationBatchRegistry(),
        active_factors=[0],
        actions=actions,
        leader_index=220,
        latent_mean=np.zeros(2),
        action_means=action_means,
        action_coefficients=coefficients,
        conditional_variances=variances,
        incumbent=0.2,
        precision_q=np.eye(2),
        reflection_blocks=np.array([[0, 1]]),
        comparison_matrix_a=np.eye(1),
        omitted_load=np.zeros(1),
        r_max=15,
        delta=0.05,
        sample_chunk_size=1_000,
    )
    rows = result.challenger_rows(actions)
    output = tmp_path / "challengers.csv"
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with output.open(newline="") as handle:
        loaded = list(csv.DictReader(handle))
    assert len(loaded) == 401
    maximizers = [row for row in loaded if row["is_maximizer"] == "True"]
    assert len(maximizers) == 1
    np.testing.assert_allclose(float(maximizers[0]["psi"]), result.u_cert)
    for row in loaded:
        np.testing.assert_allclose(
            float(row["psi"]),
            float(row["estimated_gap"])
            + float(row["b_infer"])
            + float(row["b_struct"]),
            rtol=2e-15,
            atol=2e-15,
        )
    leader_row = loaded[result.leader_index]
    assert float(leader_row["estimated_gap"]) == 0.0
    assert float(leader_row["b_infer"]) == 0.0
    assert float(leader_row["b_struct"]) == 0.0
    assert float(leader_row["psi"]) == 0.0


def test_preregistered_contract_config_and_runner_are_frozen() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    handoff = repository_root / "project/INFERENCE_CERTIFICATION_IMPLEMENTATION_HANDOFF.md"
    assert hashlib.sha256(handoff.read_bytes()).hexdigest() == (
        "ef55b6bbc55b6245b2ac5d619b2c1bc2d038aabe85132ed2150ad22169ad14b1"
    )
    config_path = (
        repository_root
        / "experiments/symmetry/configs/inference_certification_pilot.json"
    )
    config = json.loads(config_path.read_text())
    assert config["action_grid"] == {
        "count": 401,
        "minimum": -0.58,
        "maximum": -0.06,
        "optimization_remainder": 0.0,
    }
    assert config["decision"] == {
        "incumbent": 0.5,
        "epsilon": 0.01,
        "delta": 0.05,
    }
    assert config["certification_stage"]["root_seed"] == 314159265
    assert config["pass_criteria"] == {
        "maximum_u_cert": 0.01,
        "maximum_active_factors": 18,
        "maximum_b_infer_at_worst_challenger": 0.0045,
        "maximum_cumulative_gaussian_proposals": 20000000,
        "minimum_final_acceptance_rate": 0.2,
        "all_required_tests_must_pass": True,
    }
    namespace = runpy.run_path(
        str(repository_root / "experiments/symmetry/run_inference_certification_pilot.py")
    )
    assert callable(namespace["run"])
    output_directory = (
        repository_root
        / "experiments/symmetry/outputs/inference_certification_pilot"
    )
    if output_directory.exists():
        with pytest.raises(
            RuntimeError,
            match="prospective output directory already exists",
        ):
            namespace["verify_preregistration_state"]()
        manifest = json.loads((output_directory / "batch_manifest.json").read_text())
        with (output_directory / "round_history.csv").open(newline="") as handle:
            history = {
                int(row["round"]): row for row in csv.DictReader(handle)
            }
        for batch in manifest["batches"]:
            round_row = history[int(batch["round"])]
            active_factors = json.loads(round_row["active_factors"])
            assert batch["active_factors"] == active_factors
            assert batch["active_set_hash"] == round_row["active_set_hash"]
            assert batch["active_set_hash"] == active_set_sha256(active_factors)
            assert batch["leader_index"] == int(round_row["leader_index"])
    else:
        assert not output_directory.exists()
