from __future__ import annotations

import math

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import torch
from botorch.acquisition.analytic import _log_ei_helper

from energy_bo.transport.logei import OperationalMixture, stable_log_ei
from energy_bo.transport.potential import (
    SaasUnconstrainedPotential,
    tree_to_vector,
    vector_to_tree,
)
from energy_bo.transport.preflight import envelope_check
from energy_bo.transport.svgd import (
    AdamState,
    Whitening,
    adam_ascent,
    choose_tempering_increment,
    conditional_ess_fraction,
    design_retilt_cess,
    maximin_subset,
    svgd_direction,
)
from energy_bo.experiments.task02c import (
    _cached_design_objective,
    matched_factorization_budget,
)


def _training_case() -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.tensor(
        [[0.05, 0.15], [0.3, 0.8], [0.65, 0.25], [0.9, 0.7]],
        dtype=torch.double,
    )
    y = torch.tensor([-0.4, 0.8, 0.1, -0.2], dtype=torch.double)
    return x, y


def test_unconstrained_tree_round_trip_uses_canonical_coordinates() -> None:
    vector = jnp.array([-0.5, 0.2, -1.3, 0.4, -0.7])
    tree = vector_to_tree(vector)
    np.testing.assert_allclose(tree_to_vector(tree), vector, atol=0.0, rtol=0.0)
    assert tuple(tree) == (
        "outputscale",
        "mean",
        "kernel_tausq",
        "_kernel_inv_length_sq",
    )


def test_stable_logei_matches_botorch_over_wide_standardized_range() -> None:
    standardized = torch.tensor(
        [-1e7, -1e4, -100.0, -20.0, -5.0, -1.0, 0.0, 2.0, 20.0],
        dtype=torch.double,
    )
    expected = _log_ei_helper(standardized)
    actual = stable_log_ei(
        jnp.asarray(standardized.numpy()), jnp.ones(standardized.numel()), 0.0
    )
    np.testing.assert_allclose(np.asarray(actual), expected.numpy(), atol=2e-10, rtol=2e-12)
    assert np.isfinite(np.asarray(actual)).all()


def test_stable_logei_design_gradient_matches_finite_difference() -> None:
    function = lambda value: stable_log_ei(0.3 + value**2, 0.2 + 0.1 * value, 0.7)
    point = jnp.array(0.4)
    automatic = float(jax.grad(function)(point))
    step = 1e-6
    finite = (float(function(point + step)) - float(function(point - step))) / (2 * step)
    assert abs(automatic - finite) < 1e-7


def test_numpyro_potential_preserves_jacobians_and_fused_score() -> None:
    x, y = _training_case()
    potential = SaasUnconstrainedPotential.build(x, y, 1e-4, seed=3)
    vectors = potential.initialization_vectors(3, seed=20)
    validation = potential.validate_fused(vectors, jnp.array([0.2, 0.6]))
    assert validation["centered_value_max_abs"] < 2e-10
    assert validation["gradient_max_abs"] < 2e-9
    constrained = potential.postprocess(vectors[0])
    assert bool(jnp.all(constrained["lengthscale"] > 0))
    assert float(constrained["outputscale"]) > 0
    prior = potential.prior_vectors(16, seed=44)
    assert prior.shape == (16, 5)
    assert bool(jnp.all(jnp.isfinite(prior)))


def test_decision_energy_eta_gradient_matches_finite_difference() -> None:
    x, y = _training_case()
    potential = SaasUnconstrainedPotential.build(x, y, 1e-4, seed=8)
    vector = potential.initialization_vectors(1, seed=31)[0]
    design = jnp.array([0.25, 0.65])
    objective = lambda value: potential.fused_potential(value, design, 1.0)
    automatic = jax.grad(objective)(vector)
    step = 1e-5
    for coordinate in (0, 2, 4):
        direction = jnp.zeros_like(vector).at[coordinate].set(step)
        finite = (float(objective(vector + direction)) - float(objective(vector - direction))) / (2 * step)
        assert abs(float(automatic[coordinate]) - finite) < 2e-6


def test_whitening_round_trip_and_scale_floor() -> None:
    samples = jnp.array([[1.0, 2.0], [1.0, 4.0], [1.0, 8.0], [1.0, 10.0]])
    whitening = Whitening.fit(samples)
    assert float(whitening.scale[0]) == 0.25
    np.testing.assert_allclose(
        whitening.unwhiten(whitening.whiten(samples)), samples, atol=1e-14, rtol=1e-14
    )


def test_maximin_initialization_is_deterministic_and_unique() -> None:
    values = jnp.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
    first = maximin_subset(values, 3)
    second = maximin_subset(values, 3)
    np.testing.assert_array_equal(first, second)
    assert len(set(np.asarray(first).tolist())) == 3


def test_svgd_repulsion_separates_particles_when_score_is_zero() -> None:
    particles = jnp.array([[-0.5], [0.5]])
    direction, diagnostics = svgd_direction(particles, jnp.zeros_like(particles))
    assert float(direction[0, 0]) < 0
    assert float(direction[1, 0]) > 0
    assert float(diagnostics["repulsion_norm"]) > 0


def test_cess_schedule_is_stable_and_reports_forced_progress() -> None:
    log_factor = jnp.array([-20.0, -2.0, 0.0, 1.0])
    increment, cess, forced = choose_tempering_increment(
        log_factor, 0.0, maximum_increment=0.25, target_cess=0.8
    )
    assert 0 < increment < 0.25
    assert cess >= 0.8 - 1e-10
    forced_increment, forced_cess, forced = choose_tempering_increment(
        log_factor,
        0.0,
        maximum_increment=0.25,
        target_cess=0.8,
        required_minimum=0.25,
    )
    assert forced
    assert forced_increment == 0.25
    assert forced_cess < 0.8
    assert math.isclose(float(conditional_ess_fraction(jnp.zeros(8))), 1.0)
    assert math.isclose(
        float(design_retilt_cess(jnp.arange(4.0), jnp.arange(4.0), 1.0)), 1.0
    )


def test_adam_step_is_finite_and_matched_budget_is_exact() -> None:
    values = jnp.zeros((3, 2))
    proposed, state, diagnostics = adam_ascent(
        values, jnp.full_like(values, 100.0), AdamState.zeros(values), norm_clip=1.0
    )
    assert bool(jnp.all(jnp.isfinite(proposed)))
    assert state.step == 1
    assert diagnostics["clipped_fraction"] == 1.0
    budget = matched_factorization_budget(8, 4, 8, 4)
    assert budget == {
        "design_cache_builds": 2,
        "post_initialization_per_particle": 10,
        "total_per_particle": 14,
        "total": 112,
    }


def test_beta_envelope_identity_matches_autodiff_and_finite_difference() -> None:
    x, y = _training_case()
    mixture = OperationalMixture.build(
        x.numpy(),
        y.numpy(),
        1e-4,
        np.array([[0.3, 0.8], [0.7, 0.4], [1.1, 0.6]]),
        np.array([-0.2, 0.1, 0.3]),
        np.array([0.7, 1.1, 0.9]),
    )
    for beta in (0.0, 0.25, 1.0):
        check = envelope_check(
            mixture,
            np.array([0.4, 0.55]),
            best_f=0.8,
            beta=beta,
            finite_difference_coordinates=(0, 1),
        )
        assert check["autodiff_envelope_max_abs"] < 2e-12
        assert check["finite_difference_max_abs"] < 2e-7


def test_decision_design_update_does_not_apply_tilt_twice() -> None:
    x, y = _training_case()
    mixture = OperationalMixture.build(
        x.numpy(),
        y.numpy(),
        1e-4,
        np.array([[0.3, 0.8], [0.7, 0.4], [1.1, 0.6]]),
        np.array([-0.2, 0.1, 0.3]),
        np.array([0.7, 1.1, 0.9]),
    )
    point = jnp.array([0.4, 0.55])
    arguments = (
        mixture.train_x,
        mixture.lengthscales,
        mixture.means,
        mixture.outputscales,
        mixture.chol,
        mixture.alpha,
    )
    actual = jax.grad(
        lambda value: _cached_design_objective(
            *arguments, value, 0.8, 0.75, False
        )
    )(point)
    expected = jnp.mean(
        jax.jacrev(lambda value: mixture.particle_log_ei(value, 0.8))(point),
        axis=0,
    )
    np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=2e-12)
