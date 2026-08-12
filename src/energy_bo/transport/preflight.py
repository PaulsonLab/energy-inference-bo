"""Teacher tilt and envelope-gradient preflight for Task 02C."""

from __future__ import annotations

import math
from typing import Any

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import torch
from botorch.acquisition.analytic import _log_ei_helper

from energy_bo.structural.exact_gp import ExactGPBatchState
from energy_bo.structural.particles import SaasParticles

from .logei import OperationalMixture
from .svgd import conditional_ess_fraction, normalized_log_weights


def _mass_count(probabilities: np.ndarray, mass: float) -> int:
    ordered = np.sort(probabilities)[::-1]
    return int(np.searchsorted(np.cumsum(ordered), mass, side="left") + 1)


def tilt_diagnostics(
    log_ei: np.ndarray,
    features: np.ndarray,
    beta: float,
) -> dict[str, Any]:
    log_ei_jax = jnp.asarray(log_ei, dtype=jnp.float64)
    log_weights = normalized_log_weights(beta * log_ei_jax)
    weights = np.asarray(jnp.exp(log_weights))
    entropy = -float(np.sum(weights * np.log(np.maximum(weights, np.finfo(float).tiny))))
    uniform_mean = np.mean(features, axis=0)
    tilted_mean = weights @ features
    feature_scale = np.maximum(np.std(features, axis=0, ddof=0), 1e-12)
    shift = np.abs(tilted_mean - uniform_mean) / feature_scale
    top = np.argsort(shift)[::-1][: min(5, shift.size)]
    return {
        "beta": beta,
        "ess_fraction": float(1.0 / (weights.size * np.sum(weights**2))),
        "entropy_effective_fraction": float(np.exp(entropy) / weights.size),
        "maximum_weight": float(weights.max()),
        "mass_count_50": _mass_count(weights, 0.50),
        "mass_count_90": _mass_count(weights, 0.90),
        "mass_count_99": _mass_count(weights, 0.99),
        "conditional_ess_from_beta0": float(conditional_ess_fraction(beta * log_ei_jax)),
        "largest_shift_coordinates": [int(index) for index in top],
        "largest_standardized_mean_shifts": [float(shift[index]) for index in top],
    }


def envelope_check(
    mixture: OperationalMixture,
    design_x: np.ndarray,
    best_f: float,
    beta: float,
    finite_difference_coordinates: tuple[int, ...] = (0, 1, 2, 9),
    finite_difference_step: float = 1e-5,
) -> dict[str, float]:
    x = jnp.asarray(design_x, dtype=jnp.float64)
    objective = lambda value: mixture.beta_objective(value, best_f, beta)
    autodiff = jax.grad(objective)(x)
    particle_gradient = jax.jacrev(lambda value: mixture.particle_log_ei(value, best_f))(x)
    log_ei = mixture.particle_log_ei(x, best_f)
    if beta == 0.0:
        weights = jnp.full(log_ei.shape, 1.0 / log_ei.size)
    else:
        weights = jnp.exp(normalized_log_weights(beta * log_ei))
    envelope = weights @ particle_gradient

    finite_errors: list[float] = []
    for coordinate in finite_difference_coordinates:
        if coordinate >= x.size:
            continue
        base = np.asarray(x).copy()
        if base[coordinate] <= finite_difference_step:
            plus_one, plus_two = base.copy(), base.copy()
            plus_one[coordinate] += finite_difference_step
            plus_two[coordinate] += 2.0 * finite_difference_step
            finite = (
                -3.0 * float(objective(jnp.asarray(base)))
                + 4.0 * float(objective(jnp.asarray(plus_one)))
                - float(objective(jnp.asarray(plus_two)))
            ) / (2.0 * finite_difference_step)
        elif base[coordinate] >= 1.0 - finite_difference_step:
            minus_one, minus_two = base.copy(), base.copy()
            minus_one[coordinate] -= finite_difference_step
            minus_two[coordinate] -= 2.0 * finite_difference_step
            finite = (
                3.0 * float(objective(jnp.asarray(base)))
                - 4.0 * float(objective(jnp.asarray(minus_one)))
                + float(objective(jnp.asarray(minus_two)))
            ) / (2.0 * finite_difference_step)
        else:
            plus, minus = base.copy(), base.copy()
            plus[coordinate] += finite_difference_step
            minus[coordinate] -= finite_difference_step
            finite = (
                float(objective(jnp.asarray(plus)))
                - float(objective(jnp.asarray(minus)))
            ) / (2.0 * finite_difference_step)
        finite_errors.append(abs(finite - float(envelope[coordinate])))
    return {
        "autodiff_envelope_max_abs": float(jnp.max(jnp.abs(autodiff - envelope))),
        "finite_difference_max_abs": max(finite_errors, default=0.0),
    }


def torch_cross_check(
    state: ExactGPBatchState,
    design_x: np.ndarray,
    best_f: float,
    beta: float,
    jax_objective_value: float,
    jax_gradient: np.ndarray,
) -> dict[str, float]:
    x = torch.tensor(design_x, dtype=torch.double, requires_grad=True)
    mean, variance = state.predict(x[None, :])
    sigma = variance[:, 0].clamp_min(1e-12).sqrt()
    standardized = (mean[:, 0] - best_f) / sigma
    log_ei = sigma.log() + _log_ei_helper(standardized)
    if beta == 0.0:
        objective = log_ei.mean()
    else:
        objective = (torch.logsumexp(beta * log_ei, dim=0) - math.log(log_ei.numel())) / beta
    gradient = torch.autograd.grad(objective, x)[0]
    return {
        "torch_value_abs": abs(float(objective.detach()) - jax_objective_value),
        "torch_gradient_max_abs": float(
            np.max(np.abs(gradient.detach().numpy() - np.asarray(jax_gradient)))
        ),
    }


def run_point_preflight(
    mixture: OperationalMixture,
    particles: SaasParticles,
    state: ExactGPBatchState,
    design_x: np.ndarray,
    best_f: float,
    label: str,
) -> list[dict[str, Any]]:
    x = jnp.asarray(design_x, dtype=jnp.float64)
    log_ei = np.asarray(mixture.particle_log_ei(x, best_f))
    features = particles.feature_matrix.numpy()
    rows: list[dict[str, Any]] = []
    for beta in (0.0, 0.25, 0.5, 0.75, 1.0):
        envelope = envelope_check(mixture, design_x, best_f, beta)
        objective_fn = lambda value: mixture.beta_objective(value, best_f, beta)
        value, gradient = jax.value_and_grad(objective_fn)(x)
        cross = torch_cross_check(
            state, design_x, best_f, beta, float(value), np.asarray(gradient)
        )
        rows.append(
            {
                "point": label,
                **tilt_diagnostics(log_ei, features, beta),
                **envelope,
                **cross,
            }
        )
    return rows
