"""Stable differentiable q=1 LogEI and exact JAX GP calculations."""

from __future__ import annotations

import math
from dataclasses import dataclass

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.scipy as jsp

Array = jax.Array


def _log1mexp(value: Array) -> Array:
    """Stable log(1-exp(value)) for strictly negative ``value``."""

    split = -math.log(2.0)
    lower = jnp.minimum(value, split)
    upper = jnp.minimum(value, -jnp.finfo(value.dtype).eps)
    return jnp.where(
        value < split,
        jnp.log1p(-jnp.exp(lower)),
        jnp.log(-jnp.expm1(upper)),
    )


def stable_log_ei(mean: Array, variance: Array, best_f: float | Array) -> Array:
    """Stable maximization LogEI matching BoTorch's analytic construction.

    The predictive variance is latent and is floored at ``1e-12``. No epsilon is
    added to EI itself.
    """

    variance = jnp.maximum(jnp.asarray(variance), jnp.asarray(1e-12, dtype=jnp.float64))
    sigma = jnp.sqrt(variance)
    u = (jnp.asarray(mean) - jnp.asarray(best_f)) / sigma
    log_phi = -0.5 * u**2 - 0.5 * math.log(2.0 * math.pi)

    upper_u = jnp.maximum(u, -1.0)
    upper = jnp.log(
        jnp.exp(-0.5 * upper_u**2) / math.sqrt(2.0 * math.pi)
        + upper_u * jsp.special.ndtr(upper_u)
    )

    lower_u = jnp.minimum(u, -1.0)
    moderate_u = jnp.maximum(lower_u, -1e6)
    log_phi_moderate = -0.5 * moderate_u**2 - 0.5 * math.log(2.0 * math.pi)
    w = jnp.log(jnp.abs(moderate_u)) + jsp.special.log_ndtr(moderate_u) - log_phi_moderate
    lower_moderate = log_phi + jnp.where(
        lower_u > -1e6,
        _log1mexp(w),
        -2.0 * jnp.log(jnp.abs(lower_u)),
    )
    # For large negative u, log_ndtr loses the final digits needed after the
    # subtraction inside log1mexp.  The Mills-ratio expansion for EI is
    # phi(u) / u^2 * (1 - 3/u^2 + 15/u^4 - 105/u^6 + ...).
    inverse_square = 1.0 / jnp.maximum(lower_u**2, 1.0)
    series = jnp.ones_like(lower_u)
    coefficient = 1.0
    power = jnp.ones_like(lower_u)
    sign = -1.0
    for order in range(1, 13):
        coefficient *= 2 * order + 1
        power = power * inverse_square
        series = series + sign * coefficient * power
        sign *= -1.0
    lower_asymptotic = log_phi - 2.0 * jnp.log(jnp.abs(lower_u)) + jnp.log(series)
    lower = jnp.where(lower_u < -10.0, lower_asymptotic, lower_moderate)
    return jnp.log(sigma) + jnp.where(u > -1.0, upper, lower)


def matern52_cross(x1: Array, x2: Array, lengthscale: Array) -> Array:
    """ARD Matérn-5/2 correlation between two design matrices."""

    delta = (x1[:, None, :] - x2[None, :, :]) / lengthscale
    # The positive floor matches BoTorch's JAX kernel and avoids undefined
    # sqrt gradients on the training-covariance diagonal.
    distance = jnp.sqrt(jnp.maximum(jnp.sum(delta**2, axis=-1), 1e-30))
    root = math.sqrt(5.0) * distance
    return (1.0 + root + 5.0 * distance**2 / 3.0) * jnp.exp(-root)


def exact_gp_nll_and_logei(
    *,
    train_x: Array,
    train_y: Array,
    noise_variance: float,
    lengthscale: Array,
    mean: Array,
    outputscale: Array,
    design_x: Array,
    best_f: float,
) -> tuple[Array, Array]:
    """Return exact GP NLL and latent-posterior LogEI with one factorization."""

    n = train_x.shape[0]
    covariance = outputscale * matern52_cross(train_x, train_x, lengthscale)
    covariance = covariance + noise_variance * jnp.eye(n, dtype=train_x.dtype)
    chol = jnp.linalg.cholesky(covariance)
    centered = train_y - mean
    alpha = jsp.linalg.cho_solve((chol, True), centered)
    nll = 0.5 * (
        jnp.dot(centered, alpha)
        + 2.0 * jnp.sum(jnp.log(jnp.diag(chol)))
        + n * math.log(2.0 * math.pi)
    )
    cross = outputscale * matern52_cross(train_x, design_x[None, :], lengthscale)[:, 0]
    predictive_mean = mean + jnp.dot(cross, alpha)
    solved = jsp.linalg.solve_triangular(chol, cross, lower=True)
    predictive_variance = jnp.maximum(outputscale - jnp.dot(solved, solved), 1e-12)
    return nll, stable_log_ei(predictive_mean, predictive_variance, best_f)


@dataclass(frozen=True)
class OperationalMixture:
    """Cached fixed-particle exact GPs used only for teacher evaluation."""

    train_x: Array
    train_y: Array
    noise_variance: float
    lengthscales: Array
    means: Array
    outputscales: Array
    chol: Array
    alpha: Array

    @classmethod
    def build(
        cls,
        train_x: Array,
        train_y: Array,
        noise_variance: float,
        lengthscales: Array,
        means: Array,
        outputscales: Array,
    ) -> "OperationalMixture":
        train_x = jnp.asarray(train_x, dtype=jnp.float64)
        train_y = jnp.asarray(train_y, dtype=jnp.float64).reshape(-1)
        lengthscales = jnp.asarray(lengthscales, dtype=jnp.float64)
        means = jnp.asarray(means, dtype=jnp.float64).reshape(-1)
        outputscales = jnp.asarray(outputscales, dtype=jnp.float64).reshape(-1)

        def factor(lengthscale: Array, mean: Array, outputscale: Array) -> tuple[Array, Array]:
            covariance = outputscale * matern52_cross(train_x, train_x, lengthscale)
            covariance = covariance + noise_variance * jnp.eye(train_x.shape[0])
            chol = jnp.linalg.cholesky(covariance)
            alpha = jsp.linalg.cho_solve((chol, True), train_y - mean)
            return chol, alpha

        chol, alpha = jax.vmap(factor)(lengthscales, means, outputscales)
        return cls(
            train_x,
            train_y,
            float(noise_variance),
            lengthscales,
            means,
            outputscales,
            chol,
            alpha,
        )

    @property
    def particles(self) -> int:
        return int(self.means.shape[0])

    def particle_log_ei(self, design_x: Array, best_f: float) -> Array:
        return cached_particle_log_ei(
            self.train_x,
            self.lengthscales,
            self.means,
            self.outputscales,
            self.chol,
            self.alpha,
            jnp.asarray(design_x, dtype=jnp.float64),
            best_f,
        )

    def log_integrated_ei(self, design_x: Array, best_f: float) -> Array:
        values = self.particle_log_ei(design_x, best_f)
        return jsp.special.logsumexp(values) - math.log(self.particles)

    def beta_objective(self, design_x: Array, best_f: float, beta: float) -> Array:
        values = self.particle_log_ei(design_x, best_f)
        if beta == 0.0:
            return jnp.mean(values)
        return (jsp.special.logsumexp(beta * values) - math.log(self.particles)) / beta


def cached_particle_log_ei(
    train_x: Array,
    lengthscales: Array,
    means: Array,
    outputscales: Array,
    chol: Array,
    alpha: Array,
    design_x: Array,
    best_f: float,
) -> Array:
    """Per-particle LogEI from fixed GP factorizations; all arrays are explicit."""

    def predict(
        lengthscale: Array,
        mean: Array,
        outputscale: Array,
        particle_chol: Array,
        particle_alpha: Array,
    ) -> Array:
        cross = outputscale * matern52_cross(
            train_x, design_x[None, :], lengthscale
        )[:, 0]
        posterior_mean = mean + jnp.dot(cross, particle_alpha)
        solved = jsp.linalg.solve_triangular(particle_chol, cross, lower=True)
        variance = jnp.maximum(outputscale - jnp.dot(solved, solved), 1e-12)
        return stable_log_ei(posterior_mean, variance, best_f)

    return jax.vmap(predict)(
        lengthscales, means, outputscales, chol, alpha
    )
