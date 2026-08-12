"""Small transparent SVGD, tempering, and design-step utilities."""

from __future__ import annotations

from dataclasses import dataclass

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

Array = jax.Array


def normalized_log_weights(log_weights: Array) -> Array:
    values = jnp.asarray(log_weights, dtype=jnp.float64)
    return values - jax.scipy.special.logsumexp(values)


def conditional_ess_fraction(log_increment: Array) -> Array:
    """Conditional ESS fraction for equally weighted current particles."""

    normalized = normalized_log_weights(log_increment)
    return 1.0 / (normalized.size * jnp.sum(jnp.exp(2.0 * normalized)))


def design_retilt_cess(old_log_ei: Array, new_log_ei: Array, beta: float) -> Array:
    """CESS/P induced by moving x while structural particle locations stay fixed."""

    return conditional_ess_fraction(
        beta * (jnp.asarray(new_log_ei) - jnp.asarray(old_log_ei))
    )


def choose_tempering_increment(
    log_factor: Array,
    current: float,
    *,
    maximum_increment: float = 0.25,
    target_cess: float = 0.8,
    required_minimum: float = 0.0,
    bisection_steps: int = 40,
) -> tuple[float, float, bool]:
    """Largest CESS-safe increment, with explicit forced-progress reporting."""

    upper = min(maximum_increment, 1.0 - current)
    if upper <= 0:
        return 0.0, 1.0, False
    factor = jnp.asarray(log_factor, dtype=jnp.float64)
    if float(conditional_ess_fraction(upper * factor)) >= target_cess:
        chosen = upper
    else:
        low, high = 0.0, upper
        for _ in range(bisection_steps):
            middle = 0.5 * (low + high)
            if float(conditional_ess_fraction(middle * factor)) >= target_cess:
                low = middle
            else:
                high = middle
        chosen = low
    forced = required_minimum > chosen + 1e-14
    chosen = min(upper, max(chosen, required_minimum))
    return chosen, float(conditional_ess_fraction(chosen * factor)), forced


@dataclass(frozen=True)
class Whitening:
    center: Array
    scale: Array

    @classmethod
    def fit(cls, samples: Array, scale_floor: float = 0.25) -> "Whitening":
        samples = jnp.asarray(samples, dtype=jnp.float64)
        center = jnp.median(samples, axis=0)
        q25 = jnp.quantile(samples, 0.25, axis=0)
        q75 = jnp.quantile(samples, 0.75, axis=0)
        scale = jnp.maximum((q75 - q25) / 1.349, scale_floor)
        return cls(center=center, scale=scale)

    def whiten(self, values: Array) -> Array:
        return (jnp.asarray(values) - self.center) / self.scale

    def unwhiten(self, values: Array) -> Array:
        return self.center + self.scale * jnp.asarray(values)


def maximin_subset(values: Array, count: int) -> Array:
    """Deterministic farthest-first subset, seeded by the point nearest the median."""

    values = jnp.asarray(values, dtype=jnp.float64)
    if not 1 <= count <= values.shape[0]:
        raise ValueError("count must be between one and the number of rows")
    center = jnp.median(values, axis=0)
    first = int(jnp.argmin(jnp.sum((values - center) ** 2, axis=1)))
    selected = [first]
    minimum = jnp.sum((values - values[first]) ** 2, axis=1)
    for _ in range(1, count):
        minimum = minimum.at[jnp.asarray(selected)].set(-jnp.inf)
        index = int(jnp.argmax(minimum))
        selected.append(index)
        minimum = jnp.minimum(minimum, jnp.sum((values - values[index]) ** 2, axis=1))
    return jnp.asarray(selected, dtype=jnp.int32)


def rbf_kernel(particles: Array, bandwidth_floor: float = 1e-3) -> tuple[Array, Array]:
    delta = particles[:, None, :] - particles[None, :, :]
    squared = jnp.sum(delta**2, axis=-1)
    mask = ~jnp.eye(particles.shape[0], dtype=bool)
    nonzero = jnp.where(mask, squared, jnp.nan)
    median = jnp.nanmedian(nonzero)
    bandwidth = jax.lax.stop_gradient(
        jnp.maximum(median / jnp.log(particles.shape[0] + 1.0), bandwidth_floor)
    )
    return jnp.exp(-squared / bandwidth), bandwidth


def svgd_direction(particles: Array, scores: Array) -> tuple[Array, dict[str, Array]]:
    """Standard RBF-SVGD ascent direction and transparent diagnostics."""

    kernel, bandwidth = rbf_kernel(particles)
    count = particles.shape[0]
    attraction = kernel.T @ scores / count
    repulsion = (
        2.0
        / bandwidth
        * (particles * kernel.sum(axis=0)[:, None] - kernel.T @ particles)
        / count
    )
    direction = attraction + repulsion
    return direction, {
        "bandwidth": bandwidth,
        "attraction_norm": jnp.linalg.norm(attraction) / jnp.sqrt(count),
        "repulsion_norm": jnp.linalg.norm(repulsion) / jnp.sqrt(count),
        "direction_norm": jnp.linalg.norm(direction) / jnp.sqrt(count),
    }


@dataclass
class AdamState:
    first: Array
    second: Array
    step: int = 0

    @classmethod
    def zeros(cls, values: Array) -> "AdamState":
        return cls(jnp.zeros_like(values), jnp.zeros_like(values), 0)

    def copy(self) -> "AdamState":
        return AdamState(jnp.array(self.first), jnp.array(self.second), self.step)


def adam_ascent(
    values: Array,
    direction: Array,
    state: AdamState,
    *,
    learning_rate: float = 0.01,
    norm_clip: float = 10.0,
    coordinate_clip: float | None = None,
) -> tuple[Array, AdamState, dict[str, float]]:
    norms = jnp.linalg.norm(direction, axis=-1, keepdims=True)
    clipped = direction * jnp.minimum(1.0, norm_clip / jnp.maximum(norms, 1e-30))
    step = state.step + 1
    first = 0.9 * state.first + 0.1 * clipped
    second = 0.999 * state.second + 0.001 * clipped**2
    update = learning_rate * (first / (1.0 - 0.9**step)) / (
        jnp.sqrt(second / (1.0 - 0.999**step)) + 1e-8
    )
    if coordinate_clip is not None:
        update = jnp.clip(update, -coordinate_clip, coordinate_clip)
    return values + update, AdamState(first, second, step), {
        "max_raw_norm": float(jnp.max(norms)),
        "clipped_fraction": float(jnp.mean(norms[:, 0] > norm_clip)),
    }


def svgd_step(
    particles: Array,
    whitening: Whitening,
    potential_fn,
    adam: AdamState,
    *,
    learning_rate: float = 0.01,
) -> tuple[Array, AdamState, dict[str, float]]:
    """One whitened-coordinate SVGD step for a scalar potential."""

    whitened = whitening.whiten(particles)

    def potential_white(value: Array) -> Array:
        return potential_fn(whitening.unwhiten(value))

    gradients = jax.vmap(jax.grad(potential_white))(whitened)
    scores = -gradients
    direction, diagnostics = svgd_direction(whitened, scores)
    proposed, new_adam, adam_diagnostics = adam_ascent(
        whitened, direction, adam, learning_rate=learning_rate
    )
    result = whitening.unwhiten(proposed)
    if not bool(jnp.all(jnp.isfinite(result))):
        raise FloatingPointError("nonfinite SVGD proposal")
    pairwise = jnp.sqrt(
        jnp.maximum(jnp.sum((proposed[:, None] - proposed[None, :]) ** 2, axis=-1), 0.0)
    )
    mask = ~jnp.eye(proposed.shape[0], dtype=bool)
    diagnostics_out = {
        key: float(value) for key, value in diagnostics.items()
    } | adam_diagnostics | {
        "median_pairwise_distance": float(jnp.median(pairwise[mask])),
        "score_rms": float(jnp.sqrt(jnp.mean(scores**2))),
    }
    return result, new_adam, diagnostics_out
