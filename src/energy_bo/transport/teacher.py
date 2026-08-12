"""Fresh SAAS teachers, MAP initialization, and continuous EI optimization."""

from __future__ import annotations

import json
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.analytic import LogExpectedImprovement
from botorch.models.map_saas import get_map_saas_model
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from jax import random
from numpyro.infer import MCMC, NUTS
from scipy.optimize import minimize

from energy_bo.structural.acquisition import gaussian_expected_improvement
from energy_bo.structural.exact_gp import ExactGPBatchState
from energy_bo.structural.particles import SaasParticles

from .logei import OperationalMixture
from .potential import SaasUnconstrainedPotential


@dataclass(frozen=True)
class TeacherSamples:
    unconstrained: np.ndarray
    lengthscales: np.ndarray
    means: np.ndarray
    outputscales: np.ndarray
    elapsed_seconds: float
    metadata: dict[str, Any]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            unconstrained=self.unconstrained,
            lengthscales=self.lengthscales,
            means=self.means,
            outputscales=self.outputscales,
        )
        path.with_suffix(".json").write_text(json.dumps(self.metadata, indent=2) + "\n")


def fit_fresh_teacher(
    potential: SaasUnconstrainedPotential,
    *,
    warmup_steps: int,
    num_samples: int,
    thinning: int,
    max_tree_depth: int,
    seed: int,
    progress_bar: bool = False,
) -> TeacherSamples:
    """Fit fresh float64 NumPyro NUTS and retain raw hierarchical sites."""

    kernel = NUTS(potential.model_sample, max_tree_depth=max_tree_depth)
    sampler = MCMC(
        kernel,
        num_warmup=warmup_steps,
        num_samples=num_samples,
        thinning=thinning,
        progress_bar=progress_bar,
    )
    start = time.perf_counter()
    sampler.run(random.PRNGKey(seed))
    elapsed = time.perf_counter() - start
    raw = sampler.get_samples()
    retained = int(np.asarray(raw["mean"]).shape[0])
    expected = num_samples // thinning
    if retained != expected:
        raise RuntimeError(f"expected {expected} retained NUTS samples, got {retained}")
    unconstrained = np.asarray(
        jnp.concatenate(
            (
                jnp.log(raw["outputscale"]).reshape(-1, 1),
                raw["mean"].reshape(-1, 1),
                jnp.log(raw["kernel_tausq"]).reshape(-1, 1),
                jnp.log(raw["_kernel_inv_length_sq"]).reshape(-1, potential.dimension),
            ),
            axis=1,
        )
    )
    inverse = np.asarray(raw["kernel_tausq"][:, None] * raw["_kernel_inv_length_sq"])
    return TeacherSamples(
        unconstrained=unconstrained,
        lengthscales=1.0 / np.sqrt(inverse),
        means=np.asarray(raw["mean"]),
        outputscales=np.asarray(raw["outputscale"]),
        elapsed_seconds=elapsed,
        metadata={
            "seed": seed,
            "warmup_steps": warmup_steps,
            "num_samples": num_samples,
            "thinning": thinning,
            "max_tree_depth": max_tree_depth,
            "elapsed_seconds": elapsed,
            "python": platform.python_version(),
            "jax": jax.__version__,
            "jax_x64": bool(jax.config.jax_enable_x64),
            "jax_backend": jax.default_backend(),
            "jax_devices": [str(device) for device in jax.devices()],
        },
    )


def map_saas_initial_design(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    noise_variance: float,
    candidates: torch.Tensor,
    *,
    seed: int = 0,
) -> tuple[np.ndarray, dict[str, float]]:
    """Fit the established BoTorch MAP-SAAS model and select its best candidate."""

    start = time.perf_counter()
    torch.manual_seed(seed)
    y = train_y.double().reshape(-1, 1)
    model = get_map_saas_model(
        train_X=train_x.double(),
        train_Y=y,
        train_Yvar=torch.full_like(y, float(noise_variance)),
        outcome_transform=None,
        input_transform=None,
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    model.eval()
    with torch.no_grad():
        posterior = model.posterior(candidates.double())
        mean = posterior.mean.squeeze(-1)
        variance = posterior.variance.squeeze(-1)
        ei = gaussian_expected_improvement(mean, variance, float(train_y.max()))
        index = int(torch.argmax(ei))
    log_ei = LogExpectedImprovement(model=model, best_f=float(train_y.max()))
    continuous_x, continuous_value = optimize_acqf(
        acq_function=log_ei,
        bounds=torch.stack(
            (
                torch.zeros(train_x.shape[1], dtype=torch.double),
                torch.ones(train_x.shape[1], dtype=torch.double),
            )
        ),
        q=1,
        num_restarts=16,
        raw_samples=1024,
        options={"maxiter": 100},
    )
    return candidates[index].detach().cpu().numpy(), {
        "candidate_index": index,
        "candidate_ei": float(ei[index]),
        "continuous_x": continuous_x[0].detach().cpu().tolist(),
        "continuous_log_ei": float(continuous_value.detach()),
        "elapsed_seconds": time.perf_counter() - start,
    }


def continuous_teacher_optimum(
    mixture: OperationalMixture,
    best_f: float,
    candidates: np.ndarray,
    *,
    extra_candidates: np.ndarray | None = None,
    map_design: np.ndarray | None = None,
    starts: int = 16,
    maxiter: int = 100,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    """Multi-start bounded optimization of fresh-NUTS integrated EI."""

    candidates = np.asarray(candidates, dtype=np.float64)
    values = np.asarray(
        jax.vmap(lambda x: mixture.log_integrated_ei(x, best_f))(jnp.asarray(candidates))
    )
    order = np.argsort(values)[::-1]
    chosen: list[np.ndarray] = []
    for index in order:
        point = candidates[index]
        if not chosen or min(np.linalg.norm(point - old) for old in chosen) > 1e-8:
            chosen.append(point)
        if len(chosen) == starts:
            break
    if map_design is not None:
        chosen.append(np.asarray(map_design, dtype=np.float64))

    objective = jax.jit(jax.value_and_grad(lambda x: mixture.log_integrated_ei(x, best_f)))

    def scipy_value(x: np.ndarray) -> tuple[float, np.ndarray]:
        value, gradient = objective(jnp.asarray(x, dtype=jnp.float64))
        return -float(value), -np.asarray(gradient, dtype=np.float64)

    solutions: list[tuple[np.ndarray, float, bool]] = []
    for initial in chosen:
        result = minimize(
            scipy_value,
            initial,
            method="L-BFGS-B",
            jac=True,
            bounds=[(0.0, 1.0)] * candidates.shape[1],
            options={"maxiter": maxiter, "ftol": 1e-13, "gtol": 1e-8},
        )
        solutions.append((result.x, -float(result.fun), bool(result.success)))
    if extra_candidates is not None:
        extra = np.asarray(extra_candidates, dtype=np.float64)
        extra_values = np.asarray(
            jax.vmap(lambda x: mixture.log_integrated_ei(x, best_f))(jnp.asarray(extra))
        )
        best_extra = int(np.argmax(extra_values))
        solutions.append((extra[best_extra], float(extra_values[best_extra]), True))
    best_x, best_log_ei, _ = max(solutions, key=lambda item: item[1])
    return best_x, float(np.exp(best_log_ei)), {
        "starts": len(chosen),
        "successful_starts": sum(success for _, _, success in solutions),
        "best_log_ei": best_log_ei,
        "grid_best_log_ei": float(values.max()),
    }


def operational_mixture_from_samples(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    noise_variance: float,
    samples: TeacherSamples | SaasParticles,
) -> OperationalMixture:
    if isinstance(samples, TeacherSamples):
        lengthscales, means, outputscales = (
            samples.lengthscales,
            samples.means,
            samples.outputscales,
        )
    else:
        lengthscales = samples.lengthscales.numpy()
        means = samples.means.numpy()
        outputscales = samples.outputscales.numpy()
    return OperationalMixture.build(
        train_x.numpy(),
        train_y.numpy(),
        noise_variance,
        lengthscales,
        means,
        outputscales,
    )
