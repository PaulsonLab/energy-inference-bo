"""Trusted BoTorch SAAS NUTS reference fitting and extraction."""

from __future__ import annotations

import platform
import time
from dataclasses import asdict, dataclass
from typing import Any

import torch

from .particles import SaasParticles


@dataclass(frozen=True)
class NutsConfig:
    warmup_steps: int
    num_samples: int
    thinning: int
    max_tree_depth: int
    seed: int
    disable_progbar: bool = True

    @property
    def retained_particles(self) -> int:
        return self.num_samples // self.thinning

    def to_dict(self) -> dict[str, int | bool]:
        return asdict(self)


@dataclass(frozen=True)
class SaasReferenceFit:
    particles: SaasParticles
    elapsed_seconds: float
    environment: dict[str, Any]


def runtime_environment() -> dict[str, Any]:
    import botorch
    import gpytorch
    import jax
    import numpyro

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "botorch": botorch.__version__,
        "gpytorch": gpytorch.__version__,
        "jax": jax.__version__,
        "numpyro": numpyro.__version__,
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
    }


def fit_saas_reference(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    noise_variance: float,
    config: NutsConfig,
) -> SaasReferenceFit:
    """Fit a fresh SAAS model; no transform or warm-start state is reused."""

    from botorch.fit import fit_fully_bayesian_model_nuts
    from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP

    train_x = train_x.detach().double().cpu()
    train_y = train_y.detach().double().reshape(-1, 1).cpu()
    train_yvar = torch.full_like(train_y, float(noise_variance))
    torch.manual_seed(config.seed)
    model = SaasFullyBayesianSingleTaskGP(
        train_X=train_x,
        train_Y=train_y,
        train_Yvar=train_yvar,
        outcome_transform=None,
        input_transform=None,
    )
    start = time.perf_counter()
    fit_fully_bayesian_model_nuts(
        model,
        max_tree_depth=config.max_tree_depth,
        warmup_steps=config.warmup_steps,
        num_samples=config.num_samples,
        thinning=config.thinning,
        disable_progbar=config.disable_progbar,
        seed=config.seed,
    )
    elapsed = time.perf_counter() - start
    particles = SaasParticles.from_botorch(model)
    if particles.num_particles != config.retained_particles:
        raise RuntimeError(
            f"expected {config.retained_particles} retained samples, got {particles.num_particles}"
        )
    return SaasReferenceFit(
        particles=particles,
        elapsed_seconds=elapsed,
        environment=runtime_environment(),
    )
