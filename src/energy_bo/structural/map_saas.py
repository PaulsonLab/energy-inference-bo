"""Timed BoTorch Ensemble MAP-SAAS fitting and held-out prediction utilities."""

from __future__ import annotations

import time
import math
from dataclasses import asdict, dataclass

import torch
from botorch.fit import fit_gpytorch_mll
from botorch.optim.fit import fit_gpytorch_mll_scipy
from botorch.models.map_saas import EnsembleMapSaasSingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood

from energy_bo.predictive.mixture import GaussianMixtureMarginals
from .preprocessing import FrozenOutputTransform


@dataclass(frozen=True)
class MapSaasFitInfo:
    elapsed_seconds: float
    components: int
    max_iterations: int
    taus: list[float]
    device: str
    optimizer_iterations: int
    optimizer_callbacks: int
    retries: int
    forward_evaluations: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class MapSaasReference:
    model: EnsembleMapSaasSingleTaskGP
    transform: FrozenOutputTransform
    fit_info: MapSaasFitInfo

    def to(self, device: torch.device | str) -> "MapSaasReference":
        self.model = self.model.to(device)
        return self

    def posterior(self, x: torch.Tensor) -> GaussianMixtureMarginals:
        device = next(self.model.parameters()).device
        with torch.no_grad():
            posterior = self.model.posterior(x.detach().double().to(device), observation_noise=False)
            means = posterior.mean.squeeze(-1).detach()
            variances = posterior.variance.squeeze(-1).detach()
        means = self.transform.untransform(means)
        variances = variances * self.transform.scale**2
        result = GaussianMixtureMarginals(means, variances)
        if not torch.allclose(result.mean, self.transform.untransform(posterior.mixture_mean.squeeze(-1)), atol=1e-8, rtol=1e-8):
            raise RuntimeError("component and BoTorch mixture means disagree")
        return result

    def fixed_hyperparameter_loo(self) -> GaussianMixtureMarginals:
        """Analytic LOO marginals with fitted hyperparameters held fixed.

        These are observation-predictive marginals (including the fixed numerical
        noise) and are intentionally only an optimistic diagnostic in Task 03A.
        """

        model = self.model
        train_x = model.train_inputs[0]
        train_y = model.train_targets
        with torch.no_grad():
            covariance = model.covar_module(train_x).to_dense()
            noise = model.likelihood.noise
            if noise.shape != covariance.shape[:-1]:
                noise = noise.expand(covariance.shape[:-1])
            covariance = covariance + torch.diag_embed(noise)
            inverse = torch.cholesky_inverse(torch.linalg.cholesky(covariance))
            mean_constant = model.mean_module(train_x)
            residual = train_y - mean_constant
            alpha = torch.einsum("mij,mj->mi", inverse, residual)
            diagonal = inverse.diagonal(dim1=-2, dim2=-1)
            loo_mean = train_y - alpha / diagonal
            loo_variance = diagonal.reciprocal()
        return GaussianMixtureMarginals(
            self.transform.untransform(loo_mean.cpu()),
            loo_variance.cpu() * self.transform.scale**2,
        )


def seeded_half_cauchy_taus(seed: int, count: int = 4) -> torch.Tensor:
    generator = torch.Generator().manual_seed(int(seed))
    uniform = torch.rand(count, dtype=torch.double, generator=generator).clamp(1e-12, 1-1e-12)
    return 0.1 * torch.tan(0.5 * torch.pi * uniform)


def fit_map_saas_reference(
    train_x: torch.Tensor,
    raw_train_y: torch.Tensor,
    *,
    taus: torch.Tensor,
    noise_variance: float = 1e-6,
    max_iterations: int = 250,
    device: torch.device | str = "cpu",
) -> MapSaasReference:
    train_x = train_x.detach().double().to(device)
    raw_train_y = raw_train_y.detach().double().reshape(-1, 1)
    transform = FrozenOutputTransform.fit(raw_train_y)
    train_y = transform.transform(raw_train_y).to(device)
    train_yvar = torch.full_like(train_y, float(noise_variance))
    taus = taus.detach().double().to(device)
    model = EnsembleMapSaasSingleTaskGP(
        train_X=train_x,
        train_Y=train_y,
        train_Yvar=train_yvar,
        num_taus=taus.numel(),
        taus=taus,
        outcome_transform=None,
        input_transform=None,
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    callback_count = 0
    maximum_iteration = 0
    forward_evaluations = 0

    def count_forward(_module: torch.nn.Module, _inputs: tuple[object, ...]) -> None:
        nonlocal forward_evaluations
        forward_evaluations += 1

    hook = model.register_forward_pre_hook(count_forward)

    def callback(*args: object) -> None:
        nonlocal callback_count, maximum_iteration
        callback_count += 1
        result = args[-1]
        maximum_iteration = max(maximum_iteration, int(getattr(result, "step", callback_count)))

    is_cuda = str(device).startswith("cuda") and torch.cuda.is_available()
    if is_cuda:
        torch.cuda.synchronize()
    start = time.perf_counter()
    try:
        fit_gpytorch_mll(
            mll,
            optimizer=fit_gpytorch_mll_scipy,
            optimizer_kwargs={
                "options": {"maxiter": max_iterations},
                "callback": callback,
            },
            max_attempts=1,
        )
    finally:
        hook.remove()
    if is_cuda:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    model.eval()
    return MapSaasReference(
        model=model,
        transform=transform,
        fit_info=MapSaasFitInfo(
            elapsed_seconds=elapsed,
            components=taus.numel(),
            max_iterations=max_iterations,
            taus=[float(value) for value in taus.cpu()],
            device=str(device),
            optimizer_iterations=math.ceil(callback_count / taus.numel()),
            optimizer_callbacks=callback_count,
            retries=0,
            forward_evaluations=forward_evaluations,
        ),
    )


def map_component_mll(reference: MapSaasReference) -> torch.Tensor:
    """Return independently evaluated per-component exact marginal log likelihoods."""

    model = reference.model
    model.train(); model.likelihood.train()
    with torch.no_grad():
        output = model(*model.train_inputs)
        values = ExactMarginalLogLikelihood(model.likelihood, model)(output, model.train_targets)
    model.eval(); model.likelihood.eval()
    return values.detach().double().cpu().reshape(-1)
