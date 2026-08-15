"""Exact GPyTorch/BoTorch protein belief models."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import gpytorch
import torch
from botorch.acquisition.analytic import LogExpectedImprovement
from botorch.models.gpytorch import GPyTorchModel
from botorch.optim.closures import get_loss_closure_with_grads
from botorch.optim.fit import fit_gpytorch_mll_scipy, get_parameters_and_bounds
from gpytorch.constraints import GreaterThan
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ZeroMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.models import ExactGP
from gpytorch.priors import GammaPrior

from .kernels import make_covariance


@dataclass(frozen=True)
class FrozenStandardization:
    mean: float
    scale: float

    @classmethod
    def fit(cls, values: torch.Tensor) -> "FrozenStandardization":
        mean = float(values.mean())
        scale = max(float(values.std(unbiased=False)), 1e-8)
        return cls(mean, scale)

    def transform(self, values: torch.Tensor) -> torch.Tensor:
        return (values - self.mean) / self.scale

    def untransform_mean(self, values: torch.Tensor) -> torch.Tensor:
        return self.mean + self.scale * values

    def untransform_variance(self, values: torch.Tensor) -> torch.Tensor:
        return self.scale**2 * values


class ProteinExactGP(ExactGP, GPyTorchModel):
    _num_outputs = 1

    def __init__(self, train_x: torch.Tensor, train_y: torch.Tensor, model_name: str) -> None:
        likelihood = GaussianLikelihood(
            noise_prior=GammaPrior(2.0, 2.0),
            noise_constraint=GreaterThan(1e-4),
        )
        super().__init__(train_x, train_y, likelihood)
        self.model_name = model_name
        self.mean_module = ZeroMean()
        self.covar_module = make_covariance(model_name, train_x.shape[-1])
        self.likelihood.noise = torch.tensor(0.05, dtype=torch.double, device=train_x.device)
        self.to(dtype=torch.double, device=train_x.device)

    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        return MultivariateNormal(self.mean_module(x), self.covar_module(x))


@dataclass
class FitResult:
    model: ProteinExactGP
    transform: FrozenStandardization
    converged: bool
    message: str
    iterations: int
    function_evaluations: int
    wall_seconds: float
    objective: float


def fit_protein_gp(
    train_x: torch.Tensor,
    train_y_raw: torch.Tensor,
    model_name: str,
    *,
    max_iterations: int,
    initial_state: dict[str, torch.Tensor] | None = None,
) -> FitResult:
    transform = FrozenStandardization.fit(train_y_raw)
    train_y = transform.transform(train_y_raw).to(dtype=torch.double, device=train_x.device)
    model = ProteinExactGP(train_x, train_y, model_name)
    if initial_state is not None:
        model.load_state_dict(initial_state, strict=False)
        model.set_train_data(inputs=train_x, targets=train_y, strict=False)
    model.train(); model.likelihood.train()
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    started = time.perf_counter()
    evaluations = 0
    try:
        all_parameters, _ = get_parameters_and_bounds(mll)
        parameters = {name: value for name, value in all_parameters.items() if value.requires_grad}
        default_closure = get_loss_closure_with_grads(mll, parameters=parameters)

        def counted_closure():
            nonlocal evaluations
            evaluations += 1
            return default_closure()

        result = fit_gpytorch_mll_scipy(
            mll,
            parameters=parameters,
            closure=counted_closure,
            options={"maxiter": max_iterations},
        )
        converged = bool(result.status.name == "SUCCESS") if hasattr(result.status, "name") else bool(result.status == 0)
        message = str(result.message)
        iterations = int(result.step)
        objective = float(result.fval)
    except Exception as error:  # recorded as an invalid fit by the experiment
        converged = False
        message = f"{type(error).__name__}: {error}"
        iterations = 0
        evaluations = 0
        objective = math.nan
    wall = time.perf_counter() - started
    finite = all(torch.isfinite(parameter).all() for parameter in model.parameters())
    model.eval(); model.likelihood.eval()
    return FitResult(model, transform, converged and finite, message, iterations, evaluations, wall, objective)


@torch.no_grad()
def predict_raw(
    fit: FitResult,
    candidates: torch.Tensor,
    *,
    observation_noise: bool,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    means, variances = [], []
    for start in range(0, len(candidates), chunk_size):
        posterior = fit.model.posterior(candidates[start : start + chunk_size], observation_noise=observation_noise)
        means.append(fit.transform.untransform_mean(posterior.mean.squeeze(-1)).cpu())
        variances.append(fit.transform.untransform_variance(posterior.variance.squeeze(-1)).cpu())
    return torch.cat(means), torch.cat(variances).clamp_min(1e-15)


@torch.no_grad()
def log_ei(fit: FitResult, candidates: torch.Tensor, best_raw: float, chunk_size: int) -> torch.Tensor:
    best = (best_raw - fit.transform.mean) / fit.transform.scale
    acquisition = LogExpectedImprovement(fit.model, best_f=best)
    values = []
    for start in range(0, len(candidates), chunk_size):
        values.append(acquisition(candidates[start : start + chunk_size].unsqueeze(-2)).cpu())
    return torch.cat(values)
