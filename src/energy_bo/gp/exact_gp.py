"""Exact one-dimensional GP q=1 expected-improvement sanity check."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from botorch.acquisition.analytic import ExpectedImprovement, LogExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from scipy.integrate import quad
from scipy.special import ndtr

from energy_bo.metrics import normalized_grid


def toy_objective(x: torch.Tensor) -> torch.Tensor:
    """A deterministic smooth objective deliberately small enough for exact GP fitting."""
    return torch.sin(2.0 * torch.pi * x) + 0.2 * torch.cos(6.0 * torch.pi * x) + 0.1 * x


def make_toy_training_data() -> tuple[torch.Tensor, torch.Tensor]:
    train_x = torch.linspace(0.05, 0.95, 10, dtype=torch.double).unsqueeze(-1)
    train_y = toy_objective(train_x)
    return train_x, train_y


def fit_exact_matern_gp(seed: int = 0) -> SingleTaskGP:
    """Fit the requested exact Matérn-5/2 GP with ordinary MLL optimization."""
    torch.manual_seed(seed)
    train_x, train_y = make_toy_training_data()
    covariance = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=1))
    model = SingleTaskGP(train_x, train_y, covar_module=covariance)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    model.eval()
    model.likelihood.eval()
    return model


def manual_gaussian_ei(
    mean: np.ndarray, standard_deviation: np.ndarray, best_f: float
) -> np.ndarray:
    standardized = (mean - best_f) / standard_deviation
    density = np.exp(-0.5 * np.square(standardized)) / np.sqrt(2.0 * np.pi)
    return standard_deviation * (density + standardized * ndtr(standardized))


def gaussian_ei_quadrature(
    mean: np.ndarray, standard_deviation: np.ndarray, best_f: float
) -> tuple[np.ndarray, float]:
    """Independent adaptive integration of raw improvement under every posterior."""
    values = np.empty_like(mean, dtype=float)
    largest_error = 0.0
    for index, (mean_value, scale_value) in enumerate(
        zip(mean, standard_deviation, strict=True)
    ):
        threshold = (best_f - mean_value) / scale_value
        value, error = quad(
            lambda z: max(mean_value + scale_value * z - best_f, 0.0)
            * np.exp(-0.5 * z * z)
            / np.sqrt(2.0 * np.pi),
            threshold,
            np.inf,
            epsabs=1e-11,
            epsrel=1e-11,
            limit=200,
        )
        values[index] = value
        largest_error = max(largest_error, error)
    return values, largest_error


@dataclass(frozen=True)
class GPSanityResult:
    x: np.ndarray
    posterior_mean: np.ndarray
    posterior_standard_deviation: np.ndarray
    manual_ei: np.ndarray
    botorch_ei: np.ndarray
    log_ei: np.ndarray
    quadrature_ei: np.ndarray
    augmented_marginal: np.ndarray
    ei_squared_marginal: np.ndarray
    quadrature_error_bound: float
    best_f: float
    manual_ei_argmax: float
    botorch_ei_argmax: float
    log_ei_argmax: float
    augmented_argmax: float
    squared_argmax: float


def run_gp_q1_sanity(seed: int = 0, grid_points: int = 1001) -> GPSanityResult:
    """Run the complete q=1 GP EI/augmented-marginal correctness calculation."""
    model = fit_exact_matern_gp(seed)
    train_x, train_y = make_toy_training_data()
    best_f = float(torch.max(train_y))
    x = torch.linspace(0.0, 1.0, grid_points, dtype=torch.double).unsqueeze(-1)
    posterior = model.posterior(x)
    mean = posterior.mean.squeeze(-1).detach().cpu().numpy()
    standard_deviation = (
        posterior.variance.squeeze(-1).clamp_min(1e-18).sqrt().detach().cpu().numpy()
    )
    manual = manual_gaussian_ei(mean, standard_deviation, best_f)
    quadrature, quadrature_error = gaussian_ei_quadrature(mean, standard_deviation, best_f)
    batch_x = x.unsqueeze(-2)
    botorch_ei = (
        ExpectedImprovement(model, best_f=best_f)(batch_x).detach().cpu().numpy().reshape(-1)
    )
    log_ei = (
        LogExpectedImprovement(model, best_f=best_f)(batch_x)
        .detach()
        .cpu()
        .numpy()
        .reshape(-1)
    )
    x_numpy = x.squeeze(-1).numpy()
    augmented = normalized_grid(quadrature, x_numpy)
    squared = normalized_grid(np.square(quadrature), x_numpy)
    return GPSanityResult(
        x=x_numpy,
        posterior_mean=mean,
        posterior_standard_deviation=standard_deviation,
        manual_ei=manual,
        botorch_ei=botorch_ei,
        log_ei=log_ei,
        quadrature_ei=quadrature,
        augmented_marginal=augmented,
        ei_squared_marginal=squared,
        quadrature_error_bound=quadrature_error,
        best_f=best_f,
        manual_ei_argmax=float(x_numpy[int(np.argmax(manual))]),
        botorch_ei_argmax=float(x_numpy[int(np.argmax(botorch_ei))]),
        log_ei_argmax=float(x_numpy[int(np.argmax(log_ei))]),
        augmented_argmax=float(x_numpy[int(np.argmax(augmented))]),
        squared_argmax=float(x_numpy[int(np.argmax(squared))]),
    )
