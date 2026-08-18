"""Core mathematics for the frozen constrained-batch shift diagnostic."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from botorch.utils.objective import compute_smoothed_feasibility_indicator
from botorch.utils.safe_math import fatmax, log_fatplus
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import qmc


BeliefName = Literal["gaussian", "student_t"]


@dataclass(frozen=True)
class PreparedQMCBase:
    normals: torch.Tensor
    student_multipliers: torch.Tensor


def canonical_json_hash(value: dict[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def load_protocol(path: str | Path) -> tuple[dict[str, Any], str]:
    protocol = json.loads(Path(path).read_text())
    return protocol, canonical_json_hash(protocol)


def atomic_write_json(path: str | Path, value: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=target.name, dir=target.parent)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(value, handle, indent=2)
            handle.write("\n")
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_save_npz(path: str | Path, **arrays: np.ndarray) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=target.name, suffix=".npz", dir=target.parent
    )
    os.close(descriptor)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def matern52(
    x1: torch.Tensor, x2: torch.Tensor, lengthscales: torch.Tensor
) -> torch.Tensor:
    """ARD Matérn-5/2 correlation kernel."""

    delta = (x1.unsqueeze(-2) - x2.unsqueeze(-3)) / lengthscales
    distance = torch.sqrt(torch.clamp(delta.square().sum(dim=-1), min=1e-30))
    scaled = math.sqrt(5.0) * distance
    return (1.0 + scaled + scaled.square() / 3.0) * torch.exp(-scaled)


@dataclass(frozen=True)
class PredictiveMoments:
    location: torch.Tensor
    covariance: torch.Tensor
    student_scale: torch.Tensor
    degrees_of_freedom: float


@dataclass(frozen=True)
class ConjugateScaleProcess:
    """Conjugate scale-integrated GP with a matched Gaussian control."""

    train_x: torch.Tensor
    standardized_y: torch.Tensor
    output_center: float
    output_scale: float
    constant_mean: float
    lengthscales: torch.Tensor
    a0: float
    b0: float
    nugget: float
    objective_value: float
    converged: bool

    @property
    def sample_count(self) -> int:
        return int(self.train_x.shape[0])

    @property
    def a_n(self) -> float:
        return self.a0 + 0.5 * self.sample_count

    @property
    def degrees_of_freedom(self) -> float:
        return 2.0 * self.a_n

    def _training_state(
        self, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.train_x.to(device=device, dtype=torch.double)
        y = self.standardized_y.to(device=device, dtype=torch.double)
        ls = self.lengthscales.to(device=device, dtype=torch.double)
        eye = torch.eye(len(x), dtype=torch.double, device=device)
        kernel = matern52(x, x, ls) + self.nugget * eye
        chol = torch.linalg.cholesky(kernel)
        residual = y - self.constant_mean
        alpha = torch.cholesky_solve(residual[:, None], chol).squeeze(-1)
        quadratic = residual @ alpha
        b_n = torch.as_tensor(self.b0, dtype=torch.double, device=device) + 0.5 * quadratic
        return x, ls, chol, alpha, b_n

    def predict(self, candidate_x: torch.Tensor) -> PredictiveMoments:
        """Return raw-space latent moments for ``batch x q x d`` candidates."""

        if candidate_x.ndim == 2:
            candidate_x = candidate_x.unsqueeze(0)
        if candidate_x.ndim != 3:
            raise ValueError("candidate_x must have shape batch x q x d")
        device = candidate_x.device
        train_x, ls, chol, alpha, b_n = self._training_state(device)
        cross = matern52(candidate_x, train_x, ls)
        location_std = self.constant_mean + cross @ alpha
        solved = torch.cholesky_solve(cross.transpose(-1, -2), chol)
        prior = matern52(candidate_x, candidate_x, ls)
        conditional = prior - cross @ solved
        conditional = 0.5 * (conditional + conditional.transpose(-1, -2))
        # This is numerical stabilization only and is common to both beliefs.
        q = candidate_x.shape[-2]
        conditional = conditional + 1e-10 * torch.eye(
            q, dtype=torch.double, device=device
        )
        raw_scale2 = self.output_scale**2
        student_scale = (b_n / self.a_n) * conditional * raw_scale2
        covariance = (b_n / (self.a_n - 1.0)) * conditional * raw_scale2
        location = location_std * self.output_scale + self.output_center
        return PredictiveMoments(
            location=location,
            covariance=covariance,
            student_scale=student_scale,
            degrees_of_freedom=self.degrees_of_freedom,
        )

    def log_prob(
        self, candidate_x: torch.Tensor, value: torch.Tensor, belief: BeliefName
    ) -> torch.Tensor:
        predictive = self.predict(candidate_x)
        value = value.to(dtype=torch.double, device=candidate_x.device)
        delta = value - predictive.location
        matrix = (
            predictive.covariance
            if belief == "gaussian"
            else predictive.student_scale
        )
        chol = torch.linalg.cholesky(matrix)
        solve = torch.cholesky_solve(delta.unsqueeze(-1), chol).squeeze(-1)
        quadratic = (delta * solve).sum(dim=-1)
        logdet = 2.0 * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(-1)
        dimension = value.shape[-1]
        if belief == "gaussian":
            return -0.5 * (
                dimension * math.log(2.0 * math.pi) + logdet + quadratic
            )
        degrees = predictive.degrees_of_freedom
        return (
            torch.lgamma(torch.as_tensor((degrees + dimension) / 2.0, dtype=torch.double, device=value.device))
            - torch.lgamma(torch.as_tensor(degrees / 2.0, dtype=torch.double, device=value.device))
            - 0.5 * (dimension * math.log(degrees * math.pi) + logdet)
            - 0.5 * (degrees + dimension) * torch.log1p(quadratic / degrees)
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "output_center": self.output_center,
            "output_scale": self.output_scale,
            "constant_mean": self.constant_mean,
            "lengthscales": self.lengthscales.tolist(),
            "a0": self.a0,
            "b0": self.b0,
            "nugget": self.nugget,
            "objective_value": self.objective_value,
            "converged": self.converged,
            "degrees_of_freedom": self.degrees_of_freedom,
        }


def fit_conjugate_scale_process(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    protocol: dict[str, Any],
    *,
    maximum_iterations: int | None = None,
) -> ConjugateScaleProcess:
    """Fit the frozen scale-integrated correlation model."""

    config = protocol["beliefs"]
    x = train_x.detach().cpu().to(torch.double)
    raw_y = train_y.detach().cpu().to(torch.double).reshape(-1)
    center = float(raw_y.mean())
    scale = float(torch.sqrt(torch.mean((raw_y - center).square())))
    scale = max(scale, 1e-12)
    y = (raw_y - center) / scale
    a0 = float(config["inverse_gamma"]["a0"])
    b0 = float(config["inverse_gamma"]["b0"])
    nugget = float(config["relative_nugget"])
    iterations = int(maximum_iterations or config["maximum_iterations"])
    lower, upper = [math.log(v) for v in config["lengthscale_bounds"]]
    n = len(x)
    a_n = a0 + 0.5 * n

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        theta = torch.tensor(parameters, dtype=torch.double, requires_grad=True)
        mean = theta[0]
        lengthscales = torch.exp(theta[1:])
        kernel = matern52(x, x, lengthscales) + nugget * torch.eye(
            n, dtype=torch.double
        )
        chol = torch.linalg.cholesky(kernel)
        residual = y - mean
        alpha = torch.cholesky_solve(residual[:, None], chol).squeeze(-1)
        b_n = b0 + 0.5 * residual @ alpha
        loss = (
            torch.log(torch.diagonal(chol)).sum()
            + a_n * torch.log(b_n)
            + 0.5 * n * math.log(2.0 * math.pi)
            + gammaln(a0)
            - gammaln(a_n)
            - a0 * math.log(b0)
        )
        loss.backward()
        return float(loss.detach()), theta.grad.detach().numpy().copy()

    fits: list[tuple[float, Any]] = []
    for start in config["lengthscale_starts"]:
        initial = np.array([0.0] + [math.log(float(start))] * x.shape[-1])
        result = minimize(
            objective,
            initial,
            method="L-BFGS-B",
            jac=True,
            bounds=[(None, None)] + [(lower, upper)] * x.shape[-1],
            options={"maxiter": iterations, "ftol": 1e-12, "gtol": 1e-9},
        )
        if np.isfinite(result.fun) and np.isfinite(result.x).all():
            fits.append((float(result.fun), result))
    if not fits:
        raise RuntimeError("all conjugate-scale process fits failed")
    _, best = min(fits, key=lambda item: item[0])
    return ConjugateScaleProcess(
        train_x=x,
        standardized_y=y,
        output_center=center,
        output_scale=scale,
        constant_mean=float(best.x[0]),
        lengthscales=torch.exp(torch.tensor(best.x[1:], dtype=torch.double)),
        a0=a0,
        b0=b0,
        nugget=nugget,
        objective_value=float(best.fun),
        converged=bool(best.success),
    )


@dataclass(frozen=True)
class BeliefPair:
    objective: ConjugateScaleProcess
    constraint: ConjugateScaleProcess

    def validate_match(self, candidate_x: torch.Tensor, tolerance: float) -> dict[str, float]:
        discrepancies: list[float] = []
        for process in (self.objective, self.constraint):
            moments = process.predict(candidate_x)
            student_covariance = (
                moments.degrees_of_freedom
                / (moments.degrees_of_freedom - 2.0)
                * moments.student_scale
            )
            discrepancies.append(
                float(torch.max(torch.abs(student_covariance - moments.covariance)))
            )
        maximum = max(discrepancies)
        if maximum > tolerance:
            raise RuntimeError(f"Gaussian/Student-t covariance mismatch: {maximum}")
        return {"maximum_mean_error": 0.0, "maximum_covariance_error": maximum}


def scrambled_sobol_uniforms(
    sample_count: int,
    dimension: int,
    seed: int,
    *,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    power = int(round(math.log2(sample_count)))
    if 2**power != sample_count:
        raise ValueError("scrambled Sobol sample_count must be a power of two")
    values = qmc.Sobol(d=dimension, scramble=True, seed=seed).random_base2(power)
    tiny = np.nextafter(0.0, 1.0)
    values = np.clip(values, tiny, np.nextafter(1.0, 0.0))
    return torch.as_tensor(values, dtype=torch.double, device=device)


def prepare_qmc_base(
    base_uniforms: torch.Tensor, q: int, degrees_of_freedom: float
) -> PreparedQMCBase:
    """Transform uniforms once so repeated optimizer calls remain GPU-efficient."""

    if base_uniforms.shape[-1] != 2 * (q + 1):
        raise ValueError("base uniforms must provide q normals plus one scale per output")
    normals = []
    multipliers = []
    from scipy.stats import chi2

    for output_index in range(2):
        offset = output_index * (q + 1)
        normals.append(torch.special.ndtri(base_uniforms[:, offset : offset + q]))
        uniform = base_uniforms[:, offset + q].detach().cpu().numpy()
        chi = torch.as_tensor(
            chi2.ppf(uniform, df=degrees_of_freedom),
            dtype=torch.double,
            device=base_uniforms.device,
        )
        multipliers.append(
            torch.sqrt(
                torch.as_tensor(
                    degrees_of_freedom,
                    dtype=torch.double,
                    device=base_uniforms.device,
                )
                / chi
            )
        )
    return PreparedQMCBase(
        normals=torch.stack(normals, dim=1),
        student_multipliers=torch.stack(multipliers, dim=1),
    )


def sample_belief_pair(
    pair: BeliefPair,
    candidate_x: torch.Tensor,
    base_uniforms: torch.Tensor | PreparedQMCBase,
    belief: BeliefName,
) -> torch.Tensor:
    """Sample objective/constraint worlds using paired QMC coordinates."""

    if candidate_x.ndim == 2:
        candidate_x = candidate_x.unsqueeze(0)
    q = candidate_x.shape[-2]
    prepared = (
        base_uniforms
        if isinstance(base_uniforms, PreparedQMCBase)
        else prepare_qmc_base(
            base_uniforms, q, pair.objective.degrees_of_freedom
        )
    )
    outcomes = []
    for output_index, process in enumerate((pair.objective, pair.constraint)):
        normal = prepared.normals[:, output_index]
        moments = process.predict(candidate_x)
        matrix = moments.covariance if belief == "gaussian" else moments.student_scale
        chol = torch.linalg.cholesky(matrix)
        residual = torch.einsum("bqk,nk->nbq", chol, normal)
        if belief == "student_t":
            # If V ~ chi2(df), sqrt(df/V) produces the standard t scale mixture.
            residual = residual * prepared.student_multipliers[:, output_index, None, None]
        outcomes.append(moments.location.unsqueeze(0) + residual)
    return torch.stack(outcomes, dim=-1)


def log_qlogei_utility(
    samples: torch.Tensor, best_f: float, acquisition: dict[str, Any]
) -> torch.Tensor:
    """Exact per-world log utility used by pinned constrained qLogEI."""

    objective = samples[..., int(acquisition["objective_output_index"])]
    log_improvement = log_fatplus(
        objective - float(best_f), tau=float(acquisition["tau_relu"])
    )
    log_feasibility = compute_smoothed_feasibility_indicator(
        constraints=[lambda values: values[..., int(acquisition["constraint_output_index"])]],
        samples=samples,
        eta=float(acquisition["eta"]),
        log=True,
        fat=bool(acquisition["fat"]),
    )
    log_point_utility = log_improvement + log_feasibility
    return fatmax(
        log_point_utility,
        dim=-1,
        tau=float(acquisition["tau_max"]),
    )


def decision_shift_from_log_utility(log_utility: torch.Tensor) -> dict[str, torch.Tensor]:
    sample_count = log_utility.shape[0]
    log_first = torch.logsumexp(log_utility, dim=0) - math.log(sample_count)
    log_second = torch.logsumexp(2.0 * log_utility, dim=0) - math.log(sample_count)
    d2 = torch.clamp(log_second - 2.0 * log_first, min=0.0)
    chi_square = torch.expm1(d2)
    return {
        "log_acquisition": log_first,
        "acquisition": torch.exp(log_first),
        "log_second_moment": log_second,
        "chi_square": chi_square,
        "d2": d2,
        "ess_fraction": torch.exp(-d2),
    }


def evaluate_batches(
    pair: BeliefPair,
    candidate_x: torch.Tensor,
    base_uniforms: torch.Tensor | PreparedQMCBase,
    belief: BeliefName,
    best_f: float,
    acquisition: dict[str, Any],
    *,
    with_gradients: bool = True,
) -> dict[str, torch.Tensor]:
    x = candidate_x.detach().clone().to(dtype=torch.double)
    x.requires_grad_(with_gradients)
    samples = sample_belief_pair(pair, x, base_uniforms, belief)
    log_utility = log_qlogei_utility(samples, best_f, acquisition)
    result = decision_shift_from_log_utility(log_utility)
    if with_gradients:
        gradient = torch.autograd.grad(result["log_acquisition"].sum(), x)[0]
    else:
        gradient = torch.full_like(x, torch.nan)
    result["gradient"] = gradient
    return {key: value.detach() for key, value in result.items()}


def multivariate_t_log_density(
    value: torch.Tensor, location: torch.Tensor, scale: torch.Tensor, degrees: float
) -> torch.Tensor:
    """Standalone multivariate Student-t density used by numerical tests."""

    delta = value - location
    chol = torch.linalg.cholesky(scale)
    solve = torch.cholesky_solve(delta.unsqueeze(-1), chol).squeeze(-1)
    quadratic = (delta * solve).sum(-1)
    dimension = value.shape[-1]
    logdet = 2.0 * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(-1)
    return (
        torch.lgamma(torch.as_tensor((degrees + dimension) / 2.0, dtype=torch.double, device=value.device))
        - torch.lgamma(torch.as_tensor(degrees / 2.0, dtype=torch.double, device=value.device))
        - 0.5 * (dimension * math.log(degrees * math.pi) + logdet)
        - 0.5 * (degrees + dimension) * torch.log1p(quadratic / degrees)
    )
