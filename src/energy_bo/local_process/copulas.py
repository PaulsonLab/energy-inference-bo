"""Matched-marginal Gaussian and Student-t copula oracle utilities."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import torch
from scipy.optimize import brentq
from scipy.special import gammaln, ndtr, ndtri
from scipy.stats import chi2, qmc, t

DEFAULT_LATENT_RHO = 0.5132792667304108


def _clip_unit(values: np.ndarray, epsilon: float = 1e-12) -> tuple[np.ndarray, int]:
    clipped = np.clip(values, epsilon, 1.0 - epsilon)
    return clipped, int(np.count_nonzero(clipped != values))


def _student_scores(base: np.ndarray, rho: float, df: float) -> tuple[np.ndarray, np.ndarray, int]:
    unit, clamps = _clip_unit(base)
    first = ndtri(unit[:, 0])
    second = rho * first + math.sqrt(1.0 - rho * rho) * ndtri(unit[:, 1])
    scale = np.sqrt(chi2.ppf(unit[:, 2], df) / df)
    uniforms = np.column_stack((t.cdf(first / scale, df), t.cdf(second / scale, df)))
    uniforms, more = _clip_unit(uniforms, 1e-15)
    scores = ndtri(uniforms)
    return scores[:, 0], scores[:, 1], clamps + more


@dataclass(frozen=True)
class CopulaCalibration:
    latent_rho: float
    target_correlation: float
    calibration_correlation: float
    verification_correlations: tuple[float, ...]
    calibration_power: int
    verification_power: int
    clamp_count: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def calibrate_t_copula_correlation(
    *,
    target: float = 0.5,
    df: float = 3.0,
    calibration_power: int = 20,
    verification_power: int = 18,
    verification_replicates: int = 4,
    seed: int = 40_405,
) -> CopulaCalibration:
    """Deterministically calibrate latent t correlation to normal-score Pearson correlation."""
    base = qmc.Sobol(3, scramble=True, seed=seed).random_base2(calibration_power)
    clamp_count = 0

    def residual(rho: float) -> float:
        nonlocal clamp_count
        first, second, clamps = _student_scores(base, rho, df)
        clamp_count = max(clamp_count, clamps)
        return float(np.mean(first * second) - target)

    latent = float(brentq(residual, -0.99, 0.99, xtol=1e-13, rtol=1e-13))
    first, second, clamps = _student_scores(base, latent, df)
    clamp_count = max(clamp_count, clamps)
    verification: list[float] = []
    for replicate in range(verification_replicates):
        independent = qmc.Sobol(3, scramble=True, seed=seed + 1 + replicate).random_base2(verification_power)
        left, right, count = _student_scores(independent, latent, df)
        clamp_count += count
        verification.append(float(np.mean(left * right)))
    return CopulaCalibration(
        latent,
        target,
        float(np.mean(first * second)),
        tuple(verification),
        calibration_power,
        verification_power,
        clamp_count,
    )


def gaussian_log_prob(z: np.ndarray, rho: float = 0.5) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    quadratic = (z[..., 0] ** 2 - 2 * rho * z[..., 0] * z[..., 1] + z[..., 1] ** 2) / (1 - rho**2)
    return -math.log(2 * math.pi) - 0.5 * math.log(1 - rho**2) - 0.5 * quadratic


def student_normal_score_log_prob(
    z: np.ndarray,
    *,
    latent_rho: float = DEFAULT_LATENT_RHO,
    df: float = 3.0,
) -> np.ndarray:
    """Log density after applying normal scores to a bivariate Student-t copula."""
    z = np.asarray(z, dtype=np.float64)
    uniforms, _ = _clip_unit(ndtr(z), 1e-15)
    x = t.ppf(uniforms, df)
    rho = latent_rho
    quadratic = (x[..., 0] ** 2 - 2 * rho * x[..., 0] * x[..., 1] + x[..., 1] ** 2) / (1 - rho**2)
    log_joint = (
        gammaln((df + 2) / 2)
        - gammaln(df / 2)
        - math.log(df * math.pi)
        - 0.5 * math.log(1 - rho**2)
        - 0.5 * (df + 2) * np.log1p(quadratic / df)
    )
    log_marginal = (
        gammaln((df + 1) / 2)
        - gammaln(df / 2)
        - 0.5 * math.log(df * math.pi)
        - 0.5 * (df + 1) * np.log1p(x**2 / df)
    ).sum(-1)
    log_phi = (-0.5 * z**2 - 0.5 * math.log(2 * math.pi)).sum(-1)
    return log_phi + log_joint - log_marginal


@dataclass(frozen=True)
class CopulaEndpointSamples:
    gaussian: torch.Tensor
    student: torch.Tensor
    clamp_count: int


@dataclass(frozen=True)
class MatchedCopulaOracle:
    rho: float = 0.5
    df: float = 3.0
    latent_rho: float = DEFAULT_LATENT_RHO

    def endpoint_qmc(self, power: int, seed: int) -> CopulaEndpointSamples:
        base = qmc.Sobol(3, scramble=True, seed=seed).random_base2(power)
        unit, clamps = _clip_unit(base)
        first, second = ndtri(unit[:, 0]), ndtri(unit[:, 1])
        gaussian = np.column_stack((first, self.rho * first + math.sqrt(1 - self.rho**2) * second))
        student_first, student_second, more = _student_scores(base, self.latent_rho, self.df)
        student = np.column_stack((student_first, student_second))
        return CopulaEndpointSamples(torch.from_numpy(gaussian).double(), torch.from_numpy(student).double(), clamps + more)

    def log_prob(self, z: torch.Tensor, context: float | torch.Tensor) -> torch.Tensor:
        device = z.device
        values = z.detach().double().cpu().numpy()
        gaussian = gaussian_log_prob(values, self.rho)
        student = student_normal_score_log_prob(values, latent_rho=self.latent_rho, df=self.df)
        r = torch.as_tensor(context, dtype=torch.double).detach().cpu().numpy()
        with np.errstate(divide="ignore"):
            log_mix = np.logaddexp(np.log1p(-r) + gaussian, np.log(r) + student)
        log_mix = np.where(r == 0, gaussian, np.where(r == 1, student, log_mix))
        return torch.as_tensor(log_mix, dtype=torch.double, device=device)

    def sample_training(self, contexts: torch.Tensor, seed: int) -> torch.Tensor:
        contexts_np = contexts.detach().double().cpu().numpy()
        rng = np.random.default_rng(seed)
        count = len(contexts_np)
        choose_student = rng.random(count) < contexts_np
        normals = rng.standard_normal((count, 2))
        gaussian = np.column_stack((normals[:, 0], self.rho * normals[:, 0] + math.sqrt(1 - self.rho**2) * normals[:, 1]))
        scales = np.sqrt(rng.chisquare(self.df, count) / self.df)
        correlated = np.column_stack((normals[:, 0], self.latent_rho * normals[:, 0] + math.sqrt(1 - self.latent_rho**2) * normals[:, 1]))
        uniforms = t.cdf(correlated / scales[:, None], self.df)
        uniforms, _ = _clip_unit(uniforms, 1e-15)
        student = ndtri(uniforms)
        return torch.from_numpy(np.where(choose_student[:, None], student, gaussian)).double()

    @staticmethod
    def q1_ei(best: float) -> float:
        density = math.exp(-0.5 * best**2) / math.sqrt(2 * math.pi)
        tail = 0.5 * math.erfc(best / math.sqrt(2))
        return density - best * tail

    @staticmethod
    def endpoint_qei(samples: CopulaEndpointSamples, mean: torch.Tensor, scale: torch.Tensor, best: float, *, chunk: int = 16) -> tuple[torch.Tensor, torch.Tensor]:
        mean, scale = mean.double(), scale.double()

        def evaluate(z: torch.Tensor) -> torch.Tensor:
            output: list[torch.Tensor] = []
            for start in range(0, len(mean), chunk):
                values = mean[start:start+chunk, None, :] + scale[start:start+chunk, None, :] * z[None]
                output.append((values.max(-1).values - best).clamp_min(0).mean(-1))
            return torch.cat(output)

        return evaluate(samples.gaussian), evaluate(samples.student)
