"""Small PIT-normal residual families for Task 03A."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch
from numpy.polynomial.hermite import hermgauss
from scipy.optimize import minimize
from scipy.special import ndtr


LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)


@dataclass(frozen=True)
class ResidualFit:
    objective: float
    nll: float
    penalty: float
    iterations: int
    converged: bool


class ResidualDistribution(Protocol):
    is_identity: bool

    def log_prob(self, z: torch.Tensor, context: torch.Tensor) -> torch.Tensor: ...
    def cdf(self, z: torch.Tensor, context: torch.Tensor) -> torch.Tensor: ...
    def sample(self, context: torch.Tensor, generator: torch.Generator) -> torch.Tensor: ...


def normal_log_prob(z: torch.Tensor) -> torch.Tensor:
    return -0.5 * z.square() - LOG_SQRT_2PI


class ContextualResidualEnergy:
    """Convex affine energy in centered RBF and fixed context features."""

    def __init__(
        self,
        context_dimension: int,
        *,
        centers: torch.Tensor | None = None,
        bandwidth: float = 0.8,
        quadrature_points: int = 64,
        l2_precision: float = 10.0,
    ) -> None:
        if context_dimension < 1 or bandwidth <= 0 or l2_precision <= 0:
            raise ValueError("invalid residual-energy configuration")
        self.context_dimension = int(context_dimension)
        self.centers = (
            torch.linspace(-3.0, 3.0, 9, dtype=torch.double)
            if centers is None else centers.detach().double().clone()
        )
        self.bandwidth = float(bandwidth)
        self.l2_precision = float(l2_precision)
        nodes, weights = hermgauss(quadrature_points)
        self.nodes = torch.tensor(np.sqrt(2.0) * nodes, dtype=torch.double)
        self.weights = torch.tensor(weights / np.sqrt(np.pi), dtype=torch.double)
        h2 = bandwidth**2
        self.reference_basis_mean = (
            bandwidth / math.sqrt(1.0 + h2)
            * torch.exp(-self.centers.square() / (2.0 * (1.0 + h2)))
        )
        self.coefficients = torch.zeros(
            (self.centers.numel(), self.context_dimension), dtype=torch.double
        )

    @property
    def is_identity(self) -> bool:
        return bool(torch.count_nonzero(self.coefficients) == 0)

    @property
    def parameter_count(self) -> int:
        return self.coefficients.numel()

    def basis(self, z: torch.Tensor) -> torch.Tensor:
        z = torch.as_tensor(z, dtype=torch.double)
        centers = self.centers.to(z.device); reference_mean = self.reference_basis_mean.to(z.device)
        raw = torch.exp(-0.5 * ((z[..., None] - centers) / self.bandwidth).square())
        return raw - reference_mean

    def energy(
        self, z: torch.Tensor, context: torch.Tensor, coefficients: torch.Tensor | None = None
    ) -> torch.Tensor:
        matrix = (self.coefficients if coefficients is None else coefficients).to(z.device)
        return torch.einsum("...k,kh,...h->...", self.basis(z), matrix, context.double())

    def log_normalizer(
        self, context: torch.Tensor, coefficients: torch.Tensor | None = None
    ) -> torch.Tensor:
        device = (self.coefficients if coefficients is None else coefficients).device
        context = torch.as_tensor(context, dtype=torch.double, device=device)
        matrix = (self.coefficients if coefficients is None else coefficients).to(context.device)
        node_basis = self.basis(self.nodes.to(context.device))
        energy = torch.einsum("qk,kh,...h->...q", node_basis, matrix, context)
        return torch.logsumexp(self.weights.to(context.device).log() - energy, dim=-1)

    def log_prob(
        self, z: torch.Tensor, context: torch.Tensor, coefficients: torch.Tensor | None = None
    ) -> torch.Tensor:
        return (
            normal_log_prob(torch.as_tensor(z, dtype=torch.double))
            - self.energy(z, context, coefficients)
            - self.log_normalizer(context, coefficients)
        )

    def objective(
        self, z: torch.Tensor, context: torch.Tensor, coefficients: torch.Tensor
    ) -> torch.Tensor:
        return -self.log_prob(z, context, coefficients).sum() + (
            0.5 * self.l2_precision * coefficients.square().sum()
        )

    def fit(
        self,
        z: torch.Tensor,
        context: torch.Tensor,
        max_iter: int = 250,
        initial_coefficients: torch.Tensor | None = None,
    ) -> ResidualFit:
        z = torch.as_tensor(z, dtype=torch.double).reshape(-1)
        context = torch.as_tensor(context, dtype=torch.double, device=self.coefficients.device)
        if context.shape != (z.numel(), self.context_dimension):
            raise ValueError("context must have shape [samples, context_dimension]")
        initial = (
            torch.zeros_like(self.coefficients)
            if initial_coefficients is None
            else torch.as_tensor(initial_coefficients, dtype=torch.double).clone()
        )
        if initial.shape != self.coefficients.shape:
            raise ValueError("initial coefficients have the wrong shape")
        coefficients = torch.nn.Parameter(initial)
        optimizer = torch.optim.LBFGS(
            [coefficients], max_iter=max_iter, line_search_fn="strong_wolfe",
            tolerance_grad=1e-10, tolerance_change=1e-12,
        )
        calls = 0

        def closure() -> torch.Tensor:
            nonlocal calls
            optimizer.zero_grad()
            value = self.objective(z, context, coefficients)
            value.backward()
            calls += 1
            return value

        optimizer.step(closure)
        self.coefficients = coefficients.detach().clone()
        nll = float(-self.log_prob(z, context).sum())
        penalty = float(0.5 * self.l2_precision * self.coefficients.square().sum())
        return ResidualFit(nll + penalty, nll, penalty, calls, True)

    def cdf(self, z: torch.Tensor, context: torch.Tensor, points: int = 160) -> torch.Tensor:
        z = torch.as_tensor(z, dtype=torch.double)
        context = torch.as_tensor(context, dtype=torch.double)
        if self.is_identity:
            return 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
        nodes, weights = np.polynomial.legendre.leggauss(points)
        nodes = torch.tensor(nodes, dtype=torch.double, device=z.device)
        weights = torch.tensor(weights, dtype=torch.double, device=z.device)
        upper = z.clamp(-10.0, 10.0)
        transformed = -10.0 + (upper[..., None] + 10.0) * (nodes + 1.0) / 2.0
        integral = (upper + 10.0) / 2.0 * torch.sum(
            weights * torch.exp(self.log_prob(transformed, context[..., None, :])), dim=-1
        )
        return torch.where(z <= -10, torch.zeros_like(z), torch.where(z >= 10, torch.ones_like(z), integral)).clamp(0, 1)

    def sample(self, context: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        context = torch.as_tensor(context, dtype=torch.double)
        uniform = torch.rand(context.shape[:-1], dtype=torch.double, device=context.device, generator=generator)
        lower, upper = torch.full_like(uniform, -10.0), torch.full_like(uniform, 10.0)
        for _ in range(70):
            middle = 0.5 * (lower + upper)
            go_right = self.cdf(middle, context) < uniform
            lower = torch.where(go_right, middle, lower)
            upper = torch.where(go_right, upper, middle)
        return 0.5 * (lower + upper)

    def to(self, device: torch.device | str) -> "ContextualResidualEnergy":
        self.centers=self.centers.to(device); self.nodes=self.nodes.to(device); self.weights=self.weights.to(device); self.reference_basis_mean=self.reference_basis_mean.to(device); self.coefficients=self.coefficients.to(device)
        return self

    def correction_kl(self, context: torch.Tensor) -> torch.Tensor:
        context=torch.as_tensor(context,dtype=torch.double)
        nodes=self.nodes.to(context.device); weights=self.weights.to(context.device)
        energy=self.energy(nodes[:,None],context[None,:,:])
        log_normalizer=self.log_normalizer(context)[None,:]
        log_ratio=-energy-log_normalizer
        ratio=torch.exp(log_ratio)
        return torch.sum(weights[:,None]*ratio*log_ratio,dim=0).clamp_min(0)


class ConditionalGaussianResidual:
    """Global or context-matched Gaussian PIT calibration."""

    def __init__(self, context_dimension: int, l2_precision: float = 10.0) -> None:
        self.context_dimension = int(context_dimension)
        self.l2_precision = float(l2_precision)
        self.location = torch.zeros(context_dimension, dtype=torch.double)
        self.log_scale = torch.zeros(context_dimension, dtype=torch.double)

    @property
    def is_identity(self) -> bool:
        return bool(torch.count_nonzero(self.location) + torch.count_nonzero(self.log_scale) == 0)

    def _parameters(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        context = torch.as_tensor(context, dtype=torch.double, device=self.location.device)
        return context @ self.location, torch.exp((context @ self.log_scale).clamp(-5, 5))

    def log_prob(self, z: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        location, scale = self._parameters(context)
        return normal_log_prob((z - location) / scale) - scale.log()

    def cdf(self, z: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        location, scale = self._parameters(context)
        return 0.5 * (1 + torch.erf((z - location) / scale / math.sqrt(2)))

    def sample(self, context: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        location, scale = self._parameters(context)
        return location + scale * torch.randn(location.shape, dtype=torch.double, device=location.device, generator=generator)

    def to(self, device: torch.device | str) -> "ConditionalGaussianResidual":
        self.location=self.location.to(device); self.log_scale=self.log_scale.to(device); return self

    def fit(self, z: torch.Tensor, context: torch.Tensor, max_iter: int = 250) -> ResidualFit:
        z = torch.as_tensor(z, dtype=torch.double).reshape(-1)
        context = torch.as_tensor(context, dtype=torch.double)
        parameter = torch.nn.Parameter(torch.zeros(2, self.context_dimension, dtype=torch.double))
        optimizer = torch.optim.LBFGS([parameter], max_iter=max_iter, line_search_fn="strong_wolfe")
        calls = 0
        def closure() -> torch.Tensor:
            nonlocal calls
            optimizer.zero_grad()
            location = context @ parameter[0]
            log_scale = (context @ parameter[1]).clamp(-5, 5)
            nll = (0.5 * ((z - location) * torch.exp(-log_scale)).square() + log_scale + LOG_SQRT_2PI).sum()
            value = nll + 0.5 * self.l2_precision * parameter.square().sum()
            value.backward(); calls += 1
            return value
        optimizer.step(closure)
        self.location, self.log_scale = parameter.detach().clone()
        nll = float(-self.log_prob(z, context).sum())
        penalty = float(0.5 * self.l2_precision * parameter.detach().square().sum())
        return ResidualFit(nll + penalty, nll, penalty, calls, True)


class GlobalSkewNormalResidual:
    """Regularized global skew-normal calibration fitted by deterministic multistart."""

    def __init__(self, l2_precision: float = 10.0) -> None:
        self.l2_precision = float(l2_precision)
        self.parameters = np.zeros(3)
        self.is_identity = False

    def fit(self, z: torch.Tensor, context: torch.Tensor, max_iter: int = 250) -> ResidualFit:
        values = np.asarray(z.detach().double())
        def objective(p: np.ndarray) -> float:
            loc, log_scale, shape = p
            scale = np.exp(np.clip(log_scale, -5, 5))
            u = (values - loc) / scale
            logcdf = np.log(np.maximum(ndtr(shape * u), 1e-300))
            nll = np.sum(0.5 * u**2 + LOG_SQRT_2PI + np.log(scale) - math.log(2) - logcdf)
            return float(nll + 0.5 * self.l2_precision * np.dot(p, p))
        results = [minimize(objective, np.array([0.0, 0.0, a]), method="L-BFGS-B", options={"maxiter": max_iter}) for a in (-4,-2,0,2,4)]
        best = min(results, key=lambda result: result.fun)
        self.parameters = best.x
        penalty = 0.5 * self.l2_precision * float(np.dot(best.x, best.x))
        return ResidualFit(float(best.fun), float(best.fun - penalty), penalty, int(best.nit), bool(best.success))

    def log_prob(self, z: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        loc, log_scale, shape = map(float, self.parameters)
        scale = math.exp(np.clip(log_scale, -5, 5)); u = (z - loc) / scale
        return math.log(2) + normal_log_prob(u) + torch.special.log_ndtr(shape * u) - math.log(scale)

    def cdf(self, z: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        from scipy.stats import skewnorm
        loc, log_scale, shape = map(float, self.parameters)
        result = skewnorm.cdf(
            z.detach().cpu().numpy(), shape, loc=loc, scale=math.exp(log_scale)
        )
        return torch.as_tensor(result, dtype=torch.double, device=z.device)

    def sample(self, context: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        loc, log_scale, shape = map(float, self.parameters); delta = shape / math.sqrt(1 + shape**2)
        u = torch.randn(context.shape[:-1], dtype=torch.double, generator=generator)
        v = torch.randn(context.shape[:-1], dtype=torch.double, generator=generator)
        return loc + math.exp(log_scale) * (delta * u.abs() + math.sqrt(1-delta**2) * v)

    def to(self, device: torch.device | str) -> "GlobalSkewNormalResidual": return self


class GlobalGaussianMixtureResidual:
    """Strongly bounded two-Gaussian residual calibration."""

    def __init__(self, l2_precision: float = 10.0) -> None:
        self.l2_precision = float(l2_precision)
        self.parameters = np.array([0.0, -0.2, 0.2, 0.0, 0.0])
        self.is_identity = False

    def _unpack(self, p: np.ndarray | None = None) -> tuple[float,float,float,float,float]:
        p = self.parameters if p is None else p
        weight = 0.1 + 0.8 / (1 + np.exp(-p[0]))
        means = np.sort(p[1:3]); scales = np.clip(np.exp(p[3:5]), 0.2, 5.0)
        return float(weight), float(means[0]), float(means[1]), float(scales[0]), float(scales[1])

    def fit(self, z: torch.Tensor, context: torch.Tensor, max_iter: int = 250) -> ResidualFit:
        values = np.asarray(z.detach().double())
        def objective(p: np.ndarray) -> float:
            w,m1,m2,s1,s2 = self._unpack(p)
            d1=np.exp(-.5*((values-m1)/s1)**2)/(math.sqrt(2*math.pi)*s1)
            d2=np.exp(-.5*((values-m2)/s2)**2)/(math.sqrt(2*math.pi)*s2)
            return float(-np.log(np.maximum(w*d1+(1-w)*d2,1e-300)).sum()+.5*self.l2_precision*np.dot(p,p))
        starts=[np.zeros(5)]
        for gap in (.25,.75):
            for logit in (-1,1):
                for scale in (-.3,.3): starts.append(np.array([logit,-gap,gap,scale,scale]))
        starts=starts[:8]
        results=[minimize(objective,s,method="L-BFGS-B",options={"maxiter":max_iter}) for s in starts]
        best=min(results,key=lambda r:r.fun); self.parameters=best.x
        penalty=.5*self.l2_precision*float(np.dot(best.x,best.x))
        return ResidualFit(float(best.fun),float(best.fun-penalty),penalty,int(best.nit),bool(best.success))

    def log_prob(self,z:torch.Tensor,context:torch.Tensor)->torch.Tensor:
        w,m1,m2,s1,s2=self._unpack(); a=math.log(w)+normal_log_prob((z-m1)/s1)-math.log(s1); b=math.log(1-w)+normal_log_prob((z-m2)/s2)-math.log(s2)
        return torch.logsumexp(torch.stack((a,b)),dim=0)
    def cdf(self,z:torch.Tensor,context:torch.Tensor)->torch.Tensor:
        w,m1,m2,s1,s2=self._unpack(); return w*.5*(1+torch.erf((z-m1)/s1/math.sqrt(2)))+(1-w)*.5*(1+torch.erf((z-m2)/s2/math.sqrt(2)))
    def sample(self,context:torch.Tensor,generator:torch.Generator)->torch.Tensor:
        w,m1,m2,s1,s2=self._unpack(); choose=torch.rand(context.shape[:-1],dtype=torch.double,generator=generator)<w; normal=torch.randn(context.shape[:-1],dtype=torch.double,generator=generator); return torch.where(choose,m1+s1*normal,m2+s2*normal)
    def to(self,device:torch.device|str)->"GlobalGaussianMixtureResidual": return self
