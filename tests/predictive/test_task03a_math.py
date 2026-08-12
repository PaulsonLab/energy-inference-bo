"""Mathematical and provenance checks for Task 03A."""

from __future__ import annotations

import math

import numpy as np
import torch
from scipy.integrate import quad

from energy_bo.oracle.warped_gp import SparseWarpedGPOracle
from energy_bo.predictive.corrected import PITCorrectedPredictive
from energy_bo.predictive.crossfit import HeldOutPrediction, balanced_folds
from energy_bo.predictive.mixture import GaussianMixtureMarginals
from energy_bo.predictive.residuals import ContextualResidualEnergy
from energy_bo.experiments.task03a import Task03AConfig, analyze_full_gates, charged_fit_seconds


def toy_mixture(points: int = 3) -> GaussianMixtureMarginals:
    means = torch.tensor([[-0.4, 0.1, 0.7], [0.6, -0.2, 1.1]], dtype=torch.double)[:, :points]
    variances = torch.tensor([[0.5, 0.8, 0.4], [0.9, 0.6, 0.7]], dtype=torch.double)[:, :points]
    return GaussianMixtureMarginals(means, variances, torch.tensor([0.35, 0.65]))


def test_gaussian_mixture_normalizes_and_matches_moments() -> None:
    mixture = toy_mixture(1)
    mass = quad(lambda y: math.exp(float(mixture.log_prob(y))), -np.inf, np.inf)[0]
    generator = torch.Generator().manual_seed(4)
    samples = mixture.sample(150_000, generator)[:, 0]
    assert abs(mass - 1) < 1e-9
    assert abs(float(samples.mean()) - float(mixture.mean[0])) < 0.01
    assert abs(float(samples.var(unbiased=False)) - float(mixture.variance[0])) < 0.02


def test_mixture_cdf_and_pit_against_monte_carlo() -> None:
    mixture = toy_mixture(1)
    generator = torch.Generator().manual_seed(5)
    samples = mixture.sample(100_000, generator)[:, 0]
    value = 0.2
    assert abs(float(mixture.cdf(value)) - float((samples <= value).double().mean())) < 0.005
    probability = mixture.cdf(samples[:20_000]) if mixture.point_count == len(samples[:20_000]) else None
    # Repackage identical marginals so each draw has its own PIT evaluation.
    repeated = GaussianMixtureMarginals(mixture.means.expand(-1, 20_000), mixture.variances.expand(-1, 20_000), mixture.weights)
    z = torch.special.ndtri(repeated.cdf(samples[:20_000]).clamp(1e-12, 1-1e-12))
    assert abs(float(z.mean())) < 0.025
    assert abs(float(z.var(unbiased=False)) - 1) < 0.04


def test_mixture_expected_improvement_matches_quadrature() -> None:
    mixture = toy_mixture(1); best = 0.15
    analytic = float(mixture.expected_improvement(best))
    numeric = quad(lambda y: max(y-best, 0) * math.exp(float(mixture.log_prob(y))), best, np.inf, epsabs=1e-11)[0]
    assert abs(analytic - numeric) < 1e-9


def test_zero_energy_exactly_recovers_reference() -> None:
    reference = toy_mixture(); context = torch.ones((3, 1), dtype=torch.double)
    residual = ContextualResidualEnergy(1)
    corrected = PITCorrectedPredictive(reference, residual, context)
    values = torch.tensor([-0.1, 0.2, 0.9], dtype=torch.double); best = 0.1
    assert torch.equal(corrected.log_prob(values), reference.log_prob(values))
    assert torch.equal(corrected.cdf(values), reference.cdf(values))
    assert torch.equal(corrected.expected_improvement(best), reference.expected_improvement(best))
    g1=torch.Generator().manual_seed(8); g2=torch.Generator().manual_seed(8)
    assert torch.equal(corrected.sample(12,g1), reference.sample(12,g2))


def test_energy_normalizer_gradient_hessian_and_convexity() -> None:
    model = ContextualResidualEnergy(3, l2_precision=10)
    generator = torch.Generator().manual_seed(10)
    z = torch.randn(12, dtype=torch.double, generator=generator)
    context = torch.randn(12,3,dtype=torch.double,generator=generator); context[:,0]=1
    coefficients = (0.05*torch.randn(9,3,dtype=torch.double,generator=generator)).requires_grad_()
    adaptive = quad(lambda x: math.exp(-.5*x*x)/math.sqrt(2*math.pi)*math.exp(-float(model.energy(torch.tensor(x), context[0], coefficients.detach()))),-10,10,epsabs=1e-11)[0]
    assert abs(float(model.log_normalizer(context[0], coefficients.detach()).exp())-adaptive)<2e-7
    analytic=torch.autograd.grad(model.objective(z,context,coefficients),coefficients)[0]
    direction=torch.randn_like(coefficients,generator=generator); eps=1e-6
    numeric=(model.objective(z,context,coefficients.detach()+eps*direction)-model.objective(z,context,coefficients.detach()-eps*direction))/(2*eps)
    assert abs(float((analytic*direction).sum()-numeric))<2e-5
    flat=coefficients.detach().reshape(-1).requires_grad_()
    hessian=torch.autograd.functional.hessian(lambda value:model.objective(z,context,value.reshape(9,3)),flat)
    assert float(torch.linalg.eigvalsh(hessian).min()) >= 10-1e-8


def test_crossfit_partition_and_provenance_are_disjoint() -> None:
    folds=balanced_folds(17,4,12)
    assert sorted(torch.cat(folds).tolist())==list(range(17))
    HeldOutPrediction((0,1),(2,3),(2,3),toy_mixture(2),0.1)
    try:
        HeldOutPrediction((0,),(0,2),(2,),toy_mixture(1),0.1)
    except ValueError:
        pass
    else:
        raise AssertionError("leaking held-out index was accepted")
    small=balanced_folds(16,4,12); large=balanced_folds(32,4,12)
    for fold in range(4): assert torch.equal(small[fold],large[fold][large[fold]<16])


def test_warped_gp_oracle_density_cdf_moments_and_ei() -> None:
    oracle=SparseWarpedGPOracle.generate(6,2,32,alpha=.6)
    test=torch.rand((4,6),dtype=torch.double,generator=torch.Generator().manual_seed(9))
    posterior=oracle.posterior(test,16); best=float(oracle.outcomes(16).max())
    generator=torch.Generator().manual_seed(11)
    latent=posterior.mean+posterior.variance.sqrt()*torch.randn((200_000,4),dtype=torch.double,generator=generator)
    samples=posterior.transform(latent)
    assert torch.max((samples.mean(0)-posterior.predictive_mean).abs())<.015
    assert torch.max((samples.var(0,unbiased=False)-posterior.predictive_variance).abs())<.04
    empirical=(samples>best).double().mean(0); assert torch.max((empirical-posterior.probability_improvement(best)).abs())<.005
    empirical_ei=(samples-best).clamp_min(0).mean(0); assert torch.max((empirical_ei-posterior.expected_improvement(best)).abs())<.005


def test_corrected_density_cdf_and_ei_are_consistent() -> None:
    reference=toy_mixture(1); residual=ContextualResidualEnergy(1)
    residual.coefficients[:,0]=torch.linspace(-.15,.15,9)
    corrected=PITCorrectedPredictive(reference,residual,torch.ones((1,1)))
    mass=quad(lambda y:math.exp(float(corrected.log_prob(y))),-8,8,epsabs=2e-7)[0]
    best=.1; numeric=quad(lambda y:(y-best)*math.exp(float(corrected.log_prob(y))),best,8,epsabs=2e-7)[0]
    assert abs(mass-1)<2e-5
    assert abs(float(corrected.expected_improvement(best,quadrature_points=160))-numeric)<3e-5
    grid=torch.linspace(-8,8,101); repeated=GaussianMixtureMarginals(reference.means.expand(-1,101),reference.variances.expand(-1,101),reference.weights)
    c=PITCorrectedPredictive(repeated,residual,torch.ones((101,1))).cdf(grid)
    assert torch.all(c[1:]>=c[:-1]) and c[0]<1e-8 and c[-1]>1-1e-8
    samples=corrected.sample(20,torch.Generator().manual_seed(44))
    assert samples.shape==(20,1) and torch.isfinite(samples).all()


def test_charged_timing_is_exact_sum() -> None:
    assert charged_fit_seconds(3.0,2.0,.25)==5.25


def test_incomplete_smoke_cannot_evaluate_full_gates() -> None:
    status=analyze_full_gates(Task03AConfig.smoke(),[],[])
    assert not status["eligible"] and status["decision"]=="NOT_EVALUATED_FROM_SMOKE"
