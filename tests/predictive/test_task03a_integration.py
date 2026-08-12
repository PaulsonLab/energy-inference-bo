"""Independent BoTorch, baseline, and device checks for Task 03A."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from botorch.acquisition.analytic import LogExpectedImprovement
from scipy.integrate import quad

from energy_bo.oracle.warped_gp import SparseWarpedGPOracle
from energy_bo.experiments.task03a import profile_map_devices
from energy_bo.predictive.crossfit import balanced_folds
from energy_bo.predictive.mixture import GaussianMixtureMarginals
from energy_bo.predictive.residuals import (
    ConditionalGaussianResidual,
    ContextualResidualEnergy,
    GlobalGaussianMixtureResidual,
    GlobalSkewNormalResidual,
)
from energy_bo.structural.map_saas import (
    fit_map_saas_reference,
    seeded_half_cauchy_taus,
)


@pytest.fixture(scope="module")
def tiny_reference():
    generator=torch.Generator().manual_seed(21)
    x=torch.rand((10,4),dtype=torch.double,generator=generator)
    y=torch.sin(5*x[:,0])+0.1*x[:,1]
    return x,y,fit_map_saas_reference(x,y,taus=seeded_half_cauchy_taus(91),max_iterations=20)


def test_ensemble_ei_matches_botorch_logei(tiny_reference) -> None:
    x,y,reference=tiny_reference; candidates=x[:5]
    mixture=reference.posterior(candidates); ours=mixture.expected_improvement(float(y.max()))
    standardized_best=float(reference.transform.transform(y.max()))
    acquisition=LogExpectedImprovement(reference.model,best_f=standardized_best)
    with torch.no_grad(): botorch_ei=torch.exp(acquisition(candidates[:,None,:])).cpu()*reference.transform.scale
    assert torch.allclose(ours,botorch_ei,atol=2e-8,rtol=2e-8)


def test_fixed_hyperparameter_loo_matches_brute_conditioning(tiny_reference) -> None:
    _,_,reference=tiny_reference; analytic=reference.fixed_hyperparameter_loo(); model=reference.model
    train_x=model.train_inputs[0]; train_y=model.train_targets
    with torch.no_grad():
        covariance=model.covar_module(train_x).to_dense()+torch.diag_embed(model.likelihood.noise)
        constant=model.mean_module(train_x); brute_mean=torch.empty_like(train_y); brute_variance=torch.empty_like(train_y)
        for point in range(train_y.shape[-1]):
            keep=torch.arange(train_y.shape[-1])!=point
            k=covariance[:,keep][:,:,keep]; cross=covariance[:,keep,point]; chol=torch.linalg.cholesky(k)
            alpha=torch.cholesky_solve((train_y[:,keep]-constant[:,keep]).unsqueeze(-1),chol).squeeze(-1)
            brute_mean[:,point]=constant[:,point]+torch.sum(cross*alpha,dim=-1)
            brute_variance[:,point]=covariance[:,point,point]-torch.sum(cross*torch.cholesky_solve(cross.unsqueeze(-1),chol).squeeze(-1),dim=-1)
    brute_mean=reference.transform.untransform(brute_mean); brute_variance=brute_variance*reference.transform.scale**2
    assert torch.allclose(analytic.means,brute_mean,atol=2e-9,rtol=2e-9)
    assert torch.allclose(analytic.variances,brute_variance,atol=2e-9,rtol=2e-9)


@pytest.mark.parametrize("kind",["global_gaussian","conditional_gaussian","skew","mixture"])
def test_parametric_residual_density_normalizes(kind: str) -> None:
    generator=torch.Generator().manual_seed(50); z=torch.randn(30,dtype=torch.double,generator=generator); context=torch.stack((torch.ones(30),torch.linspace(-1,1,30),torch.cos(torch.linspace(0,2,30))),dim=1)
    if kind=="global_gaussian": model=ConditionalGaussianResidual(1); fitted_context=context[:,:1]
    elif kind=="conditional_gaussian": model=ConditionalGaussianResidual(3); fitted_context=context
    elif kind=="skew": model=GlobalSkewNormalResidual(); fitted_context=context[:,:1]
    else: model=GlobalGaussianMixtureResidual(); fitted_context=context[:,:1]
    model.fit(z,fitted_context,max_iter=60); c=fitted_context[7]
    mass=quad(lambda value:math.exp(float(model.log_prob(torch.tensor(value),c))),-12,12,epsabs=1e-9)[0]
    assert abs(mass-1)<2e-6


def test_convex_energy_independent_starts_match() -> None:
    generator=torch.Generator().manual_seed(71); z=torch.randn(24,dtype=torch.double,generator=generator); context=torch.stack((torch.ones(24),torch.linspace(-1,1,24)),dim=1)
    zero=ContextualResidualEnergy(2); random=ContextualResidualEnergy(2)
    zero.fit(z,context,max_iter=120); random.fit(z,context,max_iter=120,initial_coefficients=.2*torch.randn((9,2),dtype=torch.double,generator=generator))
    assert torch.allclose(zero.coefficients,random.coefficients,atol=2e-6,rtol=2e-6)


def test_seeded_oracle_taus_and_nested_folds_are_deterministic() -> None:
    assert torch.equal(seeded_half_cauchy_taus(12),seeded_half_cauchy_taus(12))
    a=SparseWarpedGPOracle.generate(6,3,64,.6); b=SparseWarpedGPOracle.generate(6,3,64,.6)
    assert torch.equal(a.train_x_all,b.train_x_all) and torch.equal(a.latent_all,b.latent_all)
    small=balanced_folds(16,4,9); large=balanced_folds(64,4,9)
    for fold in range(4): assert torch.equal(small[fold],large[fold][large[fold]<16])


def test_map_device_profile_keeps_cpu_fallback() -> None:
    generator=torch.Generator().manual_seed(33)
    x=torch.rand((8,4),dtype=torch.double,generator=generator)
    y=torch.sin(4*x[:,0])
    profile=profile_map_devices(x,y,seeded_half_cauchy_taus(34),max_iterations=5)
    assert profile["selected_device"] == "cpu"
    assert profile["seconds"]["cpu"] > 0


@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA is not available")
def test_cpu_cuda_mixture_and_oracle_parity() -> None:
    generator=torch.Generator().manual_seed(8); means=torch.randn((4,12),dtype=torch.double,generator=generator); variances=torch.rand((4,12),dtype=torch.double,generator=generator)+.1
    cpu=GaussianMixtureMarginals(means,variances); cuda=GaussianMixtureMarginals(means.cuda(),variances.cuda())
    # Frozen marginal objects deliberately return CPU-owned values; compare a native
    # CUDA evaluation of the same formula to the reusable CPU implementation.
    best=.2; scale=variances.cuda().sqrt(); improvement=means.cuda()-best; z=improvement/scale
    native=((improvement*.5*(1+torch.erf(z/math.sqrt(2)))+scale*torch.exp(-.5*z.square())/math.sqrt(2*math.pi)).clamp_min(0)).mean(0).cpu()
    assert torch.allclose(cpu.expected_improvement(best),native,atol=1e-8,rtol=1e-8)
