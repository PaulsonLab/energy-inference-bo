from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from scipy.integrate import quad

from energy_bo.local_process.energy import LocalEnergyModel, centered_rbf_basis, neighbor_summary
from energy_bo.local_process.geometry import build_geometry, candidate_geometry, gaussian_factor_log_prob, matern52, ordered_sobol
from energy_bo.local_process.truth import U_TRUE, V_TRUE, generate_oracle, inverse_grid_sample


def _fixture(count: int = 12):
    x = ordered_sobol(3, count, 17)
    values = torch.sin(3 * x[:, 0]) + 0.2 * torch.cos(5 * x[:, 1])
    return x, values


def test_full_conditioning_matches_exact_gp_and_truncation_matches_manual_product():
    x, _ = _fixture(9)
    covariance = matern52(x, x) + 1e-8 * torch.eye(len(x), dtype=torch.double)
    generator = torch.Generator().manual_seed(9)
    y = torch.linalg.cholesky(covariance) @ torch.randn(len(x), dtype=torch.double, generator=generator)
    geometry = build_geometry(x, m=len(x))
    exact = torch.distributions.MultivariateNormal(torch.zeros(len(x), dtype=torch.double), covariance_matrix=covariance).log_prob(y)
    assert torch.allclose(gaussian_factor_log_prob(y, geometry), exact, atol=1e-10, rtol=1e-10)
    truncated = build_geometry(x, m=3)
    manual = 0.0
    for i in range(len(x)):
        count=min(i,3); indices=truncated.neighbors[i,:count]
        if count:
            covariance=matern52(x[indices],x[indices])+1e-8*torch.eye(count,dtype=torch.double)
            cross=matern52(x[indices],x[i:i+1]).squeeze(-1)
            coefficient=torch.linalg.solve(covariance,cross)
            mean=(coefficient*y[indices]).sum(); variance=(1+1e-8-(coefficient*cross).sum()).clamp_min(1e-12)
            assert torch.allclose(truncated.coefficients[i,:count],coefficient,atol=1e-11,rtol=1e-11)
        else: mean=torch.tensor(0.,dtype=torch.double); variance=torch.tensor(1+1e-8,dtype=torch.double)
        manual += -0.5 * ((y[i] - mean).square() / variance + variance.log() + math.log(2 * math.pi))
    assert torch.allclose(gaussian_factor_log_prob(y, truncated), manual, atol=1e-12)
    assert torch.equal(ordered_sobol(3,9,17),ordered_sobol(3,9,17))


def test_conditionals_normalize_and_zero_correction_recovers_gaussian():
    mean, scale = torch.tensor([0.3], dtype=torch.double), torch.tensor([0.7], dtype=torch.double)
    summary = torch.linspace(-0.2, 0.2, 7, dtype=torch.double)[None]
    for pairwise in (False, True):
        model = LocalEnergyModel(pairwise)
        assert model.is_identity
        integral = quad(lambda value: float(torch.exp(model.log_prob(torch.tensor([value], dtype=torch.double), mean, scale, summary))), -8, 8, epsabs=1e-11)[0]
        assert integral == pytest.approx(1.0, abs=1e-10)
        assert torch.allclose(model.cdf(mean, mean, scale, summary), torch.tensor([0.5], dtype=torch.double), atol=1e-14)
        generator=torch.Generator().manual_seed(2); samples=model.sample(mean.expand(20000),scale.expand(20000),summary.expand(20000,-1),generator)
        assert abs(float(samples.mean())-.3)<.015 and abs(float(samples.var(unbiased=False))-.49)<.02
        model.unary=torch.linspace(-.1,.1,7,dtype=torch.double)
        if pairwise: model.pair=.02*torch.arange(49,dtype=torch.double).reshape(7,7)
        integral = quad(lambda value: float(torch.exp(model.log_prob(torch.tensor([value],dtype=torch.double),mean,scale,summary))),-8,8,epsabs=1e-11,epsrel=1e-11)[0]
        assert integral==pytest.approx(1.0,abs=1e-8)


def test_objective_gradient_hessian_and_optimizer_uniqueness():
    x, y = _fixture(18); geometry = build_geometry(x, m=8)
    mean, scale = geometry.means(y), geometry.variances.sqrt()
    summary = neighbor_summary(y, geometry.neighbors, geometry.mask, geometry.similarity_weights)
    model = LocalEnergyModel(False)
    parameter = (0.02 * torch.arange(7, dtype=torch.double)).requires_grad_()
    gradient = torch.autograd.grad(model.objective(y, mean, scale, summary, parameter), parameter)[0]
    finite = torch.empty_like(parameter); epsilon = 1e-6
    for i in range(len(parameter)):
        direction = torch.zeros_like(parameter); direction[i] = epsilon
        finite[i] = (model.objective(y, mean, scale, summary, parameter.detach()+direction)-model.objective(y, mean, scale, summary, parameter.detach()-direction))/(2*epsilon)
    assert torch.allclose(gradient, finite, atol=1e-6, rtol=1e-6)
    hessian = torch.autograd.functional.hessian(lambda p: model.objective(y, mean, scale, summary, p), parameter.detach())
    assert float(torch.linalg.eigvalsh(hessian).min()) >= 10 - 1e-8
    # At zero correction, the affine-feature Hessian is the summed quadrature
    # covariance plus the exact L2 Hessian.
    zero=torch.zeros(7,dtype=torch.double)
    auto_zero=torch.autograd.functional.hessian(lambda p:model.objective(y,mean,scale,summary,p),zero)
    features=centered_rbf_basis(model.nodes).expand(len(mean),-1,-1); weights=model.weights
    feature_mean=torch.einsum("q,nqk->nk",weights,features)
    second=torch.einsum("q,nqk,nql->nkl",weights,features,features)
    covariance=(second-feature_mean[:,:,None]*feature_mean[:,None,:]).sum(0)+10*torch.eye(7,dtype=torch.double)
    assert torch.allclose(auto_zero,covariance,atol=1e-9,rtol=1e-9)
    fits=[]
    for offset in (0.0, 0.01, -0.01):
        instance=LocalEnergyModel(False); instance.fit(y,mean,scale,summary,max_iter=100,initial=torch.full((7,),offset)); fits.append(instance.unary)
    assert max(float((fits[0]-value).abs().max()) for value in fits[1:]) < 1e-6
    pair_model=LocalEnergyModel(True); pair_parameter=torch.linspace(-.02,.02,56,dtype=torch.double).requires_grad_()
    pair_gradient=torch.autograd.grad(pair_model.objective(y,mean,scale,summary,pair_parameter),pair_parameter)[0]
    direction=torch.randn(56,generator=torch.Generator().manual_seed(20),dtype=torch.double); direction/=direction.norm(); epsilon=1e-6
    directional_fd=(pair_model.objective(y,mean,scale,summary,pair_parameter.detach()+epsilon*direction)-pair_model.objective(y,mean,scale,summary,pair_parameter.detach()-epsilon*direction))/(2*epsilon)
    assert torch.dot(pair_gradient,direction)==pytest.approx(float(directional_fd),abs=1e-6,rel=1e-6)
    pair_fits=[]
    for offset in (0.0,.005,-.005):
        instance=LocalEnergyModel(True); instance.fit(y,mean,scale,summary,max_iter=100,initial=torch.full((56,),offset)); pair_fits.append(torch.cat((instance.unary,instance.pair.reshape(-1))))
    assert max(float((pair_fits[0]-value).abs().max()) for value in pair_fits[1:])<1e-6


def test_child_energy_uses_local_standardized_residual():
    model=LocalEnergyModel(True)
    model.unary=torch.linspace(-.2,.2,7,dtype=torch.double)
    model.pair=.01*torch.arange(49,dtype=torch.double).reshape(7,7)
    z=torch.linspace(-2,2,17,dtype=torch.double)
    summary=torch.linspace(-.3,.3,7,dtype=torch.double).expand(len(z),-1)
    first=model.correction(.4+.7*z,torch.full_like(z,.4),torch.full_like(z,.7),summary)
    second=model.correction(-1.2+2.3*z,torch.full_like(z,-1.2),torch.full_like(z,2.3),summary)
    assert torch.allclose(first,second,atol=1e-14,rtol=1e-14)


def test_warped_and_interaction_oracles_are_normalized_and_identifiable():
    x = ordered_sobol(3, 20, 4)
    for regime in ("G", "W", "I"):
        oracle = generate_oracle(x, regime, 4)
        candidates = candidate_geometry(x, x[:2] + 0.001)
        source = oracle.values
        for point in range(2):
            sliced = LocalSlice(candidates,point)
            mean, scale, _, _ = oracle.conditional(sliced, source)
            if regime == "W":
                integral = quad(lambda z: float(torch.exp(oracle.log_prob(torch.expm1(.6*(mean+scale*z))/.6, sliced, source))[0] * torch.exp(.6*(mean+scale*z))[0] * scale[0]), -10, 10, epsabs=1e-11)[0]
            else:
                integral = quad(lambda z: float(torch.exp(oracle.log_prob(mean+scale*z, sliced, source))[0] * scale[0]), -10, 10, epsabs=1e-11)[0]
            assert integral == pytest.approx(1.0, abs=2e-7)
    model=LocalEnergyModel(True); model.pair=2*torch.outer(U_TRUE,V_TRUE)
    mean=torch.zeros(1,dtype=torch.double); scale=torch.ones(1,dtype=torch.double)
    contexts=torch.stack((-0.5*V_TRUE,0.5*V_TRUE))
    grid=torch.linspace(-8,8,20001,dtype=torch.double)
    logp=[model.log_prob(grid,mean.expand_as(grid),scale.expand_as(grid),context.expand(len(grid),-1)) for context in contexts]
    density=[torch.exp(value) for value in logp]
    kl=float(torch.trapz(density[0]*(logp[0]-logp[1]),grid))
    assert kl > 0.05 and float(torch.std(logp[0]-logp[1])) > 0.01


def test_gaussian_control_fit_remains_close_on_large_sample():
    generator=torch.Generator().manual_seed(31); count=2048
    mean=torch.zeros(count,dtype=torch.double); scale=torch.ones(count,dtype=torch.double); summary=torch.randn(count,7,generator=generator,dtype=torch.double)*.2
    values=torch.randn(count,generator=generator,dtype=torch.double)
    for pairwise in (False,True):
        model=LocalEnergyModel(pairwise); fit=model.fit(values,mean,scale,summary,max_iter=100)
        assert fit.converged and float(model.correction_kl(mean,scale,summary).mean())<.01


def LocalSlice(geometry, index):
    return type(geometry)(geometry.x[index:index+1],geometry.neighbors[index:index+1],geometry.mask[index:index+1],geometry.coefficients[index:index+1],geometry.variances[index:index+1],geometry.similarity_weights[index:index+1],geometry.jitter)


def test_free_energy_ei_gaussian_and_interaction_sampling():
    model=LocalEnergyModel(False); mean=torch.linspace(-2,2,21,dtype=torch.double); scale=torch.full_like(mean,.7); summary=torch.zeros(21,7,dtype=torch.double)
    estimate=model.expected_improvement(mean,scale,summary,0.1)
    delta=mean-.1; z=delta/scale; analytic=delta*.5*(1+torch.erf(z/math.sqrt(2)))+scale*torch.exp(-.5*z.square())/math.sqrt(2*math.pi)
    assert torch.allclose(estimate,analytic,atol=1e-12)
    direct=model.expected_improvement_quadrature(mean,scale,summary,0.1,points=256)
    assert torch.allclose(direct,analytic,atol=1e-8,rtol=1e-8)
    interaction=LocalEnergyModel(True); interaction.pair=2*torch.outer(U_TRUE,V_TRUE)
    context=.5*V_TRUE; generator=torch.Generator().manual_seed(10); uniforms=torch.rand(4000,generator=generator,dtype=torch.double)
    samples=inverse_grid_sample(interaction,torch.tensor(0.),torch.tensor(1.),context,uniforms,points=2049)
    grid=torch.linspace(-10,10,10001); logp=interaction.log_prob(grid,torch.zeros_like(grid),torch.ones_like(grid),context.expand(len(grid),-1)); density=torch.exp(logp)
    exact=float(torch.trapz(grid.square()*density,grid)); assert abs(float(samples.square().mean())-exact)<.08
    sample_quantiles=torch.quantile(samples,torch.tensor([.1,.5,.9],dtype=torch.double))
    cdf_at_quantiles=interaction.cdf(sample_quantiles,torch.zeros(3,dtype=torch.double),torch.ones(3,dtype=torch.double),context.expand(3,-1),points=256)
    assert torch.allclose(cdf_at_quantiles,torch.tensor([.1,.5,.9],dtype=torch.double),atol=.03)
    best=.2
    interaction_ei=float(interaction.expected_improvement(torch.tensor([0.]),torch.tensor([1.]),context[None],best,points=256))
    adaptive=quad(lambda value:(value-best)*math.exp(float(interaction.log_prob(torch.tensor(value,dtype=torch.double),torch.tensor(0.),torch.tensor(1.),context))) ,best,10,epsabs=1e-10,epsrel=1e-10)[0]
    assert interaction_ei==pytest.approx(adaptive,abs=1e-7,rel=1e-7)
    # Warped local-Gaussian EI receives an independent latent-variable integral.
    alpha=.6; latent_mean=.3; latent_scale=.7; warped_best=-.1
    threshold=math.log1p(alpha*warped_best)/alpha
    cdf=lambda value:.5*(1+math.erf(value/math.sqrt(2)))
    analytic=(math.exp(alpha*latent_mean+.5*alpha**2*latent_scale**2)*cdf((latent_mean+alpha*latent_scale**2-threshold)/latent_scale)-math.exp(alpha*threshold)*cdf((latent_mean-threshold)/latent_scale))/alpha
    numerical=quad(lambda z:max(math.expm1(alpha*(latent_mean+latent_scale*z))/alpha-warped_best,0)*math.exp(-.5*z*z)/math.sqrt(2*math.pi),-10,10,epsabs=1e-11,epsrel=1e-11)[0]
    assert analytic==pytest.approx(numerical,abs=1e-9,rel=1e-9)


def test_warped_oracle_cdf_and_moments_match_numerical_truth():
    x=ordered_sobol(3,20,14); oracle=generate_oracle(x,"W",14); geometry=candidate_geometry(x,x[:4]+.003)
    mean,scale,_,_=oracle.conditional(geometry,oracle.values)
    analytic_mean,analytic_variance=oracle.moments(geometry,oracle.values)
    nodes=torch.linspace(-8,8,40001,dtype=torch.double); density=torch.exp(-.5*nodes.square())/math.sqrt(2*math.pi)
    values=torch.expm1(.6*(mean[:,None]+scale[:,None]*nodes))/.6
    numerical_mean=torch.trapz(values*density,nodes,dim=-1); numerical_variance=torch.trapz((values-numerical_mean[:,None]).square()*density,nodes,dim=-1)
    assert torch.allclose(analytic_mean,numerical_mean,atol=1e-9,rtol=1e-9)
    assert torch.allclose(analytic_variance,numerical_variance,atol=1e-9,rtol=1e-9)
    assert torch.allclose(oracle.cdf(analytic_mean,geometry,oracle.values),.5*(1+torch.erf((torch.log1p(.6*analytic_mean)/.6-mean)/scale/math.sqrt(2))),atol=1e-12)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cpu_cuda_agree():
    mean=torch.tensor([-.2,.4]); scale=torch.tensor([.5,.8]); summary=torch.randn(2,7,generator=torch.Generator().manual_seed(3),dtype=torch.double)
    model=LocalEnergyModel(True); model.unary=torch.linspace(-.1,.1,7); model.pair=.01*torch.arange(49,dtype=torch.double).reshape(7,7)
    cpu=model.expected_improvement(mean,scale,summary,.1)
    cpu_objective=model.objective(mean,mean,scale,summary,torch.cat((model.unary,model.pair.reshape(-1))))
    gpu=model.to("cuda").expected_improvement(mean.cuda(),scale.cuda(),summary.cuda(),.1).cpu()
    gpu_objective=model.objective(mean.cuda(),mean.cuda(),scale.cuda(),summary.cuda(),torch.cat((model.unary,model.pair.reshape(-1)))).cpu()
    assert torch.allclose(cpu,gpu,atol=1e-8,rtol=1e-8)
    assert torch.allclose(cpu_objective,gpu_objective,atol=1e-8,rtol=1e-8)
