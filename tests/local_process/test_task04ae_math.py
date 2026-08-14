from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from scipy.stats import multivariate_t, norm, t

from energy_bo.local_process.copulas import (
    MatchedCopulaOracle,
    calibrate_t_copula_correlation,
    gaussian_log_prob,
    student_normal_score_log_prob,
)
from energy_bo.local_process.joint_decision import paired_batch_panel, paired_endpoint_metrics, tie_aware_decision_metrics
from energy_bo.local_process.joint_energy import BivariateEnergyModel, unpack_symmetric


def test_t_copula_calibration_and_matched_marginals():
    first = calibrate_t_copula_correlation(calibration_power=16, verification_power=14, verification_replicates=2)
    second = calibrate_t_copula_correlation(calibration_power=16, verification_power=14, verification_replicates=2)
    assert first == second
    assert abs(first.latent_rho - 0.5133) < 0.002
    assert max(abs(value - 0.5) for value in first.verification_correlations) < 0.003
    samples = MatchedCopulaOracle(latent_rho=first.latent_rho).endpoint_qmc(16, 81)
    for endpoint in (samples.gaussian, samples.student):
        assert torch.all(endpoint.mean(0).abs() < 5e-4)
        assert torch.all((endpoint.var(0, unbiased=False) - 1).abs() < 0.002)
        assert abs(float((endpoint[:, 0] * endpoint[:, 1]).mean()) - 0.5) < 0.004


def test_copula_log_density_matches_scipy_and_qei_contrast():
    oracle = MatchedCopulaOracle()
    z = torch.tensor([[-1.2, 0.4], [0.3, 1.1], [2.0, 2.3]], dtype=torch.double)
    assert np.allclose(gaussian_log_prob(z.numpy()), norm.logpdf(z.numpy()).sum(-1) + np.array([
        -0.5 * math.log(0.75) - 0.5 * ((a*a - a*b + b*b) / .75 - a*a - b*b) for a,b in z.numpy()
    ]), atol=1e-12)
    u = norm.cdf(z.numpy()); x = t.ppf(u, 3)
    trusted = multivariate_t.logpdf(x, shape=[[1, oracle.latent_rho], [oracle.latent_rho, 1]], df=3) - t.logpdf(x, 3).sum(-1) + norm.logpdf(z.numpy()).sum(-1)
    assert np.allclose(student_normal_score_log_prob(z.numpy()), trusted, atol=1e-11)
    samples = oracle.endpoint_qmc(17, 29)
    mean, scale = torch.zeros(1, 2, dtype=torch.double), torch.ones(1, 2, dtype=torch.double)
    gaussian, student = oracle.endpoint_qei(samples, mean, scale, 1.5)
    assert abs(float(student - gaussian)) / float(gaussian) > 0.05
    assert oracle.q1_ei(1.5) == pytest.approx(0.0293067937626, abs=1e-12)


def test_symmetric_coordinates_zero_recovery_and_exchange_invariance():
    unique = torch.randn(28, generator=torch.Generator().manual_seed(4), dtype=torch.double)
    matrix = unpack_symmetric(unique)
    assert torch.allclose(matrix, matrix.T)
    assert torch.allclose(matrix.square().sum(), unique.square().sum(), atol=1e-14)
    model = BivariateEnergyModel(True, quadrature_points=32)
    z = torch.randn(20, 2, generator=torch.Generator().manual_seed(5), dtype=torch.double)
    context = torch.linspace(0, 1, 20, dtype=torch.double)
    assert torch.allclose(model.log_prob(z, context), torch.tensor(gaussian_log_prob(z.numpy())), atol=1e-12)
    model.parameter = torch.linspace(-0.1, 0.1, 35, dtype=torch.double)
    assert torch.allclose(model.log_prob(z, context), model.log_prob(z.flip(-1), context), atol=1e-12)


def test_normalizer_gradient_hessian_and_optimizer_uniqueness():
    oracle = MatchedCopulaOracle()
    context = torch.tensor([0, .25, .5, .75, 1] * 8, dtype=torch.double)
    z = oracle.sample_training(context, 17)
    model = BivariateEnergyModel(True, quadrature_points=24)
    parameter = torch.linspace(-.02, .02, 35, dtype=torch.double).requires_grad_()
    gradient = torch.autograd.grad(model.objective(z, context, parameter), parameter)[0]
    direction = torch.randn(35, generator=torch.Generator().manual_seed(7), dtype=torch.double); direction /= direction.norm()
    eps = 1e-6
    finite = (model.objective(z, context, parameter.detach() + eps * direction) - model.objective(z, context, parameter.detach() - eps * direction)) / (2 * eps)
    assert torch.dot(gradient, direction) == pytest.approx(float(finite), abs=2e-6, rel=2e-6)
    hessian = torch.autograd.functional.hessian(lambda value: model.objective(z, context, value), torch.zeros(35, dtype=torch.double))
    assert float(torch.linalg.eigvalsh(hessian).min()) >= 10 - 1e-8
    fitted = []
    for offset in (0.0, .002, -.002):
        instance = BivariateEnergyModel(True, quadrature_points=24)
        result = instance.fit(z, context, max_iter=100, initial=torch.full((35,), offset))
        assert result.converged
        fitted.append(instance.parameter)
    assert max(float((fitted[0] - value).abs().max()) for value in fitted[1:]) < 2e-6


def test_marginals_and_qei_are_safe_at_identity():
    model = BivariateEnergyModel(True, quadrature_points=48)
    context = torch.tensor([0, .5, 1], dtype=torch.double)
    metrics = model.marginal_metrics(context, 1.5)
    assert torch.all(metrics["kl"] < 1e-12)
    assert torch.all(metrics["mean"].abs() < 1e-12)
    assert torch.allclose(metrics["variance"], torch.ones(3, dtype=torch.double), atol=1e-11)
    assert torch.allclose(metrics["q1_ei"], torch.full((3,), MatchedCopulaOracle.q1_ei(1.5), dtype=torch.double), atol=1e-10)
    assert torch.allclose(model.marginal_cdf(torch.zeros(3, dtype=torch.double), context), torch.full((3,), .5, dtype=torch.double), atol=2e-10)
    mean, variance, correlation = model.moments(context)
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-12)
    assert torch.allclose(variance, torch.ones_like(variance), atol=1e-11)
    assert torch.allclose(correlation, torch.full_like(correlation, .5), atol=1e-11)
    oracle=MatchedCopulaOracle();samples=oracle.endpoint_qmc(18,123)
    truth=oracle.endpoint_qei(samples,torch.zeros(1,2,dtype=torch.double),torch.ones(1,2,dtype=torch.double),1.5)[0]
    estimate=model.qei(torch.zeros(1,2,dtype=torch.double),torch.ones(1,2,dtype=torch.double),torch.zeros(1,dtype=torch.double),1.5)
    assert torch.allclose(estimate,truth,atol=3e-5,rtol=3e-5)


def test_paired_panel_is_deterministic_and_tie_aware():
    first, second = paired_batch_panel(), paired_batch_panel()
    assert torch.equal(first.mean, second.mean) and torch.equal(first.scale, second.scale)
    assert torch.equal(first.mean[0::2], first.mean[1::2])
    oracle = MatchedCopulaOracle(); samples = oracle.endpoint_qmc(16, 39)
    gaussian, student = oracle.endpoint_qei(samples, first.mean[::2], first.scale[::2], 1.5)
    truth = torch.stack((gaussian, student), -1).reshape(-1)
    baseline = torch.stack((gaussian, gaussian), -1).reshape(-1)
    pair = paired_endpoint_metrics(baseline, truth)
    decision = tie_aware_decision_metrics(baseline, truth)
    assert pair["oracle_significant_fraction"] >= .75
    assert decision["maximizer_count"] == 2
    assert decision["tie_aware_regret"] >= .01


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_joint_energy_cpu_cuda_parity():
    cpu = BivariateEnergyModel(True, quadrature_points=32)
    cpu.parameter = torch.linspace(-.1, .1, 35, dtype=torch.double)
    z = torch.randn(16, 2, generator=torch.Generator().manual_seed(44), dtype=torch.double)
    context = torch.linspace(0, 1, 16, dtype=torch.double)
    expected = cpu.log_prob(z, context)
    gpu = BivariateEnergyModel(True, quadrature_points=32).to("cuda")
    gpu.parameter = cpu.parameter.cuda()
    actual = gpu.log_prob(z.cuda(), context.cuda()).cpu()
    assert torch.allclose(expected, actual, atol=1e-8, rtol=1e-8)
