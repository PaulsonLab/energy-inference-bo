import math

import gpytorch
import pytest
import torch

from energy_bo.protein.data import DATASETS, encode_sequences, frozen_permutation, sha256_file
from energy_bo.protein.kernels import HammingRBFKernel, LOCKKernel, TanimotoKernel, blosum_scores, lock_correlation_matrix
from energy_bo.protein.metrics import gaussian_crps, gaussian_nll, normalized_regret
from energy_bo.protein.models import FrozenStandardization, fit_protein_gp, log_ei, predict_raw


torch.set_default_dtype(torch.double)


def _sequences() -> torch.Tensor:
    return encode_sequences(["ARND", "ARNE", "VRND", "VVVV", "ARND"])


def test_pinned_data_contract_and_checksum_helper(tmp_path) -> None:
    assert DATASETS["trpb"]["rows"] == 111_883
    assert DATASETS["creilov"]["rows"] == 167_530
    path = tmp_path / "payload"; path.write_bytes(b"task05a")
    assert sha256_file(path) == "4f7282a1a9f6c5fbbe65e662d3bcbbd485e69980b2a9e9e629ca579948d2a472"


def test_nested_permutations_are_dataset_specific_and_deterministic() -> None:
    first = frozen_permutation(50, "trpb", 3)
    assert torch.equal(torch.as_tensor(first), torch.as_tensor(frozen_permutation(50, "trpb", 3)))
    assert not torch.equal(torch.as_tensor(first), torch.as_tensor(frozen_permutation(50, "creilov", 3)))
    assert set(first[:10]).issubset(set(first[:20]))


def test_s0_matches_explicit_hamming_rbf() -> None:
    x = _sequences()
    kernel = HammingRBFKernel(); kernel.rate = 0.7
    observed = kernel(x, x).to_dense()
    expected = torch.exp(-0.7 * (x[:, None, :] != x[None, :, :]).sum(-1))
    torch.testing.assert_close(observed, expected)


def test_blosum_and_structured_kernels_are_symmetric_with_valid_diagonals() -> None:
    x = _sequences()[:4].double()
    for number in (50, 62):
        matrix = blosum_scores(number)
        torch.testing.assert_close(matrix, matrix.T)
    correlation = lock_correlation_matrix()
    torch.testing.assert_close(correlation.diag(), torch.ones(20))
    scores = blosum_scores(50)
    augmented = torch.full((21, 21), -5.0)
    augmented[:20, :20] = scores
    augmented[20, 20] = 1.0
    diagonal = augmented.diag()
    official_log = augmented - 0.5 * (diagonal[:, None] + diagonal[None, :])
    official_log *= 0.25 / official_log.median().abs()
    torch.testing.assert_close(correlation, official_log[:20, :20].exp())
    for kernel in (TanimotoKernel(), LOCKKernel(4)):
        gram = kernel(x, x).to_dense()
        torch.testing.assert_close(gram, gram.T, atol=1e-12, rtol=1e-12)
        assert torch.linalg.eigvalsh(gram).min() > -1e-8
        assert torch.isfinite(gram).all()


def test_s1_and_s2_match_their_explicit_formulas() -> None:
    x = _sequences()[:3]
    tanimoto = TanimotoKernel()
    residue_gram = tanimoto.residue_gram
    dot = torch.stack([torch.stack([residue_gram[left, right].sum() for right in x]) for left in x])
    norms = torch.stack([residue_gram.diag()[row].sum() for row in x])
    expected_tanimoto = dot / (norms[:, None] + norms[None, :] - dot)
    torch.testing.assert_close(tanimoto(x, x).to_dense(), expected_tanimoto)

    lock = LOCKKernel(4)
    lock.variance1 = 0.7; lock.variance2 = 1.2
    lock.log_alpha_linear1.data.fill_(math.log(0.8))
    lock.log_alpha_linear2.data.fill_(math.log(1.1))
    lock.log_alpha_global.data.fill_(math.log(0.9))
    lock.log_alpha_local_relative.data.copy_(torch.tensor([0.8, 1.0, 1.2, 1.4]).log())
    log_values = lock.log_correlation[x[:, None, :], x[None, :, :]]
    nonlinear = torch.exp((log_values * lock.local_exponents).sum(-1))
    linear1 = torch.exp(lock.alpha_linear1 * log_values).sum(-1)
    linear2 = torch.exp(lock.alpha_linear2 * log_values).sum(-1)
    expected_lock = lock.variance1 * nonlinear * linear1 + lock.variance2 * linear2
    torch.testing.assert_close(lock(x, x).to_dense(), expected_lock)


def test_lock_priors_and_gradients_are_present() -> None:
    kernel = LOCKKernel(4)
    assert len(list(kernel.named_priors())) == 6
    loss = kernel(_sequences()[:4], _sequences()[:4]).to_dense().sum()
    loss.backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in kernel.parameters())


def test_train_only_standardization_round_trip() -> None:
    train = torch.tensor([1.0, 2.0, 5.0])
    transform = FrozenStandardization.fit(train)
    standardized = transform.transform(train)
    torch.testing.assert_close(standardized.mean(), torch.tensor(0.0), atol=1e-14, rtol=0)
    torch.testing.assert_close(transform.untransform_mean(standardized), train)
    torch.testing.assert_close(transform.untransform_variance(torch.ones(3)), torch.full((3,), transform.scale**2))


def test_gaussian_metrics_and_regret_identities() -> None:
    y = torch.tensor([-1.0, 0.0, 1.0])
    mean = torch.zeros(3); variance = torch.ones(3)
    expected_nll = 0.5 * (math.log(2 * math.pi) + y.square())
    torch.testing.assert_close(gaussian_nll(y, mean, variance), expected_nll)
    assert torch.isfinite(gaussian_crps(y, mean, variance)).all()
    assert normalized_regret(3.0, 5.0, 1.0) == 0.5


def test_tiny_exact_gp_prediction_and_logei_are_finite() -> None:
    x = _sequences()[:4].double()
    y = torch.tensor([0.0, 0.3, -0.2, 0.5])
    fit = fit_protein_gp(x, y, "S0", max_iterations=5)
    mean, variance = predict_raw(fit, x, observation_noise=True, chunk_size=2)
    values = log_ei(fit, x, float(y.max()), chunk_size=2)
    assert mean.shape == variance.shape == values.shape == (4,)
    assert torch.isfinite(mean).all() and torch.isfinite(variance).all() and torch.isfinite(values).all()
    assert (variance > 0).all()
    posterior = fit.model.posterior(x, observation_noise=False)
    latent_mean = posterior.mean.squeeze(-1)
    sigma = posterior.variance.squeeze(-1).sqrt()
    best = (float(y.max()) - fit.transform.mean) / fit.transform.scale
    z = (latent_mean - best) / sigma
    normal = torch.distributions.Normal(torch.tensor(0.0), torch.tensor(1.0))
    manual = sigma * (torch.exp(normal.log_prob(z)) + z * normal.cdf(z))
    torch.testing.assert_close(values.exp(), manual, atol=1e-9, rtol=1e-8)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_protein_kernel_cpu_cuda_parity() -> None:
    x = _sequences()[:4].double()
    for constructor in (TanimotoKernel, lambda: LOCKKernel(4)):
        cpu = constructor()
        gpu = constructor().cuda()
        gpu.load_state_dict(cpu.state_dict())
        expected = cpu(x, x).to_dense()
        observed = gpu(x.cuda(), x.cuda()).to_dense().cpu()
        torch.testing.assert_close(observed, expected, atol=1e-10, rtol=1e-10)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_tiny_exact_gp_fit_and_logei_on_cuda() -> None:
    x = _sequences()[:4].double().cuda()
    y = torch.tensor([0.0, 0.3, -0.2, 0.5], device="cuda")
    fit = fit_protein_gp(x, y, "S0", max_iterations=5)
    mean, variance = predict_raw(fit, x, observation_noise=True, chunk_size=2)
    values = log_ei(fit, x, float(y.max()), chunk_size=2)
    assert torch.isfinite(mean).all() and torch.isfinite(variance).all()
    assert torch.isfinite(values).all() and (variance > 0).all()
