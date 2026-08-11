from __future__ import annotations

import numpy as np
import torch
from scipy.integrate import quad

from energy_bo.oracle.distributions import normal_pdf
from energy_bo.oracle.residual_energy import RBFResidualEnergy


def test_residual_energy_normalizes_with_nonzero_energy() -> None:
    model = RBFResidualEnergy()
    model.phi = torch.linspace(-0.3, 0.3, model.parameter_count, dtype=torch.double)
    integral, error = quad(
        lambda z: float(model.density_numpy(z)),
        -10.0,
        10.0,
        epsabs=1e-10,
        epsrel=1e-10,
        limit=300,
    )
    assert abs(integral - 1.0) < 5e-8
    assert error < 1e-8


def test_zero_energy_recovers_standard_normal_exactly() -> None:
    model = RBFResidualEnergy()
    z = torch.tensor([-2.0, -0.5, 0.0, 1.25, 3.0], dtype=torch.double)
    assert abs(float(model.log_normalizer())) < 1e-14
    assert np.max(np.abs(model.density_numpy(z.numpy()) - normal_pdf(z.numpy()))) < 1e-14


def test_centered_basis_and_constant_energy_gauge() -> None:
    model = RBFResidualEnergy()
    quadrature_mean = torch.sum(
        model._quadrature_weights.unsqueeze(-1) * model.basis(model._quadrature_nodes), dim=0
    )
    assert torch.max(torch.abs(quadrature_mean)) < 1e-10

    model.phi = torch.linspace(-0.2, 0.2, model.parameter_count, dtype=torch.double)
    z = torch.tensor([-1.0, 0.2, 1.4], dtype=torch.double)
    constant = torch.tensor(1.7, dtype=torch.double)
    unshifted = model.log_prob(z)
    shifted_log_normalizer = torch.logsumexp(
        torch.log(model._quadrature_weights)
        - model.energy(model._quadrature_nodes)
        - constant,
        dim=0,
    )
    shifted = (
        -0.5 * z.square()
        - 0.5 * np.log(2.0 * np.pi)
        - model.energy(z)
        - constant
        - shifted_log_normalizer
    )
    assert torch.max(torch.abs(unshifted - shifted)) < 1e-13
