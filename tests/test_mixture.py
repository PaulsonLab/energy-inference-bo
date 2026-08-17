import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import norm

from decision_tilt.mixture import (
    GaussianMixture1D,
    chi_square_decision_shift,
    mc_relative_variance,
    normal_improvement_moments,
    normal_softplus_moments,
    population_ess_fraction,
    softplus_utility,
)


@pytest.mark.parametrize("mean,std,incumbent", [(-0.4, 0.7, 0.1), (0.3, 0.2, 0.0)])
def test_normal_improvement_moments_match_adaptive_quadrature(
    mean: float, std: float, incumbent: float
) -> None:
    first, second = normal_improvement_moments(mean, std, incumbent)
    density = lambda value: norm.pdf(value, loc=mean, scale=std)
    expected_first = quad(
        lambda value: (value - incumbent) * density(value), incumbent, np.inf,
        epsabs=1e-13, epsrel=1e-13,
    )[0]
    expected_second = quad(
        lambda value: (value - incumbent) ** 2 * density(value), incumbent, np.inf,
        epsabs=1e-13, epsrel=1e-13,
    )[0]
    assert float(first) == pytest.approx(expected_first, abs=2e-13, rel=2e-13)
    assert float(second) == pytest.approx(expected_second, abs=2e-13, rel=2e-13)


def test_exact_mixture_ei_and_second_moment_are_weighted_component_moments() -> None:
    mixture = GaussianMixture1D(
        weights=np.array([0.8, 0.2]),
        means=np.array([-0.1, 1.4]),
        stds=np.array([0.3, 0.5]),
    )
    component_first, component_second = normal_improvement_moments(
        mixture.means, mixture.stds, 0.2
    )
    first, second = mixture.improvement_moments(0.2)
    assert first == pytest.approx(float(mixture.weights @ component_first), abs=1e-15)
    assert second == pytest.approx(float(mixture.weights @ component_second), abs=1e-15)


def test_relative_variance_equals_chi_square_over_sample_count() -> None:
    first, second = 0.35, 0.49
    shift = chi_square_decision_shift(first, second)
    assert mc_relative_variance(first, second, 256) == pytest.approx(shift / 256)
    assert second / first**2 - 1.0 == pytest.approx(shift)


def test_population_ess_fraction_is_inverse_one_plus_shift() -> None:
    first, second = 0.17, 0.11
    shift = chi_square_decision_shift(first, second)
    assert population_ess_fraction(first, second) == pytest.approx(1.0 / (1.0 + shift))
    assert population_ess_fraction(first, second) == pytest.approx(first**2 / second)


def test_utility_tilted_mixture_weights_match_component_contributions() -> None:
    mixture = GaussianMixture1D(
        weights=np.array([0.995, 0.005]),
        means=np.array([-0.25, 7.0]),
        stds=np.array([0.04, 0.2]),
    )
    component_first, _ = mixture.improvement_component_moments(0.0)
    expected = mixture.weights * component_first
    expected /= expected.sum()
    actual = mixture.tilted_component_weights(0.0)
    np.testing.assert_allclose(actual, expected, atol=1e-15, rtol=1e-15)
    assert actual.sum() == pytest.approx(1.0)


@pytest.mark.parametrize(
    "mean,std,incumbent,temperature",
    [(-0.08, 0.31, 0.0, 0.02), (0.02, 0.04, 0.0, 0.01)],
)
def test_softplus_moments_match_independent_adaptive_quadrature(
    mean: float, std: float, incumbent: float, temperature: float
) -> None:
    first, second = normal_softplus_moments(
        mean, std, incumbent, temperature, order=4096
    )
    density = lambda value: norm.pdf(value, loc=mean, scale=std)
    expected_first = quad(
        lambda value: float(softplus_utility(value, incumbent, temperature))
        * density(value),
        -np.inf,
        np.inf,
        epsabs=2e-12,
        epsrel=2e-12,
    )[0]
    expected_second = quad(
        lambda value: float(softplus_utility(value, incumbent, temperature)) ** 2
        * density(value),
        -np.inf,
        np.inf,
        epsabs=2e-12,
        epsrel=2e-12,
    )[0]
    assert float(first) == pytest.approx(expected_first, abs=2e-11, rel=2e-11)
    assert float(second) == pytest.approx(expected_second, abs=2e-11, rel=2e-11)
    assert float(softplus_utility(-1.0, incumbent, temperature)) > 0.0


def test_mixture_density_normalizes() -> None:
    mixture = GaussianMixture1D(
        weights=np.array([0.73, 0.27]),
        means=np.array([-1.0, 2.0]),
        stds=np.array([0.4, 0.8]),
    )
    integral = quad(lambda value: float(mixture.density(value)), -np.inf, np.inf)[0]
    assert integral == pytest.approx(1.0, abs=2e-12)
