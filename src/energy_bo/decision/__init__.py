"""Decision-relevant structural diagnostics for Task 02B."""

from .coresets import Coreset, acquisition_frank_wolfe, posterior_k_medoids, random_equal
from .joint_target import joint_target_marginals
from .metrics import acquisition_metrics, signature_spectrum
from .signatures import particle_ei_signatures, transformed_particle_features

__all__ = [
    "Coreset",
    "acquisition_frank_wolfe",
    "acquisition_metrics",
    "joint_target_marginals",
    "particle_ei_signatures",
    "posterior_k_medoids",
    "random_equal",
    "signature_spectrum",
    "transformed_particle_features",
]
