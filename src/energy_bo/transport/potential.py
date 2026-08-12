"""Trusted unconstrained NumPyro SAAS potential and fused exact-GP target."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import torch
from botorch.models.fully_bayesian import (
    SaasFullyBayesianSingleTaskGP,
    SaasPyroModel,
)
from jax import random
from numpyro.infer.util import initialize_model

from .logei import exact_gp_nll_and_logei

Array = jax.Array
Tree = dict[str, Array]


def vector_to_tree(vector: Array) -> Tree:
    vector = jnp.asarray(vector, dtype=jnp.float64)
    if vector.ndim != 1 or vector.shape[0] < 4:
        raise ValueError("SAAS vector must have shape [D + 3]")
    return {
        "outputscale": vector[0],
        "mean": vector[1],
        "kernel_tausq": vector[2],
        "_kernel_inv_length_sq": vector[3:],
    }


def tree_to_vector(tree: Tree) -> Array:
    return jnp.concatenate(
        (
            jnp.atleast_1d(tree["outputscale"]),
            jnp.atleast_1d(tree["mean"]),
            jnp.atleast_1d(tree["kernel_tausq"]),
            jnp.ravel(tree["_kernel_inv_length_sq"]),
        )
    ).astype(jnp.float64)


@dataclass(frozen=True)
class SaasUnconstrainedPotential:
    """Exact and fused SAAS targets in NumPyro's unconstrained coordinates."""

    train_x: Array
    train_y: Array
    noise_variance: float
    best_f: float
    dimension: int
    initial_vector: Array
    full_potential_tree: Callable[[Tree], Array]
    prior_potential_tree: Callable[[Tree], Array]
    postprocess_tree: Callable[[Tree], dict[str, Array]]
    model_sample: Callable[[], None]
    prior_model_sample: Callable[[], None]

    @classmethod
    def build(
        cls,
        train_x: torch.Tensor | np.ndarray,
        train_y: torch.Tensor | np.ndarray,
        noise_variance: float,
        *,
        seed: int = 0,
    ) -> "SaasUnconstrainedPotential":
        tx = torch.as_tensor(np.asarray(train_x), dtype=torch.double).reshape(len(train_x), -1)
        ty = torch.as_tensor(np.asarray(train_y), dtype=torch.double).reshape(-1, 1)
        if tx.shape[0] != ty.shape[0] or tx.shape[1] < 2:
            raise ValueError("training data must have shapes [n,D] and [n]")
        if noise_variance <= 0:
            raise ValueError("noise variance must be positive")
        yvar = torch.full_like(ty, float(noise_variance))
        model = SaasFullyBayesianSingleTaskGP(
            tx,
            ty,
            yvar,
            outcome_transform=None,
            input_transform=None,
        )
        full = initialize_model(random.PRNGKey(seed), model.pyro_model.sample)

        prior_model = SaasPyroModel()
        prior_model.set_inputs(
            train_X=torch.empty((0, tx.shape[1]), dtype=torch.double),
            train_Y=torch.empty((0, 1), dtype=torch.double),
            train_Yvar=torch.empty((0, 1), dtype=torch.double),
        )
        prior = initialize_model(random.PRNGKey(seed + 1), prior_model.sample)
        if set(full.param_info.z) != set(prior.param_info.z):
            raise RuntimeError("posterior and prior NumPyro latent schemas differ")
        return cls(
            train_x=jnp.asarray(tx.numpy(), dtype=jnp.float64),
            train_y=jnp.asarray(ty[:, 0].numpy(), dtype=jnp.float64),
            noise_variance=float(noise_variance),
            best_f=float(ty.max()),
            dimension=tx.shape[1],
            initial_vector=tree_to_vector(full.param_info.z),
            full_potential_tree=full.potential_fn,
            prior_potential_tree=prior.potential_fn,
            postprocess_tree=full.postprocess_fn,
            model_sample=model.pyro_model.sample,
            prior_model_sample=prior_model.sample,
        )

    @property
    def coordinate_count(self) -> int:
        return self.dimension + 3

    def postprocess(self, vector: Array) -> dict[str, Array]:
        return self.postprocess_tree(vector_to_tree(vector))

    def full_potential(self, vector: Array) -> Array:
        return self.full_potential_tree(vector_to_tree(vector))

    def prior_potential(self, vector: Array) -> Array:
        return self.prior_potential_tree(vector_to_tree(vector))

    def nll_and_log_ei(self, vector: Array, design_x: Array) -> tuple[Array, Array]:
        constrained = self.postprocess(vector)
        return exact_gp_nll_and_logei(
            train_x=self.train_x,
            train_y=self.train_y,
            noise_variance=self.noise_variance,
            lengthscale=constrained["lengthscale"],
            mean=constrained["mean"],
            outputscale=constrained["outputscale"],
            design_x=jnp.asarray(design_x, dtype=jnp.float64),
            best_f=self.best_f,
        )

    def fused_potential(self, vector: Array, design_x: Array, beta: float) -> Array:
        nll, log_ei = self.nll_and_log_ei(vector, design_x)
        return self.prior_potential(vector) + nll - beta * log_ei

    def posterior_potential(self, vector: Array, design_x: Array) -> Array:
        del design_x
        nll, _ = self.nll_and_log_ei(vector, jnp.zeros(self.dimension))
        return self.prior_potential(vector) + nll

    def operational(self, vectors: Array) -> dict[str, Array]:
        constrained = jax.vmap(self.postprocess)(vectors)
        return {
            "lengthscales": constrained["lengthscale"],
            "means": constrained["mean"],
            "outputscales": constrained["outputscale"],
        }

    def initialization_vectors(self, count: int, seed: int) -> Array:
        """Generate valid NumPyro ``init_to_uniform`` states without a teacher."""

        if count < 1:
            raise ValueError("count must be positive")
        vectors = []
        for index in range(count):
            info = initialize_model(
                random.PRNGKey(seed + index), self.model_sample, validate_grad=True
            )
            vectors.append(tree_to_vector(info.param_info.z))
        return jnp.stack(vectors)

    def prior_vectors(self, count: int, seed: int) -> Array:
        """Draw exact constrained priors and map them to unconstrained coordinates."""

        import numpyro

        if count < 1:
            raise ValueError("count must be positive")
        predictive = numpyro.infer.Predictive(self.prior_model_sample, num_samples=count)
        samples = predictive(random.PRNGKey(seed))
        return jnp.concatenate(
            (
                jnp.log(samples["outputscale"]).reshape(count, 1),
                samples["mean"].reshape(count, 1),
                jnp.log(samples["kernel_tausq"]).reshape(count, 1),
                jnp.log(samples["_kernel_inv_length_sq"]).reshape(count, self.dimension),
            ),
            axis=1,
        )

    def validate_fused(self, vectors: Array, design_x: Array) -> dict[str, float]:
        """Compare exact NumPyro and fused posterior differences and gradients."""

        vectors = jnp.asarray(vectors, dtype=jnp.float64)
        exact_values = jax.vmap(self.full_potential)(vectors)
        fused_values = jax.vmap(lambda z: self.fused_potential(z, design_x, 0.0))(vectors)
        # Potentials may differ by a target-independent additive constant.
        centered_error = (exact_values - exact_values[0]) - (fused_values - fused_values[0])
        exact_grad = jax.vmap(jax.grad(self.full_potential))(vectors)
        fused_grad = jax.vmap(jax.grad(lambda z: self.fused_potential(z, design_x, 0.0)))(vectors)
        return {
            "centered_value_max_abs": float(jnp.max(jnp.abs(centered_error))),
            "gradient_max_abs": float(jnp.max(jnp.abs(exact_grad - fused_grad))),
            "gradient_rms": float(jnp.sqrt(jnp.mean((exact_grad - fused_grad) ** 2))),
        }
