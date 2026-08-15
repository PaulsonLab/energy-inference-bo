"""Generic, BLOSUM-Tanimoto, and LOCK kernels for fixed-length proteins."""

from __future__ import annotations

import math

import gpytorch
import torch
from gpytorch.constraints import GreaterThan, Positive
from gpytorch.kernels import Kernel
from gpytorch.priors import GammaPrior, LogNormalPrior, NormalPrior

from .data import AMINO_ACIDS

# NCBI BLOSUM matrices in their native integer log-odds units, restricted to
# the twenty standard amino acids in AMINO_ACIDS order.
_BLOSUM50 = (
    (5,-2,-1,-2,-1,-1,-1,0,-2,-1,-2,-1,-1,-3,-1,1,0,-3,-2,0),
    (-2,7,-1,-2,-4,1,0,-3,0,-4,-3,3,-2,-3,-3,-1,-1,-3,-1,-3),
    (-1,-1,7,2,-2,0,0,0,1,-3,-4,0,-2,-4,-2,1,0,-4,-2,-3),
    (-2,-2,2,8,-4,0,2,-1,-1,-4,-4,-1,-4,-5,-1,0,-1,-5,-3,-4),
    (-1,-4,-2,-4,13,-3,-3,-3,-3,-2,-2,-3,-2,-2,-4,-1,-1,-5,-3,-1),
    (-1,1,0,0,-3,7,2,-2,1,-3,-2,2,0,-4,-1,0,-1,-1,-1,-3),
    (-1,0,0,2,-3,2,6,-3,0,-4,-3,1,-2,-3,-1,-1,-1,-3,-2,-3),
    (0,-3,0,-1,-3,-2,-3,8,-2,-4,-4,-2,-3,-4,-2,0,-2,-3,-3,-4),
    (-2,0,1,-1,-3,1,0,-2,10,-4,-3,0,-1,-1,-2,-1,-2,-3,2,-4),
    (-1,-4,-3,-4,-2,-3,-4,-4,-4,5,2,-3,2,0,-3,-3,-1,-3,-1,4),
    (-2,-3,-4,-4,-2,-2,-3,-4,-3,2,5,-3,3,1,-4,-3,-1,-2,-1,1),
    (-1,3,0,-1,-3,2,1,-2,0,-3,-3,6,-2,-4,-1,0,-1,-3,-2,-3),
    (-1,-2,-2,-4,-2,0,-2,-3,-1,2,3,-2,7,0,-3,-2,-1,-1,0,1),
    (-3,-3,-4,-5,-2,-4,-3,-4,-1,0,1,-4,0,8,-4,-3,-2,1,4,-1),
    (-1,-3,-2,-1,-4,-1,-1,-2,-2,-3,-4,-1,-3,-4,10,-1,-1,-4,-3,-3),
    (1,-1,1,0,-1,0,-1,0,-1,-3,-3,0,-2,-3,-1,5,2,-4,-2,-2),
    (0,-1,0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1,2,5,-3,-2,0),
    (-3,-3,-4,-5,-5,-1,-3,-3,-3,-3,-2,-3,-1,1,-4,-4,-3,15,2,-3),
    (-2,-1,-2,-3,-3,-1,-2,-3,2,-1,-1,-2,0,4,-3,-2,-2,2,8,-1),
    (0,-3,-3,-4,-1,-3,-3,-4,-4,4,1,-3,1,-1,-3,-2,0,-3,-1,5),
)
_BLOSUM62 = (
    (4,-1,-2,-2,0,-1,-1,0,-2,-1,-1,-1,-1,-2,-1,1,0,-3,-2,0),
    (-1,5,0,-2,-3,1,0,-2,0,-3,-2,2,-1,-3,-2,-1,-1,-3,-2,-3),
    (-2,0,6,1,-3,0,0,0,1,-3,-3,0,-2,-3,-2,1,0,-4,-2,-3),
    (-2,-2,1,6,-3,0,2,-1,-1,-3,-4,-1,-3,-3,-1,0,-1,-4,-3,-3),
    (0,-3,-3,-3,9,-3,-4,-3,-3,-1,-1,-3,-1,-2,-3,-1,-1,-2,-2,-1),
    (-1,1,0,0,-3,5,2,-2,0,-3,-2,1,0,-3,-1,0,-1,-2,-1,-2),
    (-1,0,0,2,-4,2,5,-2,0,-3,-3,1,-2,-3,-1,0,-1,-3,-2,-2),
    (0,-2,0,-1,-3,-2,-2,6,-2,-4,-4,-2,-3,-3,-2,0,-2,-2,-3,-3),
    (-2,0,1,-1,-3,0,0,-2,8,-3,-3,-1,-2,-1,-2,-1,-2,-2,2,-3),
    (-1,-3,-3,-3,-1,-3,-3,-4,-3,4,2,-3,1,0,-3,-2,-1,-3,-1,3),
    (-1,-2,-3,-4,-1,-2,-3,-4,-3,2,4,-2,2,0,-3,-2,-1,-2,-1,1),
    (-1,2,0,-1,-3,1,1,-2,-1,-3,-2,5,-1,-3,-1,0,-1,-3,-2,-2),
    (-1,-1,-2,-3,-1,0,-2,-3,-2,1,2,-1,5,0,-2,-1,-1,-1,-1,1),
    (-2,-3,-3,-3,-2,-3,-3,-3,-1,0,0,-3,0,6,-4,-2,-2,1,3,-1),
    (-1,-2,-2,-1,-3,-1,-1,-2,-2,-3,-3,-1,-2,-4,7,-1,-1,-4,-3,-2),
    (1,-1,1,0,-1,0,0,0,-1,-2,-2,0,-1,-2,-1,4,1,-3,-2,-2),
    (0,-1,0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1,1,5,-2,-2,0),
    (-3,-3,-4,-4,-2,-2,-3,-2,-2,-3,-2,-3,-1,1,-4,-3,-2,11,2,-3),
    (-2,-2,-2,-3,-2,-1,-2,-3,2,-1,-1,-2,-1,3,-3,-2,-2,2,7,-1),
    (0,-3,-3,-3,-1,-2,-2,-3,-3,3,1,-2,1,-1,-2,-2,0,-3,-1,4),
)


def blosum_scores(number: int, *, dtype: torch.dtype = torch.double) -> torch.Tensor:
    values = _BLOSUM50 if number == 50 else _BLOSUM62 if number == 62 else None
    if values is None:
        raise ValueError("only BLOSUM50 and BLOSUM62 are supported")
    return torch.tensor(values, dtype=dtype)


def lock_correlation_matrix() -> torch.Tensor:
    scores = blosum_scores(50)
    # Match the authors' 21-token normalization, then retain the standard-residue
    # block because the measured landscapes contain no gaps.
    augmented = torch.full((21, 21), -5.0, dtype=scores.dtype)
    augmented[:20, :20] = scores
    augmented[20, 20] = 1.0
    diagonal = augmented.diag()
    log_correlation = augmented - 0.5 * (diagonal[:, None] + diagonal[None, :])
    log_correlation = log_correlation * (0.25 / log_correlation.median().abs())
    correlation = torch.exp(log_correlation[:20, :20])
    correlation.fill_diagonal_(1.0)
    return correlation


def tanimoto_gram_matrix() -> torch.Tensor:
    scores = blosum_scores(62)
    eigenvalues, eigenvectors = torch.linalg.eigh(scores)
    positive = eigenvalues.clamp_min(0)
    embedding = eigenvectors * positive.sqrt().unsqueeze(0)
    return embedding @ embedding.T


def _pairwise_sum(matrix: torch.Tensor, x1: torch.Tensor, x2: torch.Tensor, block: int = 16) -> torch.Tensor:
    result = torch.zeros(
        torch.broadcast_shapes(x1.shape[:-2], x2.shape[:-2]) + (x1.shape[-2], x2.shape[-2]),
        dtype=matrix.dtype,
        device=x1.device,
    )
    for start in range(0, x1.shape[-1], block):
        left = x1[..., :, None, start : start + block]
        right = x2[..., None, :, start : start + block]
        result = result + matrix[left, right].sum(-1)
    return result


class HammingRBFKernel(Kernel):
    has_lengthscale = False

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.register_parameter("raw_rate", torch.nn.Parameter(torch.zeros(())))
        self.register_constraint("raw_rate", Positive())
        self.register_prior("rate_prior", LogNormalPrior(0.0, 1.0), "rate")
        self.rate = torch.tensor(1.0)

    @property
    def rate(self) -> torch.Tensor:
        return self.raw_rate_constraint.transform(self.raw_rate)

    @rate.setter
    def rate(self, value: torch.Tensor | float) -> None:
        self.initialize(raw_rate=self.raw_rate_constraint.inverse_transform(torch.as_tensor(value, dtype=self.raw_rate.dtype, device=self.raw_rate.device)))

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, diag: bool = False, **params) -> torch.Tensor:
        x1, x2 = x1.long(), x2.long()
        if diag:
            return torch.ones(x1.shape[:-1], dtype=self.raw_rate.dtype, device=x1.device)
        mismatches = (x1.unsqueeze(-2) != x2.unsqueeze(-3)).sum(-1)
        return torch.exp(-self.rate * mismatches)


class TanimotoKernel(Kernel):
    has_lengthscale = False

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.register_buffer("residue_gram", tanimoto_gram_matrix())

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        diagonal = self.residue_gram.diag()
        return diagonal[x.long()].sum(-1)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, diag: bool = False, **params) -> torch.Tensor:
        x1, x2 = x1.long(), x2.long()
        if diag:
            return torch.ones(x1.shape[:-1], dtype=self.residue_gram.dtype, device=x1.device)
        dot = _pairwise_sum(self.residue_gram, x1, x2)
        norm1 = self._norm(x1).unsqueeze(-1)
        norm2 = self._norm(x2).unsqueeze(-2)
        return dot / (norm1 + norm2 - dot).clamp_min(1e-12)


class LOCKKernel(Kernel):
    """Equation (22) of Jankowiak et al. (2026), with published priors."""

    has_lengthscale = False

    def __init__(self, sequence_length: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.sequence_length = int(sequence_length)
        self.register_buffer("log_correlation", lock_correlation_matrix().log())
        for name in ("variance1", "variance2"):
            self.register_parameter(f"raw_{name}", torch.nn.Parameter(torch.zeros(())))
            self.register_constraint(f"raw_{name}", GreaterThan(1e-4))
        for name, shape in (
            ("alpha_linear1", ()), ("alpha_linear2", ()),
            ("alpha_global", ()), ("alpha_local_relative", (self.sequence_length,)),
        ):
            self.register_parameter(f"log_{name}", torch.nn.Parameter(torch.zeros(shape)))
        self.register_prior("variance1_prior", GammaPrior(2.0, 2.0), "variance1")
        self.register_prior("variance2_prior", GammaPrior(2.0, 2.0), "variance2")
        self.register_prior("alpha_linear1_prior", NormalPrior(0.0, 1.0), "log_alpha_linear1")
        self.register_prior("alpha_linear2_prior", NormalPrior(0.0, 1.0), "log_alpha_linear2")
        self.register_prior("alpha_global_prior", NormalPrior(0.0, 1.0), "log_alpha_global")
        self.register_prior("alpha_local_relative_prior", NormalPrior(0.0, 0.5), "log_alpha_local_relative")
        for name in ("variance1", "variance2"):
            setattr(self, name, torch.ones(getattr(self, f"raw_{name}").shape or (), dtype=torch.double))

    @staticmethod
    def _positive_property(name: str):
        def getter(self):
            return getattr(self, f"raw_{name}_constraint").transform(getattr(self, f"raw_{name}"))
        def setter(self, value):
            raw = getattr(self, f"raw_{name}")
            constraint = getattr(self, f"raw_{name}_constraint")
            self.initialize(**{f"raw_{name}": constraint.inverse_transform(torch.as_tensor(value, dtype=raw.dtype, device=raw.device))})
        return property(getter, setter)

    variance1 = _positive_property("variance1")
    variance2 = _positive_property("variance2")
    @property
    def alpha_linear1(self) -> torch.Tensor:
        return self.log_alpha_linear1.exp()

    @property
    def alpha_linear2(self) -> torch.Tensor:
        return self.log_alpha_linear2.exp()

    @property
    def alpha_global(self) -> torch.Tensor:
        return self.log_alpha_global.exp()

    @property
    def alpha_local_relative(self) -> torch.Tensor:
        return self.log_alpha_local_relative.exp()

    @property
    def local_exponents(self) -> torch.Tensor:
        return self.alpha_global * self.alpha_local_relative

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, diag: bool = False, **params) -> torch.Tensor:
        x1, x2 = x1.long(), x2.long()
        length = x1.shape[-1]
        if length != self.sequence_length or x2.shape[-1] != length:
            raise ValueError("LOCK kernel received an unexpected sequence length")
        if diag:
            value = length * (self.variance1 + self.variance2)
            return value.expand(x1.shape[:-1])
        shape = torch.broadcast_shapes(x1.shape[:-2], x2.shape[:-2]) + (x1.shape[-2], x2.shape[-2])
        linear1 = torch.zeros(shape, dtype=self.log_correlation.dtype, device=x1.device)
        linear2 = torch.zeros_like(linear1)
        nonlinear_log = torch.zeros_like(linear1)
        local = self.local_exponents
        for start in range(0, length, 16):
            stop = min(start + 16, length)
            left = x1[..., :, None, start:stop]
            right = x2[..., None, :, start:stop]
            log_values = self.log_correlation[left, right]
            linear1 = linear1 + torch.exp(self.alpha_linear1 * log_values).sum(-1)
            linear2 = linear2 + torch.exp(self.alpha_linear2 * log_values).sum(-1)
            nonlinear_log = nonlinear_log + (log_values * local[start:stop]).sum(-1)
        return self.variance1 * nonlinear_log.exp() * linear1 + self.variance2 * linear2


def make_covariance(model_name: str, sequence_length: int) -> Kernel:
    prior = GammaPrior(2.0, 2.0)
    if model_name == "S0":
        return gpytorch.kernels.ScaleKernel(HammingRBFKernel(), outputscale_prior=prior)
    if model_name == "S1":
        return gpytorch.kernels.ScaleKernel(TanimotoKernel(), outputscale_prior=prior)
    if model_name == "S2":
        return LOCKKernel(sequence_length)
    raise ValueError(f"unknown belief model: {model_name}")
