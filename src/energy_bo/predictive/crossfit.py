"""Genuine held-out MAP-SAAS mixture predictions and provenance."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import torch

from energy_bo.structural.map_saas import MapSaasReference, fit_map_saas_reference
from .mixture import GaussianMixtureMarginals


@dataclass(frozen=True)
class HeldOutPrediction:
    heldout_indices: tuple[int, ...]
    training_indices: tuple[int, ...]
    transform_training_indices: tuple[int, ...]
    mixture: GaussianMixtureMarginals
    elapsed_seconds: float
    fit_info: dict[str, object] | None = None

    def __post_init__(self) -> None:
        heldout = set(self.heldout_indices)
        if heldout & set(self.training_indices) or heldout & set(self.transform_training_indices):
            raise ValueError("held-out outcomes leaked into reference or preprocessing fit")

    def to_provenance(self) -> dict[str, object]:
        value = asdict(self)
        value.pop("mixture")
        return value


@dataclass
class CrossFitResult:
    z: torch.Tensor
    log_prob: torch.Tensor
    raw_context: torch.Tensor
    folds: list[HeldOutPrediction]
    clamp_count: int
    elapsed_seconds: float
    context_center: torch.Tensor
    context_scale: torch.Tensor

    def context(self, raw_context: torch.Tensor | None = None) -> torch.Tensor:
        raw = self.raw_context if raw_context is None else raw_context.double()
        standardized = ((raw - self.context_center.to(raw.device)) / self.context_scale.to(raw.device)).clamp(-3, 3)
        return torch.cat((torch.ones((raw.shape[0], 1), dtype=torch.double, device=raw.device), standardized), dim=1)


def balanced_folds(count: int, fold_count: int, seed: int) -> list[torch.Tensor]:
    if fold_count < 2 or count < fold_count:
        raise ValueError("need at least one point per fold")
    generator = torch.Generator().manual_seed(seed)
    labels: list[int] = []
    for start in range(0, count, fold_count):
        width = min(fold_count, count - start)
        labels.extend(torch.randperm(fold_count, generator=generator)[:width].tolist())
    label_tensor = torch.tensor(labels)
    return [torch.nonzero(label_tensor == fold).reshape(-1) for fold in range(fold_count)]


def mixture_raw_context(mixture: GaussianMixtureMarginals) -> torch.Tensor:
    return torch.stack((0.5 * mixture.variance.log(), mixture.disagreement_fraction), dim=1)


def cross_fit_map_saas(
    train_x: torch.Tensor,
    raw_train_y: torch.Tensor,
    *,
    taus: torch.Tensor,
    seed: int,
    fold_count: int = 4,
    noise_variance: float = 1e-6,
    max_iterations: int = 250,
    device: torch.device | str = "cpu",
) -> CrossFitResult:
    train_x = train_x.detach().double(); raw_train_y = raw_train_y.detach().double().reshape(-1)
    folds = balanced_folds(len(train_x), fold_count, seed)
    z = torch.empty(len(train_x), dtype=torch.double); log_prob = torch.empty_like(z)
    raw_context = torch.empty((len(train_x), 2), dtype=torch.double)
    records: list[HeldOutPrediction] = []; clamps = 0; start_all = time.perf_counter()
    all_indices = torch.arange(len(train_x))
    for heldout in folds:
        mask = torch.ones(len(train_x), dtype=torch.bool); mask[heldout] = False; training = all_indices[mask]
        start = time.perf_counter()
        reference = fit_map_saas_reference(train_x[training], raw_train_y[training], taus=taus, noise_variance=noise_variance, max_iterations=max_iterations, device=device)
        mixture = reference.posterior(train_x[heldout])
        probability = mixture.cdf(raw_train_y[heldout].to(mixture.means.device)).cpu()
        clipped = probability.clamp(1e-12, 1-1e-12)
        clamps += int(torch.count_nonzero(probability != clipped)); z[heldout] = torch.special.ndtri(clipped)
        log_prob[heldout] = mixture.log_prob(raw_train_y[heldout].to(mixture.means.device)).cpu()
        raw_context[heldout] = mixture_raw_context(mixture).cpu()
        records.append(HeldOutPrediction(tuple(map(int, heldout)), tuple(map(int, training)), tuple(map(int, training)), mixture.to("cpu"), time.perf_counter()-start, reference.fit_info.to_dict()))
    center = raw_context.mean(dim=0); scale = raw_context.std(dim=0, unbiased=False).clamp_min(1e-6)
    return CrossFitResult(z, log_prob, raw_context, records, clamps, time.perf_counter()-start_all, center, scale)
