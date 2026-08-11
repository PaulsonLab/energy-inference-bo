"""Small transparent structural-particle coresets for an acquisition teacher."""

from __future__ import annotations

from dataclasses import dataclass

import torch


def _matrix(value: torch.Tensor, name: str) -> torch.Tensor:
    result = value.detach().to(dtype=torch.double, device="cpu")
    if result.ndim != 2 or min(result.shape) < 1 or not torch.isfinite(result).all():
        raise ValueError(f"{name} must be a finite nonempty matrix")
    return result


def _validate_k(k: int, particles: int) -> int:
    k = int(k)
    if not 1 <= k <= particles:
        raise ValueError("coreset size must lie between one and the particle count")
    return k


@dataclass(frozen=True)
class Coreset:
    method: str
    indices: torch.Tensor
    weights: torch.Tensor
    error_history: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        indices = self.indices.detach().to(dtype=torch.long, device="cpu").reshape(-1)
        weights = self.weights.detach().to(dtype=torch.double, device="cpu").reshape(-1)
        if indices.shape != weights.shape or indices.numel() < 1:
            raise ValueError("coreset indices and weights must be nonempty and aligned")
        if torch.unique(indices).numel() != indices.numel() or torch.any(indices < 0):
            raise ValueError("coreset indices must be unique and nonnegative")
        if not torch.isfinite(weights).all() or torch.any(weights < -1e-14):
            raise ValueError("coreset weights must be finite and nonnegative")
        weights = weights.clamp_min(0)
        if not torch.isclose(weights.sum(), torch.tensor(1.0, dtype=torch.double), atol=1e-12):
            raise ValueError("coreset weights must sum to one")
        object.__setattr__(self, "indices", indices)
        object.__setattr__(self, "weights", weights / weights.sum())

    def acquisition(self, signatures: torch.Tensor) -> torch.Tensor:
        signatures = _matrix(signatures, "signatures")
        if int(self.indices.max()) >= signatures.shape[0]:
            raise ValueError("coreset index exceeds signature particle count")
        return torch.einsum("k,kj->j", self.weights, signatures[self.indices])


def random_equal(signatures: torch.Tensor, k: int, seed: int) -> Coreset:
    signatures = _matrix(signatures, "signatures")
    k = _validate_k(k, signatures.shape[0])
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    indices = torch.randperm(signatures.shape[0], generator=generator)[:k]
    weights = torch.full((k,), 1.0 / k, dtype=torch.double)
    return Coreset("random_equal", indices, weights)


def posterior_k_medoids(features: torch.Tensor, k: int, *, max_iterations: int = 50) -> Coreset:
    """Deterministic farthest-first k-medoids with cluster-mass weights."""

    features = _matrix(features, "features")
    k = _validate_k(k, features.shape[0])
    centered = features - features.mean(dim=0, keepdim=True)
    standardized = centered / centered.square().mean(dim=0).sqrt().clamp_min(1e-15)
    distances = torch.cdist(standardized, standardized)
    first = int(torch.argmin(standardized.square().sum(dim=1)))
    medoids = [first]
    while len(medoids) < k:
        nearest = distances[:, medoids].min(dim=1).values
        nearest[torch.tensor(medoids)] = -1.0
        medoids.append(int(torch.argmax(nearest)))
    medoid_tensor = torch.tensor(medoids, dtype=torch.long)
    for _ in range(max_iterations):
        assignment = torch.argmin(distances[:, medoid_tensor], dim=1)
        updated: list[int] = []
        for cluster in range(k):
            members = torch.nonzero(assignment == cluster, as_tuple=False).reshape(-1)
            if members.numel() == 0:
                # Exact duplicate particle coordinates can make a later tied medoid
                # empty under deterministic argmin assignment. Retain that medoid;
                # its final cluster-mass weight is correctly zero.
                updated.append(int(medoid_tensor[cluster]))
                continue
            costs = distances[members][:, members].sum(dim=1)
            updated.append(int(members[int(torch.argmin(costs))]))
        new_tensor = torch.tensor(updated, dtype=torch.long)
        if torch.equal(new_tensor, medoid_tensor):
            break
        medoid_tensor = new_tensor
    assignment = torch.argmin(distances[:, medoid_tensor], dim=1)
    weights = torch.bincount(assignment, minlength=k).double() / features.shape[0]
    order = torch.argsort(medoid_tensor)
    return Coreset("posterior_medoid", medoid_tensor[order], weights[order])


def acquisition_frank_wolfe(signatures: torch.Tensor, k: int) -> Coreset:
    """Select at most ``k`` signatures by Frank-Wolfe with exact line search."""

    signatures = _matrix(signatures, "signatures")
    k = _validate_k(k, signatures.shape[0])
    target = signatures.mean(dim=0)
    initial = int(torch.argmin((signatures - target).square().mean(dim=1)))
    full_weights = torch.zeros(signatures.shape[0], dtype=torch.double)
    full_weights[initial] = 1.0
    current = signatures[initial].clone()
    support = {initial}
    history = [float(torch.mean((current - target).square()).sqrt())]
    while len(support) < k:
        residual = current - target
        scores = signatures @ residual
        scores[torch.tensor(sorted(support), dtype=torch.long)] = torch.inf
        vertex = int(torch.argmin(scores))
        direction = signatures[vertex] - current
        denominator = torch.dot(direction, direction)
        if float(denominator) <= torch.finfo(torch.double).eps:
            break
        gamma = float(torch.clamp(-torch.dot(residual, direction) / denominator, 0.0, 1.0))
        if gamma <= 1e-15:
            break
        full_weights *= 1.0 - gamma
        full_weights[vertex] += gamma
        current = (1.0 - gamma) * current + gamma * signatures[vertex]
        support.add(vertex)
        history.append(float(torch.mean((current - target).square()).sqrt()))
    indices = torch.nonzero(full_weights > 0, as_tuple=False).reshape(-1)
    return Coreset(
        "acquisition_fw",
        indices,
        full_weights[indices] / full_weights[indices].sum(),
        tuple(history),
    )
