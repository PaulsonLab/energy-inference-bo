"""Discrete teacher validation of structural-decision joint targets."""

from __future__ import annotations

import torch


def _normalize(values: torch.Tensor) -> torch.Tensor:
    total = values.sum()
    if not torch.isfinite(values).all() or not total > 0:
        raise ValueError("joint-target values must be finite, nonnegative, and nonzero")
    return values / total


def joint_target_marginals(
    signatures: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Return M=1, independent M=2, and common-particle M=2 x marginals."""

    signatures = signatures.detach().to(dtype=torch.double, device="cpu")
    if signatures.ndim != 2 or min(signatures.shape) < 1:
        raise ValueError("signatures must have shape [particles, candidates]")
    if not torch.isfinite(signatures).all() or torch.any(signatures < 0):
        raise ValueError("signatures must be finite and nonnegative")
    particles = signatures.shape[0]
    if weights is None:
        weights = torch.full((particles,), 1.0 / particles, dtype=torch.double)
    else:
        weights = weights.detach().to(dtype=torch.double, device="cpu").reshape(-1)
        if weights.shape != (particles,) or not torch.isfinite(weights).all() or torch.any(weights < 0):
            raise ValueError("weights must be finite and nonnegative with one per particle")
        if not weights.sum() > 0:
            raise ValueError("weights must have positive total mass")
        weights = weights / weights.sum()
    teacher = torch.einsum("p,pj->j", weights, signatures)
    # The uniform candidate-prior constant cancels in all normalized marginals.
    m1_joint = weights[:, None] * signatures
    m1 = _normalize(m1_joint.sum(dim=0))
    independent_m2 = _normalize(teacher.square())
    common_particle_m2 = _normalize(torch.einsum("p,pj->j", weights, signatures.square()))
    return {
        "teacher": teacher,
        "normalized_teacher": _normalize(teacher),
        "m1": m1,
        "independent_m2": independent_m2,
        "normalized_teacher_squared": _normalize(teacher.square()),
        "common_particle_m2": common_particle_m2,
    }
