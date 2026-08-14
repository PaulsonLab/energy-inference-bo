"""Deterministic oracle geometry and exact local Gaussian conditionals."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch.quasirandom import SobolEngine


def matern52(x1: torch.Tensor, x2: torch.Tensor, lengthscale: float = 0.2) -> torch.Tensor:
    x1, x2 = x1.double(), x2.double()
    distance = torch.cdist(x1[..., :2] / lengthscale, x2[..., :2] / lengthscale)
    scaled = math.sqrt(5.0) * distance
    return (1.0 + scaled + scaled.square() / 3.0) * torch.exp(-scaled)


def maximin_order(points: torch.Tensor) -> torch.Tensor:
    """Reproducible maximin order in the first two coordinates."""
    active = points[:, :2].double().cpu()
    center_distance = ((active - 0.5) ** 2).sum(-1)
    first = int(torch.argmin(center_distance))
    selected = [first]
    remaining = torch.ones(len(points), dtype=torch.bool)
    remaining[first] = False
    minimum = torch.cdist(active, active[first : first + 1]).squeeze(-1)
    for _ in range(1, len(points)):
        score = minimum.clone()
        score[~remaining] = -1.0
        nxt = int(torch.argmax(score))  # torch.argmax supplies the index tie-break.
        selected.append(nxt)
        remaining[nxt] = False
        minimum = torch.minimum(minimum, torch.linalg.vector_norm(active - active[nxt], dim=-1))
    return torch.tensor(selected, dtype=torch.long)


def ordered_sobol(dimension: int, count: int, seed: int) -> torch.Tensor:
    raw = SobolEngine(dimension, scramble=True, seed=seed).draw(count).double()
    return raw[maximin_order(raw)]


@dataclass(frozen=True)
class LocalGeometry:
    x: torch.Tensor
    neighbors: torch.Tensor  # [n,m], padded by zero with mask false
    mask: torch.Tensor
    coefficients: torch.Tensor
    variances: torch.Tensor
    similarity_weights: torch.Tensor
    jitter: float = 1e-8

    @property
    def count(self) -> int:
        return self.x.shape[0]

    @property
    def neighborhood_size(self) -> int:
        return self.neighbors.shape[1]

    def to(self, device: torch.device | str) -> "LocalGeometry":
        return LocalGeometry(*(getattr(self, field).to(device) for field in ("x", "neighbors", "mask", "coefficients", "variances", "similarity_weights")), self.jitter)

    def means(self, values: torch.Tensor) -> torch.Tensor:
        values = values.double()
        gathered = values[self.neighbors]
        return (self.coefficients * gathered * self.mask).sum(-1)


def _row_geometry(
    target: torch.Tensor,
    source: torch.Tensor,
    indices: torch.Tensor,
    *,
    m: int,
    jitter: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    count = indices.numel()
    padded = torch.zeros(m, dtype=torch.long)
    mask = torch.zeros(m, dtype=torch.bool)
    coefficients = torch.zeros(m, dtype=torch.double)
    weights = torch.zeros(m, dtype=torch.double)
    if count == 0:
        return padded, mask, coefficients, weights
    padded[:count], mask[:count] = indices, True
    neighbor_x = source[indices]
    covariance = matern52(neighbor_x, neighbor_x) + jitter * torch.eye(count, dtype=torch.double)
    cross = matern52(neighbor_x, target[None]).squeeze(-1)
    chol = torch.linalg.cholesky(covariance)
    coefficients[:count] = torch.cholesky_solve(cross[:, None], chol).squeeze(-1)
    positive = cross.clamp_min(0)
    weights[:count] = positive / positive.sum().clamp_min(1e-15)
    return padded, mask, coefficients, weights


def build_geometry(x: torch.Tensor, m: int = 8, jitter: float = 1e-8) -> LocalGeometry:
    x = x.detach().double().cpu()
    rows, masks = [], []
    for i in range(len(x)):
        if i == 0:
            indices = torch.empty(0, dtype=torch.long)
        else:
            distances = torch.linalg.vector_norm(x[:i, :2] - x[i, :2], dim=-1)
            indices = torch.argsort(distances, stable=True)[: min(m, i)]
        row = torch.zeros(m, dtype=torch.long); mask = torch.zeros(m, dtype=torch.bool)
        row[:len(indices)] = indices; mask[:len(indices)] = True
        rows.append(row); masks.append(mask)
    neighbors, mask = torch.stack(rows), torch.stack(masks)
    coefficients = torch.zeros((len(x),m),dtype=torch.double); weights = torch.zeros_like(coefficients); variances=torch.ones(len(x),dtype=torch.double)*(1+jitter)
    # The first m rows have variable neighborhood sizes; all later local systems are batched.
    for i in range(min(m,len(x))):
        count=int(mask[i].sum()); indices=neighbors[i,:count]
        _,_,coeff,weight=_row_geometry(x[i],x,indices,m=m,jitter=jitter)
        cross=matern52(x[indices],x[i:i+1]).squeeze(-1) if count else torch.empty(0)
        coefficients[i],weights[i]=coeff,weight; variances[i]=max(1+jitter-float((coeff[:count]*cross).sum()),1e-12)
    if len(x)>m:
        targets=x[m:]; indices=neighbors[m:]; local_x=x[indices]
        covariance=matern52(local_x,local_x)+jitter*torch.eye(m,dtype=torch.double)
        cross=matern52(local_x,targets[:,None,:]).squeeze(-1)
        chol=torch.linalg.cholesky(covariance); solved=torch.cholesky_solve(cross[...,None],chol).squeeze(-1)
        positive=cross.clamp_min(0); coefficients[m:]=solved; weights[m:]=positive/positive.sum(-1,keepdim=True).clamp_min(1e-15)
        variances[m:]=(1+jitter-(solved*cross).sum(-1)).clamp_min(1e-12)
    return LocalGeometry(x,neighbors,mask,coefficients,variances,weights,jitter)


def candidate_geometry(train_x: torch.Tensor, candidate_x: torch.Tensor, m: int = 8, jitter: float = 1e-8) -> LocalGeometry:
    train_x, candidate_x = train_x.detach().double().cpu(), candidate_x.detach().double().cpu()
    count=min(m,len(train_x)); distances=torch.cdist(candidate_x[:,:2],train_x[:,:2])
    selected=torch.argsort(distances,dim=-1,stable=True)[:,:count]
    neighbors=torch.zeros((len(candidate_x),m),dtype=torch.long); neighbors[:,:count]=selected
    mask=torch.zeros((len(candidate_x),m),dtype=torch.bool); mask[:,:count]=True
    local_x=train_x[selected]; covariance=matern52(local_x,local_x)+jitter*torch.eye(count,dtype=torch.double)
    cross=matern52(local_x,candidate_x[:,None,:]).squeeze(-1); chol=torch.linalg.cholesky(covariance)
    solved=torch.cholesky_solve(cross[...,None],chol).squeeze(-1)
    coefficients=torch.zeros((len(candidate_x),m),dtype=torch.double); coefficients[:,:count]=solved
    weights=torch.zeros_like(coefficients); positive=cross.clamp_min(0); weights[:,:count]=positive/positive.sum(-1,keepdim=True).clamp_min(1e-15)
    variances=(1+jitter-(solved*cross).sum(-1)).clamp_min(1e-12)
    return LocalGeometry(candidate_x,neighbors,mask,coefficients,variances,weights,jitter)


def gaussian_factor_log_prob(values: torch.Tensor, geometry: LocalGeometry) -> torch.Tensor:
    means, variances = geometry.means(values), geometry.variances
    return (-0.5 * ((values - means).square() / variances + variances.log() + math.log(2 * math.pi))).sum()
