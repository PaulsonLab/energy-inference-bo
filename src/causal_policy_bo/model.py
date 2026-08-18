"""The frozen two-regime mechanism model used by ``policy_kill``."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from scipy.special import expit, ndtr


@dataclass(frozen=True)
class TwoRegimeConfig:
    """Scientific parameters for one frozen two-regime problem."""

    delta: float
    noise_std: float
    prior_m1: float = 0.5
    incumbent: float = 0.60
    left_center: float = 0.25
    right_center: float = 0.75
    diagnostic_center: float = 0.50
    exploit_width: float = 0.06
    diagnostic_width: float = 0.035
    domain_low: float = 0.0
    domain_high: float = 1.0
    objective: str = "best_observed_improvement"

    def __post_init__(self) -> None:
        if not 0.0 < self.prior_m1 < 1.0:
            raise ValueError("prior_m1 must lie strictly between zero and one")
        if self.noise_std <= 0.0:
            raise ValueError("noise_std must be positive")
        if self.delta <= 0.0:
            raise ValueError("delta must be positive")
        if self.objective != "best_observed_improvement":
            raise ValueError("policy_kill freezes the best-observed-improvement objective")

    @property
    def prior_log_odds(self) -> float:
        return float(np.log(self.prior_m1) - np.log1p(-self.prior_m1))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _rbf_numpy(x: np.ndarray, center: float, width: float) -> np.ndarray:
    return np.exp(-0.5 * ((x - center) / width) ** 2)


def _rbf_torch(x: torch.Tensor, center: float, width: float) -> torch.Tensor:
    return torch.exp(-0.5 * ((x - center) / width) ** 2)


class TwoRegimeModel:
    """Known response functions with uncertainty only in the latent regime."""

    def __init__(self, config: TwoRegimeConfig):
        self.config = config

    def response_numpy(self, x: np.ndarray | float, regime: int) -> np.ndarray:
        x_array = np.asarray(x, dtype=np.float64)
        left = _rbf_numpy(
            x_array, self.config.left_center, self.config.exploit_width
        )
        right = _rbf_numpy(
            x_array, self.config.right_center, self.config.exploit_width
        )
        diagnostic = _rbf_numpy(
            x_array,
            self.config.diagnostic_center,
            self.config.diagnostic_width,
        )
        if regime == 0:
            return (
                1.12 * left
                + (1.10 - self.config.delta) * right
                - 0.40 * diagnostic
            )
        if regime == 1:
            return (
                (1.12 - self.config.delta) * left
                + 1.10 * right
                + 0.30 * diagnostic
            )
        raise ValueError("regime must be zero or one")

    def response_torch(self, x: torch.Tensor, regime: torch.Tensor) -> torch.Tensor:
        left = _rbf_torch(x, self.config.left_center, self.config.exploit_width)
        right = _rbf_torch(x, self.config.right_center, self.config.exploit_width)
        diagnostic = _rbf_torch(
            x,
            self.config.diagnostic_center,
            self.config.diagnostic_width,
        )
        response0 = (
            1.12 * left
            + (1.10 - self.config.delta) * right
            - 0.40 * diagnostic
        )
        response1 = (
            (1.12 - self.config.delta) * left
            + 1.10 * right
            + 0.30 * diagnostic
        )
        return torch.where(regime.to(dtype=torch.bool), response1, response0)

    def means_numpy(self, x: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
        return self.response_numpy(x, 0), self.response_numpy(x, 1)

    def posterior_log_odds_numpy(
        self,
        log_odds: np.ndarray | float,
        x: np.ndarray | float,
        y: np.ndarray | float,
    ) -> np.ndarray:
        mu0, mu1 = self.means_numpy(x)
        log_likelihood_ratio = (
            (np.asarray(y) - mu0) ** 2 - (np.asarray(y) - mu1) ** 2
        ) / (2.0 * self.config.noise_std**2)
        return np.asarray(log_odds) + log_likelihood_ratio

    def posterior_log_odds_torch(
        self, log_odds: torch.Tensor, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        regime0 = torch.zeros_like(x, dtype=torch.long)
        regime1 = torch.ones_like(x, dtype=torch.long)
        mu0 = self.response_torch(x, regime0)
        mu1 = self.response_torch(x, regime1)
        return log_odds + ((y - mu0) ** 2 - (y - mu1) ** 2) / (
            2.0 * self.config.noise_std**2
        )

    @staticmethod
    def probability_m1(log_odds: np.ndarray | float) -> np.ndarray:
        return expit(np.asarray(log_odds, dtype=np.float64))

    @staticmethod
    def normal_expected_improvement(
        mean: np.ndarray,
        incumbent: np.ndarray,
        noise_std: float,
    ) -> np.ndarray:
        z = (mean - incumbent) / noise_std
        density = np.exp(-0.5 * z**2) / np.sqrt(2.0 * np.pi)
        return (mean - incumbent) * ndtr(z) + noise_std * density

    def predictive_expected_improvement(
        self,
        x: np.ndarray,
        log_odds: np.ndarray | float,
        incumbent: np.ndarray | float,
    ) -> np.ndarray:
        probability1 = self.probability_m1(log_odds)
        mean0, mean1 = self.means_numpy(x)
        ei0 = self.normal_expected_improvement(
            mean0, np.asarray(incumbent), self.config.noise_std
        )
        ei1 = self.normal_expected_improvement(
            mean1, np.asarray(incumbent), self.config.noise_std
        )
        return (1.0 - probability1) * ei0 + probability1 * ei1
