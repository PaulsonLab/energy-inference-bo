"""The three fixed oracle EI scenarios used throughout Task 01."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class OracleScenario:
    name: str
    description: str
    mean_function: Callable[[np.ndarray], np.ndarray]
    scale_function: Callable[[np.ndarray], np.ndarray]
    best_f: float

    def mean(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.mean_function(np.asarray(x, dtype=float)), dtype=float)

    def scale(self, x: np.ndarray) -> np.ndarray:
        scale = np.asarray(self.scale_function(np.asarray(x, dtype=float)), dtype=float)
        if np.any(scale <= 0.0):
            raise ValueError("scenario scale must remain positive")
        return scale


LOCATION_CONTROL = OracleScenario(
    name="location_control",
    description="Control with smooth mean variation and constant predictive scale.",
    mean_function=lambda x: 0.15 + 0.55 * np.sin(2.0 * np.pi * x),
    scale_function=lambda x: np.full_like(x, 0.45),
    best_f=0.2,
)

SCALE_CONTROL = OracleScenario(
    name="scale_control",
    description="Control with constant mean and smoothly varying predictive scale.",
    mean_function=lambda x: np.full_like(x, -0.1),
    scale_function=lambda x: 0.45 + 0.25 * (1.0 + np.cos(2.0 * np.pi * x)) / 2.0,
    best_f=0.25,
)

TAIL_SENSITIVE = OracleScenario(
    name="tail_sensitive",
    description=(
        "Deliberately constructed upper-tail trade-off: Gaussian EI peaks at 0.75, "
        "while oracle-mixture EI peaks at 0.25."
    ),
    mean_function=lambda x: -0.25 - 0.25 * np.sin(2.0 * np.pi * x),
    scale_function=lambda x: 0.75 + 0.25 * np.sin(2.0 * np.pi * x),
    best_f=0.0,
)

PREDETERMINED_SCENARIOS = (LOCATION_CONTROL, SCALE_CONTROL, TAIL_SENSITIVE)


def scenario_grid(points: int = 1001) -> np.ndarray:
    if points < 3:
        raise ValueError("at least three points are needed for grid integration")
    return np.linspace(0.0, 1.0, points, dtype=float)
