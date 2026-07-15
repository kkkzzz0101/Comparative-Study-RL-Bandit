from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

EnvironmentKind = Literal["stochastic", "nonstationary", "adversarial"]


@dataclass(frozen=True)
class Scenario:
    """Potential outcomes for one reproducible environment realization."""

    name: str
    kind: EnvironmentKind
    expected_rewards: np.ndarray
    reward_table: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.expected_rewards.ndim != 2:
            raise ValueError("expected_rewards must have shape (horizon, n_arms)")
        if self.reward_table.shape != self.expected_rewards.shape:
            raise ValueError("reward_table and expected_rewards must have equal shapes")
        if self.expected_rewards.shape[1] < 2:
            raise ValueError("scenario must contain at least two arms")
        if not np.all(np.isfinite(self.expected_rewards)):
            raise ValueError("expected rewards must be finite")
        if not np.all((0 <= self.expected_rewards) & (self.expected_rewards <= 1)):
            raise ValueError("expected rewards must lie in [0, 1]")
        if not np.all((0 <= self.reward_table) & (self.reward_table <= 1)):
            raise ValueError("rewards must lie in [0, 1]")

    @property
    def horizon(self) -> int:
        return int(self.expected_rewards.shape[0])

    @property
    def n_arms(self) -> int:
        return int(self.expected_rewards.shape[1])


class BanditEnvironment(ABC):
    """Factory for deterministic, reproducible scenario tables."""

    name: str

    def __init__(self, n_arms: int) -> None:
        if n_arms < 2:
            raise ValueError("n_arms must be at least 2")
        self.n_arms = n_arms

    @abstractmethod
    def generate(self, horizon: int, scenario_seed: int, reward_seed: int) -> Scenario:
        """Generate means and potential rewards before an algorithm is run."""


def sample_bernoulli_rewards(expected_rewards: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.binomial(1, expected_rewards).astype(float)


def validate_horizon(horizon: int) -> None:
    if horizon < 2:
        raise ValueError("horizon must be at least 2")
