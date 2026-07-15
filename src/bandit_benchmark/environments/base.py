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


def reflect_unit_interval(values: np.ndarray) -> np.ndarray:
    """Reflect arbitrary real values into ``[0, 1]`` without clipping."""

    folded = np.mod(values, 2.0)
    return np.where(folded <= 1.0, folded, 2.0 - folded)


def mean_path_metadata(expected_rewards: np.ndarray) -> dict[str, Any]:
    """Return JSON-serializable diagnostics shared by every mean-path environment."""

    ordered = np.sort(expected_rewards, axis=1)
    gaps = ordered[:, -1] - ordered[:, -2]
    optimal_arms = np.argmax(expected_rewards, axis=1)
    optimal_switch_points = (np.flatnonzero(optimal_arms[1:] != optimal_arms[:-1]) + 1).tolist()
    step_changes = np.abs(np.diff(expected_rewards, axis=0))
    return {
        "total_variation": float(np.sum(np.max(step_changes, axis=1))),
        "max_step_change": float(np.max(step_changes, initial=0.0)),
        "minimum_gap": float(np.min(gaps)),
        "optimal_switch_points": [int(point) for point in optimal_switch_points],
        "n_optimal_switches": len(optimal_switch_points),
    }


def make_bernoulli_scenario(
    *,
    name: str,
    kind: EnvironmentKind,
    expected_rewards: np.ndarray,
    scenario_seed: int,
    reward_seed: int,
    metadata: dict[str, Any] | None = None,
) -> Scenario:
    """Build a validated Bernoulli scenario with standard diagnostics."""

    expected_rewards = np.asarray(expected_rewards, dtype=float)
    if expected_rewards.ndim != 2 or expected_rewards.shape[0] < 2:
        raise ValueError("expected_rewards must have shape (horizon >= 2, n_arms)")
    if expected_rewards.shape[1] < 2:
        raise ValueError("expected_rewards must contain at least two arms")
    if not np.all(np.isfinite(expected_rewards)):
        raise ValueError("expected rewards must be finite")
    if not np.all((0 <= expected_rewards) & (expected_rewards <= 1)):
        raise ValueError("expected rewards must lie in [0, 1]")

    combined_metadata: dict[str, Any] = {
        "scenario_seed": int(scenario_seed),
        "reward_seed": int(reward_seed),
        "reward_model": "bernoulli",
        "change_points": [],
        "changed_arms": [],
    }
    if metadata:
        combined_metadata.update(metadata)
    combined_metadata.update(mean_path_metadata(expected_rewards))
    return Scenario(
        name=name,
        kind=kind,
        expected_rewards=expected_rewards,
        reward_table=sample_bernoulli_rewards(expected_rewards, reward_seed),
        metadata=combined_metadata,
    )
