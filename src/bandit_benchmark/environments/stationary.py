from __future__ import annotations

import numpy as np

from bandit_benchmark.environments.base import (
    BanditEnvironment,
    Scenario,
    make_bernoulli_scenario,
    validate_horizon,
)


class StationaryBernoulli(BanditEnvironment):
    """Stationary Bernoulli arms with an optional controlled best-arm gap."""

    name = "stationary"

    def __init__(
        self,
        n_arms: int,
        low: float = 0.15,
        high: float = 0.85,
        gap: float | None = None,
    ) -> None:
        super().__init__(n_arms=n_arms)
        if not 0 <= low < high <= 1:
            raise ValueError("low and high must satisfy 0 <= low < high <= 1")
        if gap is not None and not 0 < gap <= high - low:
            raise ValueError("gap must be positive and no larger than high - low")
        self.low = low
        self.high = high
        self.gap = gap

    def _profile(self) -> np.ndarray:
        if self.gap is None:
            return np.linspace(self.low, self.high, self.n_arms)
        if self.n_arms == 2:
            return np.asarray([self.high - self.gap, self.high])
        suboptimal = np.linspace(self.low, self.high - self.gap, self.n_arms - 1)
        return np.concatenate((suboptimal, np.asarray([self.high])))

    def generate(self, horizon: int, scenario_seed: int, reward_seed: int) -> Scenario:
        validate_horizon(horizon)
        rng = np.random.default_rng(scenario_seed)
        profile = self._profile()[rng.permutation(self.n_arms)]
        means = np.repeat(profile[None, :], horizon, axis=0)
        sorted_profile = np.sort(profile)
        return make_bernoulli_scenario(
            name=self.name,
            kind="stochastic",
            expected_rewards=means,
            scenario_seed=scenario_seed,
            reward_seed=reward_seed,
            metadata={
                "change_family": "stationary",
                "low": self.low,
                "high": self.high,
                "configured_gap": self.gap,
                "realized_best_arm_gap": float(sorted_profile[-1] - sorted_profile[-2]),
            },
        )
