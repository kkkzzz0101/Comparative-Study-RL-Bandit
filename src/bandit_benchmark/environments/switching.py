from __future__ import annotations

from typing import Literal

import numpy as np

from bandit_benchmark.environments.base import (
    BanditEnvironment,
    Scenario,
    make_bernoulli_scenario,
    validate_horizon,
)

SwitchMode = Literal["global", "per_arm"]


class HazardSwitchingBernoulli(BanditEnvironment):
    """Bernoulli means redrawn at global or arm-specific hazard events."""

    name = "hazard_switching"

    def __init__(
        self,
        n_arms: int,
        hazard_rate: float = 0.001,
        mode: SwitchMode = "global",
        redraw_low: float = 0.0,
        redraw_high: float = 1.0,
    ) -> None:
        super().__init__(n_arms=n_arms)
        if not 0 <= hazard_rate <= 1:
            raise ValueError("hazard_rate must be in [0, 1]")
        if mode not in ("global", "per_arm"):
            raise ValueError("mode must be 'global' or 'per_arm'")
        if not 0 <= redraw_low < redraw_high <= 1:
            raise ValueError("redraw bounds must satisfy 0 <= low < high <= 1")
        self.hazard_rate = hazard_rate
        self.mode = mode
        self.redraw_low = redraw_low
        self.redraw_high = redraw_high

    def generate(self, horizon: int, scenario_seed: int, reward_seed: int) -> Scenario:
        validate_horizon(horizon)
        rng = np.random.default_rng(scenario_seed)
        means = np.empty((horizon, self.n_arms), dtype=float)
        means[0] = rng.uniform(self.redraw_low, self.redraw_high, size=self.n_arms)
        changes: list[dict[str, object]] = []

        for t in range(1, horizon):
            means[t] = means[t - 1]
            if self.mode == "global":
                mask = np.repeat(rng.random() < self.hazard_rate, self.n_arms)
            else:
                mask = rng.random(self.n_arms) < self.hazard_rate
            arms = np.flatnonzero(mask)
            if len(arms):
                means[t, arms] = rng.uniform(self.redraw_low, self.redraw_high, size=len(arms))
                changes.append({"t": int(t), "arms": [int(arm) for arm in arms]})

        return make_bernoulli_scenario(
            name=self.name,
            kind="nonstationary",
            expected_rewards=means,
            scenario_seed=scenario_seed,
            reward_seed=reward_seed,
            metadata={
                "change_family": "hazard_switching",
                "change_points": [int(change["t"]) for change in changes],
                "changed_arms": [change["arms"] for change in changes],
                "changes": changes,
                "hazard_rate": self.hazard_rate,
                "mode": self.mode,
                "redraw_low": self.redraw_low,
                "redraw_high": self.redraw_high,
            },
        )
