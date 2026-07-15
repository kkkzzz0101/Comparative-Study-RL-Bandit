from __future__ import annotations

import numpy as np

from bandit_benchmark.environments.base import (
    BanditEnvironment,
    Scenario,
    make_bernoulli_scenario,
    reflect_unit_interval,
    validate_horizon,
)


class GradualDriftBernoulli(BanditEnvironment):
    """Phase-shifted sinusoidal Bernoulli mean paths."""

    name = "gradual_drift"

    def __init__(self, n_arms: int, cycles: float = 1.5, amplitude: float = 0.3) -> None:
        super().__init__(n_arms=n_arms)
        if cycles <= 0:
            raise ValueError("cycles must be positive")
        if not 0 < amplitude <= 0.45:
            raise ValueError("amplitude must be in (0, 0.45]")
        self.cycles = cycles
        self.amplitude = amplitude

    def generate(self, horizon: int, scenario_seed: int, reward_seed: int) -> Scenario:
        validate_horizon(horizon)
        rng = np.random.default_rng(scenario_seed)
        phase_offset = rng.uniform(0, 2 * np.pi)
        arm_phases = np.linspace(0, 2 * np.pi, self.n_arms, endpoint=False)
        time_phase = np.linspace(0, 2 * np.pi * self.cycles, horizon)
        means = 0.5 + self.amplitude * np.sin(
            time_phase[:, None] + arm_phases[None, :] + phase_offset
        )
        means = np.clip(means, 0.01, 0.99)
        return make_bernoulli_scenario(
            name=self.name,
            kind="nonstationary",
            expected_rewards=means,
            scenario_seed=scenario_seed,
            reward_seed=reward_seed,
            metadata={
                "change_family": "smooth_periodic",
                "cycles": self.cycles,
                "amplitude": self.amplitude,
                "phase_offset": float(phase_offset),
            },
        )


class BoundedRandomWalkBernoulli(BanditEnvironment):
    """Independent Gaussian mean walks with reflection at 0 and 1."""

    name = "bounded_random_walk"

    def __init__(
        self,
        n_arms: int,
        step_std: float = 0.03,
        initial_low: float = 0.20,
        initial_high: float = 0.80,
    ) -> None:
        super().__init__(n_arms=n_arms)
        if step_std < 0:
            raise ValueError("step_std must be non-negative")
        if not 0 <= initial_low < initial_high <= 1:
            raise ValueError("initial bounds must satisfy 0 <= low < high <= 1")
        self.step_std = step_std
        self.initial_low = initial_low
        self.initial_high = initial_high

    def generate(self, horizon: int, scenario_seed: int, reward_seed: int) -> Scenario:
        validate_horizon(horizon)
        rng = np.random.default_rng(scenario_seed)
        means = np.empty((horizon, self.n_arms), dtype=float)
        means[0] = rng.uniform(self.initial_low, self.initial_high, size=self.n_arms)
        for t in range(1, horizon):
            step = rng.normal(0.0, self.step_std, size=self.n_arms)
            means[t] = reflect_unit_interval(means[t - 1] + step)
        return make_bernoulli_scenario(
            name=self.name,
            kind="nonstationary",
            expected_rewards=means,
            scenario_seed=scenario_seed,
            reward_seed=reward_seed,
            metadata={
                "change_family": "bounded_random_walk",
                "step_std": self.step_std,
                "boundary_rule": "reflection",
                "initial_low": self.initial_low,
                "initial_high": self.initial_high,
            },
        )
