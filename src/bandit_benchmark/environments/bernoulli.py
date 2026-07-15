from __future__ import annotations

import numpy as np

from bandit_benchmark.environments.base import (
    BanditEnvironment,
    Scenario,
    sample_bernoulli_rewards,
    validate_horizon,
)


def _random_profile(n_arms: int, rng: np.random.Generator) -> np.ndarray:
    profile = np.linspace(0.15, 0.85, n_arms)
    return profile[rng.permutation(n_arms)]


class StationaryBernoulli(BanditEnvironment):
    name = "stationary"

    def generate(self, horizon: int, scenario_seed: int, reward_seed: int) -> Scenario:
        validate_horizon(horizon)
        rng = np.random.default_rng(scenario_seed)
        means = np.repeat(_random_profile(self.n_arms, rng)[None, :], horizon, axis=0)
        return Scenario(
            name=self.name,
            kind="stochastic",
            expected_rewards=means,
            reward_table=sample_bernoulli_rewards(means, reward_seed),
            metadata={"scenario_seed": scenario_seed, "reward_seed": reward_seed},
        )


class GradualDriftBernoulli(BanditEnvironment):
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
        return Scenario(
            name=self.name,
            kind="nonstationary",
            expected_rewards=means,
            reward_table=sample_bernoulli_rewards(means, reward_seed),
            metadata={
                "cycles": self.cycles,
                "amplitude": self.amplitude,
                "scenario_seed": scenario_seed,
                "reward_seed": reward_seed,
            },
        )


class AbruptChangeBernoulli(BanditEnvironment):
    name = "abrupt_change"

    def __init__(self, n_arms: int, n_changes: int = 3) -> None:
        super().__init__(n_arms=n_arms)
        if n_changes < 1:
            raise ValueError("n_changes must be positive")
        self.n_changes = n_changes

    def generate(self, horizon: int, scenario_seed: int, reward_seed: int) -> Scenario:
        validate_horizon(horizon)
        if self.n_changes >= horizon:
            raise ValueError("n_changes must be smaller than horizon")

        rng = np.random.default_rng(scenario_seed)
        edges = np.linspace(0, horizon, self.n_changes + 2, dtype=int)
        means = np.empty((horizon, self.n_arms), dtype=float)
        previous_best: int | None = None

        for start, stop in zip(edges[:-1], edges[1:], strict=True):
            profile = _random_profile(self.n_arms, rng)
            while int(np.argmax(profile)) == previous_best:
                profile = _random_profile(self.n_arms, rng)
            means[start:stop] = profile
            previous_best = int(np.argmax(profile))

        change_points = edges[1:-1].tolist()
        return Scenario(
            name=self.name,
            kind="nonstationary",
            expected_rewards=means,
            reward_table=sample_bernoulli_rewards(means, reward_seed),
            metadata={
                "change_points": change_points,
                "scenario_seed": scenario_seed,
                "reward_seed": reward_seed,
            },
        )


class ObliviousAdversarial(BanditEnvironment):
    """Pre-generated high-variation rewards, independent of policy actions."""

    name = "oblivious_adversarial"

    def __init__(self, n_arms: int, block_size: int = 50, corruption: float = 0.1) -> None:
        super().__init__(n_arms=n_arms)
        if block_size < 1:
            raise ValueError("block_size must be positive")
        if not 0 <= corruption <= 1:
            raise ValueError("corruption must be in [0, 1]")
        self.block_size = block_size
        self.corruption = corruption

    def generate(self, horizon: int, scenario_seed: int, reward_seed: int) -> Scenario:
        validate_horizon(horizon)
        rng = np.random.default_rng(scenario_seed)
        reward_rng = np.random.default_rng(reward_seed)
        rewards = reward_rng.binomial(1, 0.15, size=(horizon, self.n_arms)).astype(float)

        n_blocks = int(np.ceil(horizon / self.block_size))
        leaders = rng.integers(0, self.n_arms, size=n_blocks)
        for block, leader in enumerate(leaders):
            start = block * self.block_size
            stop = min(start + self.block_size, horizon)
            rewards[start:stop, leader] = reward_rng.binomial(1, 0.85, size=stop - start)

        corruption_mask = reward_rng.random(rewards.shape) < self.corruption
        rewards[corruption_mask] = 1 - rewards[corruption_mask]
        return Scenario(
            name=self.name,
            kind="adversarial",
            expected_rewards=rewards.copy(),
            reward_table=rewards,
            metadata={
                "block_size": self.block_size,
                "corruption": self.corruption,
                "scenario_seed": scenario_seed,
                "reward_seed": reward_seed,
            },
        )
