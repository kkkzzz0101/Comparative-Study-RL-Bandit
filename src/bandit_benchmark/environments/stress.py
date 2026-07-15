from __future__ import annotations

import numpy as np

from bandit_benchmark.environments.base import (
    BanditEnvironment,
    Scenario,
    mean_path_metadata,
    validate_horizon,
)


class RapidSwitchingBernoulli(BanditEnvironment):
    """Blockwise Bernoulli leaders with independent outcome corruption."""

    name = "rapid_switching"

    def __init__(
        self,
        n_arms: int,
        block_size: int = 50,
        base_mean: float = 0.15,
        leader_mean: float = 0.85,
        corruption: float = 0.10,
        force_leader_change: bool = True,
    ) -> None:
        super().__init__(n_arms=n_arms)
        if block_size < 1:
            raise ValueError("block_size must be positive")
        if not 0 <= base_mean < leader_mean <= 1:
            raise ValueError("means must satisfy 0 <= base_mean < leader_mean <= 1")
        if not 0 <= corruption < 0.5:
            raise ValueError("corruption must be in [0, 0.5)")
        self.block_size = block_size
        self.base_mean = base_mean
        self.leader_mean = leader_mean
        self.corruption = corruption
        self.force_leader_change = force_leader_change

    def generate(self, horizon: int, scenario_seed: int, reward_seed: int) -> Scenario:
        validate_horizon(horizon)
        scenario_rng = np.random.default_rng(scenario_seed)
        reward_rng = np.random.default_rng(reward_seed)
        n_blocks = int(np.ceil(horizon / self.block_size))
        leaders: list[int] = []
        for _ in range(n_blocks):
            leader = int(scenario_rng.integers(self.n_arms))
            while self.force_leader_change and leaders and leader == leaders[-1]:
                leader = int(scenario_rng.integers(self.n_arms))
            leaders.append(leader)

        raw_means = np.full((horizon, self.n_arms), self.base_mean, dtype=float)
        for block, leader in enumerate(leaders):
            start = block * self.block_size
            stop = min(start + self.block_size, horizon)
            raw_means[start:stop, leader] = self.leader_mean

        rewards = reward_rng.binomial(1, raw_means).astype(float)
        corruption_mask = reward_rng.random(rewards.shape) < self.corruption
        rewards[corruption_mask] = 1.0 - rewards[corruption_mask]
        effective_means = self.corruption + (1.0 - 2.0 * self.corruption) * raw_means
        changed_blocks = [
            block for block in range(1, n_blocks) if leaders[block] != leaders[block - 1]
        ]
        change_points = [block * self.block_size for block in changed_blocks]
        metadata = {
            "scenario_seed": int(scenario_seed),
            "reward_seed": int(reward_seed),
            "reward_model": "corrupted_bernoulli",
            "change_family": "rapid_block_switching",
            "change_points": change_points,
            "changed_arms": [
                sorted([leaders[block - 1], leaders[block]]) for block in changed_blocks
            ],
            "leaders": leaders,
            "block_size": self.block_size,
            "base_mean": self.base_mean,
            "leader_mean": self.leader_mean,
            "corruption": self.corruption,
            "force_leader_change": self.force_leader_change,
        }
        metadata.update(mean_path_metadata(effective_means))
        return Scenario(
            name=self.name,
            kind="nonstationary",
            expected_rewards=effective_means,
            reward_table=rewards,
            metadata=metadata,
        )


class ObliviousAdversarial(RapidSwitchingBernoulli):
    """Deprecated registry-compatible name for :class:`RapidSwitchingBernoulli`."""

    name = "oblivious_adversarial"
