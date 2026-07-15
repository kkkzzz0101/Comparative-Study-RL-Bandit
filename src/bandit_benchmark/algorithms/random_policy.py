from __future__ import annotations

from bandit_benchmark.algorithms.base import BanditAlgorithm


class RandomPolicy(BanditAlgorithm):
    """Uniform random policy used as a lower-bound sanity check."""

    name = "random"

    def select_action(self) -> int:
        return int(self.rng.integers(self.n_arms))

    def update(self, action: int, reward: float) -> None:
        self._validate_observation(action, reward)
