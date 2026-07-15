from __future__ import annotations

from bandit_benchmark.algorithms.base import BanditAlgorithm


class ThompsonSampling(BanditAlgorithm):
    """Bernoulli Thompson Sampling implementation placeholder assigned to Xuantong."""

    name = "thompson_sampling"

    def select_action(self) -> int:
        raise NotImplementedError("implement ThompsonSampling.select_action")

    def update(self, action: int, reward: float) -> None:
        raise NotImplementedError("implement ThompsonSampling.update")
