from __future__ import annotations

from bandit_benchmark.algorithms.base import BanditAlgorithm


class EXP3(BanditAlgorithm):
    """EXP3 implementation task for Member A."""

    name = "exp3"

    def __init__(self, n_arms: int, seed: int, gamma: float = 0.07) -> None:
        super().__init__(n_arms=n_arms, seed=seed)
        if not 0 < gamma <= 1:
            raise ValueError("gamma must be in (0, 1]")
        self.gamma = gamma

    def select_action(self) -> int:
        raise NotImplementedError("T15: implement EXP3.select_action")

    def update(self, action: int, reward: float) -> None:
        raise NotImplementedError("T15: implement EXP3.update")
