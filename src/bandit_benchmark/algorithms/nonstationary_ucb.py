from __future__ import annotations

from bandit_benchmark.algorithms.base import BanditAlgorithm


class DiscountedUCB(BanditAlgorithm):
    """Discounted-UCB implementation task for Member A."""

    name = "discounted_ucb"

    def __init__(self, n_arms: int, seed: int, discount: float = 0.99) -> None:
        super().__init__(n_arms=n_arms, seed=seed)
        if not 0 < discount < 1:
            raise ValueError("discount must be between 0 and 1")
        self.discount = discount

    def select_action(self) -> int:
        raise NotImplementedError("T13: implement DiscountedUCB.select_action")

    def update(self, action: int, reward: float) -> None:
        raise NotImplementedError("T13: implement DiscountedUCB.update")


class SlidingWindowUCB(BanditAlgorithm):
    """Sliding-Window UCB implementation task for Member A."""

    name = "sliding_window_ucb"

    def __init__(self, n_arms: int, seed: int, window_size: int = 250) -> None:
        super().__init__(n_arms=n_arms, seed=seed)
        if window_size < n_arms:
            raise ValueError("window_size must be at least n_arms")
        self.window_size = window_size

    def select_action(self) -> int:
        raise NotImplementedError("T14: implement SlidingWindowUCB.select_action")

    def update(self, action: int, reward: float) -> None:
        raise NotImplementedError("T14: implement SlidingWindowUCB.update")
