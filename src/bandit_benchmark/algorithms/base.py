from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BanditAlgorithm(ABC):
    """Minimal interface shared by every policy in the benchmark.

    An algorithm receives only the reward for the action it selected. It must
    never receive the environment's full reward or expected-reward table.
    """

    name: str

    def __init__(self, n_arms: int, seed: int) -> None:
        if n_arms < 2:
            raise ValueError("n_arms must be at least 2")
        self.n_arms = n_arms
        self.rng = np.random.default_rng(seed)

    @abstractmethod
    def select_action(self) -> int:
        """Choose an arm index in ``[0, n_arms)``."""

    @abstractmethod
    def update(self, action: int, reward: float) -> None:
        """Update internal state from the observed action and reward."""

    def _validate_observation(self, action: int, reward: float) -> None:
        if not 0 <= action < self.n_arms:
            raise ValueError(f"invalid action {action} for {self.n_arms} arms")
        if not np.isfinite(reward):
            raise ValueError("reward must be finite")

    def _random_argmax(self, values: np.ndarray) -> int:
        """Break ties randomly without changing the values being compared."""

        candidates = np.flatnonzero(values == np.max(values))
        return int(self.rng.choice(candidates))
