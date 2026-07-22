from __future__ import annotations

import numpy as np
from bandit_benchmark.algorithms.base import BanditAlgorithm


class EXP3(BanditAlgorithm):
    """
    EXP3 algorithm for adversarial bandit environments.

    Maintains exponential weights for each arm based on estimated rewards.
    Uses a mixture of the weight distribution and a uniform distribution
    to ensure exploration.

    Reference: Auer, Cesa-Bianchi, Freund, Schapire (2002).
    """

    name = "exp3"

    def __init__(self, n_arms: int, seed: int, gamma: float = 0.1) -> None:
        """
        Initialize EXP3.

        Args:
            n_arms: Number of arms
            seed: Random seed
            gamma: Exploration rate (0 < gamma <= 1)
        """
        super().__init__(n_arms, seed)
        if not 0 < gamma <= 1:
            raise ValueError("gamma must be between 0 and 1")
        self.gamma = gamma

        # Initialize weights uniformly
        self.weights = np.ones(n_arms)
        self.total_steps = 0

    def select_action(self) -> int:
        """Select action according to EXP3 probability distribution."""
        # Total weight
        total_weight = np.sum(self.weights)

        # Compute probability for each arm:
        # p_i = (1 - gamma) * (w_i / total_weight) + gamma / n_arms
        probabilities = (1 - self.gamma) * (self.weights / total_weight) + self.gamma / self.n_arms

        # Ensure numerical stability
        probabilities = np.clip(probabilities, 1e-12, 1.0)

        # Sample an action according to the probability distribution
        # Use the RNG from the base class
        return int(self.rng.choice(self.n_arms, p=probabilities))

    def update(self, action: int, reward: float) -> None:
        """Update weights based on observed reward."""
        self._validate_observation(action, reward)

        self.total_steps += 1

        # Total weight before update
        total_weight = np.sum(self.weights)

        # Compute probabilities for all arms
        probabilities = (1 - self.gamma) * (self.weights / total_weight) + self.gamma / self.n_arms
        probabilities = np.clip(probabilities, 1e-12, 1.0)

        # Compute estimated reward for all arms
        # Estimated reward for arm i: if i == action, reward / p_i, else 0
        estimated_rewards = np.zeros(self.n_arms)
        p_action = probabilities[action]
        # Avoid division by zero
        if p_action > 1e-12:
            estimated_rewards[action] = reward / p_action

        # Update weights: w_i = w_i * exp(gamma * estimated_reward_i / n_arms)
        # The division by n_arms scales the update
        scaling = self.gamma / self.n_arms
        self.weights = self.weights * np.exp(scaling * estimated_rewards)

        # Clip weights to avoid numerical overflow
        self.weights = np.clip(self.weights, 1e-12, 1e12)

    @property
    def probabilities(self) -> np.ndarray:
        """Current selection probabilities for each arm."""
        total_weight = np.sum(self.weights)
        probs = (1 - self.gamma) * (self.weights / total_weight) + self.gamma / self.n_arms
        return np.clip(probs, 1e-12, 1.0)
