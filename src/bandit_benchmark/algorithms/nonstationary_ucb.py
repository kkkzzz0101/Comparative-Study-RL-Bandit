from __future__ import annotations

import numpy as np
from bandit_benchmark.algorithms.base import BanditAlgorithm


class DiscountedUCB(BanditAlgorithm):
    """
    Discounted UCB for non-stationary environments.

    Applies exponential discounting to past observations, giving more weight
    to recent rewards. The discount factor gamma controls how quickly old
    information is forgotten.

    Reference: Kocsis & Szepesvári (2006).
    """

    name = "discounted_ucb"

    def __init__(self, n_arms: int, seed: int, discount: float = 0.99) -> None:
        super().__init__(n_arms=n_arms, seed=seed)
        if not 0 < discount < 1:
            raise ValueError("discount must be between 0 and 1")
        self.discount = discount

        # Store full history of actions and rewards for discount computation
        self.history_actions = []
        self.history_rewards = []

        # Total discounted weight at each step
        self.total_discounted_weight = 0.0

    def select_action(self) -> int:
        """Select arm with highest discounted UCB score."""
        if len(self.history_actions) < self.n_arms:
            # Initial exploration: try each arm once
            return len(self.history_actions)

        # Compute discounted counts and rewards for each arm
        discounted_counts = np.zeros(self.n_arms)
        discounted_rewards = np.zeros(self.n_arms)

        # Iterate through history in reverse order (most recent first)
        # This allows us to apply discount factors correctly
        for t, (action, reward) in enumerate(zip(self.history_actions, self.history_rewards)):
            # Discount factor: gamma^(t_current - t)
            # The most recent observation gets weight 1.0
            # Observations further back get progressively smaller weights
            steps_ago = len(self.history_actions) - 1 - t
            discount_weight = self.discount ** steps_ago

            discounted_counts[action] += discount_weight
            discounted_rewards[action] += discount_weight * reward

        # Update total discounted weight
        self.total_discounted_weight = sum(self.discount ** steps_ago 
                                          for steps_ago in range(len(self.history_actions)))

        # Compute UCB scores
        # Mean = discounted_rewards / discounted_counts
        # Exploration bonus = sqrt(2 * log(total_weight) / discounted_counts)
        with np.errstate(divide='ignore', invalid='ignore'):
            # Avoid division by zero for arms not yet selected in discounted sense
            ucb_scores = np.zeros(self.n_arms)
            for arm in range(self.n_arms):
                if discounted_counts[arm] > 0:
                    mean = discounted_rewards[arm] / discounted_counts[arm]
                    bonus = np.sqrt(2 * np.log(self.total_discounted_weight) / discounted_counts[arm])
                    ucb_scores[arm] = mean + bonus
                else:
                    # If an arm has zero discounted count, give it infinite UCB to force exploration
                    ucb_scores[arm] = np.inf

        return self._random_argmax(ucb_scores)

    def update(self, action: int, reward: float) -> None:
        """Store the observation in history for later discount computation."""
        self._validate_observation(action, reward)

        self.history_actions.append(action)
        self.history_rewards.append(reward)


class SlidingWindowUCB(BanditAlgorithm):
    """
    Sliding-Window UCB for non-stationary environments.

    Uses only the most recent W observations to compute UCB scores.
    Older observations are discarded. This makes the algorithm reactive
    to abrupt changes.

    Reference: Garivier & Moulines (2011).
    """

    name = "sliding_window_ucb"

    def __init__(self, n_arms: int, seed: int, window_size: int = 250) -> None:
        super().__init__(n_arms=n_arms, seed=seed)
        if window_size < n_arms:
            raise ValueError("window_size must be at least n_arms")
        self.window_size = window_size

        # Store recent history of actions and rewards
        self.history_actions = []
        self.history_rewards = []

    def select_action(self) -> int:
        """Select arm with highest UCB score within the sliding window."""
        # Use only the most recent window_size observations
        start_idx = max(0, len(self.history_actions) - self.window_size)
        window_actions = self.history_actions[start_idx:]
        window_rewards = self.history_rewards[start_idx:]

        if len(window_actions) < self.n_arms:
            # Initial exploration: try each arm once
            return len(window_actions)

        # Compute counts and rewards within the window
        counts = np.zeros(self.n_arms)
        rewards = np.zeros(self.n_arms)

        for action, reward in zip(window_actions, window_rewards):
            counts[action] += 1
            rewards[action] += reward

        # Total steps in window
        total_steps = len(window_actions)

        # Compute UCB scores within the window
        with np.errstate(divide='ignore', invalid='ignore'):
            ucb_scores = np.zeros(self.n_arms)
            for arm in range(self.n_arms):
                if counts[arm] > 0:
                    mean = rewards[arm] / counts[arm]
                    bonus = np.sqrt(2 * np.log(total_steps) / counts[arm])
                    ucb_scores[arm] = mean + bonus
                else:
                    # Force exploration of arms not seen in the window
                    ucb_scores[arm] = np.inf

        return self._random_argmax(ucb_scores)

    def update(self, action: int, reward: float) -> None:
        """Store observation and maintain window size."""
        self._validate_observation(action, reward)

        self.history_actions.append(action)
        self.history_rewards.append(reward)

        # Optionally trim history to save memory (but we keep all for simplicity)
        # The select_action method only uses the last window_size entries
