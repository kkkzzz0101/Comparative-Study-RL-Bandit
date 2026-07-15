from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bandit_benchmark.environments.base import Scenario


@dataclass(frozen=True)
class TrajectoryMetrics:
    selected_means: np.ndarray
    optimal_arms: np.ndarray
    optimal_means: np.ndarray
    instantaneous_pseudo_regret: np.ndarray
    cumulative_pseudo_regret: np.ndarray
    cumulative_external_regret: np.ndarray
    optimal_action: np.ndarray


def calculate_trajectory_metrics(
    scenario: Scenario, actions: np.ndarray, observed_rewards: np.ndarray
) -> TrajectoryMetrics:
    if actions.shape != (scenario.horizon,):
        raise ValueError("actions must contain exactly one action per time step")
    if observed_rewards.shape != (scenario.horizon,):
        raise ValueError("observed_rewards must contain exactly one reward per time step")
    if not np.all((0 <= actions) & (actions < scenario.n_arms)):
        raise ValueError("actions contain an invalid arm index")

    time = np.arange(scenario.horizon)
    selected_means = scenario.expected_rewards[time, actions]
    optimal_arms = np.argmax(scenario.expected_rewards, axis=1)
    optimal_means = np.max(scenario.expected_rewards, axis=1)
    instantaneous_pseudo_regret = optimal_means - selected_means
    cumulative_pseudo_regret = np.cumsum(instantaneous_pseudo_regret)

    cumulative_arm_rewards = np.cumsum(scenario.reward_table, axis=0)
    best_fixed_arm_reward = np.max(cumulative_arm_rewards, axis=1)
    cumulative_external_regret = best_fixed_arm_reward - np.cumsum(observed_rewards)

    return TrajectoryMetrics(
        selected_means=selected_means,
        optimal_arms=optimal_arms,
        optimal_means=optimal_means,
        instantaneous_pseudo_regret=instantaneous_pseudo_regret,
        cumulative_pseudo_regret=cumulative_pseudo_regret,
        cumulative_external_regret=cumulative_external_regret,
        optimal_action=actions == optimal_arms,
    )
