import numpy as np

from bandit_benchmark.environments.base import Scenario
from bandit_benchmark.metrics import calculate_trajectory_metrics


def test_pseudo_regret_matches_toy_example() -> None:
    means = np.asarray([[0.2, 0.8], [0.7, 0.3], [0.4, 0.6]])
    rewards = np.asarray([[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    scenario = Scenario(
        name="toy",
        kind="nonstationary",
        expected_rewards=means,
        reward_table=rewards,
    )
    actions = np.asarray([0, 0, 1])
    observed_rewards = rewards[np.arange(3), actions]

    metrics = calculate_trajectory_metrics(scenario, actions, observed_rewards)

    np.testing.assert_allclose(metrics.instantaneous_pseudo_regret, [0.6, 0.0, 0.0])
    np.testing.assert_allclose(metrics.cumulative_pseudo_regret, [0.6, 0.6, 0.6])
    np.testing.assert_array_equal(metrics.optimal_action, [False, True, True])


def test_external_regret_uses_best_fixed_arm() -> None:
    rewards = np.asarray([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    scenario = Scenario(
        name="toy_adversarial",
        kind="adversarial",
        expected_rewards=rewards,
        reward_table=rewards,
    )
    actions = np.asarray([1, 1, 1])
    observed_rewards = rewards[np.arange(3), actions]

    metrics = calculate_trajectory_metrics(scenario, actions, observed_rewards)

    assert metrics.cumulative_external_regret[-1] == 1.0
