import numpy as np
import pytest

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


def test_any_tied_best_arm_counts_as_optimal() -> None:
    means = np.asarray([[0.7, 0.7, 0.2], [0.3, 0.8, 0.8]])
    scenario = Scenario(
        name="ties",
        kind="nonstationary",
        expected_rewards=means,
        reward_table=np.zeros_like(means),
    )
    actions = np.asarray([1, 2])

    metrics = calculate_trajectory_metrics(scenario, actions, np.asarray([0.0, 0.0]))

    np.testing.assert_array_equal(metrics.optimal_action, [True, True])
    np.testing.assert_allclose(metrics.instantaneous_pseudo_regret, [0.0, 0.0])


def test_small_but_real_gap_is_not_treated_as_a_tie() -> None:
    means = np.asarray([[0.5, 0.500001]])
    scenario = Scenario(
        name="small_gap",
        kind="stochastic",
        expected_rewards=means,
        reward_table=np.zeros_like(means),
    )

    metrics = calculate_trajectory_metrics(scenario, np.asarray([0]), np.asarray([0.0]))

    assert not metrics.optimal_action[0]
    assert metrics.instantaneous_pseudo_regret[0] == pytest.approx(0.000001)
