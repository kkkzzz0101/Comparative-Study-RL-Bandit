import numpy as np

from bandit_benchmark.environments.bernoulli import (
    AbruptChangeBernoulli,
    GradualDriftBernoulli,
    StationaryBernoulli,
)


def test_stationary_environment_is_reproducible() -> None:
    environment = StationaryBernoulli(n_arms=5)

    first = environment.generate(horizon=100, scenario_seed=1, reward_seed=2)
    second = environment.generate(horizon=100, scenario_seed=1, reward_seed=2)

    np.testing.assert_array_equal(first.expected_rewards, second.expected_rewards)
    np.testing.assert_array_equal(first.reward_table, second.reward_table)
    assert np.unique(first.expected_rewards, axis=0).shape[0] == 1


def test_abrupt_environment_changes_optimal_arm_at_each_change_point() -> None:
    scenario = AbruptChangeBernoulli(n_arms=5, n_changes=3).generate(
        horizon=400, scenario_seed=4, reward_seed=5
    )

    change_points = scenario.metadata["change_points"]
    optimal_arms = np.argmax(scenario.expected_rewards, axis=1)

    assert all(optimal_arms[point - 1] != optimal_arms[point] for point in change_points)


def test_gradual_environment_tracks_multiple_optimal_arms() -> None:
    scenario = GradualDriftBernoulli(n_arms=5).generate(horizon=500, scenario_seed=6, reward_seed=7)

    assert len(np.unique(np.argmax(scenario.expected_rewards, axis=1))) > 1
    assert np.all((0 <= scenario.expected_rewards) & (scenario.expected_rewards <= 1))
