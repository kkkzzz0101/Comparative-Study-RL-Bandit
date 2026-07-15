import numpy as np

from bandit_benchmark.algorithms.random_policy import RandomPolicy


def test_random_policy_always_selects_valid_arm() -> None:
    policy = RandomPolicy(n_arms=5, seed=7)

    actions = np.asarray([policy.select_action() for _ in range(100)])

    assert np.all((0 <= actions) & (actions < 5))


def test_random_policy_is_reproducible() -> None:
    first = RandomPolicy(n_arms=4, seed=9)
    second = RandomPolicy(n_arms=4, seed=9)

    assert [first.select_action() for _ in range(20)] == [second.select_action() for _ in range(20)]
