from __future__ import annotations

from typing import Any

from bandit_benchmark.algorithms import (
    EXP3,
    UCB1,
    DiscountedUCB,
    RandomPolicy,
    SlidingWindowUCB,
    ThompsonSampling,
)
from bandit_benchmark.algorithms.base import BanditAlgorithm
from bandit_benchmark.environments import (
    AbruptChangeBernoulli,
    GradualDriftBernoulli,
    ObliviousAdversarial,
    StationaryBernoulli,
)
from bandit_benchmark.environments.base import BanditEnvironment

ALGORITHMS = {
    "random": RandomPolicy,
    "ucb1": UCB1,
    "thompson_sampling": ThompsonSampling,
    "discounted_ucb": DiscountedUCB,
    "sliding_window_ucb": SlidingWindowUCB,
    "exp3": EXP3,
}

ENVIRONMENTS = {
    "stationary": StationaryBernoulli,
    "gradual_drift": GradualDriftBernoulli,
    "abrupt_change": AbruptChangeBernoulli,
    "oblivious_adversarial": ObliviousAdversarial,
}


def build_algorithm(name: str, n_arms: int, seed: int, params: dict[str, Any]) -> BanditAlgorithm:
    try:
        algorithm_class = ALGORITHMS[name]
    except KeyError as error:
        choices = ", ".join(sorted(ALGORITHMS))
        raise ValueError(f"unknown algorithm '{name}'; choose one of: {choices}") from error
    return algorithm_class(n_arms=n_arms, seed=seed, **params)


def build_environment(name: str, params: dict[str, Any]) -> BanditEnvironment:
    try:
        environment_class = ENVIRONMENTS[name]
    except KeyError as error:
        choices = ", ".join(sorted(ENVIRONMENTS))
        raise ValueError(f"unknown environment '{name}'; choose one of: {choices}") from error
    return environment_class(**params)
