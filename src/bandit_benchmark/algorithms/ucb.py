from __future__ import annotations

from bandit_benchmark.algorithms.base import BanditAlgorithm


class UCB1(BanditAlgorithm):
    """UCB1 implementation placeholder assigned to Xuantong.

    Acceptance criteria are listed in ``docs/sprint-plan.md``. Keep this class
    interface unchanged so it can be registered by the experiment runner.
    """

    name = "ucb1"

    def select_action(self) -> int:
        raise NotImplementedError("implement UCB1.select_action")

    def update(self, action: int, reward: float) -> None:
        raise NotImplementedError("implement UCB1.update")
