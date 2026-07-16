from __future__ import annotations

import numpy as np
from bandit_benchmark.algorithms.base import BanditAlgorithm


class UCB1(BanditAlgorithm):
    """UCB1 algorithm for stationary stochastic bandits.
    
    Core idea: Balance exploration and exploitation using confidence bounds.
    - Exploitation: choose arms with high average reward
    - Exploration: choose arms that haven't been tried enough
    
    UCB1 formula: mean_i + sqrt(2 * ln(t) / n_i)
    where t = total rounds, n_i = number of times arm i was selected.
    
    Reference: Auer, Cesa-Bianchi, and Fischer (2002).
    """
    
    name = "ucb1"
    
    def __init__(self, n_arms: int, seed: int):
        """Initialize UCB1.
        
        Args:
            n_arms: Number of arms (at least 2)
            seed: Random seed for reproducibility
        """
        super().__init__(n_arms, seed)
        # 每个臂被选择的次数
        self.counts = np.zeros(n_arms, dtype=int)
        # 每个臂的累计奖励
        self.values = np.zeros(n_arms)
        # 总步数
        self.total_steps = 0
    
    def select_action(self) -> int:
        """Select arm with highest UCB1 score.
        
        If any arm hasn't been tried yet, select it to ensure initial exploration.
        """
        # 如果还有臂从未被选择，优先探索它
        # 这保证了每个臂至少被选择一次，避免除以 0
        if self.total_steps < self.n_arms:
            return int(self.total_steps)
        
        # 计算每个臂的 UCB 分数
        # 均值 + sqrt(2 * ln(t) / n_i)
        # 注意：self.counts 是整数数组，需要转换为浮点数
        ucb_scores = self.values / self.counts + np.sqrt(
            2 * np.log(self.total_steps) / self.counts
        )
        
        # 使用基类提供的 _random_argmax 来打破平局
        # 这比 np.argmax 更公平（随机选择并列最大的）
        return self._random_argmax(ucb_scores)
    
    def update(self, action: int, reward: float) -> None:
        """Update statistics for the selected arm.
        
        Args:
            action: The arm that was selected (0 to n_arms-1)
            reward: The observed reward (typically 0 or 1 for Bernoulli)
        """
        self._validate_observation(action, reward)
        
        self.counts[action] += 1
        self.values[action] += reward
        self.total_steps += 1
    
    @property
    def means(self) -> np.ndarray:
        """Convenience property: average reward of each arm."""
        return np.divide(
            self.values,
            self.counts,
            out=np.zeros_like(self.values, dtype=float),
            where=(self.counts > 0)
        )
