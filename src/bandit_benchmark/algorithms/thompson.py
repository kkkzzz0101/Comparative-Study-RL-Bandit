from __future__ import annotations

import numpy as np
from bandit_benchmark.algorithms.base import BanditAlgorithm


class ThompsonSampling(BanditAlgorithm):
    """Bernoulli Thompson Sampling with Beta(1,1) prior.
    
    Core idea: Maintain a Beta distribution for each arm, representing
    the posterior belief about its success probability. At each round,
    sample from each arm's distribution and choose the arm with the
    highest sampled value.
    
    This naturally balances exploration and exploitation:
    - Arms with high average reward are likely to be sampled high (exploitation)
    - Arms with high uncertainty (few observations) have wide distributions (exploration)
    
    Reference: Thompson (1933), Chapelle & Li (2011).
    """
    
    name = "thompson"
    
    def __init__(self, n_arms: int, seed: int):
        """Initialize Thompson Sampling with Beta(1,1) prior.
        
        Args:
            n_arms: Number of arms (at least 2)
            seed: Random seed for reproducibility
        """
        super().__init__(n_arms, seed)
        # Beta 分布的 α 参数（成功次数 + 1）
        # 初始为 1（Beta(1,1) 是均匀分布）
        self.alphas = np.ones(n_arms)
        # Beta 分布的 β 参数（失败次数 + 1）
        # 初始为 1（Beta(1,1) 是均匀分布）
        self.betas = np.ones(n_arms)
    
    def select_action(self) -> int:
        """Select arm by sampling from each arm's Beta posterior.
        
        For each arm i:
            1. Sample θ_i ~ Beta(α_i, β_i)
            2. Select arm with largest θ_i
        """
        # 为每个臂从 Beta 分布中采样
        samples = self.rng.beta(self.alphas, self.betas)
        # 选择采样值最大的臂（平局时随机打破）
        return self._random_argmax(samples)
    
    def update(self, action: int, reward: float) -> None:
        """Update Beta posterior for the selected arm.
        
        For Bernoulli rewards (0 or 1):
        - If reward = 1: increment α (success)
        - If reward = 0: increment β (failure)
        
        Args:
            action: The arm that was selected (0 to n_arms-1)
            reward: The observed reward (must be 0 or 1 for Bernoulli)
        """
        self._validate_observation(action, reward)
        
        # 更新 Beta 分布参数
        if reward == 1:
            self.alphas[action] += 1
        else:
            self.betas[action] += 1
    
    @property
    def means(self) -> np.ndarray:
        """Convenience property: posterior mean of each arm.
        
        For Beta(α, β), the mean is α / (α + β).
        This represents the expected success probability of each arm.
        """
        return self.alphas / (self.alphas + self.betas)
