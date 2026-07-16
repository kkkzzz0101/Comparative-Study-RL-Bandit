import numpy as np
from bandit_benchmark.algorithms.ucb import UCB1


def test_ucb1_initial_exploration():
    """测试：UCB1 应该先探索所有臂"""
    ucb = UCB1(n_arms=3, seed=42)
    
    # 前3步应该分别选臂0、1、2
    assert ucb.select_action() == 0
    ucb.update(0, 1.0)
    assert ucb.select_action() == 1
    ucb.update(1, 0.0)
    assert ucb.select_action() == 2
    ucb.update(2, 0.5)
    
    # 此时总步数 = 3
    assert ucb.total_steps == 3


def test_ucb1_counts_update():
    """测试：counts 和 values 是否正确更新"""
    ucb = UCB1(n_arms=2, seed=42)
    
    ucb.update(0, 1.0)
    ucb.update(1, 0.5)
    ucb.update(0, 0.0)
    
    # 臂0被选2次，累计奖励1.0
    assert ucb.counts[0] == 2
    assert ucb.values[0] == 1.0
    # 臂1被选1次，累计奖励0.5
    assert ucb.counts[1] == 1
    assert ucb.values[1] == 0.5
    # 总步数 = 3
    assert ucb.total_steps == 3


def test_ucb1_random_argmax():
    """测试：平局时使用随机选择（_random_argmax）"""
    ucb = UCB1(n_arms=3, seed=42)
    # 跳过探索阶段，手动设置 counts
    ucb.total_steps = 10
    ucb.counts = np.array([5, 5, 5])
    ucb.values = np.array([0.5, 0.5, 0.5])
    
    # 所有臂的 UCB 分数相同，应该随机选择
    actions = [ucb.select_action() for _ in range(20)]
    # 应该不是每次都选同一个（但随机性可能导致偶尔集中）
    # 我们检查是否至少选过2个不同的臂
    assert len(set(actions)) >= 2
