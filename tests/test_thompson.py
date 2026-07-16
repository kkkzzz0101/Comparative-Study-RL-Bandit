import sys
from pathlib import Path

# 将项目的 src 目录添加到 Python 的搜索路径中
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import numpy as np
from bandit_benchmark.algorithms.thompson import ThompsonSampling


def test_thompson_initial_prior():
    """测试：初始时 Beta(1,1) 是均匀分布"""
    ts = ThompsonSampling(n_arms=3, seed=42)
    
    # 初始 α = 1, β = 1
    assert np.all(ts.alphas == 1)
    assert np.all(ts.betas == 1)
    # 均值 = 1/(1+1) = 0.5
    assert np.all(ts.means == 0.5)


def test_thompson_update():
    """测试：更新是否正确"""
    ts = ThompsonSampling(n_arms=2, seed=42)
    
    # 臂0获得奖励1（成功）
    ts.update(0, 1.0)
    # 臂1获得奖励0（失败）
    ts.update(1, 0.0)
    
    # 臂0: α=2, β=1, 均值=2/3
    assert ts.alphas[0] == 2
    assert ts.betas[0] == 1
    assert ts.means[0] == 2/3
    
    # 臂1: α=1, β=2, 均值=1/3
    assert ts.alphas[1] == 1
    assert ts.betas[1] == 2
    assert ts.means[1] == 1/3


def test_thompson_exploration():
    """测试：Thompson Sampling 能自动探索"""
    ts = ThompsonSampling(n_arms=3, seed=42)
    
    # 所有臂初始均值都是0.5，选择是随机的
    actions = [ts.select_action() for _ in range(30)]
    
    # 应该选择过所有臂（至少各一次）
    assert len(set(actions)) == 3


def test_thompson_prefers_best_arm():
    """测试：Thompson Sampling 能识别最优臂"""
    ts = ThompsonSampling(n_arms=2, seed=42)
    
    # 让臂0一直成功，臂1一直失败
    for _ in range(10):
        ts.update(0, 1.0)
        ts.update(1, 0.0)
    
    # 臂0的均值应该接近1，臂1的均值接近0
    assert ts.means[0] > 0.8
    assert ts.means[1] < 0.2
    
    # 后续选择应该偏向臂0
    actions = [ts.select_action() for _ in range(20)]
    count_0 = sum(1 for a in actions if a == 0)
    count_1 = sum(1 for a in actions if a == 1)
    
    assert count_0 > count_1
