import sys
from pathlib import Path

# Add the project's src directory to Python path
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import numpy as np
from bandit_benchmark.algorithms.thompson import ThompsonSampling


def test_thompson_initial_prior():
    """Test: Initial Beta(1,1) prior is uniform."""
    ts = ThompsonSampling(n_arms=3, seed=42)
    
    # Initial alpha = 1, beta = 1
    assert np.all(ts.alphas == 1)
    assert np.all(ts.betas == 1)
    # Mean = 1/(1+1) = 0.5
    assert np.all(ts.means == 0.5)


def test_thompson_update():
    """Test: Updates are correct."""
    ts = ThompsonSampling(n_arms=2, seed=42)
    
    # Arm 0 gets reward 1 (success)
    ts.update(0, 1.0)
    # Arm 1 gets reward 0 (failure)
    ts.update(1, 0.0)
    
    # Arm 0: alpha=2, beta=1, mean=2/3
    assert ts.alphas[0] == 2
    assert ts.betas[0] == 1
    assert ts.means[0] == 2/3
    
    # Arm 1: alpha=1, beta=2, mean=1/3
    assert ts.alphas[1] == 1
    assert ts.betas[1] == 2
    assert ts.means[1] == 1/3


def test_thompson_exploration():
    """Test: Thompson Sampling explores automatically."""
    ts = ThompsonSampling(n_arms=3, seed=42)
    
    # All arms start with mean 0.5, selection is random
    actions = [ts.select_action() for _ in range(30)]
    
    # Should have selected all arms at least once
    assert len(set(actions)) == 3


def test_thompson_prefers_best_arm():
    """Test: Thompson Sampling identifies the best arm."""
    ts = ThompsonSampling(n_arms=2, seed=42)
    
    # Arm 0 always succeeds, Arm 1 always fails
    for _ in range(10):
        ts.update(0, 1.0)
        ts.update(1, 0.0)
    
    # Arm 0 mean should be near 1, Arm 1 mean near 0
    assert ts.means[0] > 0.8
    assert ts.means[1] < 0.2
    
    # Subsequent selections should favor Arm 0
    actions = [ts.select_action() for _ in range(20)]
    count_0 = sum(1 for a in actions if a == 0)
    count_1 = sum(1 for a in actions if a == 1)
    
    assert count_0 > count_1
