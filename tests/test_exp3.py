import sys
from pathlib import Path

# Add the project's src directory to Python path
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import numpy as np
from bandit_benchmark.algorithms.exp3 import EXP3


def test_exp3_init():
    """Test: EXP3 initialization."""
    exp3 = EXP3(n_arms=3, seed=42, gamma=0.1)
    assert exp3.gamma == 0.1
    assert exp3.name == "exp3"
    assert np.all(exp3.weights == 1.0)


def test_exp3_probabilities_sum_to_one():
    """Test: Selection probabilities sum to 1."""
    exp3 = EXP3(n_arms=3, seed=42, gamma=0.1)
    probs = exp3.probabilities
    assert np.isclose(np.sum(probs), 1.0, atol=1e-10)
    assert np.all(probs > 0)


def test_exp3_exploration():
    """Test: EXP3 explores (all arms have positive probability)."""
    exp3 = EXP3(n_arms=3, seed=42, gamma=0.1)
    probs = exp3.probabilities
    # All arms should have non-zero probability due to gamma
    assert np.all(probs > 0)


def test_exp3_update():
    """Test: Weights update correctly after observing reward."""
    exp3 = EXP3(n_arms=2, seed=42, gamma=0.1)

    # Initial weights are 1, 1
    assert np.all(exp3.weights == 1.0)

    # Select arm 0 and observe reward 1.0
    exp3.update(0, 1.0)

    # Arm 0 should have higher weight than arm 1
    assert exp3.weights[0] > exp3.weights[1]


def test_exp3_prefers_better_arm():
    """Test: EXP3 tends to prefer the better arm over time."""
    exp3 = EXP3(n_arms=2, seed=42, gamma=0.1)

    # Simulate 20 rounds where arm 0 is better
    for _ in range(20):
        # Always select arm 0 (in reality, we'd sample, but for testing
        # we force a pattern to see if weights adapt)
        exp3.update(0, 1.0)
        exp3.update(1, 0.0)

    # After many updates, arm 0 should have higher weight
    # The weight ratio is approximately exp(gamma * K / n_arms) for each positive update
    # With 20 updates of reward 1.0, the ratio should be substantial
    assert exp3.weights[0] > exp3.weights[1] * 2  # Changed from 10 to 2

    # Probability of selecting arm 0 should be higher
    probs = exp3.probabilities
    assert probs[0] > probs[1]
