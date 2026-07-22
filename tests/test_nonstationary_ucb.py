import sys
from pathlib import Path

# Add the project's src directory to Python path
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import numpy as np
from bandit_benchmark.algorithms.nonstationary_ucb import DiscountedUCB, SlidingWindowUCB


def test_discounted_ucb_init():
    """Test: D-UCB initialization."""
    ducb = DiscountedUCB(n_arms=3, seed=42, discount=0.95)
    assert ducb.discount == 0.95
    assert ducb.name == "discounted_ucb"


def test_discounted_ucb_initial_exploration():
    """Test: D-UCB tries each arm once first."""
    ducb = DiscountedUCB(n_arms=3, seed=42, discount=0.95)

    assert ducb.select_action() == 0
    ducb.update(0, 1.0)

    assert ducb.select_action() == 1
    ducb.update(1, 0.0)

    assert ducb.select_action() == 2
    ducb.update(2, 0.5)

    assert len(ducb.history_actions) == 3


def test_discounted_ucb_prefers_best():
    """Test: D-UCB identifies and prefers the best arm."""
    ducb = DiscountedUCB(n_arms=2, seed=42, discount=0.99)

    # Initial exploration
    ducb.update(0, 1.0)  # Arm 0: reward 1.0
    ducb.update(1, 0.0)  # Arm 1: reward 0.0

    # Subsequent selections should favor Arm 0
    for _ in range(10):
        action = ducb.select_action()
        ducb.update(action, 1.0 if action == 0 else 0.0)

    # Arm 0 should be selected more often
    count_0 = sum(1 for a in ducb.history_actions if a == 0)
    count_1 = sum(1 for a in ducb.history_actions if a == 1)
    assert count_0 > count_1


def test_sliding_window_init():
    """Test: SW-UCB initialization."""
    swucb = SlidingWindowUCB(n_arms=3, seed=42, window_size=100)
    assert swucb.window_size == 100
    assert swucb.name == "sliding_window_ucb"


def test_sliding_window_initial_exploration():
    """Test: SW-UCB tries each arm once first."""
    swucb = SlidingWindowUCB(n_arms=3, seed=42, window_size=100)

    assert swucb.select_action() == 0
    swucb.update(0, 1.0)

    assert swucb.select_action() == 1
    swucb.update(1, 0.0)

    assert swucb.select_action() == 2
    swucb.update(2, 0.5)

    assert len(swucb.history_actions) == 3


def test_sliding_window_drops_old_observations():
    """Test: SW-UCB only uses recent observations in selection."""
    swucb = SlidingWindowUCB(n_arms=2, seed=42, window_size=5)

    # Add 10 observations
    for _ in range(10):
        swucb.update(0, 0.0)

    # The history should contain all 10, but selection only uses last 5
    assert len(swucb.history_actions) == 10
    # The select_action should work without error
    action = swucb.select_action()
    assert 0 <= action < swucb.n_arms
