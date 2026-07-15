"""Backward-compatible imports for Bernoulli environment classes.

New code may import from :mod:`bandit_benchmark.environments` directly. The
classes live in focused modules so environment families are easier to extend.
"""

from bandit_benchmark.environments.drift import (
    BoundedRandomWalkBernoulli,
    GradualDriftBernoulli,
)
from bandit_benchmark.environments.piecewise import (
    AbruptChangeBernoulli,
    CanonicalAbruptBernoulli,
    RandomAbruptBernoulli,
)
from bandit_benchmark.environments.stationary import StationaryBernoulli
from bandit_benchmark.environments.stress import (
    ObliviousAdversarial,
    RapidSwitchingBernoulli,
)
from bandit_benchmark.environments.switching import HazardSwitchingBernoulli

__all__ = [
    "AbruptChangeBernoulli",
    "BoundedRandomWalkBernoulli",
    "CanonicalAbruptBernoulli",
    "GradualDriftBernoulli",
    "HazardSwitchingBernoulli",
    "ObliviousAdversarial",
    "RandomAbruptBernoulli",
    "RapidSwitchingBernoulli",
    "StationaryBernoulli",
]
