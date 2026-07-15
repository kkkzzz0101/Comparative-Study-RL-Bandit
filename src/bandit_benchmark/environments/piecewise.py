from __future__ import annotations

import numpy as np

from bandit_benchmark.environments.base import (
    BanditEnvironment,
    Scenario,
    make_bernoulli_scenario,
    validate_horizon,
)


def _legacy_profile(n_arms: int, rng: np.random.Generator) -> np.ndarray:
    profile = np.linspace(0.15, 0.85, n_arms)
    return profile[rng.permutation(n_arms)]


def _sample_spaced_change_points(
    *, horizon: int, n_changes: int, min_segment_length: int, rng: np.random.Generator
) -> list[int]:
    minimum_horizon = (n_changes + 1) * min_segment_length
    if horizon < minimum_horizon:
        raise ValueError(
            "horizon must be at least (n_changes + 1) * min_segment_length"
        )
    upper = horizon - minimum_horizon + n_changes
    compressed = np.sort(rng.choice(np.arange(1, upper + 1), size=n_changes, replace=False))
    points = compressed + np.arange(1, n_changes + 1) * (min_segment_length - 1)
    return [int(point) for point in points]


class CanonicalAbruptBernoulli(BanditEnvironment):
    """Garivier--Moulines three-arm, two-breakpoint Bernoulli benchmark."""

    name = "canonical_abrupt"

    def __init__(self, n_arms: int = 3) -> None:
        super().__init__(n_arms=n_arms)
        if n_arms != 3:
            raise ValueError("canonical_abrupt requires exactly 3 arms")

    def generate(self, horizon: int, scenario_seed: int, reward_seed: int) -> Scenario:
        validate_horizon(horizon)
        first = max(1, int(0.30 * horizon))
        second = min(horizon - 1, max(first + 1, int(0.50 * horizon)))
        if first >= second:
            raise ValueError("horizon is too short for two distinct canonical change points")

        means = np.repeat(np.asarray([[0.5, 0.3, 0.4]]), horizon, axis=0)
        means[first:second, 2] = 0.9
        return make_bernoulli_scenario(
            name=self.name,
            kind="nonstationary",
            expected_rewards=means,
            scenario_seed=scenario_seed,
            reward_seed=reward_seed,
            metadata={
                "change_family": "canonical_piecewise",
                "change_points": [first, second],
                "changed_arms": [[2], [2]],
                "literature_case": "Garivier-Moulines-2011",
                "reference_horizon": 10_000,
                "scaled_from_reference": horizon != 10_000,
                "n_arms": self.n_arms,
            },
        )


class RandomAbruptBernoulli(BanditEnvironment):
    """Irregular piecewise means with sparse, bounded arm-level jumps."""

    name = "random_abrupt"

    def __init__(
        self,
        n_arms: int,
        n_changes: int = 5,
        min_segment_length: int = 100,
        arm_change_probability: float = 0.5,
        min_jump: float = 0.05,
        max_jump: float = 0.30,
        initial_low: float = 0.10,
        initial_high: float = 0.90,
    ) -> None:
        super().__init__(n_arms=n_arms)
        if n_changes < 1:
            raise ValueError("n_changes must be positive")
        if min_segment_length < 1:
            raise ValueError("min_segment_length must be positive")
        if not 0 < arm_change_probability <= 1:
            raise ValueError("arm_change_probability must be in (0, 1]")
        if not 0 < min_jump <= max_jump <= 0.5:
            raise ValueError("jumps must satisfy 0 < min_jump <= max_jump <= 0.5")
        if not 0 <= initial_low < initial_high <= 1:
            raise ValueError("initial bounds must satisfy 0 <= low < high <= 1")
        self.n_changes = n_changes
        self.min_segment_length = min_segment_length
        self.arm_change_probability = arm_change_probability
        self.min_jump = min_jump
        self.max_jump = max_jump
        self.initial_low = initial_low
        self.initial_high = initial_high

    @staticmethod
    def _jump(
        profile: np.ndarray, arms: np.ndarray, magnitudes: np.ndarray, signs: np.ndarray
    ) -> None:
        candidates = profile[arms] + signs * magnitudes
        invalid = (candidates < 0) | (candidates > 1)
        candidates[invalid] = profile[arms][invalid] - signs[invalid] * magnitudes[invalid]
        profile[arms] = candidates

    def generate(self, horizon: int, scenario_seed: int, reward_seed: int) -> Scenario:
        validate_horizon(horizon)
        rng = np.random.default_rng(scenario_seed)
        change_points = _sample_spaced_change_points(
            horizon=horizon,
            n_changes=self.n_changes,
            min_segment_length=self.min_segment_length,
            rng=rng,
        )
        edges = [0, *change_points, horizon]
        profile = rng.uniform(self.initial_low, self.initial_high, size=self.n_arms)
        means = np.empty((horizon, self.n_arms), dtype=float)
        changed_arms: list[list[int]] = []
        realized_jumps: list[list[float]] = []

        for segment, (start, stop) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
            if segment > 0:
                mask = rng.random(self.n_arms) < self.arm_change_probability
                if not np.any(mask):
                    mask[int(rng.integers(self.n_arms))] = True
                arms = np.flatnonzero(mask)
                before = profile[arms].copy()
                magnitudes = rng.uniform(self.min_jump, self.max_jump, size=len(arms))
                signs = rng.choice(np.asarray([-1.0, 1.0]), size=len(arms))
                self._jump(profile, arms, magnitudes, signs)
                changed_arms.append([int(arm) for arm in arms])
                realized_jumps.append([float(value) for value in np.abs(profile[arms] - before)])
            means[start:stop] = profile

        return make_bernoulli_scenario(
            name=self.name,
            kind="nonstationary",
            expected_rewards=means,
            scenario_seed=scenario_seed,
            reward_seed=reward_seed,
            metadata={
                "change_family": "random_piecewise",
                "change_points": change_points,
                "changed_arms": changed_arms,
                "realized_jumps": realized_jumps,
                "n_changes": self.n_changes,
                "min_segment_length": self.min_segment_length,
                "arm_change_probability": self.arm_change_probability,
                "min_jump": self.min_jump,
                "max_jump": self.max_jump,
                "initial_low": self.initial_low,
                "initial_high": self.initial_high,
            },
        )


class AbruptChangeBernoulli(BanditEnvironment):
    """Legacy evenly spaced global redraw; retained for config compatibility."""

    name = "abrupt_change"

    def __init__(self, n_arms: int, n_changes: int = 3) -> None:
        super().__init__(n_arms=n_arms)
        if n_changes < 1:
            raise ValueError("n_changes must be positive")
        self.n_changes = n_changes

    def generate(self, horizon: int, scenario_seed: int, reward_seed: int) -> Scenario:
        validate_horizon(horizon)
        if self.n_changes >= horizon:
            raise ValueError("n_changes must be smaller than horizon")

        rng = np.random.default_rng(scenario_seed)
        edges = np.linspace(0, horizon, self.n_changes + 2, dtype=int)
        means = np.empty((horizon, self.n_arms), dtype=float)
        previous_best: int | None = None
        previous_profile: np.ndarray | None = None
        changed_arms: list[list[int]] = []

        for segment, (start, stop) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
            profile = _legacy_profile(self.n_arms, rng)
            while int(np.argmax(profile)) == previous_best:
                profile = _legacy_profile(self.n_arms, rng)
            means[start:stop] = profile
            if segment > 0 and previous_profile is not None:
                changed_arms.append(
                    [int(arm) for arm in np.flatnonzero(profile != previous_profile)]
                )
            previous_best = int(np.argmax(profile))
            previous_profile = profile

        change_points = [int(point) for point in edges[1:-1]]
        return make_bernoulli_scenario(
            name=self.name,
            kind="nonstationary",
            expected_rewards=means,
            scenario_seed=scenario_seed,
            reward_seed=reward_seed,
            metadata={
                "change_family": "legacy_even_global_redraw",
                "change_points": change_points,
                "changed_arms": changed_arms,
                "n_changes": self.n_changes,
                "legacy": True,
            },
        )
