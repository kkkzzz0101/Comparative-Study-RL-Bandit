import json
from pathlib import Path

import numpy as np
import pytest

from bandit_benchmark.environments import (
    BoundedRandomWalkBernoulli,
    CanonicalAbruptBernoulli,
    CSVMeanPathEnvironment,
    GradualDriftBernoulli,
    HazardSwitchingBernoulli,
    LoggedBanditEvent,
    RandomAbruptBernoulli,
    RapidSwitchingBernoulli,
    StationaryBernoulli,
)
from bandit_benchmark.registry import build_environment


@pytest.mark.parametrize(
    ("name", "params", "horizon"),
    [
        ("stationary", {"n_arms": 4, "gap": 0.10}, 120),
        ("canonical_abrupt", {"n_arms": 3}, 120),
        (
            "random_abrupt",
            {"n_arms": 4, "n_changes": 2, "min_segment_length": 30},
            120,
        ),
        ("gradual_drift", {"n_arms": 4}, 120),
        ("bounded_random_walk", {"n_arms": 4}, 120),
        (
            "hazard_switching",
            {"n_arms": 4, "hazard_rate": 0.1, "mode": "per_arm"},
            120,
        ),
        ("rapid_switching", {"n_arms": 4, "block_size": 20}, 120),
    ],
)
def test_official_environment_contract_and_reproducibility(
    name: str, params: dict[str, object], horizon: int
) -> None:
    environment = build_environment(name, params)

    first = environment.generate(horizon=horizon, scenario_seed=11, reward_seed=21)
    second = environment.generate(horizon=horizon, scenario_seed=11, reward_seed=21)
    new_rewards = environment.generate(horizon=horizon, scenario_seed=11, reward_seed=22)

    assert first.expected_rewards.shape == (horizon, environment.n_arms)
    assert first.reward_table.shape == first.expected_rewards.shape
    assert np.all(np.isfinite(first.expected_rewards))
    assert np.all((0 <= first.expected_rewards) & (first.expected_rewards <= 1))
    assert set(np.unique(first.reward_table)).issubset({0.0, 1.0})
    np.testing.assert_array_equal(first.expected_rewards, second.expected_rewards)
    np.testing.assert_array_equal(first.reward_table, second.reward_table)
    np.testing.assert_array_equal(first.expected_rewards, new_rewards.expected_rewards)
    json.dumps(first.metadata)


def test_stationary_environment_controls_best_arm_gap() -> None:
    scenario = StationaryBernoulli(n_arms=5, low=0.10, high=0.80, gap=0.12).generate(
        horizon=100, scenario_seed=1, reward_seed=2
    )

    profile = np.sort(scenario.expected_rewards[0])
    assert np.unique(scenario.expected_rewards, axis=0).shape[0] == 1
    assert profile[-1] - profile[-2] == pytest.approx(0.12)
    assert scenario.metadata["realized_best_arm_gap"] == pytest.approx(0.12)


def test_canonical_abrupt_matches_published_three_arm_case() -> None:
    scenario = CanonicalAbruptBernoulli().generate(
        horizon=10_000, scenario_seed=3, reward_seed=4
    )

    np.testing.assert_allclose(scenario.expected_rewards[2_999], [0.5, 0.3, 0.4])
    np.testing.assert_allclose(scenario.expected_rewards[3_000], [0.5, 0.3, 0.9])
    np.testing.assert_allclose(scenario.expected_rewards[4_999], [0.5, 0.3, 0.9])
    np.testing.assert_allclose(scenario.expected_rewards[5_000], [0.5, 0.3, 0.4])
    assert scenario.metadata["change_points"] == [3_000, 5_000]
    assert np.argmax(scenario.expected_rewards[[0, 3_000, 5_000]], axis=1).tolist() == [0, 2, 0]

    short = CanonicalAbruptBernoulli().generate(horizon=5, scenario_seed=3, reward_seed=4)
    assert short.metadata["change_points"] == [1, 2]


def test_random_abrupt_respects_spacing_changed_arms_and_jump_bounds() -> None:
    environment = RandomAbruptBernoulli(
        n_arms=5,
        n_changes=3,
        min_segment_length=50,
        arm_change_probability=0.4,
        min_jump=0.05,
        max_jump=0.30,
    )
    scenario = environment.generate(horizon=300, scenario_seed=5, reward_seed=6)

    points = scenario.metadata["change_points"]
    edges = [0, *points, scenario.horizon]
    assert all(stop - start >= 50 for start, stop in zip(edges[:-1], edges[1:], strict=True))
    actual_points = (np.flatnonzero(np.any(np.diff(scenario.expected_rewards, axis=0), axis=1)) + 1)
    assert actual_points.tolist() == points

    for index, point in enumerate(points):
        actual_arms = np.flatnonzero(
            scenario.expected_rewards[point] != scenario.expected_rewards[point - 1]
        ).tolist()
        assert actual_arms == scenario.metadata["changed_arms"][index]
        jumps = np.asarray(scenario.metadata["realized_jumps"][index])
        assert np.all((environment.min_jump <= jumps) & (jumps <= environment.max_jump))


def test_gradual_environment_tracks_multiple_optimal_arms() -> None:
    scenario = GradualDriftBernoulli(n_arms=5).generate(
        horizon=500, scenario_seed=6, reward_seed=7
    )

    assert len(np.unique(np.argmax(scenario.expected_rewards, axis=1))) > 1
    assert scenario.metadata["change_points"] == []


def test_bounded_random_walk_reflects_large_steps_into_unit_interval() -> None:
    scenario = BoundedRandomWalkBernoulli(n_arms=4, step_std=3.0).generate(
        horizon=200, scenario_seed=8, reward_seed=9
    )

    assert np.all((0 <= scenario.expected_rewards) & (scenario.expected_rewards <= 1))
    assert scenario.metadata["boundary_rule"] == "reflection"
    assert not np.allclose(scenario.expected_rewards[0], scenario.expected_rewards[-1])


def test_hazard_switching_global_and_per_arm_event_metadata() -> None:
    stationary = HazardSwitchingBernoulli(n_arms=3, hazard_rate=0.0).generate(
        horizon=20, scenario_seed=10, reward_seed=11
    )
    assert np.unique(stationary.expected_rewards, axis=0).shape[0] == 1
    assert stationary.metadata["change_points"] == []

    global_case = HazardSwitchingBernoulli(n_arms=3, hazard_rate=1.0, mode="global").generate(
        horizon=20, scenario_seed=12, reward_seed=13
    )
    assert global_case.metadata["change_points"] == list(range(1, 20))
    assert all(arms == [0, 1, 2] for arms in global_case.metadata["changed_arms"])

    per_arm = HazardSwitchingBernoulli(n_arms=4, hazard_rate=0.4, mode="per_arm").generate(
        horizon=50, scenario_seed=14, reward_seed=15
    )
    for change in per_arm.metadata["changes"]:
        t = change["t"]
        actual = np.flatnonzero(
            per_arm.expected_rewards[t] != per_arm.expected_rewards[t - 1]
        ).tolist()
        assert actual == change["arms"]


def test_rapid_switching_uses_effective_corrupted_means() -> None:
    environment = RapidSwitchingBernoulli(
        n_arms=4,
        block_size=5,
        base_mean=0.20,
        leader_mean=0.80,
        corruption=0.10,
    )
    scenario = environment.generate(horizon=23, scenario_seed=16, reward_seed=17)

    assert scenario.kind == "nonstationary"
    leaders = scenario.metadata["leaders"]
    assert all(first != second for first, second in zip(leaders[:-1], leaders[1:], strict=True))
    np.testing.assert_allclose(np.unique(scenario.expected_rewards), [0.26, 0.74])
    for block, leader in enumerate(leaders):
        start = block * environment.block_size
        if start < scenario.horizon:
            assert scenario.expected_rewards[start, leader] == pytest.approx(0.74)


def test_rapid_switching_records_only_realized_leader_changes() -> None:
    scenario = RapidSwitchingBernoulli(
        n_arms=2, block_size=3, force_leader_change=False
    ).generate(horizon=60, scenario_seed=1, reward_seed=2)

    actual_points = (
        np.flatnonzero(np.any(np.diff(scenario.expected_rewards, axis=0), axis=1)) + 1
    ).tolist()
    assert scenario.metadata["change_points"] == actual_points
    assert len(scenario.metadata["changed_arms"]) == len(actual_points)


def test_csv_mean_path_adapter_is_callable(tmp_path: Path) -> None:
    source = tmp_path / "ctr.csv"
    source.write_text(
        "time,item_a,item_b,item_c\n"
        "0,0.1,0.2,0.3\n"
        "1,0.2,0.3,0.4\n"
        "2,0.3,0.4,0.5\n",
        encoding="utf-8",
    )
    environment = build_environment(
        "csv_mean_path",
        {"path": str(source), "reward_columns": ["item_a", "item_b", "item_c"]},
    )

    assert isinstance(environment, CSVMeanPathEnvironment)

    scenario = environment.generate(horizon=3, scenario_seed=18, reward_seed=19)

    np.testing.assert_allclose(
        scenario.expected_rewards,
        [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.3, 0.4, 0.5]],
    )
    assert scenario.metadata["source_type"] == "csv_mean_path"


def test_csv_mean_path_rejects_invalid_probabilities(tmp_path: Path) -> None:
    source = tmp_path / "invalid-ctr.csv"
    source.write_text("a,b\n0.2,1.2\n0.3,0.4\n", encoding="utf-8")
    environment = CSVMeanPathEnvironment(path=str(source), reward_columns=["a", "b"])

    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        environment.generate(horizon=2, scenario_seed=1, reward_seed=2)

    with pytest.raises(ValueError, match="sequence of column names"):
        CSVMeanPathEnvironment(path=str(source), reward_columns="ab")


def test_logged_event_reserves_replay_schema() -> None:
    event = LoggedBanditEvent(
        timestamp="2026-01-01T00:00:00Z",
        available_actions=("a", "b"),
        logged_action="b",
        reward=1.0,
        propensity=0.5,
        context={"device": "mobile"},
    )

    assert event.logged_action == "b"
    with pytest.raises(ValueError, match="available_actions"):
        LoggedBanditEvent(
            timestamp="t",
            available_actions=("a",),
            logged_action="b",
            reward=0.0,
            propensity=1.0,
        )
