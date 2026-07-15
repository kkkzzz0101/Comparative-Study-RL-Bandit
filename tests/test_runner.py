import csv
import json
from pathlib import Path

from bandit_benchmark.runner import run_experiment


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_smoke_runner_writes_expected_rows(tmp_path: Path) -> None:
    config = tmp_path / "smoke.toml"
    output = tmp_path / "output"
    config.write_text(
        """
[experiment]
name = "test-smoke"
horizon = 20
seeds = [0, 1]
save_trajectories = true

[[algorithms]]
name = "random"

[[environments]]
name = "stationary"
[environments.params]
n_arms = 3

[[environments]]
name = "abrupt_change"
[environments.params]
n_arms = 3
n_changes = 1
""".strip(),
        encoding="utf-8",
    )

    run_experiment(config, output_override=output)

    runs = _read_csv(output / "runs.csv")
    trajectories = _read_csv(output / "trajectories.csv")
    summaries = _read_csv(output / "summary.csv")
    assert len(runs) == 4
    assert len(trajectories) == 2 * 2 * 20
    assert len(summaries) == 2
    assert {row["algorithm"] for row in runs} == {"random"}
    assert {row["algorithm_name"] for row in runs} == {"random"}
    assert {row["environment_name"] for row in runs} == {"stationary", "abrupt_change"}
    assert all("total_variation" in json.loads(row["scenario_metadata"]) for row in runs)


def test_runner_supports_parameterized_environment_ids(tmp_path: Path) -> None:
    config = tmp_path / "variants.toml"
    output = tmp_path / "output"
    config.write_text(
        """
[experiment]
name = "variants"
horizon = 20
seeds = [0]
save_trajectories = false

[[algorithms]]
name = "random"
id = "random_a"

[[algorithms]]
name = "random"
id = "random_b"

[[environments]]
name = "hazard_switching"
id = "hazard_global"
[environments.params]
n_arms = 3
hazard_rate = 0.1
mode = "global"

[[environments]]
name = "hazard_switching"
id = "hazard_per_arm"
[environments.params]
n_arms = 3
hazard_rate = 0.1
mode = "per_arm"
""".strip(),
        encoding="utf-8",
    )

    run_experiment(config, output_override=output)

    runs = _read_csv(output / "runs.csv")
    assert {row["algorithm"] for row in runs} == {"random_a", "random_b"}
    assert {row["algorithm_name"] for row in runs} == {"random"}
    assert {row["environment"] for row in runs} == {"hazard_global", "hazard_per_arm"}
    assert {row["environment_name"] for row in runs} == {"hazard_switching"}
    assert len({row["run_id"] for row in runs}) == 4
