import csv
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
