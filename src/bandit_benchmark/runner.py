from __future__ import annotations

import json
import subprocess
import time
import tomllib
from pathlib import Path
from typing import Any

import numpy as np

from bandit_benchmark.metrics import calculate_trajectory_metrics
from bandit_benchmark.registry import build_algorithm, build_environment
from bandit_benchmark.results import summarize_runs, write_csv

RUN_FIELDS = [
    "run_id",
    "experiment",
    "algorithm",
    "algorithm_name",
    "environment",
    "environment_name",
    "environment_kind",
    "horizon",
    "n_arms",
    "scenario_seed",
    "reward_seed",
    "policy_seed",
    "cumulative_reward",
    "final_pseudo_regret",
    "final_external_regret",
    "optimal_action_rate",
    "runtime_seconds",
    "git_commit",
    "algorithm_params",
    "environment_params",
    "scenario_metadata",
]

TRAJECTORY_FIELDS = [
    "run_id",
    "t",
    "action",
    "observed_reward",
    "selected_mean",
    "optimal_arm",
    "optimal_mean",
    "instantaneous_pseudo_regret",
    "cumulative_pseudo_regret",
    "cumulative_external_regret",
    "optimal_action",
]

SUMMARY_FIELDS = [
    "environment",
    "environment_kind",
    "algorithm",
    "n_runs",
    "primary_metric",
    "mean_regret",
    "regret_ci95",
    "mean_cumulative_reward",
    "reward_ci95",
    "mean_optimal_action_rate",
]


def load_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        config = tomllib.load(handle)
    for key in ("experiment", "algorithms", "environments"):
        if key not in config:
            raise ValueError(f"config is missing required section '{key}'")
    return config


def _enabled(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [entry for entry in entries if entry.get("enabled", True)]


def _seeds(experiment: dict[str, Any]) -> list[int]:
    if "seeds" in experiment:
        return [int(seed) for seed in experiment["seeds"]]
    start = int(experiment.get("seed_start", 0))
    count = int(experiment.get("n_seeds", 1))
    if count < 1:
        raise ValueError("n_seeds must be positive")
    return list(range(start, start + count))


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__} to JSON")


def run_experiment(config_path: Path, output_override: Path | None = None) -> Path:
    config = load_config(config_path)
    experiment = config["experiment"]
    name = str(experiment["name"])
    horizon = int(experiment["horizon"])
    output_dir = output_override or Path(experiment.get("output_dir", f"results/{name}"))
    save_trajectories = bool(experiment.get("save_trajectories", True))
    algorithms = _enabled(config["algorithms"])
    environments = _enabled(config["environments"])

    if horizon < 2 or not algorithms or not environments:
        raise ValueError("horizon, enabled algorithms, and enabled environments are required")

    algorithm_ids = [str(entry.get("id", entry["name"])) for entry in algorithms]
    if len(algorithm_ids) != len(set(algorithm_ids)):
        raise ValueError("enabled algorithm entries must have unique 'id' values")
    environment_ids = [str(entry.get("id", entry["name"])) for entry in environments]
    if len(environment_ids) != len(set(environment_ids)):
        raise ValueError("enabled environment entries must have unique 'id' values")

    run_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    commit = _git_commit()

    for environment_entry in environments:
        environment_name = environment_entry["name"]
        environment_id = str(environment_entry.get("id", environment_name))
        environment_params = dict(environment_entry.get("params", {}))
        environment = build_environment(environment_name, environment_params)

        for base_seed in _seeds(experiment):
            scenario_seed = base_seed
            reward_seed = base_seed + 10_000
            scenario = environment.generate(horizon, scenario_seed, reward_seed)

            for algorithm_entry in algorithms:
                algorithm_name = algorithm_entry["name"]
                algorithm_id = str(algorithm_entry.get("id", algorithm_name))
                algorithm_params = dict(algorithm_entry.get("params", {}))
                policy_seed = base_seed + 20_000
                algorithm = build_algorithm(
                    algorithm_name, scenario.n_arms, policy_seed, algorithm_params
                )
                run_id = f"{name}-{environment_id}-{algorithm_id}-{base_seed}"
                actions = np.empty(horizon, dtype=int)
                observed_rewards = np.empty(horizon, dtype=float)

                started = time.perf_counter()
                for t in range(horizon):
                    action = algorithm.select_action()
                    if not 0 <= action < scenario.n_arms:
                        raise ValueError(f"{algorithm_name} selected invalid arm {action} at t={t}")
                    reward = float(scenario.reward_table[t, action])
                    algorithm.update(action, reward)
                    actions[t] = action
                    observed_rewards[t] = reward
                runtime_seconds = time.perf_counter() - started

                metrics = calculate_trajectory_metrics(scenario, actions, observed_rewards)
                pseudo_regret = (
                    ""
                    if scenario.kind == "adversarial"
                    else float(metrics.cumulative_pseudo_regret[-1])
                )
                external_regret = (
                    float(metrics.cumulative_external_regret[-1])
                    if scenario.kind == "adversarial"
                    else ""
                )
                run_rows.append(
                    {
                        "run_id": run_id,
                        "experiment": name,
                        "algorithm": algorithm_id,
                        "algorithm_name": algorithm_name,
                        "environment": environment_id,
                        "environment_name": environment_name,
                        "environment_kind": scenario.kind,
                        "horizon": horizon,
                        "n_arms": scenario.n_arms,
                        "scenario_seed": scenario_seed,
                        "reward_seed": reward_seed,
                        "policy_seed": policy_seed,
                        "cumulative_reward": float(np.sum(observed_rewards)),
                        "final_pseudo_regret": pseudo_regret,
                        "final_external_regret": external_regret,
                        "optimal_action_rate": float(np.mean(metrics.optimal_action)),
                        "runtime_seconds": runtime_seconds,
                        "git_commit": commit,
                        "algorithm_params": json.dumps(
                            algorithm_params, sort_keys=True, default=_json_default
                        ),
                        "environment_params": json.dumps(
                            environment_params, sort_keys=True, default=_json_default
                        ),
                        "scenario_metadata": json.dumps(
                            scenario.metadata, sort_keys=True, default=_json_default
                        ),
                    }
                )

                if save_trajectories:
                    for t in range(horizon):
                        trajectory_rows.append(
                            {
                                "run_id": run_id,
                                "t": t,
                                "action": int(actions[t]),
                                "observed_reward": float(observed_rewards[t]),
                                "selected_mean": float(metrics.selected_means[t]),
                                "optimal_arm": int(metrics.optimal_arms[t]),
                                "optimal_mean": float(metrics.optimal_means[t]),
                                "instantaneous_pseudo_regret": (
                                    ""
                                    if scenario.kind == "adversarial"
                                    else float(metrics.instantaneous_pseudo_regret[t])
                                ),
                                "cumulative_pseudo_regret": (
                                    ""
                                    if scenario.kind == "adversarial"
                                    else float(metrics.cumulative_pseudo_regret[t])
                                ),
                                "cumulative_external_regret": (
                                    float(metrics.cumulative_external_regret[t])
                                    if scenario.kind == "adversarial"
                                    else ""
                                ),
                                "optimal_action": int(metrics.optimal_action[t]),
                            }
                        )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "runs.csv", run_rows, RUN_FIELDS)
    if save_trajectories:
        write_csv(output_dir / "trajectories.csv", trajectory_rows, TRAJECTORY_FIELDS)
    write_csv(output_dir / "summary.csv", summarize_runs(run_rows), SUMMARY_FIELDS)
    (output_dir / "config.toml").write_bytes(config_path.read_bytes())
    return output_dir
