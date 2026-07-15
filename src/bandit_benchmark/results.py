from __future__ import annotations

import csv
import math
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_runs(run_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in run_rows:
        grouped[(row["environment"], row["environment_kind"], row["algorithm"])].append(row)

    summaries: list[dict[str, Any]] = []
    for (environment, kind, algorithm), rows in sorted(grouped.items()):
        primary_metric = "final_external_regret" if kind == "adversarial" else "final_pseudo_regret"
        values = np.asarray([float(row[primary_metric]) for row in rows], dtype=float)
        rewards = np.asarray([float(row["cumulative_reward"]) for row in rows], dtype=float)
        n_runs = len(rows)
        standard_error = float(np.std(values, ddof=1) / math.sqrt(n_runs)) if n_runs > 1 else 0.0
        reward_standard_error = (
            float(np.std(rewards, ddof=1) / math.sqrt(n_runs)) if n_runs > 1 else 0.0
        )
        summaries.append(
            {
                "environment": environment,
                "environment_kind": kind,
                "algorithm": algorithm,
                "n_runs": n_runs,
                "primary_metric": primary_metric,
                "mean_regret": float(np.mean(values)),
                "regret_ci95": 1.96 * standard_error,
                "mean_cumulative_reward": float(np.mean(rewards)),
                "reward_ci95": 1.96 * reward_standard_error,
                "mean_optimal_action_rate": float(
                    np.mean([float(row["optimal_action_rate"]) for row in rows])
                ),
            }
        )
    return summaries
