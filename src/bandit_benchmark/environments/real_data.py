from __future__ import annotations

import csv
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from bandit_benchmark.environments.base import (
    BanditEnvironment,
    Scenario,
    make_bernoulli_scenario,
    validate_horizon,
)


class MeanPathDataEnvironment(BanditEnvironment):
    """Bridge for semi-synthetic data that supplies a full ``T x K`` mean path."""

    @abstractmethod
    def load_expected_rewards(
        self, horizon: int, scenario_seed: int
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Load or estimate the full counterfactual mean table and source metadata."""

    def generate(self, horizon: int, scenario_seed: int, reward_seed: int) -> Scenario:
        validate_horizon(horizon)
        means, source_metadata = self.load_expected_rewards(horizon, scenario_seed)
        means = np.asarray(means, dtype=float)
        if means.shape != (horizon, self.n_arms):
            raise ValueError(
                f"loaded mean path must have shape {(horizon, self.n_arms)}, got {means.shape}"
            )
        return make_bernoulli_scenario(
            name=self.name,
            kind="nonstationary",
            expected_rewards=means,
            scenario_seed=scenario_seed,
            reward_seed=reward_seed,
            metadata={"change_family": "data_mean_path", **source_metadata},
        )


class CSVMeanPathEnvironment(MeanPathDataEnvironment):
    """Load per-arm CTR/mean columns from a CSV and sample Bernoulli outcomes."""

    name = "csv_mean_path"

    def __init__(
        self,
        path: str,
        reward_columns: Sequence[str],
        delimiter: str = ",",
    ) -> None:
        if isinstance(reward_columns, str):
            raise ValueError("reward_columns must be a sequence of column names, not a string")
        columns = tuple(str(column) for column in reward_columns)
        super().__init__(n_arms=len(columns))
        if len(set(columns)) != len(columns):
            raise ValueError("reward_columns must be unique")
        if len(delimiter) != 1:
            raise ValueError("delimiter must be a single character")
        self.path = Path(path)
        self.reward_columns = columns
        self.delimiter = delimiter

    def load_expected_rewards(
        self, horizon: int, scenario_seed: int
    ) -> tuple[np.ndarray, dict[str, Any]]:
        del scenario_seed
        rows: list[list[float]] = []
        with self.path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter=self.delimiter)
            if reader.fieldnames is None:
                raise ValueError("CSV must contain a header row")
            missing = [column for column in self.reward_columns if column not in reader.fieldnames]
            if missing:
                raise ValueError(f"CSV is missing reward columns: {', '.join(missing)}")
            for row in reader:
                rows.append([float(row[column]) for column in self.reward_columns])
                if len(rows) == horizon:
                    break
        if len(rows) < horizon:
            raise ValueError(f"CSV contains {len(rows)} rows but horizon={horizon}")
        return np.asarray(rows, dtype=float), {
            "source_type": "csv_mean_path",
            "source_path": str(self.path),
            "reward_columns": list(self.reward_columns),
            "delimiter": self.delimiter,
        }


@dataclass(frozen=True)
class LoggedBanditEvent:
    """Future replay/OPE record; not consumed by the current full-table runner."""

    timestamp: str
    available_actions: tuple[str, ...]
    logged_action: str
    reward: float
    propensity: float
    context: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.logged_action not in self.available_actions:
            raise ValueError("logged_action must be present in available_actions")
        if not 0 <= self.reward <= 1:
            raise ValueError("reward must lie in [0, 1]")
        if not 0 < self.propensity <= 1:
            raise ValueError("propensity must lie in (0, 1]")


class LoggedBanditDataset(ABC):
    """Interface reserved for a future chronological replay/OPE runner."""

    @abstractmethod
    def __iter__(self) -> Iterator[LoggedBanditEvent]:
        """Yield events in chronological order."""
