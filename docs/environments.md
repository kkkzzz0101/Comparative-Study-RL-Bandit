# Environment Reference

[中文版](environments.zh-CN.md)

This document is the code-backed reference for every environment currently callable by the benchmark. The corresponding Python implementations live under `src/bandit_benchmark/environments/`; configuration names are defined in `src/bandit_benchmark/registry.py`.

## 1. Data model

Every full-table environment implements:

```python
generate(horizon: int, scenario_seed: int, reward_seed: int) -> Scenario
```

A `Scenario` contains:

| Field | Shape/type | Meaning |
| --- | --- | --- |
| `expected_rewards` | `(T, K)` float array | Hidden mean reward `mu[t, arm]`, known to the evaluator but not the policy |
| `reward_table` | `(T, K)` float array | Pre-generated potential reward for every arm and round |
| `kind` | string | `stochastic`, `nonstationary`, or `adversarial` |
| `metadata` | dictionary | Seeds, changes, variation, gaps, and generator-specific information |

At round `t`, an algorithm selects one arm and receives only:

```python
scenario.reward_table[t, selected_arm]
```

It never receives the full row or `expected_rewards`. Pre-generating the table ensures that every policy faces the same potential outcomes for the same seeds.

Three random sources remain separate:

- `scenario_seed`: controls the hidden mean path, change points, leaders, or phase when applicable;
- `reward_seed`: controls Bernoulli potential outcomes;
- `policy_seed`: controls algorithm randomization and is assigned by the runner.

## 2. Quick start

Run all core environments with the random sanity policy:

```bash
bandit-benchmark run --config configs/smoke.toml
```

The smoke configuration currently runs stationary, canonical abrupt, random abrupt, gradual drift, bounded random walk, both hazard modes, and rapid switching.

A TOML environment entry has a registry `name`, an optional experiment `id`, and constructor parameters:

```toml
[[environments]]
name = "hazard_switching"
id = "hazard_per_arm"

[environments.params]
n_arms = 5
hazard_rate = 0.01
mode = "per_arm"
```

Use a unique `id` when the same generator appears more than once with different parameters. The ID is used in run IDs and summary groups, while `environment_name` records the underlying registry name.

Direct Python use:

```python
from bandit_benchmark.registry import build_environment

environment = build_environment(
    "random_abrupt",
    {
        "n_arms": 5,
        "n_changes": 3,
        "min_segment_length": 100,
        "min_jump": 0.05,
        "max_jump": 0.30,
    },
)
scenario = environment.generate(
    horizon=1000,
    scenario_seed=7,
    reward_seed=10_007,
)

print(scenario.expected_rewards.shape)       # (1000, 5)
print(scenario.metadata["change_points"])
print(scenario.metadata["total_variation"])
```

## 3. Available environments

| Registry name | Python class | Kind | Mechanism |
| --- | --- | --- | --- |
| `stationary` | `StationaryBernoulli` | stochastic | Fixed Bernoulli means with an optional controlled top-arm gap |
| `canonical_abrupt` | `CanonicalAbruptBernoulli` | nonstationary | Published three-arm, two-breakpoint regression case |
| `random_abrupt` | `RandomAbruptBernoulli` | nonstationary | Irregular breakpoints, sparse arm changes, bounded jumps |
| `gradual_drift` | `GradualDriftBernoulli` | nonstationary | Phase-shifted sinusoidal means |
| `bounded_random_walk` | `BoundedRandomWalkBernoulli` | nonstationary | Independent Gaussian mean walks with reflection |
| `hazard_switching` | `HazardSwitchingBernoulli` | nonstationary | Random global or arm-specific redraw events |
| `rapid_switching` | `RapidSwitchingBernoulli` | nonstationary | Rapid block leaders with outcome corruption |
| `csv_mean_path` | `CSVMeanPathEnvironment` | nonstationary | CTR/mean columns loaded from CSV, then Bernoulli sampled |

Compatibility names:

| Registry name | Meaning |
| --- | --- |
| `abrupt_change` | Legacy evenly spaced global-redraw generator; kept for old configs |
| `oblivious_adversarial` | Deprecated name for `rapid_switching`; now correctly evaluated as nonstationary |

`abrupt_change` accepts required `n_arms` and optional `n_changes=3`. Compatibility generators are maintained for old configs but are not part of the official experiment suite; new work should use the documented official names.

No generator in the current official suite is presented as a canonical strategic adversary. A deterministic/adaptive adversarial suite would need a separately declared payoff and comparator model.

## 4. Generator details

### 4.1 `stationary`

All arm means remain constant. Their assignment to arm indices is permuted by `scenario_seed`.

| Parameter | Default | Validation/effect |
| --- | ---: | --- |
| `n_arms` | required | At least 2 |
| `low` | `0.15` | Lower candidate/bound; with two arms and explicit `gap`, the lower realized mean is `high-gap` |
| `high` | `0.85` | Best mean; `0 <= low < high <= 1` |
| `gap` | omitted | If supplied, fixes the difference between the best and second-best means |

Use several `gap` values to create easy and hard identification cases.

### 4.2 `canonical_abrupt`

This is the Garivier–Moulines three-arm Bernoulli instance. At `T=10,000`:

```text
t < 3000 or t >= 5000: [0.5, 0.3, 0.4]
3000 <= t < 5000:      [0.5, 0.3, 0.9]
best arm:                 0 -> 2 -> 0
```

For another horizon, breakpoints scale to 30% and 50%. `n_arms` defaults to and must equal `3`, and `horizon` must be at least `3`. The mean path is deterministic; `scenario_seed` is still recorded for a uniform interface.

### 4.3 `random_abrupt`

This is the main irregular piecewise-stationary generator. It samples breakpoint locations subject to a minimum segment length. At each breakpoint, every arm changes independently with a declared probability; at least one arm is forced to change. The optimal arm is not forced to change.

| Parameter | Default | Validation/effect |
| --- | ---: | --- |
| `n_arms` | required | At least 2 |
| `n_changes` | `5` | Positive number of breakpoints |
| `min_segment_length` | `100` | Minimum rounds between boundaries |
| `arm_change_probability` | `0.5` | Per-arm change probability in `(0, 1]` |
| `min_jump` | `0.05` | Smallest absolute mean jump |
| `max_jump` | `0.30` | Largest jump; no greater than `0.5` |
| `initial_low` | `0.10` | Lower initial-mean bound |
| `initial_high` | `0.90` | Upper initial-mean bound |

The horizon must satisfy:

```text
horizon >= (n_changes + 1) * min_segment_length
```

### 4.4 `gradual_drift`

Each arm follows a phase-shifted sinusoid:

```text
mu[t, a] = 0.5 + amplitude * sin(time_phase[t] + arm_phase[a] + phase_offset)
```

| Parameter | Default | Validation/effect |
| --- | ---: | --- |
| `n_arms` | required | At least 2 |
| `cycles` | `1.5` | Positive number of cycles across the horizon |
| `amplitude` | `0.3` | In `(0, 0.45]` |

This is useful for controlled smooth drift, but a policy should not be judged only on periodic paths.

### 4.5 `bounded_random_walk`

Every arm takes an independent Gaussian step:

```text
mu[t, a] = reflect(mu[t-1, a] + Normal(0, step_std))
```

Reflection, rather than clipping, keeps every value in `[0,1]` without accumulating artificial mass exactly at the boundaries.

| Parameter | Default | Validation/effect |
| --- | ---: | --- |
| `n_arms` | required | At least 2 |
| `step_std` | `0.03` | Non-negative drift scale |
| `initial_low` | `0.20` | Lower initial-mean bound |
| `initial_high` | `0.80` | Upper initial-mean bound |

### 4.6 `hazard_switching`

Initial means are drawn uniformly. At every later round, a hazard event redraws either all arms or individual arms.

| Parameter | Default | Validation/effect |
| --- | ---: | --- |
| `n_arms` | required | At least 2 |
| `hazard_rate` | `0.001` | Event probability in `[0,1]` |
| `mode` | `"global"` | `"global"` or `"per_arm"` |
| `redraw_low` | `0.0` | Lower redraw bound |
| `redraw_high` | `1.0` | Upper redraw bound |

In global mode, one event redraws every arm. In per-arm mode, each arm has an independent event at every round.

### 4.7 `rapid_switching`

Each block has one high-mean leader and low-mean non-leaders. Consecutive leaders differ by default. After Bernoulli sampling, every outcome is independently flipped with probability `corruption`.

The effective mean recorded for evaluation is:

```text
p_effective = corruption + (1 - 2 * corruption) * p
```

| Parameter | Default | Validation/effect |
| --- | ---: | --- |
| `n_arms` | required | At least 2 |
| `block_size` | `50` | Positive rounds per leader block |
| `base_mean` | `0.15` | Non-leader Bernoulli mean |
| `leader_mean` | `0.85` | Leader mean; greater than `base_mean` |
| `corruption` | `0.10` | Independent flip rate in `[0,0.5)` |
| `force_leader_change` | `true` | Prevent consecutive blocks from sharing a leader |

This is a high-variation stochastic stress test, not proof of robustness against a strategic adversary. It uses dynamic pseudo-regret in the current runner.

## 5. Metadata and results

Every generated scenario records the following JSON-serializable metadata:

- `scenario_seed`, `reward_seed`, and `reward_model`;
- `change_family`, `change_points`, and `changed_arms`;
- `total_variation` and `max_step_change`;
- `minimum_gap`;
- `optimal_switch_points` and `n_optimal_switches`;
- generator-specific details and realized events needed to audit the scenario.

`change_points` contains declared discrete generator events such as breakpoints or redraws. It is empty for smooth drift, random walk, and CSV mean paths. `optimal_switch_points` is computed from the mean table and records changes in the argmax, so it may be non-empty even when `change_points` is empty.

The runner writes this dictionary to the `scenario_metadata` column of `runs.csv`. Random change schedules can therefore be reconstructed and audited later.

## 6. Real-data interfaces

### 6.1 Implemented: full mean-path/CTR bridge

`MeanPathDataEnvironment` is an abstract bridge for semi-synthetic sources that can supply a complete `(T,K)` expected-reward table. Its shared `generate()` validates the table and samples comparable Bernoulli potential outcomes.

`CSVMeanPathEnvironment` is the first callable adapter. The CSV must have a header and at least `horizon` rows:

```csv
time,item_a,item_b,item_c
0,0.10,0.20,0.30
1,0.12,0.18,0.32
2,0.16,0.17,0.29
```

```toml
[[environments]]
name = "csv_mean_path"

[environments.params]
path = "data/ctr_mean_path.csv"
reward_columns = ["item_a", "item_b", "item_c"]
delimiter = ","
```

| Parameter | Default | Validation/effect |
| --- | ---: | --- |
| `path` | required | CSV source path |
| `reward_columns` | required | At least two unique header names, in arm order |
| `delimiter` | `","` | Exactly one character |

The selected columns must be finite and lie in `[0,1]`. A relative `path` resolves from the process working directory, normally the repository root when using the documented command; it is not resolved relative to the TOML file. The source mean path is deterministic, so `scenario_seed` is recorded but does not alter it. This adapter is appropriate for time-binned CTR curves, not raw logged interactions.

### 6.2 Reserved: logged replay/OPE

`LoggedBanditEvent` and `LoggedBanditDataset` reserve a future event contract containing:

```text
timestamp, context, available_actions, logged_action, reward, propensity
```

Logged data does not contain counterfactual rewards for unselected arms, so it cannot truthfully produce the current full `Scenario`. A separate chronological replay/OPE runner is intentionally not implemented yet.

## 7. Adding another environment

1. Subclass `BanditEnvironment` or `MeanPathDataEnvironment`.
2. Implement `generate(...)`, or implement `load_expected_rewards(...)` for a data mean path.
3. Return a validated `Scenario` and keep metadata JSON-serializable.
4. Export the class from `environments/__init__.py`.
5. Register its callable name in `registry.py`.
6. Add reproducibility, bounds, metadata, and mechanism-specific tests.
7. Add the exact constructor parameters to both environment guides.

Environment code must never depend on policy actions when it pre-generates an oblivious/full-table scenario.
