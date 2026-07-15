# Comparative Study of RL and Bandit Algorithms in Non-Stochastic Environments

Reproducible Final Year Project benchmark for evaluating bandit policies under stationary, drifting, abrupt-change, and adversarial reward sequences.

## Current research question

Which bandit policy is most reliable under each type of environmental change, and can those results support a later environment-aware algorithm-selection framework?

The current 14-day sprint evaluates fixed policies. Environment detection and dynamic policy routing are the next phase, not part of this first benchmark milestone.

## Repository map

```text
configs/                    Experiment definitions
docs/                       Project note, protocol, sprint plan, meeting log
results/                    Generated CSV outputs (runs are Git-ignored)
src/bandit_benchmark/
  algorithms/               Shared interface and policy implementations
  environments/             Reproducible potential-outcome generators
  metrics.py                Pseudo-regret and external-regret calculations
  runner.py                 Config-driven evaluation runner
tests/                      Deterministic unit and smoke tests
```

## Setup

Python 3.11 or newer is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pytest
```

## Run the smoke benchmark

```bash
bandit-benchmark run --config configs/smoke.toml
```

The command writes `runs.csv`, `trajectories.csv`, `summary.csv`, and a copy of the exact configuration to `results/smoke/`.

The smoke configuration currently enables only the implemented random policy. Enable policies in `configs/pilot.toml` after their implementation and tests pass.

## Implementation ownership

| Module | Initial owner | Status |
| --- | --- | --- |
| Random baseline and experiment framework | Shared scaffold | Ready |
| UCB1 and Thompson Sampling | Member B | TODO |
| Discounted UCB, Sliding-Window UCB, EXP3 | Member A | TODO |
| Environment validation and plotting | Both | TODO |

See [`docs/sprint-plan.md`](docs/sprint-plan.md) for acceptance criteria, [`docs/experiment-protocol.md`](docs/experiment-protocol.md) for the evaluation contract, and [`docs/literature-review-simulation-data.md`](docs/literature-review-simulation-data.md) for the primary-source review of simulation-data protocols.

## Fair-comparison rules

- Algorithms receive only the reward of the arm they selected.
- Every algorithm faces the same pre-generated potential reward table for a given environment seed.
- Scenario, reward, and policy randomness use separate seeds.
- Tuning seeds and final evaluation seeds must remain disjoint.
- Dynamic pseudo-regret and adversarial external regret are reported separately.

## Collaboration workflow

1. Create or assign a task in [`docs/task-board.md`](docs/task-board.md).
2. Work on a short branch such as `feature/ucb1`.
3. Add unit tests with the implementation.
4. Open a pull request; the other member reviews it before merging to `main`.
5. Run `pytest` and the smoke config before every merge.

Remote repository: [kkkzzz0101/Comparative-Study-RL-Bandit](https://github.com/kkkzzz0101/Comparative-Study-RL-Bandit)
