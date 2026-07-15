# Generated results

Experiment runs create one subdirectory containing:

- `runs.csv` — one row per algorithm/environment/seed run.
- `trajectories.csv` — long-form time-step results.
- `summary.csv` — aggregate mean and 95% confidence interval.
- `config.toml` — exact configuration used for the run.

`runs.csv` distinguishes configured algorithm/environment `id` values from
their registry `algorithm_name`/`environment_name`. Its `scenario_metadata`
JSON column records realized change points, changed arms, variation, gaps, and
all generator seeds.

Generated result directories are ignored by Git. Commit only deliberately selected small summary tables or final figures under `reports/`.
