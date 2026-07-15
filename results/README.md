# Generated results

Experiment runs create one subdirectory containing:

- `runs.csv` — one row per algorithm/environment/seed run.
- `trajectories.csv` — long-form time-step results.
- `summary.csv` — aggregate mean and 95% confidence interval.
- `config.toml` — exact configuration used for the run.

Generated result directories are ignored by Git. Commit only deliberately selected small summary tables or final figures under `reports/`.
