# Comparative Study of RL and Bandit Algorithms in Non-Stochastic Environments

Final Year Project (FYP) workspace for a two-person team.

## Project question

How do reinforcement-learning (RL) and multi-armed bandit algorithms adapt when reward distributions or environment dynamics change in non-stochastic, non-stationary, or otherwise unmodelled ways?

## Repository guide

- [`docs/project-note.md`](docs/project-note.md) — project scope, research plan, deliverables, and decisions.
- [`docs/task-board.md`](docs/task-board.md) — shared task assignment and status.
- [`docs/meeting-log.md`](docs/meeting-log.md) — meeting notes and decision record.
- `src/` — simulation and evaluation code (to be added).
- `experiments/` — runnable experiment configurations and outputs (to be added).
- `reports/` — report source, figures, and final deliverables (to be added).

## Collaboration workflow

1. Create an issue for each task or add it to the task board.
2. Work in a short-lived branch: `feature/<topic>` or `fix/<topic>`.
3. Open a pull request (PR); the other member reviews before merging to `main`.
4. Update the task board and meeting log whenever task ownership, scope, or decisions change.
5. Keep raw or large generated data out of Git; commit code, configs, small summary tables, and reproducible instructions.

## Getting started after a remote GitHub repository exists

```bash
git remote add origin https://github.com/<OWNER>/comparative-study-rl-bandit.git
git push -u origin main
```

Then invite the other member as a repository collaborator in GitHub: **Settings → Collaborators → Add people**.
