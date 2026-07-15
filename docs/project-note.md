# FYP Project Note

> **Working title:** Comparative Study of RL and Bandit Algorithms in Non-Stochastic Environments
> **Team:** two members (replace `Member A` / `Member B` in the task board)
> **Status:** Sprint 1 — fixed-policy evaluation benchmark (2026-07-15 to 2026-07-28)

## 1. Background and motivation

Practical decision systems—such as e-commerce recommenders, financial trading platforms, and adaptive clinical trials—face changing and sometimes abrupt environmental behaviour. Standard RL and multi-armed bandit methods often assume stationary, stochastic rewards; those assumptions may not hold when distributions drift, regimes switch, trends go viral, or behaviour is adversarial/unmodelled.

This FYP will examine how RL and bandit methods behave after moving from idealized stochastic settings to complex non-stochastic environments, using reproducible Python simulations and rigorous comparative analysis.

## 2. Core objective

Evaluate the adaptability, robustness, and performance degradation of selected RL/bandit algorithms under controlled non-stochastic environmental shifts. Produce a benchmark codebase, empirical analysis, and practical recommendations for unstable live-data deployments.

## 3. Research questions

1. How quickly and reliably do classical methods adapt after different types of environmental shift?
2. Which shift type produces the largest degradation: gradual drift, abrupt change point, recurring regime switch, or adversarial/non-stochastic reward sequence?
3. Which robust or non-stationary variants reduce cumulative regret and recovery time relative to stationary baselines?
4. How do empirical findings translate into algorithm-selection guidance for real-world settings?

The longer-term system hypothesis is an environment-aware controller that identifies the current operating regime and routes decisions to a suitable policy. The first sprint deliberately evaluates fixed policies before building that controller, so its routing rules are evidence-based rather than assumed.

## 4. Proposed scope

### Environments

Start with a synthetic K-armed bandit benchmark so changes are observable and experiments are easy to reproduce. Implement a small RL/MDP extension only if the core benchmark is stable and time permits.

- Stationary stochastic control baseline.
- Gradual reward drift.
- Abrupt change points.
- Periodic or regime-switching rewards.
- Optional: deterministic/adversarial reward sequences.

### Candidate algorithms

Use a compact, defensible comparison set; confirm final selection after literature review.

| Category | Candidate algorithms |
| --- | --- |
| Stationary baselines | Epsilon-greedy, UCB1, Thompson Sampling |
| Non-stationary/robust | Sliding-Window UCB, Discounted UCB, EXP3 |
| Optional RL extension | Tabular Q-learning with constant learning rate; an adaptation-aware variant |

### Evaluation metrics

- Cumulative and per-step regret.
- Cumulative reward.
- Post-change recovery time.
- Probability/rate of selecting the optimal current action.
- Sensitivity to hyperparameters and random seeds.
- Runtime and computational cost (where meaningful).

## 5. Expected deliverables

1. Reproducible Python benchmark codebase with experiment configurations.
2. Simulation results, plots, and statistical summary across environments and seeds.
3. Analytical FYP report describing methods, failure modes, results, limitations, and deployment recommendations.
4. Optional theoretical component: derive or adapt regret-bound reasoning under a clearly stated non-stochastic assumption.
5. Final presentation/demo materials if required by the programme.

## 6. Suggested milestones

| Milestone | Completion evidence |
| --- | --- |
| M1 — Scope locked | Research questions, algorithm set, environments, and evaluation protocol agreed |
| M2 — Literature review | Annotated reading list and concise related-work synthesis |
| M3 — Benchmark MVP | Reproducible stationary + one non-stationary environment, baseline algorithms, initial plots |
| M4 — Full experiments | All selected scenarios and algorithms, seed repetitions, stored results |
| M5 — Analysis | Figures, statistical comparison, failure-mode and sensitivity analysis |
| M6 — Report and handover | Draft reviewed, code reproducible from a clean environment, final submission package |

## 7. Decisions to make early

- [ ] Confirm supervisor expectations, assessment rubric, and deadlines.
- [ ] Decide whether “non-stochastic” includes adversarial rewards or focuses on non-stationary stochastic processes.
- [ ] Lock the core algorithm set (recommended: 5–6 algorithms maximum).
- [ ] Decide the RL extension boundary; do not let it delay the bandit benchmark.
- [ ] Define the experiment protocol: horizon, arm count, change schedule, seeds, tuning budget, and significance reporting.

## 8. Reproducibility standard

Each experiment should declare its random seed, environment parameters, algorithm hyperparameters, code version/commit, and output path. Include a single command or notebook that regenerates every reported figure.

## 9. Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| Scope becomes too broad (bandits + full RL theory) | Treat bandits as core; time-box the RL/theory extension |
| Unfair algorithm tuning | Predefine tuning ranges and report all selected hyperparameters |
| Results vary heavily by seed | Use multiple seeds, confidence intervals, and robust summaries |
| “Non-stochastic” is underspecified | State an operational definition for every simulated environment |

## 10. Change log

| Date | Decision/change | Owner |
| --- | --- | --- |
| 2026-07-15 | Initial project note created from FYP description | Team |
| 2026-07-15 | Defined Sprint 1 as fixed-policy evaluation across four environment classes | Team |
| 2026-07-15 | Deferred online environment classification and policy routing to the next phase | Team |
