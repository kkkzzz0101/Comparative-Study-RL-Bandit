# Experiment Protocol — Fixed-Policy Evaluation

> **Protocol status:** draft v0.1 for supervisor/team review
> **Sprint window:** 2026-07-15 to 2026-07-28

## 1. Objective

Test rather than assume the following mapping:

| Hypothesis | Environment | Expected strong policy |
| --- | --- | --- |
| H1 | Stationary stochastic | UCB1 / Thompson Sampling |
| H2 | Gradual drift | Discounted UCB |
| H3 | Abrupt changes | Sliding-Window UCB |
| H4 | Rapid-switching/high-variation stress test | EXP3 (hypothesis only) |

The result of this sprint will define the candidate policies and environment signals used by the later adaptive router.

## 2. Core algorithms

- Uniform Random (sanity baseline).
- UCB1.
- Bernoulli Thompson Sampling with Beta priors.
- Discounted UCB.
- Sliding-Window UCB.
- EXP3.

Epsilon-Greedy may be added only if it does not delay the core six policies.

## 3. Environments

All core stochastic experiments use rewards in `[0, 1]` and default to Bernoulli potential outcomes so they can be interpreted as click/no-click recommendation feedback.

### E0 — Stationary

Each arm keeps one fixed mean throughout the horizon. This is the control condition.

### E1 — Gradual drift

Arm means evolve smoothly and the identity of the optimal arm changes over time. `cycles` and `amplitude` control change speed and severity.

### E2 — Abrupt changes

The horizon contains piecewise-stationary segments. Arm means are reassigned at hidden change points, with a different optimal arm after every change.

### E3 — Rapid-switching/high variation

The complete reward sequence is generated before the policy runs and does not react to the selected actions. This preserves fair common-table comparisons. The current internal generator name is `oblivious_adversarial`, but its mechanism—blockwise Bernoulli leaders plus controlled corruption—is scientifically a high-variation stochastic stress test, not evidence against a strategic adversary.

Standard EXP3 external regret is measured against the best fixed arm. If the report makes a canonical adversarial claim, add explicit precommitted deterministic payoff tables; if it makes a tracking claim, report dynamic or bounded-switch comparator regret separately. See the [simulation-data literature review](literature-review-simulation-data.md).

## 4. Default experiment scale

| Stage | Horizon | Arms | Seeds | Purpose |
| --- | ---: | ---: | ---: | --- |
| Smoke | 200 | 5 | 2 | Interface and output validation |
| Pilot/tuning | 10,000 | 10 | 10 | Debugging and parameter selection |
| Final | 10,000 | 10 | 50 | Locked evaluation |

Final scale may increase only after runtime is measured. Final seeds start at 1000 so they do not overlap the pilot seeds.

## 5. Randomness and fairness

For base seed `s`:

- scenario seed = `s`;
- reward seed = `s + 10,000`;
- policy seed = `s + 20,000`.

The environment generates a full `T × K` potential-outcome table once. Each policy sees only its selected reward, while the runner retains the full table for evaluation. Every algorithm uses the same scenario and reward seeds.

## 6. Metrics

For stochastic and non-stationary environments, report dynamic pseudo-regret:

\[
R_T = \sum_{t=1}^{T}\left(\max_a \mu_{t,a} - \mu_{t,A_t}\right).
\]

Also report:

- cumulative observed reward;
- optimal-action rate;
- mean and 95% confidence interval over seeds;
- runtime;
- recovery time after abrupt changes (to be added during analysis work);
- tracking delay for gradual drift (to be defined before final runs).

For adversarial environments, report external regret relative to the best fixed arm in hindsight. Do not compare its numeric value directly with dynamic pseudo-regret.

## 7. Parameter-selection rule

Initial grids:

- Discounted UCB discount: `{0.95, 0.97, 0.99, 0.995}`.
- Sliding-Window UCB window: `{100, 250, 500, 1000}`.
- EXP3: theoretical setting plus a small predeclared gamma grid.

Select parameters using pilot seeds only. Lock the chosen values in `configs/final.toml` before running final seeds. Do not change them after seeing final results.

## 8. Required checks before final runs

- [ ] Every policy selects only valid actions.
- [ ] UCB1 samples every arm during initialization.
- [ ] Thompson posterior updates match a hand-computed example.
- [ ] Sliding-Window UCB drops expired observations.
- [ ] EXP3 probabilities are finite, positive, and sum to one.
- [ ] All environments reproduce exactly from fixed seeds.
- [ ] Drift and abrupt scenarios actually change the optimal arm.
- [ ] Pilot plots are plausible and free from off-by-one errors.
- [ ] Supervisor/team has approved the adversarial environment interpretation.
