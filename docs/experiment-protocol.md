# Experiment Protocol — Fixed-Policy Evaluation

> **Protocol status:** draft v0.1 for supervisor/team review
> **Owners:** Zuting (methodology) and Xuantong (experiments/reproducibility)

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

### E1 — Canonical and random abrupt changes

The canonical three-arm case is an exact literature regression test. The random case uses irregular breakpoints, sparse arm changes, bounded jump sizes, and does not force the best arm to change at every breakpoint.

### E2 — Smooth and random-walk drift

Sinusoidal paths provide controlled periodic drift; reflected Gaussian random walks provide non-periodic gradual change.

### E3 — Hazard switching

Means are redrawn at random events, either globally or independently per arm. This removes the artificial assumption of equally spaced changes.

### E4 — Rapid-switching/high variation

Blockwise Bernoulli leaders and independent outcome corruption create a high-variation stochastic stress test. The official registry name is `rapid_switching`; `oblivious_adversarial` remains only as a deprecated compatibility name.

The current rapid-switching case uses dynamic pseudo-regret. Standard EXP3 external regret against the best fixed arm applies only if a future canonical `kind="adversarial"` suite is added. If the report makes a tracking claim, report dynamic or bounded-switch comparator regret separately. See the [simulation-data literature review](literature-review-simulation-data.md).

### E5 — CTR-shaped semi-synthetic data

The `csv_mean_path` adapter reads a complete time-indexed CTR/mean path and generates Bernoulli potential outcomes. Logged replay remains a separate future interface because real logs lack full counterfactual reward tables.

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
- [ ] Environment metadata matches realized breakpoints, changed arms, variation, and optimal-arm switches.
- [ ] Pilot plots are plausible and free from off-by-one errors.
- [ ] Supervisor/team has approved whether a separate canonical adversarial suite is required.
