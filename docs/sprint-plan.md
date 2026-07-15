# Sprint 1 — Algorithm Evaluation Benchmark

> **Dates:** 2026-07-15 to 2026-07-28
> **Goal:** complete a reproducible comparison of six policies across four environment classes.

Replace `Member A` and `Member B` with names during the kickoff meeting. Member A is the teammate already familiar with bandits; Member B is the teammate learning the topic.

## Division of work

### Member A — methodology and non-stationary policies

- T10: own the experiment protocol and fairness review.
- T13: implement Discounted UCB.
- T14: implement Sliding-Window UCB.
- T15: implement EXP3.
- T18: lead parameter tuning and statistical interpretation.

### Member B — foundations and baseline policies

- T11: implement and test UCB1.
- T12: implement and test Bernoulli Thompson Sampling.
- T16: validate stationary and abrupt environments with diagnostic plots.
- T19: prepare result tables, figures, and reproducibility notes.

### Shared

- Review each other's pull requests.
- Run integration and smoke tests.
- Write the literature rationale for every algorithm/environment pairing.
- Review conclusions together; neither member should interpret only their own implementation.

## Acceptance criteria by task

| ID | Task | Acceptance criteria |
| --- | --- | --- |
| T10 | Lock protocol | Supervisor/team agrees on algorithms, environment definitions, metrics, and tuning split |
| T11 | UCB1 | Pulls every arm initially; correct confidence bonus; deterministic test passes |
| T12 | Thompson Sampling | Beta posterior initialized and updated correctly; seeded sampling reproducible |
| T13 | Discounted UCB | Counts and rewards decay every step; discount validation and toy test pass |
| T14 | Sliding-Window UCB | Only the most recent `window_size` observations contribute; expiry test passes |
| T15 | EXP3 | Importance-weighted update is correct; probability vector is valid and numerically stable |
| T16 | Environment QA | Seed reproduction, reward bounds, change locations, and optimal-arm transitions verified |
| T17 | Runner integration | All enabled policies complete smoke and pilot configs with expected row counts |
| T18 | Tuning/final runs | Tuning grid recorded; final parameters locked before evaluation seeds run |
| T19 | Analysis outputs | Reward/regret curves, summary table, CI, recovery/tracking analysis generated from saved CSV |
| T20 | Sprint report | 3–5 page note states findings, failure modes, limitations, and router implications |

## Fourteen-day schedule

| Day | Date | Expected outcome |
| ---: | --- | --- |
| 1 | Jul 15 | Scaffold, interfaces, protocol draft, and assignments ready |
| 2 | Jul 16 | Formula review; UCB1/TS and non-stationary policy branches opened |
| 3 | Jul 17 | UCB1 and D-UCB first implementations |
| 4 | Jul 18 | Thompson Sampling, SW-UCB, and EXP3 first implementations |
| 5 | Jul 19 | Unit tests and environment QA |
| 6 | Jul 20 | All policies integrated into smoke config |
| 7 | Jul 21 | Ten-seed pilot and supervisor/team checkpoint |
| 8 | Jul 22 | Fixes from pilot; parameter grid finalized |
| 9 | Jul 23 | Tuning runs |
| 10 | Jul 24 | Parameters locked; final run begins |
| 11 | Jul 25 | Final run completed and checked |
| 12 | Jul 26 | Figures, confidence intervals, and failure cases |
| 13 | Jul 27 | Evaluation note and reproducibility audit |
| 14 | Jul 28 | Mutual review, merge, sprint conclusion, router requirements |

## Definition of done

- `pytest` passes from a clean environment.
- One documented command reproduces the core results.
- Final results use unseen seeds and locked parameters.
- Raw run metadata includes code commit, parameters, and all three seeds.
- The team can explain why every policy behaves as observed.
- The next adaptive-framework sprint has evidence-based routing hypotheses.
