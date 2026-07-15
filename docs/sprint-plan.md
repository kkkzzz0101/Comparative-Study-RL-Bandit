# Project Work Plan

## Goal

Use a reproducible fixed-policy benchmark to identify where bandit algorithms fail, then test whether an environment-aware adaptive framework can outperform robust fixed policies on unseen changing environments.

## Division of responsibility

### Zuting — methodology and adaptive-system owner

- Own the research question, comparison protocol, regret definitions, and fairness rules.
- Verify the implementations of Discounted UCB, Sliding-Window UCB, EXP3, and change-detection/routing logic.
- Design the failure-boundary experiments and the adaptive framework.
- Lead interpretation of results, limitations, and the technical argument in the report.

### Xuantong — experiment, data, and reproducibility owner

- Build bandit foundations by verifying UCB1 and Thompson Sampling against hand-worked examples.
- Validate environment generators, seeds, change metadata, and reward distributions.
- Own batch execution, result tables, plots, confidence intervals, and reproducibility checks.
- Lead the practical CTR-shaped data case study and document its assumptions.

### Shared responsibility

- Review each other's code and reject results that neither member can explain.
- Agree on tuning ranges before final evaluation.
- Inspect surprising results together and distinguish implementation bugs from real failure modes.
- Write and rehearse the final conclusions together.

## Work sequence

### 1. Benchmark v1

Implement and validate UCB1, Bernoulli Thompson Sampling, Discounted UCB, Sliding-Window UCB, and EXP3 against stationary, abrupt, gradual, and high-variation environments.

Completion evidence:

- deterministic/unit tests for algorithm update rules;
- environment diagnostic plots and metadata checks;
- one command that reproduces a small comparison run;
- dynamic and external regret kept separate.

### 2. Failure-boundary study

Sweep meaningful difficulty variables rather than merely fitting one dataset:

- reward gap;
- change magnitude;
- change frequency or hazard rate;
- drift speed or variation budget;
- number of arms and observation noise.

Produce a map showing where each policy succeeds, degrades, or becomes statistically indistinguishable from another policy.

### 3. Adaptive framework

Use observable signals such as short/long-window reward differences, volatility, or a declared change detector to select or reset policies. Compare it against:

- the best fixed policy selected in hindsight;
- one robust fixed-policy baseline;
- a type-aware oracle router;
- a no-switch controller;
- the same controller with switching cost.

Development and final evaluation must use different generator instances and parameter ranges so the router cannot memorize visible curve shapes.

### 4. Practical validation

Construct a clearly labelled semi-synthetic recommendation case from time-binned CTR paths, or use chronological logged replay if suitable randomized logs are available. Treat this as external validation, not as the source of ground-truth environment labels.

## Definition of done

- `pytest` passes from a clean environment.
- One documented command reproduces every reported figure and table.
- Tuning and final evaluation use disjoint seeds.
- Every run records the code version, generator parameters, algorithm parameters, and seeds.
- Both members can explain the algorithms, environment assumptions, comparisons, failure modes, and limitations.
