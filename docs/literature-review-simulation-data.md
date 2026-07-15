# Literature Review — How Bandit Papers Generate Non-Stationary Simulation Data

> **Review date:** 2026-07-15  
> **Scope:** primary papers on synthetic, semi-synthetic, and logged-data evaluation for non-stationary and adversarial multi-armed bandits.

## Executive finding

There is no single standard “non-stochastic dataset.” Prior work starts from a structural assumption and constructs data to match it:

1. **Piecewise-stationary:** Bernoulli means are fixed within segments and jump at breakpoints.
2. **Hazard switching:** means are redrawn at random change times, globally or independently per arm.
3. **Smooth drift:** means follow cosine/sinusoidal paths or bounded random walks.
4. **Variation-budget:** any mean path is allowed if its total variation is below a controlled budget.
5. **Adversarial:** the payoff vector at each round is arbitrary; the canonical model does not prescribe a stochastic generator.
6. **Semi-synthetic:** time-binned CTR from real logs defines the mean path, then fresh Bernoulli rewards are sampled.
7. **Logged replay/OPE:** real randomized interaction logs are replayed or reweighted without inventing counterfactual rewards.

The literature also changes one assumption at a time. Papers usually control the number of arms, horizon, reward gap, number/rate of changes, drift speed, or variation budget, then repeat Monte Carlo trials.

The main implication for this FYP is important: the initial routing table “stationary → UCB/TS, gradual → D-UCB, abrupt → SW-UCB, unpredictable → EXP3” should be treated as a hypothesis. Classic experiments show D-UCB and SW-UCB working in both abrupt and smooth environments, while EXP3's adversarial guarantee uses a different comparator from non-stationary dynamic regret.

## 1. Representative protocols

| Work | Environment construction | Experimental scale / metric | Relevance to us |
| --- | --- | --- | --- |
| Garivier & Moulines (2011) | Bernoulli piecewise means with two fixed breakpoints; separately, a two-arm smooth cosine mean path | `K=3`, `T=10,000` for abrupt case; dynamic regret | Canonical small, interpretable abrupt and smooth sanity cases |
| Mellor & Shapiro (2013) | At a hazard event, Bernoulli mean is redrawn from `Uniform(0,1)`; changes are global or per-arm; also a truncated normal random walk | `T=10^6`, 100 repetitions, hazard `10^-4`; random-walk variance `0.03` | Adds random change timing, local/global changes, and non-periodic drift |
| Besbes, Gur & Zeevi (2014/2019) | Mean paths belong to a total-variation class; numerical work uses controlled two-arm sinusoidal paths | Dynamic oracle regret under known variation budget | Gives a principled severity axis independent of “drift shape” |
| Liu, Lee & Shroff (2018) | A two-arm flipping problem and a hazard-switching Bernoulli problem | 1,000 synthetic trials; dynamic regret, change-detection analysis | Direct precedent for testing change magnitude and detectability |
| Cao et al. (2019) | Random piecewise Bernoulli means; separately, Yahoo CTR is binned and converted to a piecewise Bernoulli simulator | Large Monte Carlo studies; dynamic regret | Strong precedent for a real-data-shaped simulator |
| Besson et al. (2022) | Random breakpoints with minimum spacing; each arm changes with a declared probability and bounded magnitude | `T=20,000`, 2,000 random instances; dynamic regret and restart count | Most directly reusable random piecewise generator |
| Auer et al. (2002) | Arbitrary payoff sequence in `[0,1]^K`; no required probability model | External regret against the best fixed arm | Defines what “adversarial” means; not a recipe saying “switch arms rapidly” |
| Li et al. (2011) | Replay uniformly randomized Yahoo events in timestamp order; accept only matching target-policy actions | Real clicks/CTR; no full counterfactual table | Avoids simulator bias but cannot provide exact dynamic regret/type labels |
| Saito et al. (2021) | Real e-commerce logs from Random and Bernoulli-TS policies, including propensities | Off-policy value estimation | Practical validation layer, not a replacement for controlled simulation |

## 2. Exact synthetic generators used in prior work

### 2.1 Scripted piecewise Bernoulli means

[Garivier and Moulines](https://arxiv.org/pdf/0805.3415) use an especially reproducible three-arm example:

\[
p_t(1)=0.5,\qquad p_t(2)=0.3,
\]

and

\[
p_t(3)=
\begin{cases}
0.4,&t<3000\text{ or }t\ge 5000,\\
0.9,&3000\le t<5000.
\end{cases}
\]

Rewards are independent Bernoulli draws. The best arm is arm 1, then arm 3, then arm 1 again. Because only one arm changes, this case separates adaptation from the easier signal where every arm jumps simultaneously.

This is worth reproducing exactly as a regression benchmark. Our current abrupt generator redraws every arm and enforces a new best arm, which is useful but typically easier to detect.

### 2.2 Flipping one arm and varying change magnitude

[Liu, Lee and Shroff](https://arxiv.org/pdf/1711.03539) use two Bernoulli arms:

\[
\mu_t(1)=0.5,
\]

while arm 2 is `0.8` outside the middle third of the horizon and `0.5 - Δ` within it. Breakpoints are `T/3` and `2T/3`; `Δ` ranges from `0.02` to `0.30`.

This design is valuable because it measures difficulty through change magnitude. Small changes are harder to detect, even if the number and locations of changes remain fixed.

### 2.3 Random hazard switching

[Mellor and Shapiro](https://proceedings.mlr.press/v31/mellor13a.pdf), later reused by Liu et al., generate Bernoulli means as

\[
\mu_t(i)=
\begin{cases}
\mu_{t-1}(i),&\text{with probability }1-\beta(t),\\
U(0,1),&\text{with probability }\beta(t).
\end{cases}
\]

They distinguish:

- **Global switching:** one hazard event redraws all arms.
- **Per-arm switching:** each arm has an independent change process.

Their large synthetic studies use a constant switch rate of `10^-4`, horizon `10^6`, and 100 repetitions. Global switching supplies a strong shared signal; per-arm switching is more realistic and harder because an unplayed arm may change unnoticed.

Our current generator has equally spaced global changes. Adding hazard-driven and per-arm variants would prevent the evaluation from depending on known regular spacing.

### 2.4 Smooth periodic drift

Garivier and Moulines also use two Bernoulli arms: one remains at `0.5`, while the other follows a cosine path centered at `0.5` with amplitude `0.4`. The identity of the best arm changes smoothly and periodically.

The PASCAL Exploration–Exploitation challenge environments summarized by Mellor and Shapiro similarly used periodic Gaussian, sinusoidal, and constant expected-payoff signals.

Our sinusoidal generator is therefore literature-aligned. However, a router evaluated only on sinusoidal drift might learn periodicity rather than general gradual change.

### 2.5 Truncated-normal random walk

Mellor and Shapiro additionally let every Bernoulli success probability drift using a normal random walk truncated to `[0,1]`:

\[
\mu_t(i)\sim \operatorname{TruncNormal}(\mu_{t-1}(i),\sigma^2;0,1).
\]

They use `σ² = 0.03`, horizon `10^6`, and 100 repetitions. Unlike a sine wave, this path is stochastic and non-periodic.

For our benchmark, this should be a second gradual-drift family. Boundary handling must be declared explicitly: truncated sampling, clipping, and reflection generate different stationary behaviour near 0 and 1.

### 2.6 Variation-budget paths

[Besbes, Gur and Zeevi](https://papers.nips.cc/paper_files/paper/2014/file/91ba7292e5388b90b58d0b839a7f19ec-Paper.pdf) define a class of paths satisfying

\[
\sum_{t=1}^{T-1}\max_a\left|\mu_{t+1,a}-\mu_{t,a}\right|\le V_T.
\]

This is a model class rather than a single random-data recipe. The same `V_T` can be spent through one large jump, many small jumps, or a continuous ramp. Their dynamic-regret analysis motivates restarted EXP3 (Rexp3) and shows why the amount of change—not only its label—must be controlled.

The expanded [Stochastic Systems paper](https://pubsonline.informs.org/doi/10.1287/stsy.2019.0033) gives an exact two-arm Bernoulli trajectory:

\[
\mu_{t,1}=\frac{1}{2}+\frac{3}{10}\sin\left(\frac{5V_T\pi t}{3T}\right),
\qquad
\mu_{t,2}=\frac{1}{2}+\frac{3}{10}\sin\left(\frac{5V_T\pi t}{3T}+\pi\right).
\]

They fix `V_T=3`, compare it with another path that spends the same variation only in the first and last thirds, vary `T` up to `3 × 10^8`, and use 100 replications. This is a particularly clean demonstration that **where** variation occurs can matter even when the total budget is identical.

A useful experiment for us is to hold `V_T` constant and compare:

- one abrupt jump;
- several medium jumps;
- continuous linear or random-walk drift.

If rankings change even under equal variation, the shape and concentration of change matter beyond total severity.

### 2.7 Random piecewise means for scaling studies

[Cao et al.](https://proceedings.mlr.press/v89/cao19a/cao19a.pdf) use evenly spaced segments and Bernoulli rewards. In one experiment, `K=10`, segment length is 20,000, and a random mean vector `μ` is alternated with `1-μ` across segments; the difference between its largest and smallest entries must exceed `0.6`. They average over 100 problem instances and 50 stochastic repetitions per instance.

This illustrates two distinct sources of uncertainty that our design should also separate:

- **scenario instances:** different mean paths/change schedules;
- **reward repetitions:** different Bernoulli outcomes for the same mean path.

Our current seed design separates scenario and reward randomness, which already supports this distinction.

### 2.8 Random breakpoints, sparse arm changes, and bounded jump sizes

[Besson et al.](https://www.jmlr.org/papers/v23/20-1384.html) provide the most directly reusable random piecewise-stationary recipe:

1. Draw initial arm means.
2. Sample breakpoint locations uniformly, subject to minimum segment length `d_min`.
3. At each breakpoint, let each arm change independently with probability `p`.
4. Sample its change magnitude uniformly from `[Δ_min, Δ_max]` and keep the result within `[0,1]`.
5. Draw bounded/Bernoulli rewards from the resulting segment means.

Their robustness study uses `T=20,000`, `K=5`, at most six breakpoints, `d_min=1,000`, `p=0.5`, `Δ_min=0.05`, `Δ_max=0.30`, and averages over 2,000 random problem instances. They separately report dynamic regret and the number of algorithm restarts.

This design is stronger than redrawing every mean at fixed intervals because it independently controls:

- how often the world changes;
- how many arms are affected at a breakpoint;
- how detectable each change is;
- whether the optimal arm actually changes.

The authors also publish their [reference implementation](https://github.com/EmilieKaufmann/PiecewiseStationaryBandits). Its simulation workflow pre-generates reward realizations and reuses comparable instances across policies, which supports our common-reward-table design.

## 3. What “adversarial simulation” means

[Auer et al.](https://cseweb.ucsd.edu/~yfreund/papers/bandits.pdf) define a payoff vector

\[
g_t=(g_{t,1},\ldots,g_{t,K})\in[0,1]^K
\]

for every round, with no stochastic assumption. EXP3 is evaluated theoretically against the best single fixed arm in hindsight:

\[
\max_a\sum_{t=1}^{T}g_{t,a}-\sum_{t=1}^{T}g_{t,A_t}.
\]

Consequently, classic adversarial-bandit theory does not prescribe one canonical simulator such as “randomly switch the winning arm every 50 rounds.” An oblivious adversary fixes the entire payoff table before interaction; an adaptive adversary can depend on past interaction, subject to the exact model definition.

This produces three cautions:

1. A rapidly switching Bernoulli process is still a stochastic non-stationary model when analyzed through its generating distribution.
2. Dynamic regret against the best arm at every time and external regret against one best fixed arm answer different questions.
3. If the best arm rotates evenly, the best fixed arm may be weak, making external regret look small even when the policy tracks the current best arm poorly.

Our current `ObliviousAdversarial` generator pre-generates its table correctly, but its mechanism—blockwise Bernoulli leaders plus random bit flips—is more accurately described as a **rapid-switching corrupted stress test** or a high-variation piecewise-stationary process. It is close to the blockwise “one slightly better arm” construction used in the lower-bound argument of Besbes et al.; it is not evidence against a strategic adversary. The generator should be renamed or accompanied by explicit deterministic payoff-table cases. We should not assume EXP3 will win merely because the table is pre-generated.

If we later evaluate a switching comparator, [Auer et al.](https://cseweb.ucsd.edu/~yfreund/papers/bandits.pdf) provide EXP3.S, whose comparator is an action sequence with a bounded number of switches. That switching regret is again distinct from both standard EXP3 external regret and unrestricted dynamic regret.

## 4. Semi-synthetic and real-data protocols

### 4.1 Yahoo CTR converted into Bernoulli mean paths

Cao et al. provide the clearest precedent for a practical simulator:

1. Select six articles with positive CTR over five days.
2. Estimate each article's CTR in 43,200-second (12-hour) bins.
3. If an adjacent-bin change is below `0.01`, carry forward the preceding CTR.
4. Treat the resulting piecewise CTR curve as `μ_{t,a}`.
5. Generate fresh Bernoulli rewards and average regret over 100 Monte Carlo trials.

This produces `K=6`, `T=432,000`, and nine stationary segments in their experiment. It supplies full counterfactual rewards for simulation, while real traffic shapes the means.

The tradeoff is that bin size, smoothing threshold, article selection, and the mapping from clock time to rounds manufacture part of the non-stationarity. Empirical CTR is also treated as ground truth despite estimation error.

### 4.2 Logged replay instead of reward simulation

[Li et al.](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/Published-3.pdf) evaluate policies using uniformly randomized Yahoo traffic:

1. Read logged `(context, displayed action, click)` events in time order.
2. Ask the target policy which action it would select.
3. If it matches the logged action, reveal the click and update the policy.
4. Otherwise skip the event without updating.

This avoids fitting a reward simulator and can give unbiased policy-value estimates under the logging assumptions. With `K` uniform actions, only about `1/K` events are retained. It also provides no full reward table, exact dynamic oracle, or ground-truth environment-type label.

Replay is therefore suitable for a practical CTR case study, but not for training or validating our proposed environment classifier.

### 4.3 Open Bandit Dataset

[Saito et al.](https://arxiv.org/pdf/2008.07146) release real ZOZOTOWN fashion-recommendation logs: roughly 26 million impressions collected over seven days from Random and Bernoulli-TS logging policies. Records include timestamp, selected item/position, binary click, context features, and action propensity.

The paper's benchmark is off-policy evaluation: estimate one deployed policy's value from the other policy's logs and compare it with observed on-policy CTR. It is not originally a non-stationary-regret simulator.

For our Phase 2, the dataset could support two separate analyses:

- chronological replay/OPE as realistic policy-value evidence;
- a clearly labeled semi-synthetic benchmark where smoothed time-bin CTRs define Bernoulli means.

The second approach should use only sufficiently exposed items, report uncertainty in estimated CTR, and avoid rescaling tiny differences without labeling the result as an amplified stress test.

## 5. Comparison with our current implementation

| Current generator | Literature alignment | Main limitation | Recommended action |
| --- | --- | --- | --- |
| `StationaryBernoulli` | Standard stochastic baseline | A single gap does not cover identification difficulty | Implemented configurable `gap`, `low`, and `high`; sweep multiple gaps |
| `CanonicalAbruptBernoulli` / `RandomAbruptBernoulli` | Matches scripted and randomized piecewise Bernoulli precedents | One generator cannot cover all breakpoint sparsity and jump sizes | Implemented separate canonical regression and irregular sparse-change cases |
| `GradualDriftBernoulli` / `BoundedRandomWalkBernoulli` | Matches periodic and non-periodic drift precedents | Boundary rule and drift scale affect conclusions | Implemented sinusoidal and reflected Gaussian paths with explicit metadata |
| `HazardSwitchingBernoulli` | Matches global/per-arm redraw protocols | Realized change count is random | Implemented both modes and records every realized event |
| `RapidSwitchingBernoulli` | Matches a high-variation block-leader stress construction | It is not a strategic adversary and should not use the wrong comparator | Implemented as nonstationary with corrected corrupted expectations; old name is deprecated |

The common potential-outcome table and three-seed separation in the current runner are strong design choices and consistent with fair Monte Carlo comparison. The runner now also stores scenario metadata and supports distinct IDs for parameterized instances of the same generator.

## 6. Recommended benchmark suite for this FYP

### Core suite for the current two-week sprint

1. **S0 Stationary-gap:** Bernoulli means with at least two gap levels.
2. **S1 Canonical two-swap:** exact Garivier–Moulines three-arm instance.
3. **S2 Random abrupt:** Besson-style irregular breakpoints, sparse arm changes, and bounded jump sizes.
4. **D1 Smooth drift:** current sinusoidal generator at several speeds/amplitudes.
5. **X1 Rapid-switching corruption:** retain current stress test under a more precise name.

If the five required cases are stable, add these robustness cases before the final run:

1. **D2 Random-walk drift:** bounded/truncated Gaussian walk.
2. **H1 Hazard switching:** global and per-arm redraw from `Uniform(0,1)`.

For every generator, record:

- number and location of changes;
- which arms changed;
- minimum reward gap;
- total variation `V_T`;
- number of optimal-arm switches;
- scenario, reward, and policy seed.

### Second-stage additions

1. Equal-`V_T` step-versus-ramp experiments.
2. Deterministic oblivious payoff tables evaluated with external regret.
3. Open Bandit Dataset chronological/OPE case study.
4. OBD-derived time-binned Bernoulli simulator with uncertainty and sensitivity analysis.

## 7. Implications for the future environment-aware router

The router must not be trained and tested on the same visible generator shapes. Otherwise, it may learn “sine wave means gradual” or “equally spaced jumps mean abrupt” without learning a transferable environment property.

A defensible protocol is:

- develop on scripted flips and sinusoidal drift;
- tune thresholds on separate hazard rates/change magnitudes;
- test on unseen random-walk paths, irregular breakpoints, and different gaps;
- report classification accuracy, detection delay, false alarms, switching cost, and end-to-end regret;
- compare against the best fixed policy, a type-aware oracle, and no-switch baselines.

The most meaningful research question emerging from the literature is therefore not simply “which algorithm belongs to which environment?” It is:

> Under what change rate, magnitude, reward gap, and detection uncertainty does selecting or switching policies outperform the best robust fixed policy?

## References

1. A. Garivier and E. Moulines. [On Upper-Confidence Bound Policies for Non-Stationary Bandit Problems](https://arxiv.org/abs/0805.3415). ALT, 2011. DOI: [10.1007/978-3-642-24412-4_16](https://doi.org/10.1007/978-3-642-24412-4_16).
2. J. Mellor and J. Shapiro. [Thompson Sampling in Switching Environments with Bayesian Online Change Point Detection](https://proceedings.mlr.press/v31/mellor13a.html). AISTATS, 2013.
3. O. Besbes, Y. Gur, and A. Zeevi. [Stochastic Multi-Armed-Bandit Problem with Non-stationary Rewards](https://papers.nips.cc/paper_files/paper/2014/hash/91ba7292e5388b90b58d0b839a7f19ec-Abstract.html). NeurIPS, 2014.
4. F. Liu, J. Lee, and N. Shroff. [A Change-Detection Based Framework for Piecewise-Stationary Multi-Armed Bandit Problem](https://arxiv.org/abs/1711.03539). AAAI, 2018.
5. Y. Cao, Z. Wen, B. Kveton, and Y. Xie. [Nearly Optimal Adaptive Procedure with Change Detection for Piecewise-Stationary Bandit](https://proceedings.mlr.press/v89/cao19a.html). AISTATS, 2019.
6. P. Auer, N. Cesa-Bianchi, Y. Freund, and R. Schapire. [The Nonstochastic Multiarmed Bandit Problem](https://doi.org/10.1137/S0097539701398375). SIAM Journal on Computing, 2002.
7. L. Li, W. Chu, J. Langford, and X. Wang. [Unbiased Offline Evaluation of Contextual-Bandit-Based News Article Recommendation Algorithms](https://arxiv.org/abs/1003.5956). WSDM, 2011.
8. Y. Saito, S. Aihara, M. Matsutani, and Y. Narita. [Open Bandit Dataset and Pipeline: Towards Realistic and Reproducible Off-Policy Evaluation](https://arxiv.org/abs/2008.07146). NeurIPS Datasets and Benchmarks, 2021.
9. O. Besbes, Y. Gur, and A. Zeevi. [Optimal Exploration–Exploitation in a Multi-armed Bandit Problem with Non-stationary Rewards](https://pubsonline.informs.org/doi/10.1287/stsy.2019.0033). *Stochastic Systems*, 2019.
10. L. Besson, E. Kaufmann, O.-A. Maillard, and J. Seznec. [Efficient Change-Point Detection for Tackling Piecewise-Stationary Bandits](https://www.jmlr.org/papers/v23/20-1384.html). *JMLR*, 2022.
