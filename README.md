# Paris Statistic

Paris Statistic is a dynasty fantasy-football valuation project designed to turn player production, opportunity, age, draft capital, availability, preseason role, positional scarcity, and repeated elite performance into a transparent dynasty ranking system.

## Current research design

- League: 12 teams
- Starters: 1 QB, 2 RB, 3 WR, 2 TE, 2 FLEX, 1 Superflex
- Scoring: 1 PPR, 1 point per 25 passing yards, 5 per passing TD, -2 per interception, 1 per 10 rushing/receiving yards, 6 per rushing/receiving TD, -2 per lost fumble, +1 for a 50+ yard TD
- Historical development uses football outcomes only through 2023
- Locked football-outcome holdout seasons: 2024-2025
- Outputs: positional rankings plus one overall Top 150 on a 0-1,000 Paris scale

**No 2024 or 2025 football outcomes have been used to choose or tune the architecture documented below.**

## Paris v2 philosophy

Paris v2 separates football prediction from dynasty valuation. First it predicts future player performance, then estimates role/availability risk, converts the result to league-specific replacement value, adds a separate rookie pathway, and finally applies a modest multi-year longevity layer.

The goal is not to hand-pick weights until the rankings look right. The goal is to test assumptions historically, keep only features that improve out-of-sample performance, reject extra complexity when it does not help, and freeze the architecture before opening the holdout seasons.

## Paris v2.1.1 performance model

Selected position models:

- QB: linear age, baseline opportunity, continuous elite persistence, Ridge alpha 30
- RB: linear age, expanded opportunity, elite-tier persistence, Ridge alpha 100
- WR: quadratic age, baseline opportunity, elite-tier persistence, Ridge alpha 100
- TE: piecewise age curve, usage-share opportunity, elite-tier persistence, Ridge alpha 300

The v2.1.1 whole-ranking Spearman results were 0.702, 0.718, 0.720, 0.742, and 0.704 for 2019 through 2023. Those tests measured next-season conditional PPG/VOR among returning players with enough playing time.

## Paris v2.2 role-survival layer

Paris v2.2 added a separate logistic classifier for the probability that a player retains a meaningful role the following season. Under the same availability-adjusted season-value target, the role layer improved average pre-2023 Spearman from about 0.703 to about 0.734 and 2023 validation from about 0.745 to about 0.757.

A veteran/contract stress test found that historical contract information improved narrow roster-security classification but did not improve the complete ranking enough to replace the simpler role model. Contract information was therefore not forced into the main equation.

## Paris v2.2.2-v2.2.5: roster, depth chart, and QB role refinement

A Week-1 nflverse roster filter removes players who were already retired, unsigned, or otherwise not on an NFL roster while retaining injured/reserve players who still represented dynasty assets.

Historical Week-1 depth-chart information produced the largest role-layer gain. Paris then tested stricter QB definitions of a meaningful role while leaving RB/WR/TE unchanged. The selected threshold was:

- QB: at least four games and at least the 55th percentile of next-season QB opportunity
- RB/WR/TE: at least four games and at least the 25th percentile of position opportunity
- logistic C = 0.1
- full role-probability multiplier on projected PPG
- Week-1 roster eligibility and Week-1 depth-chart features included

Selected v2.2.5 results:

| Year | Overall Spearman | Top-24 overlap |
| --- | ---: | ---: |
| 2019 | 0.783 | 29.2% |
| 2020 | 0.830 | 45.8% |
| 2021 | 0.817 | 45.8% |
| 2022 | 0.841 | 37.5% |
| 2023 validation | 0.817 | 41.7% |

**The returning-player performance and role-survival architecture is frozen at v2.2.5.**

## Paris v2.3 rookie pathway

Rookies require a separate pathway because they have no NFL lagged production or opportunity. Candidate feature groups tested draft capital/age, combine measurements, and preseason role information.

The selected rookie model was **draft capital + preseason depth chart**, using Ridge alpha 300. Combine additions did not improve the model enough to keep them.

Selected rookie-only results:

| Year | Rookie Spearman | Top-10 overlap | Top-20 overlap |
| --- | ---: | ---: | ---: |
| 2019 | 0.634 | 60% | 60% |
| 2020 | 0.734 | 40% | 70% |
| 2021 | 0.715 | 70% | 65% |
| 2022 | 0.754 | 50% | 60% |
| 2023 validation | 0.790 | 50% | 65% |

After merging the rookie pathway with the frozen returning-player model:

| Year | Merged Spearman | Top-24 overlap | Top-50 overlap |
| --- | ---: | ---: | ---: |
| 2019 | 0.800 | 33.3% | 46% |
| 2020 | 0.838 | 45.8% | 46% |
| 2021 | 0.812 | 41.7% | 60% |
| 2022 | 0.842 | 41.7% | 58% |
| 2023 validation | 0.831 | 37.5% | 52% |

**The rookie one-year architecture is frozen at v2.3.**

## Paris v2.4-v2.4.1: multi-year dynasty horizon

The first multi-year attempt used a new Ridge model to predict a discounted three-year sum of future replacement value from the one-year Paris signal, age, position, draft capital, and role information. It was rejected because the extra regression made the full ranking worse than the simpler one-year VOR baseline.

Paris v2.4.1 instead estimates empirical **age-by-position surplus-retention curves**. For each position and age bucket, it measures how much positive above-replacement value historically survived into the following year, with sparse age buckets shrunk toward the position-wide average.

For players at or above replacement, the dynasty horizon is:

`DynastyScore = VOR × [1 + d × R(age,pos) + d² × R(age,pos) × R(age+1,pos)]`

Selected settings:

- future-year discount `d = 0.50`
- age-bucket shrink parameter = 5
- retention capped at 1.00
- full use of the retention multiplier
- three-year evaluation target = discounted sum of positive future VOR

The Spearman improvement was deliberately small rather than transformative, but the retention version improved all three complete three-year cohorts and improved Top-50 identification in all three. The broader multi-year Ridge model was therefore rejected while the modest retention adjustment was kept.

## Paris v2.4.2: exact 50+ yard touchdown scoring

The historical scoring pass now uses nflverse play-by-play to implement the league's exact long-touchdown rule.

- A 50+ yard passing TD gives the passer the normal passing TD plus the +1 long-TD bonus.
- The receiver on that play also gets the normal receiving TD plus the +1 long-TD bonus.
- A 50+ yard rushing TD gives the rusher the normal rushing TD plus the +1 long-TD bonus.
- Defensive/return TDs are not included in the Paris offensive-player scoring specification.

This correction added about 123-171 fantasy bonus credits league-wide per season from 2012-2023 and changed the backtest only slightly, as expected for a rare scoring event. The architecture and hyperparameters were not reopened because of the scoring correction.

## Paris v2.4.3: real Top-150 tail and 0-1,000 scale

A structural audit found that the league has 132 starting slots. The v2.4.1 positive-VOR formulation therefore produced 132 above-replacement players and then tied almost every player below replacement at zero. That meant positions 133-150 were not meaningfully ordered.

Before the holdout was opened, Paris tested four contained tail rules. Every rule left all positive-VOR players and the v2.4.1 longevity calculation unchanged; the only question was how to order players at or below replacement. The selected rule was the simplest **signed VOR tail**:

- if `VOR > 0`: keep the frozen v2.4.1 longevity-adjusted score;
- if `VOR <= 0`: use the raw signed VOR to order the below-replacement tail.

This produced the best pre-holdout Top-150 capture among the tested tail rules:

| Forecast | Top-50 | Top-100 | Top-150 |
| --- | ---: | ---: | ---: |
| 2019 -> 2019-21 | 52% | 66% | 66.7% |
| 2020 -> 2020-22 | 56% | 68% | 72.7% |
| 2021 -> 2021-23 | 70% | 68% | 70.7% |

For comparison, the old clipped tail had Top-150 overlap of 64%, 68%, and 66% in those three cohorts. Top-50 and Top-100 order were unchanged by the tail fix.

Whole-pool Spearman is not the selection metric for this stage. Resolving hundreds of previously tied fringe players lowers whole-pool rank correlation against a target that also contains many zero-value fringe ties. The v2.4.3 tail stage exists specifically to make the requested Top 150 meaningful, so Top-150 capture is the relevant test while the top of the board is held fixed.

### Public 0-1,000 Paris scale

The 0-1,000 score is a monotonic presentation transform of the frozen dynasty signal; it does not alter ranking order.

Within each preseason Top 150:

- rank 1's raw dynasty signal maps to **1000**;
- rank 150's raw dynasty signal maps to **1**;
- values in between are linearly scaled between those anchors;
- players outside the Top 150 receive **0** on the public Top-150 scale.

Historical audits showed a Spearman correlation of exactly 1.000 between raw dynasty signal and public Paris Score within every Top 150, confirming that the scale does not change order.

## Pre-holdout freeze

The statistical architecture is now frozen before opening 2024-2025 football outcomes:

1. returning-player performance + role survival: **v2.2.5**;
2. rookie pathway: **v2.3**, draft capital + preseason depth chart, Ridge alpha 300;
3. dynasty longevity: **v2.4.1**, age-position retention with `d=.50`, shrink 5, cap 1.00;
4. exact league scoring: **v2.4.2**, including the +1 50+ yard TD rule;
5. Top-150 tail: **v2.4.3 signed VOR below replacement**;
6. public scale: monotonic Top-150 1-1000 transform.

Architecture code was frozen at commit `f139dbec901c1d8a516bff6948b19893752a1483` before any 2024 or 2025 football outcome was opened for the final holdout evaluation.

## Files

- `paris_v2_core.py` — league scoring, elite-persistence features, dynamic replacement logic, VOR, and ranking helpers
- `paris_v2_backtest.py` — original historical feature engineering and rolling Ridge backtests
- `run_backtest_v21.py` — corrected exact-season v2.1 pipeline
- `run_backtest_v211.py` — expanded age-curve, opportunity, and elite-tier testing
- `run_backtest_v22.py` — original role-survival and availability-adjusted value testing
- `run_stress_v221.py` / `run_stress_v221_current_contracts.py` — veteran and contract stress tests
- `run_backtest_v222.py` — Week-1 roster eligibility test
- `run_backtest_v223.py` — Week-1 depth-chart role test
- `run_backtest_v224.py` — rejected interaction-heavy QB transition test
- `run_backtest_v225.py` — selected position-specific QB role-threshold test
- `run_backtest_v23_rookies.py` — separate rookie model and merged veteran/rookie backtest
- `run_backtest_v24_horizon.py` — rejected broad multi-year Ridge experiment
- `run_backtest_v241_retention.py` — selected empirical age-position longevity layer
- `paris_long_td_bonus.py` — exact 50+ yard touchdown scoring pass
- `run_backtest_v242_exact_scoring.py` — exact-scoring integration and audit
- `run_backtest_v243_top150.py` — below-replacement Top-150 tail test and final public scale
- `test_paris_v2_core.py` — unit tests for the core model logic

## Status

**Paris v2.4.3 is the pre-holdout frozen architecture.** The 2024 and 2025 football outcomes remain sealed. The next step is a one-shot holdout evaluation; its results should be reported whether they are strong, mediocre, or poor, without changing the frozen architecture afterward and pretending the holdout remained untouched.
