# Paris Statistic

Paris Statistic is a dynasty fantasy-football valuation project designed to turn player production, opportunity, age, draft capital, availability, team context, positional scarcity, and repeated elite performance into a transparent dynasty ranking system.

## Current research design

- League: 12 teams
- Starters: 1 QB, 2 RB, 3 WR, 2 TE, 2 FLEX, 1 Superflex
- Scoring: 1 PPR, 1 point per 25 passing yards, 5 per passing TD, -2 per interception, 1 per 10 rushing/receiving yards, 6 per rushing/receiving TD, -2 per lost fumble, +1 for a 50+ yard TD
- Historical development uses football outcomes only through 2023
- Locked football-outcome holdout seasons: 2024-2025
- Outputs: positional rankings plus one overall Top 150

No 2024 or 2025 football outcomes have been used to choose or tune the model.

## Paris v2 philosophy

Paris v2 separates football prediction from dynasty valuation. First it predicts future player performance, then estimates role/availability risk, converts the result to league-specific replacement value, adds a separate rookie pathway, and finally applies a modest multi-year longevity layer.

The goal is not to hand-pick weights until the rankings look right. The goal is to test assumptions historically, keep only features that improve out-of-sample performance, and reject extra complexity when it does not help.

## Paris v2.1.1 performance model

Selected position models:

- QB: linear age, baseline opportunity, continuous elite persistence, Ridge alpha 30
- RB: linear age, expanded opportunity, elite-tier persistence, Ridge alpha 100
- WR: quadratic age, baseline opportunity, elite-tier persistence, Ridge alpha 100
- TE: piecewise age curve, usage-share opportunity, elite-tier persistence, Ridge alpha 300

The v2.1.1 whole-ranking Spearman results were 0.702, 0.718, 0.720, 0.742, and 0.704 for 2019 through 2023. Those tests measured next-season conditional PPG/VOR among returning players with enough playing time.

## Paris v2.2 role-survival layer

Paris v2.2 added a separate logistic classifier for the probability that a player retains a meaningful role the following season. Under the same availability-adjusted season-value target, the role layer improved average pre-2023 Spearman from about 0.703 to about 0.734 and 2023 validation from about 0.745 to about 0.757.

A veteran/contract stress test found that historical contract information improves narrow roster-security classification but did not improve the complete ranking enough to replace the simpler role model. Contract information is therefore reserved as a possible secondary career-continuation signal rather than forced into the main equation.

## Paris v2.2.2-v2.2.5: roster, depth chart, and QB role refinement

The role layer was stress-tested using information that would have been known at the start of each season.

### Week-1 roster eligibility

A Week-1 nflverse roster filter removes players who were already retired, unsigned, or otherwise not on an NFL roster while retaining injured/reserve players who still represented dynasty assets. This modestly improved the full ranking but materially improved veteran ordering.

### Week-1 depth chart

Historical nflverse depth-chart information was added as a role input. This produced the largest role-layer gain:

- 2019-2022 average overall Spearman: about 0.736 before depth -> about 0.816 with depth
- 2023 overall Spearman: about 0.760 -> about 0.810
- 2019-2022 average veteran Spearman: about 0.724 -> about 0.803

### Position-specific QB role definition

Paris tested stricter QB definitions of a meaningful role while leaving RB/WR/TE unchanged. The selected threshold was:

- QB: at least four games and at least the 55th percentile of next-season QB opportunity
- RB/WR/TE: at least four games and at least the 25th percentile of position opportunity
- logistic C = 0.1
- full role-probability multiplier on projected PPG
- Week-1 roster eligibility and Week-1 depth-chart features included

With the selected QB threshold, Paris achieved:

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

The selected rookie model was **draft capital + preseason depth chart**, using Ridge alpha 300. The combine additions did not improve the model enough to keep them.

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

For example, the rejected model's three-year Spearman was 0.590 vs 0.629 for the baseline in 2019, 0.635 vs 0.653 in 2020, and 0.638 vs 0.672 in 2021. This is an intentional negative result: a more complicated model was not kept merely because it looked more sophisticated.

Paris v2.4.1 instead estimates empirical **age-by-position surplus-retention curves**. For each position and age bucket, it measures how much positive above-replacement value historically survived into the following year, with sparse age buckets shrunk toward the position-wide average.

The dynasty horizon is then:

`DynastyScore = CurrentPositiveVOR × [1 + d × R(age,pos) + d² × R(age,pos) × R(age+1,pos)]`

where `R` is the historically estimated retention rate. The no-adjustment one-year score was included as an explicit candidate, so the longevity layer would only be selected if it improved the historical tests.

The selected v2.4.1 settings were:

- future-year discount `d = 0.50`
- age-bucket shrink parameter = 5
- retention capped at 1.00
- full use of the retention multiplier
- three-year target = discounted sum of positive future VOR; being below replacement contributes zero rather than negative dynasty asset value

Complete three-year backtests:

| Forecast | v2.4.1 Spearman | One-year baseline | Top-50 v2.4.1 | Top-50 baseline |
| --- | ---: | ---: | ---: | ---: |
| 2019 -> 2019-21 | 0.634 | 0.633 | 54% | 52% |
| 2020 -> 2020-22 | 0.655 | 0.654 | 56% | 52% |
| 2021 -> 2021-23 | 0.676 | 0.675 | 70% | 68% |

The Spearman gains are deliberately described as **small**, not transformative. The useful finding is that the retention layer improved all three complete test cohorts and improved Top-50 identification in all three, while the broader multi-year regression failed. Paris therefore keeps v2.4.1 as a modest longevity adjustment rather than allowing it to override the stronger one-year model.

2022 has only a two-year observable horizon through 2023 and 2023 only a one-year observable horizon; those later cohorts are retained as partial diagnostics rather than pretending that a full three-year target is available.

## Before the holdout is opened

The main statistical architecture is now in place: returning-player model, role survival, rookie pathway, and multi-year retention horizon. Before evaluating the sealed 2024-2025 outcomes, two implementation steps remain:

1. add the rare +1 50-yard-touchdown scoring bonus through a play-by-play scoring pass so the historical scoring exactly matches the intended league settings;
2. map the final continuous dynasty score to the public-facing 0-1,000 Paris scale without changing player order.

Only after those implementation steps are fixed should 2024-2025 be opened for the final holdout test.

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
- `download_nflverse_data.py` — standard historical-data downloader capped at 2023
- `test_paris_v2_core.py` — unit tests for the core model logic

## Status

Returning-player architecture is frozen at v2.2.5, rookie one-year architecture at v2.3, and the modest longevity layer at v2.4.1. The 2024-2025 football outcomes remain sealed. Exact long-touchdown scoring and final 0-1,000 presentation scaling are the last implementation steps before the holdout evaluation.
