# Paris Statistic

Paris Statistic is a dynasty fantasy-football valuation project designed to turn player production, opportunity, age, draft capital, availability, team context, positional scarcity, and repeated elite performance into a transparent dynasty ranking system.

## Current research design

- League: 12 teams
- Starters: 1 QB, 2 RB, 3 WR, 2 TE, 2 FLEX, 1 Superflex
- Scoring: 1 PPR, 1 point per 25 passing yards, 5 per passing TD, -2 per interception, 1 per 10 rushing/receiving yards, 6 per rushing/receiving TD, -2 per lost fumble, +1 for a 50+ yard TD
- Historical rolling backtests: 2019-2022
- Final validation/tuning season: 2023
- Locked holdout seasons: 2024-2025
- Outputs: positional rankings plus one overall Top 150

The current historical scripts intentionally download only through 2023. The 2024 and 2025 seasons remain untouched holdouts.

## Paris v2 philosophy

Paris v2 separates football prediction from dynasty valuation. First it predicts future player performance using historical features. Then it applies league-specific replacement value and dynasty adjustments to create an overall value signal.

The goal is not to hand-pick weights until the rankings look right. The goal is to test assumptions historically, keep only features that improve out-of-sample performance, lock the model after 2023, and then evaluate it honestly on 2024-2025.

## Paris v2.1.1

The current performance model tests:

- linear, quadratic, and piecewise-linear position-specific age curves;
- baseline, expanded, and usage-share opportunity features;
- no superstar term, top-five persistence, continuous elite persistence, and an elite-tier feature where a positional top-five season scores more than a 6-10 finish;
- Ridge regularization strengths from 1 to 300;
- exact calendar-year lagging so a missing season cannot accidentally cause a model to skip forward a year.

Selected position models:

- QB: linear age, baseline opportunity, continuous elite persistence, Ridge alpha 30
- RB: linear age, expanded opportunity, elite-tier persistence, Ridge alpha 100
- WR: quadratic age, baseline opportunity, elite-tier persistence, Ridge alpha 100
- TE: piecewise age curve, usage-share opportunity, elite-tier persistence, Ridge alpha 300

The v2.1.1 whole-ranking Spearman results were 0.702, 0.718, 0.720, 0.742, and 0.704 for 2019 through 2023 respectively. Those tests measured next-season conditional PPG/VOR among returning players with enough playing time.

## Paris v2.2 role-survival layer

Paris v2.2 keeps the v2.1.1 football-performance architecture fixed and adds a second model that estimates whether a player will retain a meaningful role the following season.

A candidate next-season role is defined from both games played and position-relative opportunity. The role classifier is trained only on information available before the forecast season. Its probability then adjusts the player's projected PPG before league-specific replacement value is calculated.

The selected preliminary v2.2 settings were:

- role threshold: at least four games and at least the 25th percentile of next-season positional opportunity;
- logistic regularization C = 0.1;
- full role-probability adjustment to projected PPG.

Under the availability-adjusted season-value target, selected v2.2 rolling Spearman results were:

| Year | Overall Spearman | Top-24 overlap | Top-50 overlap |
| --- | ---: | ---: | ---: |
| 2019 | 0.704 | 29.2% | 46% |
| 2020 | 0.737 | 41.7% | 50% |
| 2021 | 0.724 | 50.0% | 62% |
| 2022 | 0.772 | 33.3% | 58% |
| 2023 validation | 0.757 | 41.7% | 54% |

Within this same v2.2 season-value target, adding the role layer improved average pre-2023 Spearman from about 0.703 with no role adjustment to about 0.734, and improved 2023 validation from about 0.745 to about 0.757.

These numbers are not directly comparable to the v2.1.1 numbers above because v2.2 deliberately changes the target from conditional PPG to availability-adjusted season value. The improvement claim is only made against the no-role-adjustment baseline under the same v2.2 target.

The role layer is useful on average but is not a perfect predictor of sudden job changes, trades, or injuries. Cases such as a veteran starter becoming a backup the next year can still be missed.

The current season-level backtest also does not yet add the rare +1 bonus for 50+ yard touchdowns because that requires the play-by-play scoring pass. The league definition above remains the intended final scoring format.

## Files

- `paris_v2_core.py` — league scoring, elite-persistence features, dynamic replacement logic, VOR, and ranking helpers
- `paris_v2_backtest.py` — original historical feature engineering and rolling Ridge backtests
- `run_backtest_v21.py` — corrected exact-season v2.1 pipeline
- `run_backtest_v211.py` — expanded age-curve, opportunity, and elite-tier testing
- `run_backtest_v22.py` — role-survival and availability-adjusted value testing
- `download_nflverse_data.py` — downloads historical data only through 2023
- `test_paris_v2_core.py` — unit tests for the core model logic

## Status

Work in progress. Paris v2.2 supports keeping the role-survival layer, but the 2024-2025 holdouts should remain sealed until the rookie pathway and multi-year dynasty-value layer are built and the role definition is stress-tested further.
