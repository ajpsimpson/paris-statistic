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

The current refinement tests:

- linear, quadratic, and piecewise-linear position-specific age curves;
- baseline, expanded, and usage-share opportunity features;
- no superstar term, top-five persistence, continuous elite persistence, and an elite-tier feature where a positional top-five season scores more than a 6-10 finish;
- Ridge regularization strengths from 1 to 300;
- exact calendar-year lagging so a missing season cannot accidentally cause a model to skip forward a year.

Model selection uses 60% rolling 2019-2022 performance, 40% 2023 validation performance, and a small stability penalty for models that vary sharply across historical seasons.

### Current selected position models

- QB: linear age, baseline opportunity, continuous elite persistence, Ridge alpha 30
- RB: linear age, expanded opportunity, elite-tier persistence, Ridge alpha 100
- WR: quadratic age, baseline opportunity, elite-tier persistence, Ridge alpha 100
- TE: piecewise age curve, usage-share opportunity, elite-tier persistence, Ridge alpha 300

### Current overall rolling results

| Year | Overall Spearman | Top-24 overlap | Top-50 overlap |
| --- | ---: | ---: | ---: |
| 2019 | 0.702 | 41.7% | 54% |
| 2020 | 0.718 | 50.0% | 62% |
| 2021 | 0.720 | 54.2% | 66% |
| 2022 | 0.742 | 50.0% | 60% |
| 2023 validation | 0.704 | 41.7% | 66% |

The 2019-2022 average overall Spearman is about 0.720. The 2023 validation result is about 0.704.

These results apply only to returning NFL players with enough next-season playing time for evaluation. They are not yet a complete dynasty model because rookies, full availability/role-survival risk, and the multi-year dynasty horizon still need separate treatment.

The current season-level backtest also does not yet add the rare +1 bonus for 50+ yard touchdowns because that requires the play-by-play scoring pass. The league definition above remains the intended final scoring format.

## Files

- `paris_v2_core.py` — league scoring, elite-persistence features, dynamic replacement logic, VOR, and ranking helpers
- `paris_v2_backtest.py` — original historical feature engineering and rolling Ridge backtests
- `run_backtest_v21.py` — corrected exact-season v2.1 pipeline
- `run_backtest_v211.py` — expanded age-curve, opportunity, and elite-tier testing
- `download_nflverse_data.py` — downloads historical data only through 2023
- `test_paris_v2_core.py` — unit tests for the core model logic

## Status

Work in progress. Paris v2.1.1 is strong enough to continue developing, but the 2024-2025 holdouts should remain sealed until availability/role survival, rookies, and the multi-year dynasty-value layer are addressed.
