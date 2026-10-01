# Paris Statistic

Paris Statistic is a dynasty fantasy-football valuation project designed to turn player production, opportunity, age, draft capital, availability, team context, positional scarcity, and repeated elite performance into a transparent dynasty ranking system.

## Current research design

- League: 12 teams
- Starters: 1 QB, 2 RB, 3 WR, 2 TE, 2 FLEX, 1 Superflex
- Scoring: 1 PPR, 1 point per 25 passing yards, 5 per passing TD, -2 per interception, 1 per 10 rushing/receiving yards, 6 per rushing/receiving TD, -2 per lost fumble, +1 for a 50+ yard TD
- Historical rolling backtests: 2019-2022
- Final validation/tuning season: 2023
- Locked football-outcome holdout seasons: 2024-2025
- Outputs: positional rankings plus one overall Top 150

The historical football-stat scripts intentionally download only through 2023. No 2024 or 2025 football outcomes have been used to choose or tune the model.

## Paris v2 philosophy

Paris v2 separates football prediction from dynasty valuation. First it predicts future player performance using historical features. Then it applies league-specific replacement value and dynasty adjustments to create an overall value signal.

The goal is not to hand-pick weights until the rankings look right. The goal is to test assumptions historically, keep only features that improve out-of-sample performance, lock the model after 2023, and then evaluate it honestly on 2024-2025.

## Paris v2.1.1 performance model

The performance model tested:

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

The selected preliminary role model used a role threshold of at least four games and at least the 25th percentile of next-season positional opportunity, logistic C = 0.1, and a full role-probability adjustment to projected PPG.

Under the availability-adjusted season-value target, selected v2.2 rolling Spearman results were:

| Year | Overall Spearman | Top-24 overlap | Top-50 overlap |
| --- | ---: | ---: | ---: |
| 2019 | 0.704 | 29.2% | 46% |
| 2020 | 0.737 | 41.7% | 50% |
| 2021 | 0.724 | 50.0% | 62% |
| 2022 | 0.772 | 33.3% | 58% |
| 2023 validation | 0.757 | 41.7% | 54% |

Within the same v2.2 target, adding the role layer improved average pre-2023 Spearman from about 0.703 with no role adjustment to about 0.734, and improved 2023 validation from about 0.745 to about 0.757.

## Paris v2.2.1 veteran and contract stress test

The role model was stress-tested specifically on veteran players. The test compared three architectures:

1. the original v2.2 role model;
2. a veteran-specific model adding age-over-threshold and declining games, PPG, and opportunity interactions;
3. the veteran-specific model plus historical OverTheCap contract information from nflverse.

Veteran thresholds were QB 32, RB 27, WR 29, and TE 30. A second late-veteran group used QB 35, RB 29, WR 31, and TE 32.

Contract features included years remaining, expiring-contract status, recent signing, APY as a percentage of the cap, guaranteed percentage, total contract value, and APY. Only contracts signed before the forecast year were eligible. The current `is_active` field was deliberately excluded because it would introduce future information into older forecasts.

The current nflverse Parquet contract file matched about 98-100% of relevant player-season rows from 2019-2023, so the contract ablation had sufficient coverage.

The stress test did **not** support replacing the original role model. The original base model remained the best combined choice:

- pre-2023 overall Spearman: 0.734
- pre-2023 veteran Spearman: 0.714
- 2023 overall Spearman: 0.757
- 2023 veteran Spearman: 0.734
- pre-2023 veteran role AUC: about 0.820
- 2023 veteran role AUC: about 0.897

The strongest contract-heavy candidate improved pure veteran role classification in some settings. One candidate reached roughly 0.842 pre-2023 veteran AUC and 0.927 on 2023 validation, with a lower 2023 Brier error than the base model. However, it did not improve the full dynasty-value ranking enough: its 2023 overall Spearman was about 0.750 versus 0.757 for the base model, and its Top-24 overlap was lower.

This suggests contract information contains real roster-security signal, but it is better treated as a possible separate career-continuation or roster-security input than forced into the main role-survival equation.

The veteran stress test also showed that many remaining misses are event risk rather than gradual decline. High-confidence misses included retirements, severe injuries, and sudden external role changes. Those are not well solved by simply adding a stronger age penalty.

The current season-level backtest still does not add the rare +1 bonus for 50+ yard touchdowns because that requires the play-by-play scoring pass. The league definition above remains the intended final scoring format.

## Files

- `paris_v2_core.py` — league scoring, elite-persistence features, dynamic replacement logic, VOR, and ranking helpers
- `paris_v2_backtest.py` — original historical feature engineering and rolling Ridge backtests
- `run_backtest_v21.py` — corrected exact-season v2.1 pipeline
- `run_backtest_v211.py` — expanded age-curve, opportunity, and elite-tier testing
- `run_backtest_v22.py` — role-survival and availability-adjusted value testing
- `run_stress_v221.py` — veteran role-survival stress test
- `run_stress_v221_current_contracts.py` — current historical-contract Parquet ablation and coverage diagnostics
- `download_nflverse_data.py` — downloads historical football data only through 2023
- `test_paris_v2_core.py` — unit tests for the core model logic

## Status

Work in progress. The veteran stress test supports keeping the simpler v2.2 role-survival model as the main role layer. Contract information appears useful as a secondary roster-security signal rather than as a replacement for the base classifier. The next major modeling steps are a rookie pathway and a multi-year dynasty-value horizon before the 2024-2025 football holdouts are opened.
