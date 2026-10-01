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

## Paris v2 philosophy

Paris v2 separates football prediction from dynasty valuation. First it predicts future player performance using historical features. Then it applies league-specific replacement value and dynasty adjustments to create an overall value signal.

The model tests three approaches to superstar value:

1. No explicit superstar feature
2. Top-five persistence
3. Continuous elite persistence based on positional percentile

The goal is not to hand-pick weights until the rankings look right. The goal is to test assumptions historically, keep only features that improve out-of-sample performance, lock the model after 2023, and then evaluate it honestly on 2024-2025.

## Files

- `paris_v2_core.py` — league scoring, elite-persistence features, dynamic replacement logic, VOR, and ranking helpers
- `paris_v2_backtest.py` — historical feature engineering and rolling Ridge-regression backtests
- `test_paris_v2_core.py` — unit tests for the core model logic

## Status

Work in progress. The equations and hyperparameters are not being presented as empirically optimal until they survive the planned historical validation process.
