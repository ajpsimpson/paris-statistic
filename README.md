# Paris Statistic

Paris Statistic is a dynasty fantasy-football valuation project designed to turn player production, opportunity, age, draft capital, availability, team context, positional scarcity, and repeated elite performance into a transparent dynasty ranking system.

## Current research design

- League: 12 teams
- Starters: 1 QB, 2 RB, 3 WR, 2 TE, 2 FLEX, 1 Superflex
- Scoring: 1 PPR, 1 point per 25 passing yards, 5 per passing TD, -2 per interception, 1 per 10 rushing/receiving yards, 6 per rushing/receiving TD, -2 per lost fumble, +1 for a 50+ yard TD
- Historical rolling backtests: 2019-2022
- Development/validation season: 2023
- Locked football-outcome holdout seasons: 2024-2025
- Outputs: positional rankings plus one overall Top 150

The historical football-stat scripts intentionally download only through 2023. No 2024 or 2025 football outcomes have been used to choose or tune the model.

## Paris v2 philosophy

Paris v2 separates football prediction from dynasty valuation. First it predicts future player performance using historical features. Then it estimates role/availability risk, applies league-specific replacement value, and eventually adds a multi-year dynasty horizon.

The goal is not to hand-pick weights until the rankings look right. The goal is to test assumptions historically, keep only features that improve out-of-sample performance, lock the model after 2023, and then evaluate it honestly on 2024-2025.

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

The role layer was then stress-tested using information that would have been known at the start of each season.

### Week-1 roster eligibility

A Week-1 nflverse roster filter removes players who were already retired, unsigned, or otherwise not on an NFL roster while retaining injured/reserve players who still represented dynasty assets. This modestly improved the full ranking but materially improved veteran ordering.

### Week-1 depth chart

Historical nflverse depth-chart information was added as a role input. This produced the largest role-layer gain:

- 2019-2022 average overall Spearman: about 0.736 before depth -> about 0.816 with depth
- 2023 overall Spearman: about 0.760 -> about 0.810
- 2019-2022 average veteran Spearman: about 0.724 -> about 0.803

The gain showed that role information known at the start of a season can matter more than simply adding stronger age penalties.

### Position-specific QB role definition

The depth-chart test exposed a specific problem: a former starting quarterback who moved to a backup job could still receive too much credit from his prior-year production. A special interaction-heavy QB model improved metrics slightly but added complexity without solving the problem cleanly.

Paris therefore tested stricter QB definitions of a meaningful role while leaving RB/WR/TE unchanged. The selected threshold was:

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

The 2019-2022 average overall Spearman is about 0.818. The 2023 result is about 0.817. This version also reduced the Marcus Mariota-style former-starter/now-backup problem without an ad hoc player adjustment.

**The returning-player role-survival architecture is frozen at v2.2.5.** Further role tuning is intentionally stopped to reduce overfitting risk. Remaining misses such as unpredictable injuries are treated as event risk rather than reasons to keep modifying the historical model.

## Remaining major steps

1. Build and validate a separate rookie pathway because rookies have no NFL lagged production.
2. Merge rookie and returning-player values into one league-relative ranking.
3. Build and validate a multi-year dynasty horizon rather than only next-season value.
4. Freeze the full Paris architecture.
5. Only then open the 2024 and 2025 football-outcome holdouts.

The current season-level backtest still does not add the rare +1 bonus for 50+ yard touchdowns because that requires the play-by-play scoring pass. The league definition above remains the intended final scoring format.

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
- `download_nflverse_data.py` — downloads historical football data only through 2023
- `test_paris_v2_core.py` — unit tests for the core model logic

## Status

Returning-player performance and role-survival architecture are now frozen through v2.2.5. Rookie modeling is the next active development stage. The 2024-2025 football outcomes remain sealed.
