from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import run_backtest_v21 as v21
import run_backtest_v211 as v211
import run_backtest_v22 as v22
import run_backtest_v223 as v223
import run_backtest_v23_rookies as v23
import run_backtest_v24_horizon as v24
import run_backtest_v241_retention as v241
from paris_long_td_bonus import apply_long_td_bonus, load_long_td_bonus, value_scale_1000


# v2.4.1 settings are frozen. Exact scoring is an implementation correction,
# not permission to reopen the candidate grid.
DISCOUNT = 0.50
SHRINK = 5.0
CAP = 1.00
STRENGTH = 1.00
COMPLETE_EVAL = [2019, 2020, 2021]
RANKING_YEARS = [2019, 2020, 2021, 2022, 2023]
PBP_YEARS = list(range(2012, 2024))


def add_selected_scores(table: pd.DataFrame, annual: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_rows: list[pd.DataFrame] = []
    curve_rows: list[pd.DataFrame] = []

    for year in RANKING_YEARS:
        g = table[table.forecast_year.eq(year)].copy().reset_index(drop=True)
        curves, pos = v241.retention_table(annual, year, SHRINK, CAP)
        r1 = v241.lookup_retention(g, curves, pos, 0)
        r2 = v241.lookup_retention(g, curves, pos, 1)
        full = 1.0 + DISCOUNT * r1 + (DISCOUNT ** 2) * r1 * r2
        g["retention_y2"] = r1.to_numpy()
        g["retention_y3_conditional"] = r2.to_numpy()
        g["horizon_multiplier"] = 1.0 + STRENGTH * (full - 1.0)
        g["dynasty_score"] = g.pred_vor_pos * g.horizon_multiplier
        g["paris_score"] = value_scale_1000(g.dynasty_score)
        g["overall_rank"] = g.dynasty_score.rank(ascending=False, method="min")
        g["position_rank"] = g.groupby("position").dynasty_score.rank(ascending=False, method="min")
        selected_rows.append(g)

        c = curves.copy()
        c["forecast_year"] = year
        curve_rows.append(c)

    return pd.concat(selected_rows, ignore_index=True), pd.concat(curve_rows, ignore_index=True)


def exact_metrics(table: pd.DataFrame, annual: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year in COMPLETE_EVAL:
        row, _, _ = v241.evaluate_candidate(table, annual, year, DISCOUNT, SHRINK, CAP, STRENGTH)
        rows.append(row)
    return pd.DataFrame(rows)


def partial_metrics(selected: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year, horizon in [(2022, 2), (2023, 1)]:
        g = selected[selected.forecast_year.eq(year)].copy()
        g["partial_target"] = v241.target(g, DISCOUNT, horizon)
        rows.append({
            "year": year,
            "observed_horizon": horizon,
            "spearman": g.dynasty_score.corr(g.partial_target, method="spearman"),
            "baseline_spearman": g.pred_vor_pos.corr(g.partial_target, method="spearman"),
            "top50": v241.overlap(g, "dynasty_score", "partial_target", 50),
            "baseline_top50": v241.overlap(g, "pred_vor_pos", "partial_target", 50),
        })
    return pd.DataFrame(rows)


def position_metrics(selected: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year in COMPLETE_EVAL:
        g = selected[selected.forecast_year.eq(year)].copy()
        g["target"] = v241.target(g, DISCOUNT, 3)
        for pos, s in g.groupby("position"):
            rows.append({
                "year": year,
                "position": pos,
                "n": len(s),
                "spearman": s.dynasty_score.corr(s.target, method="spearman"),
                "baseline_spearman": s.pred_vor_pos.corr(s.target, method="spearman"),
            })
    return pd.DataFrame(rows)


def scale_audit(selected: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year, g in selected.groupby("forecast_year"):
        active = g[g.dynasty_score.gt(0)].copy()
        rows.append({
            "year": int(year),
            "n": len(g),
            "positive_value_players": len(active),
            "top_raw_value": float(g.dynasty_score.max()),
            "top_paris_score": float(g.paris_score.max()),
            "min_paris_score": float(g.paris_score.min()),
            "rank_correlation_raw_vs_scaled": active.dynasty_score.corr(active.paris_score, method="spearman") if len(active) > 1 else 1.0,
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--players-csv", required=True)
    ap.add_argument("--draft-parquet", required=True)
    ap.add_argument("--combine-parquet", required=True)
    ap.add_argument("--roster-dir", required=True)
    ap.add_argument("--depth-dir", required=True)
    ap.add_argument("--pbp-dir", required=True)
    ap.add_argument("--out-dir", default="outputs_v242_exact")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Historical football outcomes stop at 2023 by construction.
    raw = v21.old.load_weekly_csvs(args.data_dir)
    seasons_base = v21.aggregate_v21(raw)
    bonus_counts, bonus_diag = load_long_td_bonus(args.pbp_dir, PBP_YEARS, threshold=50)
    seasons = apply_long_td_bonus(seasons_base, bonus_counts)

    players = pd.read_csv(args.players_csv, low_memory=False)
    draft = pd.read_parquet(args.draft_parquet)
    combine = pd.read_parquet(args.combine_parquet)

    ages = v241.age_metadata(players)
    eligible_all = v241.load_week1_sets(args.roster_dir, v241.ACTUAL_YEARS)
    eligible_forecast = {y: eligible_all[y] for y in v241.FORECAST_YEARS}

    # Frozen rookie pathway and frozen returning-player architecture are simply
    # refit on the corrected historical fantasy-point totals.
    rook = v23.rookie_universe(players, draft, combine)
    rook = v23.attach_preseason_role(rook, args.roster_dir, args.depth_dir)
    rook = v23.attach_outcomes(rook, seasons)

    vet_table = v211.augment_table(seasons, players)
    vet_table = v22.augment_targets(vet_table, seasons)
    depth, _ = v223.load_preseason_depth(args.depth_dir)
    vet_table = v223.attach_depth(vet_table, depth)
    meta = v24.player_metadata(players)

    forecasts = []
    for year in v241.FORECAST_YEARS:
        forecasts.append(v24.build_forecast_pool(year, vet_table, rook, eligible_forecast, meta))
    forecasts = pd.concat(forecasts, ignore_index=True)

    annual = v241.build_actual_annual(seasons, eligible_all, ages)
    table = v24.engineer_features(v24.add_horizon_targets(forecasts, annual))

    metrics = exact_metrics(table, annual)
    selected, curves = add_selected_scores(table, annual)
    partial = partial_metrics(selected)
    pos = position_metrics(selected)
    audit = scale_audit(selected)

    top150 = selected[selected.overall_rank.le(150)].copy().sort_values(["forecast_year", "overall_rank"])
    positional = selected.sort_values(["forecast_year", "position", "position_rank"]).copy()
    td_leaders = bonus_counts.sort_values(["season", "long_td_50_count"], ascending=[True, False]).copy()

    metrics.to_csv(out / "v242_exact_complete_metrics.csv", index=False)
    partial.to_csv(out / "v242_exact_partial_diagnostics.csv", index=False)
    pos.to_csv(out / "v242_exact_position_metrics.csv", index=False)
    curves.to_csv(out / "v242_exact_retention_curves.csv", index=False)
    selected.to_csv(out / "v242_exact_all_rankings.csv", index=False)
    top150.to_csv(out / "v242_exact_top150.csv", index=False)
    positional.to_csv(out / "v242_exact_positional_rankings.csv", index=False)
    bonus_counts.to_csv(out / "v242_long_td_player_counts.csv", index=False)
    bonus_diag.to_csv(out / "v242_long_td_diagnostics.csv", index=False)
    td_leaders.to_csv(out / "v242_long_td_leaders.csv", index=False)
    audit.to_csv(out / "v242_scale_audit.csv", index=False)

    manifest = {
        "version": "Paris v2.4.2",
        "status": "pre-holdout freeze candidate",
        "football_outcomes_used_through": 2023,
        "holdout_outcomes_opened": False,
        "league": {
            "teams": 12,
            "starters": "1 QB, 2 RB, 3 WR, 2 TE, 2 FLEX, 1 Superflex",
            "ppr": 1.0,
            "passing_yards_per_point": 25,
            "passing_td": 5,
            "interception": -2,
            "rush_receive_yards_per_point": 10,
            "rush_receive_td": 6,
            "fumble_lost": -2,
            "long_td_bonus": 1,
            "long_td_threshold_yards": 50,
            "long_pass_td_credit": "passer and receiver each receive +1",
            "long_rush_td_credit": "rusher receives +1",
        },
        "frozen_architecture": {
            "returning_players": "v2.2.5",
            "rookies": "v2.3 capital_depth Ridge alpha 300",
            "horizon": {
                "version": "v2.4.1 retention",
                "discount": DISCOUNT,
                "shrink": SHRINK,
                "cap": CAP,
                "strength": STRENGTH,
            },
            "presentation_scale": "ParisScore = 1000 * max(DynastyScore, 0) / max_pool_DynastyScore",
        },
    }
    (out / "PARIS_V242_FREEZE_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("=== LONG-TD BONUS DIAGNOSTICS ===")
    print(bonus_diag.to_string(index=False))
    print("\n=== EXACT-SCORING COMPLETE 3-YEAR METRICS ===")
    print(metrics.to_string(index=False))
    print("\n=== PARTIAL 2022/2023 DIAGNOSTICS ===")
    print(partial.to_string(index=False))
    print("\n=== 0-1000 SCALE AUDIT ===")
    print(audit.to_string(index=False))
    print("\n=== 2023 TOP 50 ON 0-1000 PARIS SCALE ===")
    cols = ["overall_rank", "player_name", "position", "position_rank", "is_rookie", "age", "pred_vor", "horizon_multiplier", "dynasty_score", "paris_score"]
    print(selected[selected.forecast_year.eq(2023)].sort_values("overall_rank").head(50)[cols].to_string(index=False))
    print("\n2024 and 2025 football outcomes were not downloaded or used.")


if __name__ == "__main__":
    main()
