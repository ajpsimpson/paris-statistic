from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import run_backtest_v21 as v21
import run_backtest_v211 as v211
import run_backtest_v22 as v22
import run_backtest_v223 as v223
import run_backtest_v225 as v225
import run_backtest_v241_retention as v241
import run_backtest_v243_top150 as v243
from paris_long_td_bonus import apply_long_td_bonus, load_long_td_bonus
from paris_v2_core import DEFAULT_CONFIG, assign_starters

YEAR = 2016
ACTUAL_YEARS = list(range(2012, 2019))
PBP_YEARS = list(range(2012, 2019))
POSITIONS = ["QB", "RB", "WR", "TE"]
DISCOUNT = 0.50
SHRINK = 5.0
CAP = 1.00
QB_Q = 0.55
OTHER_Q = 0.25


def veteran_predictions(table: pd.DataFrame, year: int) -> pd.DataFrame:
    rows = []
    for pos in POSITIONS:
        pp = v22.fit_ppg_predict(table, year, pos)
        q = QB_Q if pos == "QB" else OTHER_Q
        rp = v225.role_predictions(table, year, pos, q)
        if len(pp) == 0 or len(rp) == 0:
            continue

        keep = [
            "target_year", "player_id", "player_name", "position",
            "predicted_ppg", "actual_week_value", "target_ppg_all", "target_games_all",
            "age_next"
        ]
        m = pp[[c for c in keep if c in pp.columns]].merge(
            rp[["player_id", "role_prob"] + v223.DEPTH_FEATURES],
            on="player_id", how="inner"
        )
        if "age_next" not in m.columns:
            age = table[["target_year", "player_id", "age_next"]].drop_duplicates(["target_year", "player_id"])
            m = m.merge(age, on=["target_year", "player_id"], how="left")
        m["age"] = m["age_next"]
        m["predicted_value"] = m.predicted_ppg * m.role_prob
        rows.append(m)
    return pd.concat(rows, ignore_index=True)


def add_same_cohort_future_targets(
    g: pd.DataFrame,
    seasons: pd.DataFrame,
    horizon_years: list[int],
) -> pd.DataFrame:
    out = g.copy()
    ids = set(out.player_id.astype(str))

    for k, season_year in enumerate(horizon_years, start=1):
        s = seasons[
            seasons.season.eq(season_year)
            & seasons.position.isin(POSITIONS)
            & seasons.player_id.astype(str).isin(ids)
        ][["player_id", "position", "paris_fp"]].drop_duplicates("player_id").copy()

        scheduled = 16.0 if season_year <= 2020 else 17.0
        s[f"actual_value_y{k}"] = pd.to_numeric(s.paris_fp, errors="coerce").fillna(0.0) / scheduled

        base = out[["player_id", "position"]].copy()
        base = base.merge(s[["player_id", f"actual_value_y{k}"]], on="player_id", how="left")
        base[f"actual_value_y{k}"] = base[f"actual_value_y{k}"].fillna(0.0)

        _, repl = assign_starters(
            base[["player_id", "position", f"actual_value_y{k}"]],
            DEFAULT_CONFIG,
            f"actual_value_y{k}",
        )
        base[f"actual_vor_y{k}"] = base[f"actual_value_y{k}"] - base.position.map(repl)
        base[f"actual_surplus_y{k}"] = base[f"actual_vor_y{k}"].clip(lower=0.0)

        out = out.merge(
            base[["player_id", f"actual_value_y{k}", f"actual_vor_y{k}", f"actual_surplus_y{k}"]],
            on="player_id",
            how="left",
        )
    return out


def overlap(g: pd.DataFrame, pred: str, actual: str, k: int) -> float:
    kk = min(k, len(g))
    if kk <= 0:
        return np.nan
    return len(
        set(g.nlargest(kk, pred).player_id)
        & set(g.nlargest(kk, actual).player_id)
    ) / kk


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--players-csv", required=True)
    ap.add_argument("--roster-dir", required=True)
    ap.add_argument("--depth-dir", required=True)
    ap.add_argument("--pbp-dir", required=True)
    ap.add_argument("--out-dir", default="outputs_retro_2016")
    args = ap.parse_args()
    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load only information needed for the retrospective 2016 test.
    raw = v21.old.load_weekly_csvs(args.data_dir)
    seasons_base = v21.aggregate_v21(raw)
    bonus_counts, bonus_diag = load_long_td_bonus(args.pbp_dir, PBP_YEARS, threshold=50)
    seasons = apply_long_td_bonus(seasons_base, bonus_counts)

    players = pd.read_csv(args.players_csv, low_memory=False)

    # The depth loader normally expects all later years. Limit it to the only
    # preseason snapshot used by this backcast.
    v223.DEPTH_YEARS = [YEAR]
    v223.OPENING_DATES[YEAR] = "2016-09-08"

    table = v211.augment_table(seasons, players)
    table = v22.augment_targets(table, seasons)
    depth, depth_diag = v223.load_preseason_depth(args.depth_dir)
    table = v223.attach_depth(table, depth)

    eligible = v241.load_week1_sets(args.roster_dir, ACTUAL_YEARS)
    pred = veteran_predictions(table, YEAR)
    pred = pred[pred.player_id.astype(str).isin(eligible[YEAR])].drop_duplicates("player_id").copy()

    # One-year frozen returning-player rank.
    _, pred_repl = assign_starters(
        pred[["player_id", "position", "predicted_value"]],
        DEFAULT_CONFIG,
        "predicted_value",
    )
    actual1 = pred[["player_id", "position", "actual_week_value"]].copy()
    _, actual_repl = assign_starters(actual1, DEFAULT_CONFIG, "actual_week_value")

    pred["pred_replacement"] = pred.position.map(pred_repl)
    pred["actual_replacement_y1"] = pred.position.map(actual_repl)
    pred["pred_vor"] = pred.predicted_value - pred.pred_replacement
    pred["actual_vor_y1_direct"] = pred.actual_week_value - pred.actual_replacement_y1

    # Rebuild the same 2016 returning cohort across 2016-2018 so the three-year
    # target does not silently add later rookies/free agents into its denominator.
    pred = add_same_cohort_future_targets(pred, seasons, [2016, 2017, 2018])

    # Frozen age/position longevity rule. Its retention curves are trained only
    # on seasons whose next-year outcome was already known before 2016.
    old_actual_years = v241.ACTUAL_YEARS
    v241.ACTUAL_YEARS = ACTUAL_YEARS
    ages = v241.age_metadata(players)
    annual = v241.build_actual_annual(seasons, eligible, ages)
    v241.ACTUAL_YEARS = old_actual_years

    curves, pos_fb = v241.retention_table(annual, YEAR, SHRINK, CAP)
    r1 = v241.lookup_retention(pred, curves, pos_fb, 0)
    r2 = v241.lookup_retention(pred, curves, pos_fb, 1)
    pred["retention_y2"] = r1.to_numpy()
    pred["retention_y3_conditional"] = r2.to_numpy()
    pred["horizon_multiplier"] = 1.0 + DISCOUNT*r1 + (DISCOUNT**2)*r1*r2
    pred["dynasty_signal"] = v243.dynasty_signal(pred.pred_vor, pred.horizon_multiplier, "signed")
    pred["paris_score"] = v243.top150_scale(pred.dynasty_signal, 150)
    pred["overall_rank"] = pred.dynasty_signal.rank(ascending=False, method="min")
    pred["position_rank"] = pred.groupby("position").dynasty_signal.rank(ascending=False, method="min")

    pred["three_year_target"] = (
        pred.actual_surplus_y1
        + DISCOUNT * pred.actual_surplus_y2
        + (DISCOUNT**2) * pred.actual_surplus_y3
    )
    pred["actual_three_year_rank"] = pred.three_year_target.rank(ascending=False, method="min")

    metrics = pd.DataFrame([{
        "forecast_year": YEAR,
        "pool": "returning_players_only",
        "n": len(pred),
        "one_year_value_spearman": pred.predicted_value.corr(pred.actual_week_value, method="spearman"),
        "one_year_vor_spearman": pred.pred_vor.corr(pred.actual_vor_y1_direct, method="spearman"),
        "three_year_dynasty_spearman": pred.dynasty_signal.corr(pred.three_year_target, method="spearman"),
        "top24_overlap_3y": overlap(pred, "dynasty_signal", "three_year_target", 24),
        "top50_overlap_3y": overlap(pred, "dynasty_signal", "three_year_target", 50),
        "top100_overlap_3y": overlap(pred, "dynasty_signal", "three_year_target", 100),
        "top150_overlap_3y": overlap(pred, "dynasty_signal", "three_year_target", 150),
    }])

    pos_rows = []
    for pos, p in pred.groupby("position"):
        pos_rows.append({
            "position": pos,
            "n": len(p),
            "one_year_value_spearman": p.predicted_value.corr(p.actual_week_value, method="spearman"),
            "three_year_dynasty_spearman": p.dynasty_signal.corr(p.three_year_target, method="spearman"),
        })
    pos_metrics = pd.DataFrame(pos_rows)

    top30 = pred.sort_values("overall_rank").head(30).copy()
    misses = pred.copy()
    misses["rank_error"] = misses.overall_rank - misses.actual_three_year_rank
    biggest_under = misses.sort_values("rank_error", ascending=False).head(15)
    biggest_over = misses.sort_values("rank_error", ascending=True).head(15)

    metrics.to_csv(outdir / "retro_2016_metrics.csv", index=False)
    pos_metrics.to_csv(outdir / "retro_2016_position_metrics.csv", index=False)
    pred.to_csv(outdir / "retro_2016_all_rankings.csv", index=False)
    top30.to_csv(outdir / "retro_2016_top30.csv", index=False)
    biggest_under.to_csv(outdir / "retro_2016_biggest_underestimates.csv", index=False)
    biggest_over.to_csv(outdir / "retro_2016_biggest_overestimates.csv", index=False)
    depth_diag.to_csv(outdir / "retro_2016_depth_diagnostics.csv", index=False)
    bonus_diag.to_csv(outdir / "retro_2016_long_td_diagnostics.csv", index=False)

    print("=== PARIS v2.4.3 RETROSPECTIVE 2016 RETURNING-PLAYER TEST ===")
    print("This is a backcast robustness check, NOT an untouched holdout.")
    print("Rookies are excluded to avoid using later depth-chart history to train a 2016 rookie model.")
    print("\n=== METRICS ===")
    print(metrics.to_string(index=False))
    print("\n=== POSITION METRICS ===")
    print(pos_metrics.to_string(index=False))
    print("\n=== FROZEN 2016 PRESEASON TOP 30 (RETURNERS) ===")
    cols = [
        "overall_rank", "player_name", "position", "position_rank", "age",
        "predicted_ppg", "role_prob", "predicted_value", "pred_vor",
        "horizon_multiplier", "dynasty_signal", "paris_score",
        "actual_week_value", "three_year_target", "actual_three_year_rank"
    ]
    print(top30[cols].to_string(index=False))
    print("\n=== BIGGEST 3-YEAR UNDERESTIMATES ===")
    print(biggest_under[["player_name","position","overall_rank","actual_three_year_rank","rank_error","dynasty_signal","three_year_target"]].to_string(index=False))
    print("\n=== BIGGEST 3-YEAR OVERESTIMATES ===")
    print(biggest_over[["player_name","position","overall_rank","actual_three_year_rank","rank_error","dynasty_signal","three_year_target"]].to_string(index=False))


if __name__ == "__main__":
    main()
