from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import run_backtest_v21 as v21
import run_backtest_v211 as v211
import run_backtest_v22 as v22
import run_backtest_v223 as v223
import run_backtest_v23_rookies as v23
import run_backtest_v24_horizon as v24
import run_backtest_v241_retention as v241
from paris_long_td_bonus import apply_long_td_bonus, load_long_td_bonus


DISCOUNT = 0.50
SHRINK = 5.0
CAP = 1.00
STRENGTH = 1.00
COMPLETE_EVAL = [2019, 2020, 2021]
RANKING_YEARS = [2019, 2020, 2021, 2022, 2023]
PBP_YEARS = list(range(2012, 2024))
TAIL_MODES = ["clipped", "signed", "inverse_sqrt", "inverse_full"]


def dynasty_signal(vor: pd.Series, multiplier: pd.Series, mode: str) -> pd.Series:
    vor = pd.to_numeric(vor, errors="coerce").fillna(0.0)
    multiplier = pd.to_numeric(multiplier, errors="coerce").fillna(1.0).clip(lower=1e-6)
    positive = vor.ge(0)
    if mode == "clipped":
        return vor.clip(lower=0.0) * multiplier
    if mode == "signed":
        return pd.Series(np.where(positive, vor * multiplier, vor), index=vor.index)
    if mode == "inverse_sqrt":
        return pd.Series(np.where(positive, vor * multiplier, vor / np.sqrt(multiplier)), index=vor.index)
    if mode == "inverse_full":
        return pd.Series(np.where(positive, vor * multiplier, vor / multiplier), index=vor.index)
    raise ValueError(mode)


def top_k_overlap(g: pd.DataFrame, pred_col: str, actual_col: str, k: int) -> float:
    k = min(k, len(g))
    return len(set(g.nlargest(k, pred_col).player_id) & set(g.nlargest(k, actual_col).player_id)) / max(k, 1)


def top150_scale(values: pd.Series, k: int = 150) -> pd.Series:
    """Map the public Top-150 universe to 1..1000; everyone outside it is 0.

    This is presentation only. It is a monotonic linear transform of the selected
    dynasty signal and therefore cannot change ranking order.
    """
    x = pd.to_numeric(values, errors="coerce").fillna(-np.inf)
    n = len(x)
    if n == 0:
        return pd.Series(dtype=float, index=values.index)
    kk = min(k, n)
    order = x.sort_values(ascending=False)
    top = float(order.iloc[0])
    cutoff = float(order.iloc[kk - 1])
    out = pd.Series(0.0, index=x.index)
    inside = x.ge(cutoff)
    if not np.isfinite(top) or not np.isfinite(cutoff) or top <= cutoff:
        out.loc[inside] = 1000.0
        return out
    out.loc[inside] = 1.0 + 999.0 * (x.loc[inside] - cutoff) / (top - cutoff)
    return out.clip(0.0, 1000.0)


def build_exact_table(args) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = v21.old.load_weekly_csvs(args.data_dir)
    base = v21.aggregate_v21(raw)
    bonus_counts, bonus_diag = load_long_td_bonus(args.pbp_dir, PBP_YEARS, threshold=50)
    seasons = apply_long_td_bonus(base, bonus_counts)

    players = pd.read_csv(args.players_csv, low_memory=False)
    draft = pd.read_parquet(args.draft_parquet)
    combine = pd.read_parquet(args.combine_parquet)
    ages = v241.age_metadata(players)
    eligible_all = v241.load_week1_sets(args.roster_dir, v241.ACTUAL_YEARS)
    eligible_forecast = {y: eligible_all[y] for y in v241.FORECAST_YEARS}

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
    return table, annual, bonus_diag


def add_horizon_multiplier(g: pd.DataFrame, annual: pd.DataFrame, year: int) -> pd.DataFrame:
    g = g.copy().reset_index(drop=True)
    curves, pos = v241.retention_table(annual, year, SHRINK, CAP)
    r1 = v241.lookup_retention(g, curves, pos, 0)
    r2 = v241.lookup_retention(g, curves, pos, 1)
    g["retention_y2"] = r1.to_numpy()
    g["retention_y3_conditional"] = r2.to_numpy()
    g["horizon_multiplier"] = 1.0 + DISCOUNT * r1 + (DISCOUNT ** 2) * r1 * r2
    return g


def target(g: pd.DataFrame, horizon: int = 3) -> pd.Series:
    return v241.target(g, DISCOUNT, horizon)


def candidate_metrics(table: pd.DataFrame, annual: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    prediction_rows = []
    for year in COMPLETE_EVAL:
        base = add_horizon_multiplier(table[table.forecast_year.eq(year)], annual, year)
        base["target"] = target(base, 3)
        for mode in TAIL_MODES:
            g = base.copy()
            g["dynasty_signal"] = dynasty_signal(g.pred_vor, g.horizon_multiplier, mode)
            rows.append({
                "year": year,
                "tail_mode": mode,
                "n": len(g),
                "whole_pool_spearman": g.dynasty_signal.corr(g.target, method="spearman"),
                "top50": top_k_overlap(g, "dynasty_signal", "target", 50),
                "top100": top_k_overlap(g, "dynasty_signal", "target", 100),
                "top150": top_k_overlap(g, "dynasty_signal", "target", 150),
            })
            g["tail_mode"] = mode
            prediction_rows.append(g)
    return pd.DataFrame(rows), pd.concat(prediction_rows, ignore_index=True)


def choose_tail(metrics: pd.DataFrame) -> pd.DataFrame:
    pre = metrics[metrics.year.lt(2021)].groupby("tail_mode", as_index=False).agg(
        pre_top150=("top150", "mean"),
        pre_top100=("top100", "mean"),
        pre_top50=("top50", "mean"),
        pre_whole_pool_spearman=("whole_pool_spearman", "mean"),
    )
    val = metrics[metrics.year.eq(2021)][[
        "tail_mode", "top150", "top100", "top50", "whole_pool_spearman"
    ]].rename(columns={
        "top150": "val_top150",
        "top100": "val_top100",
        "top50": "val_top50",
        "whole_pool_spearman": "val_whole_pool_spearman",
    })
    z = pre.merge(val, on="tail_mode", how="inner")
    # This stage exists only because Top-150 ranks 133-150 were tied. Selection
    # therefore prioritizes Top-150 capture; top-50/top-100 are guardrails and
    # should remain unchanged because positive-VOR scores are untouched.
    z["selection_score"] = 0.50 * z.pre_top150 + 0.50 * z.val_top150
    return z.sort_values(
        ["selection_score", "val_top150", "pre_top150", "val_top100", "val_top50"],
        ascending=False,
    )


def build_selected(table: pd.DataFrame, annual: pd.DataFrame, mode: str) -> pd.DataFrame:
    rows = []
    for year in RANKING_YEARS:
        g = add_horizon_multiplier(table[table.forecast_year.eq(year)], annual, year)
        g["dynasty_signal"] = dynasty_signal(g.pred_vor, g.horizon_multiplier, mode)
        g["overall_rank"] = g.dynasty_signal.rank(ascending=False, method="min")
        g["position_rank"] = g.groupby("position").dynasty_signal.rank(ascending=False, method="min")
        g["paris_score"] = top150_scale(g.dynasty_signal, 150)
        rows.append(g)
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--players-csv", required=True)
    ap.add_argument("--draft-parquet", required=True)
    ap.add_argument("--combine-parquet", required=True)
    ap.add_argument("--roster-dir", required=True)
    ap.add_argument("--depth-dir", required=True)
    ap.add_argument("--pbp-dir", required=True)
    ap.add_argument("--out-dir", default="outputs_v243_top150")
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    table, annual, bonus_diag = build_exact_table(args)
    metrics, predictions = candidate_metrics(table, annual)
    summary = choose_tail(metrics)
    best = str(summary.iloc[0].tail_mode)
    selected = build_selected(table, annual, best)

    selected_metrics = metrics[metrics.tail_mode.eq(best)].sort_values("year")
    top150 = selected[selected.overall_rank.le(150)].sort_values(["forecast_year", "overall_rank"]).copy()

    partial_rows = []
    for year, h in [(2022, 2), (2023, 1)]:
        g = selected[selected.forecast_year.eq(year)].copy()
        g["partial_target"] = target(g, h)
        partial_rows.append({
            "year": year,
            "observed_horizon": h,
            "top50": top_k_overlap(g, "dynasty_signal", "partial_target", 50),
            "top100": top_k_overlap(g, "dynasty_signal", "partial_target", 100),
            "top150": top_k_overlap(g, "dynasty_signal", "partial_target", 150),
            "whole_pool_spearman": g.dynasty_signal.corr(g.partial_target, method="spearman"),
        })
    partial = pd.DataFrame(partial_rows)

    scale_rows = []
    for year, g in selected.groupby("forecast_year"):
        top = g.nsmallest(150, "overall_rank").copy()
        scale_rows.append({
            "year": int(year), "n": len(g), "top150_n": len(top),
            "top_score": float(top.paris_score.max()), "rank150_score": float(top.paris_score.min()),
            "raw_scaled_spearman_top150": top.dynasty_signal.corr(top.paris_score, method="spearman"),
        })
    scale_audit = pd.DataFrame(scale_rows)

    metrics.to_csv(out/"v243_tail_candidate_metrics.csv", index=False)
    summary.to_csv(out/"v243_tail_candidate_summary.csv", index=False)
    selected_metrics.to_csv(out/"v243_selected_complete_metrics.csv", index=False)
    partial.to_csv(out/"v243_partial_diagnostics.csv", index=False)
    selected.to_csv(out/"v243_all_rankings.csv", index=False)
    top150.to_csv(out/"v243_top150.csv", index=False)
    scale_audit.to_csv(out/"v243_scale_audit.csv", index=False)
    bonus_diag.to_csv(out/"v243_long_td_diagnostics.csv", index=False)

    manifest = {
        "version": "Paris v2.4.3",
        "status": "pre-holdout frozen architecture",
        "football_outcomes_used_through": 2023,
        "holdout_2024_2025_opened": False,
        "exact_scoring": {
            "long_td_bonus": "+1 at 50+ yards",
            "passing_long_td": "passer and receiver each credited",
            "rushing_long_td": "rusher credited",
        },
        "architecture": {
            "returning_players": "v2.2.5 frozen",
            "rookies": "v2.3 capital_depth Ridge alpha 300 frozen",
            "longevity": {"version": "v2.4.1", "discount": DISCOUNT, "shrink": SHRINK, "cap": CAP, "strength": STRENGTH},
            "top150_tail": best,
            "tail_definition": "positive VOR keeps v2.4.1 longevity score; below replacement uses selected raw-VOR tail rule",
            "public_scale": "within each preseason Top 150, rank-150 raw value maps to 1 and rank-1 maps to 1000; outside Top 150 maps to 0",
        },
    }
    (out/"PARIS_V243_FREEZE_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("=== TOP-150 TAIL CANDIDATES ===")
    print(summary.to_string(index=False))
    print("\n=== SELECTED TAIL MODE ===")
    print(best)
    print("\n=== SELECTED COMPLETE METRICS ===")
    print(selected_metrics.to_string(index=False))
    print("\n=== PARTIAL 2022/2023 DIAGNOSTICS ===")
    print(partial.to_string(index=False))
    print("\n=== SCALE AUDIT ===")
    print(scale_audit.to_string(index=False))
    print("\n=== 2023 TOP 150 ===")
    cols = ["overall_rank", "player_name", "position", "position_rank", "is_rookie", "age", "pred_vor", "horizon_multiplier", "dynasty_signal", "paris_score"]
    print(top150[top150.forecast_year.eq(2023)][cols].to_string(index=False))
    print("\n2024 and 2025 football outcomes were not downloaded or used.")


if __name__ == "__main__":
    main()
