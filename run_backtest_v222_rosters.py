from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import run_backtest_v22 as v22
import run_backtest_v211 as v211
import run_backtest_v21 as v21
from paris_v2_core import DEFAULT_CONFIG, assign_starters

EVAL_YEARS = [2019, 2020, 2021, 2022, 2023]
VETERAN_AGE = {"QB": 32.0, "RB": 27.0, "WR": 29.0, "TE": 30.0}
SELECTED_Q = 0.25
SELECTED_C = 0.1
SELECTED_RISK_WEIGHT = 1.0


def load_week1_rosters(roster_dir: str) -> pd.DataFrame:
    pieces = []
    for y in EVAL_YEARS:
        p = Path(roster_dir) / f"roster_weekly_{y}.parquet"
        r = pd.read_parquet(p)
        r.columns = [str(c).lower() for c in r.columns]
        if "season" not in r.columns:
            r["season"] = y
        if "gsis_id" in r.columns:
            r = r.rename(columns={"gsis_id": "player_id"})
        elif "player_id" not in r.columns:
            raise ValueError(f"No gsis_id/player_id in {p}; columns={list(r.columns)}")
        if "week" not in r.columns:
            raise ValueError(f"No week in {p}; columns={list(r.columns)}")
        # Week 1 is the first regular-season roster snapshot. It is used only as
        # information that would have been available entering the season.
        w1 = r[pd.to_numeric(r["week"], errors="coerce").eq(1)].copy()
        w1 = w1[w1.player_id.notna()].copy()
        cols = [c for c in ["season", "player_id", "team", "status", "position", "full_name"] if c in w1.columns]
        w1 = w1[cols].drop_duplicates(["season", "player_id"])
        w1["week1_rostered"] = 1
        pieces.append(w1)
    return pd.concat(pieces, ignore_index=True)


def selected_v22_predictions(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year in EVAL_YEARS:
        for pos in ["QB", "RB", "WR", "TE"]:
            pp = v22.fit_ppg_predict(table, year, pos)
            rp, _ = v22.role_predictions(table, year, pos, SELECTED_Q, SELECTED_C)
            if len(pp) == 0 or len(rp) == 0:
                continue
            m = pp[["target_year", "player_id", "player_name", "position", "predicted_ppg", "actual_week_value", "age_next"]].merge(
                rp[["player_id", "role_prob"]], on="player_id", how="inner"
            )
            m["predicted_value"] = m["predicted_ppg"] * ((1.0 - SELECTED_RISK_WEIGHT) + SELECTED_RISK_WEIGHT * m["role_prob"])
            rows.append(m)
    return pd.concat(rows, ignore_index=True)


def apply_eligibility(preds: pd.DataFrame, rosters: pd.DataFrame, mode: str) -> pd.DataFrame:
    g = preds.merge(
        rosters[["season", "player_id", "week1_rostered"]].rename(columns={"season": "target_year"}),
        on=["target_year", "player_id"], how="left"
    )
    g["week1_rostered"] = g["week1_rostered"].fillna(0).astype(int)
    g["veteran_cut"] = g["position"].map(VETERAN_AGE)
    g["is_veteran"] = g["age_next"].ge(g["veteran_cut"]).astype(int)

    if mode == "none":
        g["eligible"] = 1
    elif mode == "hard":
        g["eligible"] = g["week1_rostered"]
    elif mode == "veteran_only":
        # Younger players remain eligible even if they are not in the week-1 file;
        # veteran players must be attached to an NFL roster. This is intended to
        # remove known retirements/cuts without over-filtering young fringe assets.
        g["eligible"] = np.where(g["is_veteran"].eq(1), g["week1_rostered"], 1)
    else:
        raise ValueError(mode)
    return g[g.eligible.eq(1)].copy()


def evaluate(preds: pd.DataFrame, rosters: pd.DataFrame):
    metrics, ranks, drops = [], [], []
    for mode in ["none", "hard", "veteran_only"]:
        gm = apply_eligibility(preds, rosters, mode)
        # record who the filter removes for auditability
        full = apply_eligibility(preds, rosters, "none")
        kept = set(zip(gm.target_year, gm.player_id))
        d = full[["target_year","player_id","player_name","position","age_next","predicted_value","actual_week_value","week1_rostered","is_veteran"]].copy()
        d = d[[ (y,p) not in kept for y,p in zip(d.target_year,d.player_id) ]]
        d["mode"] = mode
        drops.append(d)

        for year, g0 in gm.groupby("target_year"):
            g = g0.drop_duplicates("player_id").copy()
            if not {"QB","RB","WR","TE"}.issubset(set(g.position)):
                continue
            _, pr = assign_starters(g[["player_id","position","predicted_value"]], DEFAULT_CONFIG, "predicted_value")
            actual = g[["player_id","position","actual_week_value"]].copy()
            _, ar = assign_starters(actual, DEFAULT_CONFIG, "actual_week_value")
            g["pred_vor"] = g.predicted_value - g.position.map(pr)
            g["actual_vor"] = g.actual_week_value - g.position.map(ar)
            g["overall_rank"] = g.pred_vor.rank(ascending=False, method="min")
            g["actual_rank"] = g.actual_vor.rank(ascending=False, method="min")
            def ov(k):
                k = min(k, len(g))
                return len(set(g.nlargest(k,"pred_vor").player_id) & set(g.nlargest(k,"actual_vor").player_id)) / max(k,1)
            v = g[g.is_veteran.eq(1)]
            metrics.append({
                "mode": mode, "year": int(year), "n": len(g),
                "overall_spearman": g.pred_vor.corr(g.actual_vor, method="spearman"),
                "veteran_spearman": v.pred_vor.corr(v.actual_vor, method="spearman") if len(v)>=5 else np.nan,
                "top24_overlap": ov(24), "top50_overlap": ov(50),
                "week1_match_rate": float(g.week1_rostered.mean()),
            })
            g["mode"] = mode
            ranks.append(g)
    return pd.DataFrame(metrics), pd.concat(ranks, ignore_index=True), pd.concat(drops, ignore_index=True)


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    pre = metrics[metrics.year.lt(2023)].groupby("mode", as_index=False).agg(
        pre_spearman=("overall_spearman","mean"),
        pre_vet_spearman=("veteran_spearman","mean"),
        pre_top24=("top24_overlap","mean"), pre_top50=("top50_overlap","mean"),
        pre_n=("n","mean"),
    )
    val = metrics[metrics.year.eq(2023)][["mode","overall_spearman","veteran_spearman","top24_overlap","top50_overlap","n"]].rename(columns={
        "overall_spearman":"val_spearman","veteran_spearman":"val_vet_spearman","top24_overlap":"val_top24","top50_overlap":"val_top50","n":"val_n"
    })
    z = pre.merge(val,on="mode")
    z["selection_score"] = .35*z.pre_spearman + .30*z.val_spearman + .10*z.pre_vet_spearman + .10*z.val_vet_spearman + .075*z.pre_top24 + .075*z.val_top24
    return z.sort_values("selection_score", ascending=False)


def roster_diagnostics(rosters: pd.DataFrame) -> pd.DataFrame:
    return rosters.groupby("season", as_index=False).agg(players=("player_id","nunique"))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--players-csv")
    ap.add_argument("--roster-dir", required=True)
    ap.add_argument("--out-dir", default="outputs_v222")
    args=ap.parse_args(); out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)

    raw=v21.old.load_weekly_csvs(args.data_dir)
    seasons=v21.aggregate_v21(raw)
    players=pd.read_csv(args.players_csv,low_memory=False) if args.players_csv else None
    table=v211.augment_table(seasons,players)
    table=v22.augment_targets(table,seasons)
    preds=selected_v22_predictions(table)
    rosters=load_week1_rosters(args.roster_dir)
    metrics,ranks,drops=evaluate(preds,rosters)
    summary=summarize(metrics)

    roster_diagnostics(rosters).to_csv(out/"v222_roster_diagnostics.csv",index=False)
    metrics.to_csv(out/"v222_roster_filter_metrics.csv",index=False)
    summary.to_csv(out/"v222_roster_filter_summary.csv",index=False)
    ranks.to_csv(out/"v222_rankings.csv",index=False)
    drops.to_csv(out/"v222_filter_drops.csv",index=False)

    print("=== WEEK-1 ROSTER DIAGNOSTICS ===")
    print(roster_diagnostics(rosters).to_string(index=False))
    print("\n=== V2.2.2 ROSTER FILTER SUMMARY ===")
    print(summary.to_string(index=False))
    print("\n=== YEAR-BY-YEAR METRICS ===")
    print(metrics.sort_values(["mode","year"]).to_string(index=False))
    for mode in ["hard","veteran_only"]:
        d=drops[drops.mode.eq(mode)].sort_values(["target_year","predicted_value"],ascending=[True,False]).head(40)
        print(f"\n=== HIGHEST-VALUE PLAYERS REMOVED: {mode} ===")
        print(d[["target_year","player_name","position","age_next","predicted_value","actual_week_value"]].to_string(index=False))

if __name__ == "__main__":
    main()
