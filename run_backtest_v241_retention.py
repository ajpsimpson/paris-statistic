from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

import run_backtest_v21 as v21
import run_backtest_v211 as v211
import run_backtest_v22 as v22
import run_backtest_v223 as v223
import run_backtest_v23_rookies as v23
import run_backtest_v24_horizon as v24
from paris_v2_core import DEFAULT_CONFIG, assign_starters

POSITIONS = ["QB", "RB", "WR", "TE"]
ACTUAL_YEARS = list(range(2012, 2024))
FORECAST_YEARS = list(range(2016, 2024))
COMPLETE_EVAL = [2019, 2020, 2021]
DISCOUNTS = [0.50, 0.65, 0.80, 0.90, 1.00]
SHRINKS = [5.0, 15.0, 30.0, 60.0]
CAPS = [1.00, 1.25, 1.50]
STRENGTHS = [0.0, 0.25, 0.50, 0.75, 1.00]


def load_week1_sets(roster_dir: str, years: list[int]) -> dict[int, set[str]]:
    out: dict[int, set[str]] = {}
    for year in years:
        p = Path(roster_dir) / f"roster_weekly_{year}.parquet"
        df = pd.read_parquet(p)
        id_col = "gsis_id" if "gsis_id" in df.columns else "player_id"
        week = pd.to_numeric(df.week, errors="coerce")
        snap = df[week.eq(1)].copy()
        if "season" in snap.columns:
            snap = snap[pd.to_numeric(snap.season, errors="coerce").eq(year)]
        if "game_type" in snap.columns:
            reg = snap.game_type.astype(str).str.upper().eq("REG")
            if reg.any(): snap = snap[reg].copy()
        ids = snap[id_col].dropna().astype(str)
        out[year] = set(ids[~ids.isin(["nan", "None", ""])].unique())
    return out


def age_metadata(players: pd.DataFrame) -> pd.DataFrame:
    p = players.copy()
    p["player_id"] = p.gsis_id.astype(str)
    p["birth_date_parsed"] = pd.to_datetime(p.birth_date, errors="coerce")
    return p[["player_id", "birth_date_parsed"]].drop_duplicates("player_id")


def build_actual_annual(seasons: pd.DataFrame, eligible: dict[int, set[str]], ages: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year in ACTUAL_YEARS:
        s = seasons[seasons.season.eq(year) & seasons.position.isin(POSITIONS)].copy()
        s = s[s.player_id.astype(str).isin(eligible[year])].drop_duplicates("player_id")
        scheduled = 16.0 if year <= 2020 else 17.0
        s["actual_week_value"] = pd.to_numeric(s.paris_fp, errors="coerce").fillna(0.0) / scheduled
        _, repl = assign_starters(s[["player_id", "position", "actual_week_value"]], DEFAULT_CONFIG, "actual_week_value")
        s["actual_vor"] = s.actual_week_value - s.position.map(repl)
        s["actual_surplus"] = s.actual_vor.clip(lower=0)
        s = s.merge(ages, on="player_id", how="left")
        cutoff = pd.Timestamp(f"{year}-09-01")
        s["age"] = (cutoff - s.birth_date_parsed).dt.days / 365.25
        s["age_bucket"] = np.floor(s.age).clip(18, 45)
        rows.append(s[["season", "player_id", "position", "age", "age_bucket", "actual_vor", "actual_surplus"]])
    annual = pd.concat(rows, ignore_index=True)
    nxt = annual[["season", "player_id", "actual_surplus"]].copy()
    nxt["season"] = nxt.season - 1
    nxt = nxt.rename(columns={"actual_surplus": "next_surplus"})
    annual = annual.merge(nxt, on=["season", "player_id"], how="left")
    annual["next_surplus"] = annual.next_surplus.fillna(0.0)
    return annual


def retention_table(annual: pd.DataFrame, forecast_year: int, shrink: float, cap: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    t = annual[(annual.season.add(1).lt(forecast_year)) & annual.actual_surplus.gt(0)].copy()
    pos = t.groupby("position", as_index=False).agg(
        pos_current=("actual_surplus", "sum"), pos_next=("next_surplus", "sum"), pos_n=("player_id", "size")
    )
    pos["pos_ratio"] = (pos.pos_next / pos.pos_current.replace(0, np.nan)).fillna(0.0).clip(0, cap)

    grp = t.groupby(["position", "age_bucket"], as_index=False).agg(
        current_sum=("actual_surplus", "sum"), next_sum=("next_surplus", "sum"), n=("player_id", "size")
    )
    grp["raw_ratio"] = (grp.next_sum / grp.current_sum.replace(0, np.nan)).fillna(0.0).clip(0, cap)
    grp = grp.merge(pos[["position", "pos_ratio"]], on="position", how="left")
    grp["weight"] = grp.n / (grp.n + shrink)
    grp["retention"] = (grp.weight * grp.raw_ratio + (1-grp.weight) * grp.pos_ratio).clip(0, cap)
    return grp[["position", "age_bucket", "n", "raw_ratio", "pos_ratio", "retention"]], pos


def lookup_retention(df: pd.DataFrame, curves: pd.DataFrame, pos_fallback: pd.DataFrame, age_shift: int) -> pd.Series:
    temp = df[["position", "age"]].copy()
    temp["age_bucket"] = np.floor(temp.age + age_shift).clip(18, 45)
    temp["_idx"] = np.arange(len(temp))
    m = temp.merge(curves[["position", "age_bucket", "retention"]], on=["position", "age_bucket"], how="left")
    fb = pos_fallback.set_index("position").pos_ratio
    m["retention"] = m.retention.fillna(m.position.map(fb)).fillna(0.5)
    return m.sort_values("_idx").retention.reset_index(drop=True)


def target(df: pd.DataFrame, discount: float, horizon: int = 3) -> pd.Series:
    total = pd.Series(0.0, index=df.index)
    for k in range(horizon):
        total += (discount ** k) * df[f"actual_surplus_y{k+1}"]
    return total


def overlap(df: pd.DataFrame, pred_col: str, actual_col: str, k: int) -> float:
    k = min(k, len(df))
    return len(set(df.nlargest(k, pred_col).player_id) & set(df.nlargest(k, actual_col).player_id)) / max(k, 1)


def evaluate_candidate(table: pd.DataFrame, annual: pd.DataFrame, year: int, discount: float, shrink: float, cap: float, strength: float):
    g = table[table.forecast_year.eq(year)].copy().reset_index(drop=True)
    curves, pos = retention_table(annual, year, shrink, cap)
    r1 = lookup_retention(g, curves, pos, 0)
    r2 = lookup_retention(g, curves, pos, 1)
    full_multiplier = 1.0 + discount*r1 + (discount**2)*r1*r2
    multiplier = 1.0 + strength*(full_multiplier - 1.0)
    g["retention_y2"] = r1.to_numpy()
    g["retention_y3_conditional"] = r2.to_numpy()
    g["horizon_multiplier"] = multiplier.to_numpy()
    g["dynasty_score"] = g.pred_vor_pos * g.horizon_multiplier
    g["baseline_score"] = g.pred_vor_pos
    g["target"] = target(g, discount, 3)
    row = {
        "year": year, "discount": discount, "shrink": shrink, "cap": cap, "strength": strength, "n": len(g),
        "spearman": g.dynasty_score.corr(g.target, method="spearman"),
        "baseline_spearman": g.baseline_score.corr(g.target, method="spearman"),
        "top24": overlap(g,"dynasty_score","target",24), "top50": overlap(g,"dynasty_score","target",50),
        "top100": overlap(g,"dynasty_score","target",100), "top150": overlap(g,"dynasty_score","target",150),
        "baseline_top50": overlap(g,"baseline_score","target",50),
        "baseline_top100": overlap(g,"baseline_score","target",100),
    }
    g["discount"] = discount; g["shrink"] = shrink; g["cap"] = cap; g["strength"] = strength
    return row, g, curves


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    keys = ["discount","shrink","cap","strength"]
    pre = metrics[metrics.year.lt(2021)].groupby(keys,as_index=False).agg(
        pre_spearman=("spearman","mean"), pre_baseline=("baseline_spearman","mean"),
        pre_top50=("top50","mean"), pre_top100=("top100","mean")
    )
    val = metrics[metrics.year.eq(2021)][keys+["spearman","baseline_spearman","top50","top100","top150"]].rename(columns={
        "spearman":"val_spearman","baseline_spearman":"val_baseline","top50":"val_top50","top100":"val_top100","top150":"val_top150"})
    z = pre.merge(val,on=keys)
    z["pre_gain"] = z.pre_spearman-z.pre_baseline
    z["val_gain"] = z.val_spearman-z.val_baseline
    z["selection_score"] = .45*z.pre_spearman + .45*z.val_spearman + .05*z.pre_top50 + .05*z.val_top50
    return z.sort_values(["selection_score","val_gain","pre_gain"],ascending=False)


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument("--data-dir",required=True); ap.add_argument("--players-csv",required=True)
    ap.add_argument("--draft-parquet",required=True); ap.add_argument("--combine-parquet",required=True)
    ap.add_argument("--roster-dir",required=True); ap.add_argument("--depth-dir",required=True)
    ap.add_argument("--out-dir",default="outputs_v241_retention")
    args=ap.parse_args(); out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)

    raw=v21.old.load_weekly_csvs(args.data_dir); seasons=v21.aggregate_v21(raw)
    players=pd.read_csv(args.players_csv,low_memory=False); draft=pd.read_parquet(args.draft_parquet); combine=pd.read_parquet(args.combine_parquet)
    ages=age_metadata(players)
    eligible_all=load_week1_sets(args.roster_dir,ACTUAL_YEARS)
    eligible_forecast={y:eligible_all[y] for y in FORECAST_YEARS}

    rook=v23.rookie_universe(players,draft,combine); rook=v23.attach_preseason_role(rook,args.roster_dir,args.depth_dir); rook=v23.attach_outcomes(rook,seasons)
    vet_table=v211.augment_table(seasons,players); vet_table=v22.augment_targets(vet_table,seasons)
    depth,_=v223.load_preseason_depth(args.depth_dir); vet_table=v223.attach_depth(vet_table,depth)
    meta=v24.player_metadata(players)

    forecasts=[]
    for y in FORECAST_YEARS:
        forecasts.append(v24.build_forecast_pool(y,vet_table,rook,eligible_forecast,meta))
    forecasts=pd.concat(forecasts,ignore_index=True)
    annual=build_actual_annual(seasons,eligible_all,ages)
    table=v24.engineer_features(v24.add_horizon_targets(forecasts,annual))

    rows=[]; selected_rows=[]; curve_rows=[]
    for year,discount,shrink,cap,strength in product(COMPLETE_EVAL,DISCOUNTS,SHRINKS,CAPS,STRENGTHS):
        row,g,curves=evaluate_candidate(table,annual,year,discount,shrink,cap,strength)
        rows.append(row)
    metrics=pd.DataFrame(rows); summary=summarize(metrics); best=summary.iloc[0]

    for year in COMPLETE_EVAL+[2022,2023]:
        g=table[table.forecast_year.eq(year)].copy().reset_index(drop=True)
        curves,pos=retention_table(annual,year,float(best.shrink),float(best.cap))
        r1=lookup_retention(g,curves,pos,0); r2=lookup_retention(g,curves,pos,1)
        full=1+float(best.discount)*r1+(float(best.discount)**2)*r1*r2
        g["horizon_multiplier"]=1+float(best.strength)*(full-1)
        g["dynasty_score"]=g.pred_vor_pos*g.horizon_multiplier
        g["dynasty_rank"]=g.dynasty_score.rank(ascending=False,method="min")
        g["retention_y2"]=r1.to_numpy(); g["retention_y3_conditional"]=r2.to_numpy()
        selected_rows.append(g)
        curves=curves.copy(); curves["forecast_year"]=year; curve_rows.append(curves)
    selected=pd.concat(selected_rows,ignore_index=True); curves_df=pd.concat(curve_rows,ignore_index=True)

    chosen_metrics=metrics[(metrics.discount.eq(best.discount))&(metrics.shrink.eq(best.shrink))&(metrics.cap.eq(best.cap))&(metrics.strength.eq(best.strength))].sort_values("year")

    partial=[]
    for year,h in [(2022,2),(2023,1)]:
        g=selected[selected.forecast_year.eq(year)].copy(); g["partial_target"]=target(g,float(best.discount),h)
        partial.append({"year":year,"observed_horizon":h,"spearman":g.dynasty_score.corr(g.partial_target,method="spearman"),
                        "baseline_spearman":g.pred_vor_pos.corr(g.partial_target,method="spearman"),
                        "top50":overlap(g,"dynasty_score","partial_target",50),"baseline_top50":overlap(g,"pred_vor_pos","partial_target",50)})
    partial_df=pd.DataFrame(partial)

    pos_rows=[]
    for year in COMPLETE_EVAL:
        g=selected[selected.forecast_year.eq(year)].copy(); g["target"]=target(g,float(best.discount),3)
        for pos,s in g.groupby("position"):
            pos_rows.append({"year":year,"position":pos,"n":len(s),"spearman":s.dynasty_score.corr(s.target,method="spearman"),
                             "baseline_spearman":s.pred_vor_pos.corr(s.target,method="spearman")})
    pos_df=pd.DataFrame(pos_rows)

    metrics.to_csv(out/"v241_candidate_metrics.csv",index=False); summary.to_csv(out/"v241_candidate_summary.csv",index=False)
    chosen_metrics.to_csv(out/"v241_selected_complete_metrics.csv",index=False); partial_df.to_csv(out/"v241_partial_diagnostics.csv",index=False)
    curves_df.to_csv(out/"v241_retention_curves.csv",index=False); pos_df.to_csv(out/"v241_position_metrics.csv",index=False)
    selected.to_csv(out/"v241_selected_rankings.csv",index=False)

    print("=== V2.4.1 RETENTION CANDIDATES ==="); print(summary.head(30).to_string(index=False))
    print("\n=== SELECTED ==="); print(best.to_string())
    print("\n=== COMPLETE 3-YEAR METRICS ==="); print(chosen_metrics.to_string(index=False))
    print("\n=== POSITION METRICS ==="); print(pos_df.to_string(index=False))
    print("\n=== PARTIAL 2022/2023 DIAGNOSTICS ==="); print(partial_df.to_string(index=False))

    y21=selected[selected.forecast_year.eq(2021)].copy(); y21["target"]=target(y21,float(best.discount),3); y21=y21.sort_values("dynasty_rank").head(40)
    print("\n=== 2021 TOP 40 ==="); print(y21[["dynasty_rank","player_name","position","age","is_rookie","pred_vor","retention_y2","retention_y3_conditional","horizon_multiplier","dynasty_score","target"]].to_string(index=False))
    y23=selected[selected.forecast_year.eq(2023)].sort_values("dynasty_rank").head(40)
    print("\n=== 2023 PRESEASON DYNASTY TOP 40 (NO 2024/2025 OUTCOMES USED) ==="); print(y23[["dynasty_rank","player_name","position","age","is_rookie","pred_vor","retention_y2","retention_y3_conditional","horizon_multiplier","dynasty_score"]].to_string(index=False))

if __name__=="__main__":
    main()
