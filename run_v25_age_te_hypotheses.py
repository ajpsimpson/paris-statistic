from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import run_backtest_v21 as v21
import run_backtest_v211 as v211
import run_backtest_v22 as v22
import run_backtest_v223 as v223
import run_backtest_v241_retention as v241
import run_backtest_v243_top150 as v243
from paris_long_td_bonus import apply_long_td_bonus, load_long_td_bonus
from paris_v2_core import DEFAULT_CONFIG, assign_starters

YEARS = list(range(2016, 2024))
COMPLETE_3Y = list(range(2016, 2022))
DEV_YEARS = [2016, 2017, 2018]
VALID_YEARS = [2019, 2020, 2021]
PBP_YEARS = list(range(2012, 2024))
ACTUAL_YEARS = list(range(2012, 2024))
POSITIONS = ["QB", "RB", "WR", "TE"]
DISCOUNT = 0.50
SHRINK = 5.0
CAP = 1.0

RB_VARIANTS = ["baseline", "steep_age", "steep_age_elite"]
TE_VARIANTS = ["baseline", "top8", "top8_age"]

BASE_CFG = v22.PPG_CONFIG


def ridge(alpha: float) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
        ("ridge", Ridge(alpha=alpha)),
    ])


def logit() -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
        ("logit", LogisticRegression(C=0.1, class_weight="balanced", max_iter=3000)),
    ])


def augment_experimental(table: pd.DataFrame, seasons: pd.DataFrame) -> pd.DataFrame:
    out = table.copy()

    # RB aging basis: deliberately piecewise rather than a fixed manual deduction.
    # Coefficients are learned from prior seasons; the knots merely let the slope
    # change as RBs move through their late 20s.
    out["rb_age25"] = (out.age_next - 25.0).clip(lower=0.0)
    out["rb_age27"] = (out.age_next - 27.0).clip(lower=0.0)
    out["rb_age29"] = (out.age_next - 29.0).clip(lower=0.0)
    out["rb_old_elite"] = out.rb_age27 * out.elite_tier.fillna(0.0)

    # TE top-8 persistence: reward repeated membership in the narrow upper tier.
    te = seasons[["season", "player_id", "position", "ppg"]].copy()
    te["pos_rank"] = te.groupby(["season", "position"])["ppg"].rank(ascending=False, method="min")
    te["top8"] = (te.pos_rank.le(8) & te.position.eq("TE")).astype(float)
    for lag, offset in [(1, 0), (2, 1), (3, 2)]:
        t = te[["season", "player_id", "top8"]].copy()
        t["season"] += offset
        t = t.rename(columns={"top8": f"te_top8_lag{lag}"})
        out = out.merge(t, on=["season", "player_id"], how="left")
    out["te_top8_persist"] = (
        0.50*out.te_top8_lag1.fillna(0.0)
        + 0.30*out.te_top8_lag2.fillna(0.0)
        + 0.20*out.te_top8_lag3.fillna(0.0)
    )
    out["te_top8_age"] = out.te_top8_persist * out.age_next
    return out


def ppg_features(pos: str, rb_variant: str, te_variant: str) -> list[str]:
    cfg = BASE_CFG[pos]
    feats = list(v211.features(pos, cfg["age_mode"], cfg["opp_mode"], cfg["elite_mode"]))
    if pos == "RB" and rb_variant in {"steep_age", "steep_age_elite"}:
        feats += ["rb_age25", "rb_age27", "rb_age29"]
    if pos == "RB" and rb_variant == "steep_age_elite":
        feats += ["rb_old_elite"]
    if pos == "TE" and te_variant in {"top8", "top8_age"}:
        feats += ["te_top8_persist"]
    if pos == "TE" and te_variant == "top8_age":
        feats += ["te_top8_age"]
    return list(dict.fromkeys(feats))


def role_features(pos: str, rb_variant: str, te_variant: str) -> list[str]:
    feats = list(v22.role_features(pos) + v223.DEPTH_FEATURES)
    if pos == "RB" and rb_variant in {"steep_age", "steep_age_elite"}:
        feats += ["rb_age25", "rb_age27", "rb_age29"]
    if pos == "RB" and rb_variant == "steep_age_elite":
        feats += ["rb_old_elite"]
    if pos == "TE" and te_variant in {"top8", "top8_age"}:
        feats += ["te_top8_persist"]
    if pos == "TE" and te_variant == "top8_age":
        feats += ["te_top8_age"]
    return list(dict.fromkeys(feats))


def role_q(pos: str) -> float:
    return 0.55 if pos == "QB" else 0.25


def fit_year_pos(table: pd.DataFrame, year: int, pos: str, rb_variant: str, te_variant: str) -> pd.DataFrame:
    cfg = BASE_CFG[pos]
    pf = ppg_features(pos, rb_variant, te_variant)
    train_ppg = table[
        table.position.eq(pos) & table.target_year.lt(year) & table.target_ppg.notna()
    ].copy()
    test = table[table.position.eq(pos) & table.target_year.eq(year)].copy()
    if len(train_ppg) < 30 or len(test) < 5:
        return pd.DataFrame()

    pm = ridge(float(cfg["alpha"]))
    pm.fit(train_ppg[pf], train_ppg.target_ppg)
    test["predicted_ppg"] = pm.predict(test[pf])

    q = role_q(pos)
    rt = v22.add_role_label(table, q)
    rf = role_features(pos, rb_variant, te_variant)
    train_role = rt[rt.position.eq(pos) & rt.target_year.lt(year)].copy()
    if len(train_role) < 30 or train_role.role_target.nunique() < 2:
        return pd.DataFrame()
    rm = logit()
    rm.fit(train_role[rf], train_role.role_target)
    test["role_prob"] = rm.predict_proba(test[rf])[:, 1]
    test["predicted_value"] = test.predicted_ppg * test.role_prob
    return test


def build_prediction_pool(table: pd.DataFrame, eligible: dict[int, set[str]], rb_variant: str, te_variant: str) -> pd.DataFrame:
    rows = []
    for year in YEARS:
        parts = []
        for pos in POSITIONS:
            p = fit_year_pos(table, year, pos, rb_variant, te_variant)
            if len(p):
                parts.append(p)
        if not parts:
            continue
        g = pd.concat(parts, ignore_index=True)
        g = g[g.player_id.astype(str).isin(eligible[year])].drop_duplicates("player_id").copy()
        _, repl = assign_starters(g[["player_id","position","predicted_value"]], DEFAULT_CONFIG, "predicted_value")
        g["pred_vor"] = g.predicted_value - g.position.map(repl)
        g["rb_variant"] = rb_variant
        g["te_variant"] = te_variant
        rows.append(g)
    return pd.concat(rows, ignore_index=True)


def add_dynasty_signal(pred: pd.DataFrame, annual: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year, g0 in pred.groupby("target_year"):
        g = g0.copy().reset_index(drop=True)
        curves, pos = v241.retention_table(annual, int(year), SHRINK, CAP)
        r1 = v241.lookup_retention(g, curves, pos, 0)
        r2 = v241.lookup_retention(g, curves, pos, 1)
        g["horizon_multiplier"] = 1.0 + DISCOUNT*r1 + (DISCOUNT**2)*r1*r2
        g["dynasty_signal"] = v243.dynasty_signal(g.pred_vor, g.horizon_multiplier, "signed")
        rows.append(g)
    return pd.concat(rows, ignore_index=True)


def attach_targets(pred: pd.DataFrame, annual: pd.DataFrame) -> pd.DataFrame:
    out = pred.copy()
    for k in [0,1,2]:
        a = annual[["season","player_id","actual_vor","actual_surplus"]].copy()
        a["target_year"] = a.season - k
        a = a.rename(columns={
            "actual_vor":f"actual_vor_y{k+1}",
            "actual_surplus":f"actual_surplus_y{k+1}",
        })
        out = out.merge(a[["target_year","player_id",f"actual_vor_y{k+1}",f"actual_surplus_y{k+1}"]],
                        on=["target_year","player_id"],how="left")
        out[f"actual_vor_y{k+1}"] = out[f"actual_vor_y{k+1}"].fillna(0.0)
        out[f"actual_surplus_y{k+1}"] = out[f"actual_surplus_y{k+1}"].fillna(0.0)
    out["target3"] = out.actual_surplus_y1 + DISCOUNT*out.actual_surplus_y2 + (DISCOUNT**2)*out.actual_surplus_y3
    return out


def ov(g: pd.DataFrame, k: int) -> float:
    kk = min(k, len(g))
    if kk == 0: return np.nan
    return len(set(g.nlargest(kk,"dynasty_signal").player_id) & set(g.nlargest(kk,"target3").player_id))/kk


def summarize_variant(g: pd.DataFrame) -> list[dict]:
    rows=[]
    for year in YEARS:
        y=g[g.target_year.eq(year)].copy()
        if len(y)==0: continue
        row={
            "year":year,
            "rb_variant":str(y.rb_variant.iloc[0]),
            "te_variant":str(y.te_variant.iloc[0]),
            "n":len(y),
            "one_year_spearman":y.pred_vor.corr(y.actual_vor_y1,method="spearman"),
        }
        for pos in ["RB","TE"]:
            p=y[y.position.eq(pos)]
            row[f"{pos.lower()}_one_year_spearman"]=p.pred_vor.corr(p.actual_vor_y1,method="spearman")
        if year in COMPLETE_3Y:
            row["three_year_spearman"]=y.dynasty_signal.corr(y.target3,method="spearman")
            row["rb_three_year_spearman"]=y[y.position.eq("RB")].dynasty_signal.corr(
                y[y.position.eq("RB")].target3,method="spearman")
            row["te_three_year_spearman"]=y[y.position.eq("TE")].dynasty_signal.corr(
                y[y.position.eq("TE")].target3,method="spearman")
            row["top50_3y"]=ov(y,50)
            row["top100_3y"]=ov(y,100)
            row["top150_3y"]=ov(y,150)
        rows.append(row)
    return rows


def aggregate_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    keys=["rb_variant","te_variant"]
    dev=metrics[metrics.year.isin(DEV_YEARS)].groupby(keys,as_index=False).agg(
        dev_3y=("three_year_spearman","mean"),
        dev_rb_3y=("rb_three_year_spearman","mean"),
        dev_te_3y=("te_three_year_spearman","mean"),
        dev_top150=("top150_3y","mean"),
    )
    val=metrics[metrics.year.isin(VALID_YEARS)].groupby(keys,as_index=False).agg(
        val_3y=("three_year_spearman","mean"),
        val_rb_3y=("rb_three_year_spearman","mean"),
        val_te_3y=("te_three_year_spearman","mean"),
        val_top150=("top150_3y","mean"),
    )
    recent=metrics[metrics.year.isin([2022,2023])].groupby(keys,as_index=False).agg(
        recent_one_year=("one_year_spearman","mean"),
        recent_rb_one_year=("rb_one_year_spearman","mean"),
        recent_te_one_year=("te_one_year_spearman","mean"),
    )
    z=dev.merge(val,on=keys).merge(recent,on=keys)
    # Diagnostic score: develop on 2016-18; newer years are reported as confirmation,
    # not used to select the experiment.
    z["dev_score"] = .40*z.dev_3y + .30*z.dev_rb_3y + .20*z.dev_te_3y + .10*z.dev_top150
    z["dev_rank"] = z.dev_score.rank(ascending=False,method="min")
    return z.sort_values(["dev_rank","val_3y"],ascending=[True,False])


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data-dir",required=True)
    ap.add_argument("--players-csv",required=True)
    ap.add_argument("--roster-dir",required=True)
    ap.add_argument("--depth-dir",required=True)
    ap.add_argument("--pbp-dir",required=True)
    ap.add_argument("--out-dir",default="outputs_v25_age_te")
    args=ap.parse_args()
    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)

    raw=v21.old.load_weekly_csvs(args.data_dir)
    base=v21.aggregate_v21(raw)
    bonus,bonus_diag=load_long_td_bonus(args.pbp_dir,PBP_YEARS,threshold=50)
    seasons=apply_long_td_bonus(base,bonus)
    players=pd.read_csv(args.players_csv,low_memory=False)

    # Extend the frozen table back to the older diagnostic years.
    table=v211.augment_table(seasons,players)
    table=v22.augment_targets(table,seasons)
    v223.DEPTH_YEARS=list(range(2016,2024))
    depth,depth_diag=v223.load_preseason_depth(args.depth_dir)
    table=v223.attach_depth(table,depth)
    table=augment_experimental(table,seasons)

    eligible=v241.load_week1_sets(args.roster_dir,ACTUAL_YEARS)
    ages=v241.age_metadata(players)
    annual=v241.build_actual_annual(seasons,eligible,ages)

    all_metrics=[]
    all_preds=[]
    for rbv,tev in product(RB_VARIANTS,TE_VARIANTS):
        p=build_prediction_pool(table,eligible,rbv,tev)
        p=add_dynasty_signal(p,annual)
        p=attach_targets(p,annual)
        all_preds.append(p)
        all_metrics.extend(summarize_variant(p))

    metrics=pd.DataFrame(all_metrics)
    summary=aggregate_summary(metrics)
    preds=pd.concat(all_preds,ignore_index=True)

    # Focused age diagnostics for development-selected RB variant versus baseline.
    best=summary.iloc[0]
    compare=preds[
        ((preds.rb_variant.eq("baseline") & preds.te_variant.eq("baseline"))
        | (preds.rb_variant.eq(best.rb_variant) & preds.te_variant.eq(best.te_variant)))
        & preds.target_year.isin(DEV_YEARS+VALID_YEARS)
    ].copy()

    metrics.to_csv(out/"v25_yearly_metrics.csv",index=False)
    summary.to_csv(out/"v25_variant_summary.csv",index=False)
    preds.to_csv(out/"v25_all_predictions.csv",index=False)
    compare.to_csv(out/"v25_selected_vs_baseline.csv",index=False)
    bonus_diag.to_csv(out/"v25_long_td_diagnostics.csv",index=False)
    depth_diag.to_csv(out/"v25_depth_diagnostics.csv",index=False)

    print("=== PARIS v2.5 HYPOTHESIS TEST: RB AGE + TE ELITE PERSISTENCE ===")
    print("Development years: 2016-2018. Validation years: 2019-2021. 2022-23 one-year diagnostics only.")
    print("\n=== VARIANT SUMMARY ===")
    print(summary.to_string(index=False))
    print("\n=== YEARLY METRICS FOR BASELINE ===")
    print(metrics[(metrics.rb_variant=="baseline")&(metrics.te_variant=="baseline")].to_string(index=False))
    print("\n=== DEVELOPMENT-SELECTED VARIANT ===")
    print(best.to_string())
    print("\n=== YEARLY METRICS FOR SELECTED ===")
    print(metrics[(metrics.rb_variant==best.rb_variant)&(metrics.te_variant==best.te_variant)].to_string(index=False))


if __name__=="__main__":
    main()
