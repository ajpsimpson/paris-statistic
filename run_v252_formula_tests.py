from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import run_backtest_v21 as v21
import run_backtest_v211 as v211
import run_backtest_v22 as v22
import run_backtest_v223 as v223
import run_backtest_v241_retention as v241
import run_backtest_v243_top150 as v243
import run_v25_age_te_hypotheses as v25
from paris_long_td_bonus import apply_long_td_bonus, load_long_td_bonus

YEARS = list(range(2016, 2024))
COMPLETE_3Y = list(range(2016, 2022))
DEV_YEARS = [2016, 2017, 2018]
VAL_YEARS = [2019, 2020, 2021]
ACTUAL_YEARS = list(range(2012, 2024))
PBP_YEARS = list(range(2012, 2024))
DISCOUNT = .50

VARIANTS = [
    "baseline",
    "rb_age",
    "rb_age_workload",
    "rb_age_workload_elite",
    "rb_age_workload_elite_te_top8",
]


def retention_pipe(alpha: float = 30.0) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
        ("ridge", Ridge(alpha=alpha)),
    ])


def workload_history(seasons: pd.DataFrame) -> pd.DataFrame:
    s = seasons[[
        "season","player_id","position","carries","receptions","ppg","games"
    ]].copy()
    s["touches"] = pd.to_numeric(s.carries,errors="coerce").fillna(0) + pd.to_numeric(s.receptions,errors="coerce").fillna(0)
    s["pos_rank"] = s.groupby(["season","position"])["ppg"].rank(ascending=False,method="min")
    s["rb_elite"] = (s.position.eq("RB") & s.pos_rank.le(10)).astype(float)
    s["te_top8"] = (s.position.eq("TE") & s.pos_rank.le(8)).astype(float)

    # Exact-calendar three-year workload.
    base=s[["season","player_id","position","touches","rb_elite","te_top8"]].copy()
    out=base.copy()
    for lag,offset in [(2,1),(3,2)]:
        t=base[["season","player_id","touches","rb_elite","te_top8"]].copy()
        t["season"] += offset
        t=t.rename(columns={
            "touches":f"touches_l{lag}",
            "rb_elite":f"rb_elite_l{lag}",
            "te_top8":f"te_top8_l{lag}",
        })
        out=out.merge(t,on=["season","player_id"],how="left")
    out["touches_l1"]=out.touches
    out["touches_3yr"]=out[["touches_l1","touches_l2","touches_l3"]].fillna(0).sum(axis=1)
    out["touches_trend"]=out.touches_l1 - out.touches_l2.fillna(out.touches_l1)
    out["rb_elite_persist"]=(.5*out.rb_elite+.3*out.rb_elite_l2.fillna(0)+.2*out.rb_elite_l3.fillna(0))
    out["te_top8_persist"]=(.5*out.te_top8+.3*out.te_top8_l2.fillna(0)+.2*out.te_top8_l3.fillna(0))
    return out[[
        "season","player_id","position","touches_l1","touches_3yr","touches_trend",
        "rb_elite_persist","te_top8_persist"
    ]]


def build_transition_frame(annual: pd.DataFrame, wh: pd.DataFrame) -> pd.DataFrame:
    t=annual.merge(wh,on=["season","player_id","position"],how="left")
    t=t[t.actual_surplus.gt(0)].copy()
    t["retention_ratio"]=(t.next_surplus/t.actual_surplus.replace(0,np.nan)).fillna(0).clip(0,1)
    t["age_next"]=t.age+1.0
    t["age27"]=(t.age_next-27).clip(lower=0)
    t["age29"]=(t.age_next-29).clip(lower=0)
    t["log_touches3"]=np.log1p(t.touches_3yr.fillna(0))
    t["age27_x_logtouch"]=t.age27*t.log_touches3
    t["touches_trend_scaled"]=t.touches_trend.fillna(0)/100.0
    return t


def rb_feats(variant: str) -> list[str]:
    f=["age_next","age27","age29"]
    if variant in {"rb_age_workload","rb_age_workload_elite","rb_age_workload_elite_te_top8"}:
        f += ["log_touches3","age27_x_logtouch","touches_trend_scaled"]
    if variant in {"rb_age_workload_elite","rb_age_workload_elite_te_top8"}:
        f += ["rb_elite_persist"]
    return f


def te_feats() -> list[str]:
    return ["age_next","te_top8_persist"]


def predict_retention(train: pd.DataFrame, test: pd.DataFrame, feats: list[str], age_shift: int = 0) -> pd.Series:
    if len(train)<30 or len(test)==0:
        return pd.Series(.5,index=test.index,dtype=float)
    tr=train.copy()
    te=test.copy()
    if age_shift:
        te["age_next"]=pd.to_numeric(te.age_next,errors="coerce")+age_shift
        if "age27" in feats:
            te["age27"]=(te.age_next-27).clip(lower=0)
        if "age29" in feats:
            te["age29"]=(te.age_next-29).clip(lower=0)
        if "age27_x_logtouch" in feats:
            te["age27_x_logtouch"]=te.age27*te.log_touches3
    m=retention_pipe(30.0)
    m.fit(tr[feats],tr.retention_ratio)
    return pd.Series(np.clip(m.predict(te[feats]),0,1),index=test.index)


def enrich_predictions(pred: pd.DataFrame, wh: pd.DataFrame) -> pd.DataFrame:
    # Forecast year Y uses source-season Y-1 workload only.
    f=wh.copy()
    f["target_year"]=f.season+1
    f=f.rename(columns={
        "touches_3yr":"pred_touches_3yr",
        "touches_trend":"pred_touches_trend",
        "rb_elite_persist":"pred_rb_elite_persist",
        "te_top8_persist":"pred_te_top8_persist",
    })
    out=pred.merge(
        f[["target_year","player_id","pred_touches_3yr","pred_touches_trend","pred_rb_elite_persist","pred_te_top8_persist"]],
        on=["target_year","player_id"],how="left"
    )
    out["age_next_for_ret"]=pd.to_numeric(out.age,errors="coerce")
    out["age27"]=(out.age_next_for_ret-27).clip(lower=0)
    out["age29"]=(out.age_next_for_ret-29).clip(lower=0)
    out["log_touches3"]=np.log1p(out.pred_touches_3yr.fillna(0))
    out["age27_x_logtouch"]=out.age27*out.log_touches3
    out["touches_trend_scaled"]=out.pred_touches_trend.fillna(0)/100.0
    out["rb_elite_persist"]=out.pred_rb_elite_persist.fillna(0)
    out["te_top8_persist"]=out.pred_te_top8_persist.fillna(0)
    out["age_next"]=out.age_next_for_ret
    return out


def apply_variant(pred: pd.DataFrame, annual: pd.DataFrame, trans: pd.DataFrame, variant: str) -> pd.DataFrame:
    rows=[]
    for year,g0 in pred.groupby("target_year"):
        g=g0.copy().reset_index(drop=True)

        curves,posfb=v241.retention_table(annual,int(year),5.0,1.0)
        r1=v241.lookup_retention(g,curves,posfb,0)
        r2=v241.lookup_retention(g,curves,posfb,1)

        hist=trans[trans.season.add(1).lt(year)].copy()

        if variant!="baseline":
            rbidx=g.index[g.position.eq("RB")]
            if len(rbidx):
                tr=hist[hist.position.eq("RB")].copy()
                feats=rb_feats(variant)
                te=g.loc[rbidx].copy()
                rr1=predict_retention(tr,te,feats,0)
                rr2=predict_retention(tr,te,feats,1)
                r1.loc[rbidx]=rr1.to_numpy()
                r2.loc[rbidx]=rr2.to_numpy()

        if variant=="rb_age_workload_elite_te_top8":
            teidx=g.index[g.position.eq("TE")]
            if len(teidx):
                tr=hist[hist.position.eq("TE")].copy()
                te=g.loc[teidx].copy()
                tr=tr.copy()
                # Align transition-frame naming.
                tr["te_top8_persist"]=tr.te_top8_persist.fillna(0)
                tt1=predict_retention(tr,te,te_feats(),0)
                tt2=predict_retention(tr,te,te_feats(),1)
                r1.loc[teidx]=tt1.to_numpy()
                r2.loc[teidx]=tt2.to_numpy()

        g["r1"]=r1.to_numpy(); g["r2"]=r2.to_numpy()
        g["horizon_multiplier"]=1+DISCOUNT*r1+(DISCOUNT**2)*r1*r2
        g["dynasty_signal"]=v243.dynasty_signal(g.pred_vor,g.horizon_multiplier,"signed")
        g["variant"]=variant
        rows.append(g)
    return pd.concat(rows,ignore_index=True)


def attach_targets(pred: pd.DataFrame, annual: pd.DataFrame) -> pd.DataFrame:
    out=pred.copy()
    for k in [0,1,2]:
        a=annual[["season","player_id","actual_surplus","actual_vor"]].copy()
        a["target_year"]=a.season-k
        a=a.rename(columns={"actual_surplus":f"s{k+1}","actual_vor":f"v{k+1}"})
        out=out.merge(a[["target_year","player_id",f"s{k+1}",f"v{k+1}"]],on=["target_year","player_id"],how="left")
        out[f"s{k+1}"]=out[f"s{k+1}"].fillna(0)
        out[f"v{k+1}"]=out[f"v{k+1}"].fillna(0)
    out["target3"]=out.s1+DISCOUNT*out.s2+(DISCOUNT**2)*out.s3
    return out


def overlap(g,k):
    kk=min(k,len(g))
    return len(set(g.nlargest(kk,"dynasty_signal").player_id)&set(g.nlargest(kk,"target3").player_id))/max(kk,1)


def metrics(pred: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for (variant,year),g in pred.groupby(["variant","target_year"]):
        row={
            "variant":variant,"year":int(year),"n":len(g),
            "one_year_spearman":g.pred_vor.corr(g.v1,method="spearman"),
        }
        if year in COMPLETE_3Y:
            row.update({
                "three_year_spearman":g.dynasty_signal.corr(g.target3,method="spearman"),
                "rb_three_year_spearman":g[g.position.eq("RB")].dynasty_signal.corr(g[g.position.eq("RB")].target3,method="spearman"),
                "te_three_year_spearman":g[g.position.eq("TE")].dynasty_signal.corr(g[g.position.eq("TE")].target3,method="spearman"),
                "top50":overlap(g,50),"top100":overlap(g,100),"top150":overlap(g,150),
            })
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(m: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for variant,g in m.groupby("variant"):
        dev=g[g.year.isin(DEV_YEARS)]
        val=g[g.year.isin(VAL_YEARS)]
        rows.append({
            "variant":variant,
            "dev_overall":dev.three_year_spearman.mean(),
            "dev_rb":dev.rb_three_year_spearman.mean(),
            "dev_te":dev.te_three_year_spearman.mean(),
            "dev_top150":dev.top150.mean(),
            "val_overall":val.three_year_spearman.mean(),
            "val_rb":val.rb_three_year_spearman.mean(),
            "val_te":val.te_three_year_spearman.mean(),
            "val_top150":val.top150.mean(),
        })
    z=pd.DataFrame(rows)
    z["dev_score"]=.50*z.dev_overall+.30*z.dev_rb+.10*z.dev_te+.10*z.dev_top150
    return z.sort_values("dev_score",ascending=False)


def coef_table(trans: pd.DataFrame, cutoff_year: int, variant: str) -> pd.DataFrame:
    rows=[]
    hist=trans[trans.season.add(1).lt(cutoff_year)].copy()
    for pos,feats in [("RB",rb_feats(variant)),("TE",te_feats())]:
        if pos=="TE" and variant!="rb_age_workload_elite_te_top8": continue
        tr=hist[hist.position.eq(pos)].copy()
        if len(tr)<30: continue
        m=retention_pipe(30.0); m.fit(tr[feats],tr.retention_ratio)
        ridge=m.named_steps["ridge"]; scale=m.named_steps["scale"]
        for f,c in zip(feats,ridge.coef_[:len(feats)]):
            rows.append({"position":pos,"feature":f,"standardized_coefficient":c,"cutoff_year":cutoff_year,"variant":variant})
    return pd.DataFrame(rows)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data-dir",required=True)
    ap.add_argument("--players-csv",required=True)
    ap.add_argument("--roster-dir",required=True)
    ap.add_argument("--depth-dir",required=True)
    ap.add_argument("--pbp-dir",required=True)
    ap.add_argument("--out-dir",default="outputs_v252_formula")
    args=ap.parse_args(); out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)

    raw=v21.old.load_weekly_csvs(args.data_dir)
    base=v21.aggregate_v21(raw)
    bonus,_=load_long_td_bonus(args.pbp_dir,PBP_YEARS,threshold=50)
    seasons=apply_long_td_bonus(base,bonus)
    players=pd.read_csv(args.players_csv,low_memory=False)

    table=v211.augment_table(seasons,players)
    table=v22.augment_targets(table,seasons)
    v223.DEPTH_YEARS=list(range(2016,2024))
    depth,_=v223.load_preseason_depth(args.depth_dir)
    table=v223.attach_depth(table,depth)
    table=v25.augment_experimental(table,seasons)

    eligible=v241.load_week1_sets(args.roster_dir,ACTUAL_YEARS)
    ages=v241.age_metadata(players)
    annual=v241.build_actual_annual(seasons,eligible,ages)
    wh=workload_history(seasons)
    trans=build_transition_frame(annual,wh)

    # Frozen one-year model. Only longevity math changes.
    pred=v25.build_prediction_pool(table,eligible,"baseline","baseline")
    pred=enrich_predictions(pred,wh)

    allp=[]
    for variant in VARIANTS:
        x=apply_variant(pred,annual,trans,variant)
        x=attach_targets(x,annual)
        allp.append(x)
    allp=pd.concat(allp,ignore_index=True)
    m=metrics(allp)
    summary=summarize(m)

    selected=str(summary.iloc[0].variant)
    coefs=coef_table(trans,2019,selected)

    m.to_csv(out/"v252_yearly_metrics.csv",index=False)
    summary.to_csv(out/"v252_summary.csv",index=False)
    coefs.to_csv(out/"v252_selected_coefficients.csv",index=False)
    allp.to_csv(out/"v252_predictions.csv",index=False)

    print("=== PARIS v2.5.2 FORMULA TEST ===")
    print("Development 2016-18; validation 2019-21. 2024-25 are not used.")
    print("\n=== SUMMARY ===")
    print(summary.to_string(index=False))
    print("\nDevelopment-selected:",selected)
    print("\n=== YEARLY BASELINE ===")
    print(m[m.variant.eq("baseline")].to_string(index=False))
    print("\n=== YEARLY SELECTED ===")
    print(m[m.variant.eq(selected)].to_string(index=False))
    print("\n=== SELECTED RETENTION COEFFICIENTS (trained only before 2019) ===")
    print(coefs.to_string(index=False))


if __name__=="__main__":
    main()
