from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

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
VALID_YEARS = [2019, 2020, 2021]
PBP_YEARS = list(range(2012, 2024))
ACTUAL_YEARS = list(range(2012, 2024))
DISCOUNT = 0.50
SHRINK = 8.0
MODES = ["baseline", "conditional_ratio", "conditional_survival", "conditional_hybrid"]


def add_transition_signals(seasons: pd.DataFrame, annual: pd.DataFrame) -> pd.DataFrame:
    s = seasons[["season","player_id","position","ppg"]].copy()
    s["pos_rank"] = s.groupby(["season","position"])["ppg"].rank(ascending=False, method="min")
    s["rb_elite"] = (s.position.eq("RB") & s.pos_rank.le(10)).astype(float)
    s["te_top8"] = (s.position.eq("TE") & s.pos_rank.le(8)).astype(float)

    # Weighted TE top-8 history known at the end of each source season.
    base = s[["season","player_id","te_top8"]].copy()
    hist = s[["season","player_id","position","rb_elite"]].copy()
    for lag, offset, wt in [(1,0,.50),(2,1,.30),(3,2,.20)]:
        t = base.copy()
        t["season"] += offset
        t = t.rename(columns={"te_top8":f"te8_l{lag}"})
        hist = hist.merge(t,on=["season","player_id"],how="left")
    hist["te8_persist"] = .50*hist.te8_l1.fillna(0)+.30*hist.te8_l2.fillna(0)+.20*hist.te8_l3.fillna(0)
    hist["te_established"] = (hist.te8_persist.ge(.50)).astype(float)

    out = annual.merge(
        hist[["season","player_id","rb_elite","te8_persist","te_established"]],
        on=["season","player_id"], how="left"
    )
    out[["rb_elite","te8_persist","te_established"]] = out[["rb_elite","te8_persist","te_established"]].fillna(0.0)
    return out


def rb_band(age: pd.Series) -> pd.Series:
    return pd.cut(age, [-np.inf,24.999,26.999,28.999,30.999,np.inf],
                  labels=["<=24","25-26","27-28","29-30","31+"])


def te_band(age: pd.Series) -> pd.Series:
    return pd.cut(age, [-np.inf,24.999,27.999,30.999,np.inf],
                  labels=["<=24","25-27","28-30","31+"])


def historical_tables(trans: pd.DataFrame, forecast_year: int, mode: str):
    t = trans[(trans.season.add(1).lt(forecast_year)) & trans.actual_surplus.gt(0)].copy()
    t["next_positive"] = t.next_surplus.gt(0).astype(float)
    t["rb_age_band"] = rb_band(t.age + 1.0)
    t["te_age_band"] = te_band(t.age + 1.0)

    tables = {}
    for pos in ["RB","TE"]:
        x=t[t.position.eq(pos)].copy()
        band="rb_age_band" if pos=="RB" else "te_age_band"
        elite="rb_elite" if pos=="RB" else "te_established"

        age=x.groupby(band, observed=False).agg(
            n_age=("player_id","size"),
            current_sum=("actual_surplus","sum"),
            next_sum=("next_surplus","sum"),
            survival_age=("next_positive","mean"),
        ).reset_index()
        age["ratio_age"]=(age.next_sum/age.current_sum.replace(0,np.nan)).fillna(0).clip(0,1)

        grp=x.groupby([band,elite], observed=False).agg(
            n=("player_id","size"),
            current_sum=("actual_surplus","sum"),
            next_sum=("next_surplus","sum"),
            survival_group=("next_positive","mean"),
        ).reset_index()
        grp["ratio_group"]=(grp.next_sum/grp.current_sum.replace(0,np.nan)).fillna(0).clip(0,1)
        grp=grp.merge(age[[band,"ratio_age","survival_age"]],on=band,how="left")
        grp["weight"]=grp.n/(grp.n+SHRINK)
        grp["ratio_shrunk"]=(grp.weight*grp.ratio_group+(1-grp.weight)*grp.ratio_age).clip(0,1)
        grp["survival_shrunk"]=(grp.weight*grp.survival_group+(1-grp.weight)*grp.survival_age).clip(0,1)
        grp["hybrid_shrunk"]=.5*grp.ratio_shrunk+.5*grp.survival_shrunk
        tables[pos]=(grp, age)
    return tables


def lookup_custom(g: pd.DataFrame, tables, pos: str, age_shift: int, mode: str) -> pd.Series:
    x=g[g.position.eq(pos)].copy()
    if len(x)==0: return pd.Series(dtype=float)
    age=pd.to_numeric(x.age,errors="coerce")+age_shift
    band=rb_band(age) if pos=="RB" else te_band(age)
    elite=(x.elite_tier.ge(.50)).astype(float) if pos=="RB" else (x.te_top8_persist.ge(.50)).astype(float)

    # For RBs, do not carry an "elite exception" into age 29+; the historical
    # sample shows very poor retention there and the exception sample is tiny.
    if pos=="RB":
        elite=np.where(age.ge(29),0.0,elite)

    temp=pd.DataFrame({"_idx":x.index,"band":band.astype(str),"elite":elite})
    grp,age_tbl=tables[pos]
    band_col="rb_age_band" if pos=="RB" else "te_age_band"
    gg=grp.rename(columns={band_col:"band", ("rb_elite" if pos=="RB" else "te_established"):"elite"}).copy()
    gg["band"]=gg.band.astype(str)
    a=age_tbl.rename(columns={band_col:"band"}).copy(); a["band"]=a.band.astype(str)
    temp=temp.merge(gg[["band","elite","ratio_shrunk","survival_shrunk","hybrid_shrunk"]],on=["band","elite"],how="left")
    temp=temp.merge(a[["band","ratio_age","survival_age"]],on="band",how="left")
    col={"conditional_ratio":"ratio_shrunk","conditional_survival":"survival_shrunk","conditional_hybrid":"hybrid_shrunk"}[mode]
    fallback={"conditional_ratio":"ratio_age","conditional_survival":"survival_age","conditional_hybrid":None}[mode]
    vals=temp[col]
    if fallback:
        vals=vals.fillna(temp[fallback])
    else:
        vals=vals.fillna(.5*(temp.ratio_age+temp.survival_age))
    vals=vals.fillna(.5).clip(0,1)
    return pd.Series(vals.to_numpy(),index=x.index)


def add_signal(pred: pd.DataFrame, annual: pd.DataFrame, trans: pd.DataFrame, mode: str) -> pd.DataFrame:
    rows=[]
    for year,g0 in pred.groupby("target_year"):
        g=g0.copy().reset_index(drop=True)
        curves,posfb=v241.retention_table(annual,int(year),5.0,1.0)
        r1=v241.lookup_retention(g,curves,posfb,0)
        r2=v241.lookup_retention(g,curves,posfb,1)
        if mode!="baseline":
            tables=historical_tables(trans,int(year),mode)
            for pos in ["RB","TE"]:
                i=g.position.eq(pos)
                c1=lookup_custom(g,tables,pos,0,mode)
                c2=lookup_custom(g,tables,pos,1,mode)
                r1.loc[c1.index]=c1
                r2.loc[c2.index]=c2
        g["r1"]=r1.to_numpy(); g["r2"]=r2.to_numpy()
        g["horizon_multiplier"]=1+DISCOUNT*r1+(DISCOUNT**2)*r1*r2
        g["dynasty_signal"]=v243.dynasty_signal(g.pred_vor,g.horizon_multiplier,"signed")
        g["mode"]=mode
        rows.append(g)
    return pd.concat(rows,ignore_index=True)


def attach_targets(pred: pd.DataFrame, annual: pd.DataFrame) -> pd.DataFrame:
    out=pred.copy()
    for k in [0,1,2]:
        a=annual[["season","player_id","actual_surplus","actual_vor"]].copy()
        a["target_year"]=a.season-k
        a=a.rename(columns={"actual_surplus":f"surplus_y{k+1}","actual_vor":f"vor_y{k+1}"})
        out=out.merge(a[["target_year","player_id",f"surplus_y{k+1}",f"vor_y{k+1}"]],
                      on=["target_year","player_id"],how="left")
        out[f"surplus_y{k+1}"]=out[f"surplus_y{k+1}"].fillna(0.0)
        out[f"vor_y{k+1}"]=out[f"vor_y{k+1}"].fillna(0.0)
    out["target3"]=out.surplus_y1+DISCOUNT*out.surplus_y2+(DISCOUNT**2)*out.surplus_y3
    return out


def overlap(g,k):
    kk=min(k,len(g))
    return len(set(g.nlargest(kk,"dynasty_signal").player_id)&set(g.nlargest(kk,"target3").player_id))/kk


def metrics_for(g):
    rows=[]
    for year,y in g.groupby("target_year"):
        row={"year":int(year),"mode":str(y["mode"].iloc[0]),"n":len(y),
             "one_year_spearman":y.pred_vor.corr(y.vor_y1,method="spearman")}
        if year in COMPLETE_3Y:
            row.update({
                "three_year_spearman":y.dynasty_signal.corr(y.target3,method="spearman"),
                "rb_3y":y[y.position.eq("RB")].dynasty_signal.corr(y[y.position.eq("RB")].target3,method="spearman"),
                "te_3y":y[y.position.eq("TE")].dynasty_signal.corr(y[y.position.eq("TE")].target3,method="spearman"),
                "top50":overlap(y,50),"top100":overlap(y,100),"top150":overlap(y,150)
            })
        rows.append(row)
    return rows


def summarize(m):
    parts=[]
    for mode,g in m.groupby("mode"):
        dev=g[g.year.isin(DEV_YEARS)]
        val=g[g.year.isin(VALID_YEARS)]
        recent=g[g.year.isin([2022,2023])]
        parts.append({
            "mode":mode,
            "dev_3y":dev.three_year_spearman.mean(),
            "dev_rb":dev.rb_3y.mean(),
            "dev_te":dev.te_3y.mean(),
            "dev_top150":dev.top150.mean(),
            "val_3y":val.three_year_spearman.mean(),
            "val_rb":val.rb_3y.mean(),
            "val_te":val.te_3y.mean(),
            "val_top150":val.top150.mean(),
            "recent_one_year":recent.one_year_spearman.mean(),
        })
    z=pd.DataFrame(parts)
    z["dev_score"]=.45*z.dev_3y+.25*z.dev_rb+.20*z.dev_te+.10*z.dev_top150
    return z.sort_values("dev_score",ascending=False)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data-dir",required=True)
    ap.add_argument("--players-csv",required=True)
    ap.add_argument("--roster-dir",required=True)
    ap.add_argument("--depth-dir",required=True)
    ap.add_argument("--pbp-dir",required=True)
    ap.add_argument("--out-dir",default="outputs_v251_retention")
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
    trans=add_transition_signals(seasons,annual)

    # Frozen one-year model only. No RB/TE feature modifications here.
    pred=v25.build_prediction_pool(table,eligible,"baseline","baseline")

    all_rows=[]; all_preds=[]
    for mode in MODES:
        x=add_signal(pred,annual,trans,mode)
        x=attach_targets(x,annual)
        all_preds.append(x); all_rows.extend(metrics_for(x))
    metrics=pd.DataFrame(all_rows); summary=summarize(metrics)
    preds=pd.concat(all_preds,ignore_index=True)

    metrics.to_csv(out/"v251_yearly_metrics.csv",index=False)
    summary.to_csv(out/"v251_summary.csv",index=False)
    preds.to_csv(out/"v251_predictions.csv",index=False)

    print("=== PARIS v2.5.1 SURVIVAL-AWARE LONGEVITY TEST ===")
    print("Frozen one-year model; only RB/TE multi-year retention changes.")
    print("\n=== SUMMARY ===")
    print(summary.to_string(index=False))
    best=str(summary.iloc[0]["mode"])
    print("\nDevelopment-selected mode:",best)
    print("\n=== BASELINE YEARLY ===")
    print(metrics[metrics["mode"].eq("baseline")].to_string(index=False))
    print("\n=== SELECTED YEARLY ===")
    print(metrics[metrics["mode"].eq(best)].to_string(index=False))


if __name__=="__main__":
    main()
