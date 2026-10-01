from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import run_backtest_v22 as v22
import run_backtest_v211 as v211
import run_backtest_v21 as v21
from paris_v2_core import DEFAULT_CONFIG, assign_starters

POSITIONS = ["QB", "RB", "WR", "TE"]
EVAL_YEARS = [2019, 2020, 2021, 2022, 2023]
ROLE_QUANTILES = [0.25, 0.40]
ROLE_C = [0.1, 1.0, 10.0]
RISK_WEIGHTS = [0.50, 0.75, 1.0]
ARCHITECTURES = ["base", "veteran", "veteran_contract"]

VETERAN_AGE = {"QB": 32.0, "RB": 27.0, "WR": 29.0, "TE": 30.0}
LATE_VETERAN_AGE = {"QB": 35.0, "RB": 29.0, "WR": 31.0, "TE": 32.0}


def role_pipe(c: float) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
        ("logit", LogisticRegression(C=c, class_weight="balanced", max_iter=4000)),
    ])


def add_veteran_features(table: pd.DataFrame) -> pd.DataFrame:
    out = table.copy()
    vet_cut = out["position"].map(VETERAN_AGE)
    late_cut = out["position"].map(LATE_VETERAN_AGE)
    out["veteran_excess_age"] = (out["age_next"] - vet_cut).clip(lower=0)
    out["late_veteran_excess_age"] = (out["age_next"] - late_cut).clip(lower=0)
    out["is_veteran"] = out["age_next"].ge(vet_cut).astype(int)
    out["is_late_veteran"] = out["age_next"].ge(late_cut).astype(int)

    out["games_trend"] = out["games_lag1"] - out["games_lag2"]
    out["negative_ppg_trend"] = (-out["ppg_trend"]).clip(lower=0)
    out["negative_opp_trend"] = (-out["opp_trend"]).clip(lower=0)
    out["veteran_ppg_decline"] = out["veteran_excess_age"] * out["negative_ppg_trend"]
    out["veteran_opp_decline"] = out["veteran_excess_age"] * out["negative_opp_trend"]
    out["late_veteran_opp_decline"] = out["late_veteran_excess_age"] * out["negative_opp_trend"]
    return out


def attach_contract_features(table: pd.DataFrame, contracts: pd.DataFrame | None) -> pd.DataFrame:
    out = table.copy()
    defaults = {
        "contract_match": 0.0,
        "contract_years_remaining": 0.0,
        "contract_expiring": 1.0,
        "contract_recent": 0.0,
        "contract_apy_cap_pct": np.nan,
        "contract_guaranteed_pct": np.nan,
        "contract_value": np.nan,
        "contract_apy": np.nan,
    }
    if contracts is None or len(contracts) == 0:
        for c, v in defaults.items(): out[c] = v
        return out

    c = contracts.copy()
    if "gsis_id" not in c.columns:
        for col, v in defaults.items(): out[col] = v
        return out

    c = c.rename(columns={"gsis_id": "player_id"})
    for col in ["year_signed", "years", "value", "apy", "guaranteed", "apy_cap_pct"]:
        if col not in c.columns:
            c[col] = np.nan
        c[col] = pd.to_numeric(c[col], errors="coerce")
    c = c[c.player_id.notna() & c.year_signed.notna()].copy()

    # Deliberately DO NOT use is_active: in the current historical-contract file,
    # that flag reflects present-day knowledge and would leak future information.
    pieces = []
    for y in sorted(out.target_year.dropna().astype(int).unique()):
        base = out[out.target_year.eq(y)].copy()
        known = c[c.year_signed.le(y - 1)].copy()
        if len(known):
            known = known.sort_values(["player_id", "year_signed", "guaranteed", "value"]).groupby("player_id", as_index=False).tail(1)
            keep = known[["player_id", "year_signed", "years", "value", "apy", "guaranteed", "apy_cap_pct"]].copy()
            base = base.merge(keep, on="player_id", how="left")
            base["contract_match"] = base["year_signed"].notna().astype(float)
            end_year = base["year_signed"] + base["years"] - 1.0
            base["contract_years_remaining"] = (end_year - y + 1.0).clip(lower=0).fillna(0)
            base["contract_expiring"] = base["contract_years_remaining"].le(1).astype(float)
            base["contract_recent"] = (y - 1.0 - base["year_signed"]).le(1).fillna(False).astype(float)
            base["contract_apy_cap_pct"] = base["apy_cap_pct"]
            base["contract_guaranteed_pct"] = (base["guaranteed"] / base["value"].replace(0, np.nan)).clip(0, 1)
            base["contract_value"] = base["value"]
            base["contract_apy"] = base["apy"]
            base = base.drop(columns=["year_signed", "years", "value", "apy", "guaranteed", "apy_cap_pct"])
        else:
            for col, v in defaults.items(): base[col] = v
        pieces.append(base)
    return pd.concat(pieces, ignore_index=True)


def role_features(pos: str, arch: str) -> list[str]:
    base = v22.role_features(pos)
    if arch in ["veteran", "veteran_contract"]:
        base = base + [
            "veteran_excess_age", "late_veteran_excess_age", "is_veteran", "is_late_veteran",
            "games_trend", "negative_ppg_trend", "negative_opp_trend",
            "veteran_ppg_decline", "veteran_opp_decline", "late_veteran_opp_decline",
        ]
    if arch == "veteran_contract":
        base = base + [
            "contract_match", "contract_years_remaining", "contract_expiring", "contract_recent",
            "contract_apy_cap_pct", "contract_guaranteed_pct", "contract_value", "contract_apy",
        ]
    return list(dict.fromkeys(base))


def safe_auc(y: pd.Series, p: np.ndarray) -> float:
    if pd.Series(y).nunique() < 2:
        return np.nan
    return float(roc_auc_score(y, p))


def fit_role(table: pd.DataFrame, year: int, pos: str, q: float, c: float, arch: str):
    t = v22.add_role_label(table, q)
    f = role_features(pos, arch)
    train = t[t.position.eq(pos) & t.target_year.lt(year)].copy()
    test = t[t.position.eq(pos) & t.target_year.eq(year)].copy()
    if len(train) < 30 or len(test) < 5 or train.role_target.nunique() < 2:
        return pd.DataFrame(), []
    m = role_pipe(c); m.fit(train[f], train.role_target)
    p = m.predict_proba(test[f])[:, 1]
    test["role_prob"] = p

    rows = []
    for subgroup, mask in [
        ("all", pd.Series(True, index=test.index)),
        ("veteran", test.is_veteran.eq(1)),
        ("late_veteran", test.is_late_veteran.eq(1)),
    ]:
        s = test[mask].copy()
        if len(s) < 5: continue
        rows.append({
            "year": year, "position": pos, "role_quantile": q, "C": c, "architecture": arch,
            "subgroup": subgroup, "n": len(s), "positive_rate": float(s.role_target.mean()),
            "auc": safe_auc(s.role_target, s.role_prob.to_numpy()),
            "brier": float(brier_score_loss(s.role_target, s.role_prob)),
            "mean_prob": float(s.role_prob.mean()),
        })
    return test, rows


def build_predictions(table: pd.DataFrame):
    role_metrics, rows = [], []
    ppg_cache = {}
    for year in EVAL_YEARS:
        for pos in POSITIONS:
            pp = v22.fit_ppg_predict(table, year, pos)
            if len(pp) == 0: continue
            ppg_cache[(year,pos)] = pp[["target_year","player_id","player_name","position","predicted_ppg","actual_week_value","age_next","is_veteran","is_late_veteran"]].copy()
            for q,c,arch in product(ROLE_QUANTILES,ROLE_C,ARCHITECTURES):
                rp, mets = fit_role(table,year,pos,q,c,arch)
                role_metrics.extend(mets)
                if len(rp)==0: continue
                m = ppg_cache[(year,pos)].merge(rp[["player_id","role_prob","role_target"]],on="player_id",how="inner")
                m["role_quantile"]=q; m["C"]=c; m["architecture"]=arch
                rows.append(m)
    return pd.concat(rows,ignore_index=True), pd.DataFrame(role_metrics)


def evaluate_value(preds: pd.DataFrame):
    metrics=[]; rank_rows=[]
    expanded = preds.assign(_k=1).merge(pd.DataFrame({"risk_weight":RISK_WEIGHTS,"_k":1}),on="_k").drop(columns="_k")
    for (year,q,c,arch,rw),g0 in expanded.groupby(["target_year","role_quantile","C","architecture","risk_weight"]):
        g=g0.drop_duplicates("player_id").copy()
        g["predicted_value"] = g.predicted_ppg * ((1-rw)+rw*g.role_prob)
        if not set(POSITIONS).issubset(set(g.position)): continue
        _,pr=assign_starters(g[["player_id","position","predicted_value"]],DEFAULT_CONFIG,"predicted_value")
        actual=g[["player_id","position","actual_week_value"]].copy()
        _,ar=assign_starters(actual,DEFAULT_CONFIG,"actual_week_value")
        g["pred_vor"]=g.predicted_value-g.position.map(pr)
        g["actual_vor"]=g.actual_week_value-g.position.map(ar)
        g["overall_rank"]=g.pred_vor.rank(ascending=False,method="min")
        g["actual_rank"]=g.actual_vor.rank(ascending=False,method="min")
        def ov(k):
            k=min(k,len(g)); return len(set(g.nlargest(k,"pred_vor").player_id)&set(g.nlargest(k,"actual_vor").player_id))/max(k,1)
        row={"year":int(year),"role_quantile":q,"C":c,"architecture":arch,"risk_weight":rw,"n":len(g),
             "overall_spearman":g.pred_vor.corr(g.actual_vor,method="spearman"),"top24_overlap":ov(24),"top50_overlap":ov(50)}
        for subgroup,col in [("veteran","is_veteran"),("late_veteran","is_late_veteran")]:
            s=g[g[col].eq(1)]
            row[f"{subgroup}_n"]=len(s)
            row[f"{subgroup}_spearman"]=s.pred_vor.corr(s.actual_vor,method="spearman") if len(s)>=5 else np.nan
        metrics.append(row); rank_rows.append(g)
    return pd.DataFrame(metrics), pd.concat(rank_rows,ignore_index=True)


def summarize(metrics: pd.DataFrame, role_metrics: pd.DataFrame):
    keys=["role_quantile","C","architecture","risk_weight"]
    pre=metrics[metrics.year.lt(2023)].groupby(keys,as_index=False).agg(
        pre_spearman=("overall_spearman","mean"),pre_vet_spearman=("veteran_spearman","mean"),
        pre_late_vet_spearman=("late_veteran_spearman","mean"),pre_top24=("top24_overlap","mean")
    )
    val=metrics[metrics.year.eq(2023)][keys+["overall_spearman","veteran_spearman","late_veteran_spearman","top24_overlap"]].rename(columns={
        "overall_spearman":"val_spearman","veteran_spearman":"val_vet_spearman","late_veteran_spearman":"val_late_vet_spearman","top24_overlap":"val_top24"})
    z=pre.merge(val,on=keys)

    r=role_metrics[role_metrics.subgroup.eq("veteran")].groupby(["role_quantile","C","architecture","year"],as_index=False).agg(vet_auc=("auc","mean"),vet_brier=("brier","mean"))
    rp=r[r.year.lt(2023)].groupby(["role_quantile","C","architecture"],as_index=False).agg(pre_vet_auc=("vet_auc","mean"),pre_vet_brier=("vet_brier","mean"))
    rv=r[r.year.eq(2023)][["role_quantile","C","architecture","vet_auc","vet_brier"]].rename(columns={"vet_auc":"val_vet_auc","vet_brier":"val_vet_brier"})
    z=z.merge(rp,on=["role_quantile","C","architecture"],how="left").merge(rv,on=["role_quantile","C","architecture"],how="left")
    z["stress_score"]=(.25*z.pre_spearman+.20*z.val_spearman+.15*z.pre_vet_spearman+.15*z.val_vet_spearman+.075*z.pre_vet_auc+.075*z.val_vet_auc+.05*z.pre_top24+.05*z.val_top24)
    return z.sort_values("stress_score",ascending=False)


def false_positive_report(preds: pd.DataFrame, best: pd.Series) -> pd.DataFrame:
    s=preds[(preds.role_quantile.eq(best.role_quantile))&(preds.C.eq(best.C))&(preds.architecture.eq(best.architecture))].copy()
    s=s[s.is_veteran.eq(1) & s.role_target.eq(0)].copy()
    return s.sort_values(["target_year","role_prob"],ascending=[True,False])[["target_year","player_name","position","age_next","role_prob","predicted_ppg","actual_week_value"]]


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--data-dir",required=True); ap.add_argument("--players-csv"); ap.add_argument("--contracts-csv"); ap.add_argument("--out-dir",default="outputs_v221")
    args=ap.parse_args(); out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    raw=v21.old.load_weekly_csvs(args.data_dir); seasons=v21.aggregate_v21(raw)
    players=pd.read_csv(args.players_csv,low_memory=False) if args.players_csv else None
    contracts=pd.read_csv(args.contracts_csv,compression="infer",low_memory=False) if args.contracts_csv else None
    table=v211.augment_table(seasons,players); table=v22.augment_targets(table,seasons); table=add_veteran_features(table); table=attach_contract_features(table,contracts)
    preds,role_metrics=build_predictions(table); metrics,ranks=evaluate_value(preds); summary=summarize(metrics,role_metrics); best=summary.iloc[0]
    fp=false_positive_report(preds,best)
    role_metrics.to_csv(out/"v221_role_subgroup_metrics.csv",index=False); metrics.to_csv(out/"v221_value_stress_metrics.csv",index=False); summary.to_csv(out/"v221_stress_summary.csv",index=False); fp.to_csv(out/"v221_veteran_false_positives.csv",index=False)
    best_ranks=ranks[(ranks.role_quantile.eq(best.role_quantile))&(ranks.C.eq(best.C))&(ranks.architecture.eq(best.architecture))&(ranks.risk_weight.eq(best.risk_weight))].copy(); best_ranks.to_csv(out/"v221_selected_rankings.csv",index=False)
    print("=== PARIS V2.2.1 VETERAN STRESS TEST ==="); print(summary.head(20).to_string(index=False)); print("\n=== SELECTED ==="); print(best.to_string())
    print("\n=== VETERAN ROLE METRICS BY ARCHITECTURE (MEAN 2019-2022 / 2023) ===")
    r=role_metrics[role_metrics.subgroup.eq("veteran")].groupby(["architecture","year"],as_index=False).agg(auc=("auc","mean"),brier=("brier","mean"))
    print(r.to_string(index=False))
    print("\n=== HIGHEST-CONFIDENCE VETERAN ROLE FAILURES ==="); print(fp.sort_values("role_prob",ascending=False).head(30).to_string(index=False))

if __name__=="__main__": main()
