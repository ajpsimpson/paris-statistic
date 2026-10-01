from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import run_backtest_v211 as v211
import run_backtest_v21 as v21
from paris_v2_core import DEFAULT_CONFIG, assign_starters

POSITIONS = ["QB", "RB", "WR", "TE"]
EVAL_YEARS = [2019, 2020, 2021, 2022, 2023]

# Paris v2.1.1 performance architecture is frozen while the v2.2 role layer is tested.
PPG_CONFIG = {
    "QB": dict(age_mode="linear", opp_mode="baseline", elite_mode="continuous", alpha=30.0),
    "RB": dict(age_mode="linear", opp_mode="expanded", elite_mode="tier", alpha=100.0),
    "WR": dict(age_mode="quadratic", opp_mode="baseline", elite_mode="tier", alpha=100.0),
    "TE": dict(age_mode="hinge", opp_mode="share", elite_mode="tier", alpha=300.0),
}

ROLE_QUANTILES = [0.25, 0.40, 0.55]
ROLE_C = [0.1, 1.0, 10.0]
RISK_WEIGHTS = [0.0, 0.25, 0.50, 0.75, 1.0]
MIN_ROLE_GAMES = 4


def augment_targets(table: pd.DataFrame, seasons: pd.DataFrame) -> pd.DataFrame:
    """Attach exact next-season role and season-value outcomes.

    Missing next-season rows are genuine zero outcomes for the role layer rather
    than being silently dropped. This is the central v2.2 change.
    """
    nxt = seasons[[
        "season", "player_id", "position", "games", "paris_fp", "ppg", "opportunity"
    ]].copy()
    nxt["opp_pct"] = nxt.groupby(["season", "position"])["opportunity"].rank(pct=True, method="average")
    nxt = nxt.rename(columns={
        "season": "target_year", "games": "target_games_all", "paris_fp": "target_fp_all",
        "ppg": "target_ppg_all", "opportunity": "target_opp_all", "opp_pct": "target_opp_pct",
    })
    out = table.merge(
        nxt[["target_year", "player_id", "target_games_all", "target_fp_all", "target_ppg_all",
             "target_opp_all", "target_opp_pct"]],
        on=["target_year", "player_id"], how="left"
    )
    for c in ["target_games_all", "target_fp_all", "target_ppg_all", "target_opp_all", "target_opp_pct"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)

    # One point of "actual season value" means one fantasy point per scheduled
    # team game, so missed games and role loss count without turning every injury
    # into a permanent talent downgrade.
    scheduled = np.where(out["target_year"].le(2020), 16.0, 17.0)
    out["actual_week_value"] = out["target_fp_all"] / scheduled
    return out


def add_role_label(table: pd.DataFrame, q: float) -> pd.DataFrame:
    out = table.copy()
    out["role_target"] = (
        out["target_games_all"].ge(MIN_ROLE_GAMES)
        & out["target_opp_pct"].ge(q)
    ).astype(int)
    return out


def ridge_pipe(alpha: float) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
        ("ridge", Ridge(alpha=alpha)),
    ])


def role_pipe(c: float) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
        ("logit", LogisticRegression(C=c, class_weight="balanced", max_iter=3000)),
    ])


def role_features(pos: str) -> list[str]:
    """Features available before the target season that describe role durability."""
    cfg = PPG_CONFIG[pos]
    base = v211.features(pos, cfg["age_mode"], cfg["opp_mode"], cfg["elite_mode"])
    extras = [
        "games_lag1", "games_lag2", "ppg_lag1", "ppg_lag2", "ppg_trend",
        "opportunity_lag1", "opportunity_lag2", "opp_trend", "draft_pick", "age_next",
    ]
    return list(dict.fromkeys(base + extras))


def fit_ppg_predict(table: pd.DataFrame, year: int, pos: str) -> pd.DataFrame:
    cfg = PPG_CONFIG[pos]
    f = v211.features(pos, cfg["age_mode"], cfg["opp_mode"], cfg["elite_mode"])
    train = table[
        table.position.eq(pos) & table.target_year.lt(year) & table.target_ppg.notna()
    ].copy()
    test = table[table.position.eq(pos) & table.target_year.eq(year)].copy()
    if len(train) < 30 or len(test) < 5:
        return pd.DataFrame()
    m = ridge_pipe(float(cfg["alpha"]))
    m.fit(train[f], train.target_ppg)
    test["predicted_ppg"] = m.predict(test[f])
    return test


def role_predictions(table: pd.DataFrame, year: int, pos: str, q: float, c: float) -> tuple[pd.DataFrame, dict]:
    t = add_role_label(table, q)
    f = role_features(pos)
    train = t[t.position.eq(pos) & t.target_year.lt(year)].copy()
    test = t[t.position.eq(pos) & t.target_year.eq(year)].copy()
    if len(train) < 30 or len(test) < 5 or train.role_target.nunique() < 2:
        return pd.DataFrame(), {}
    m = role_pipe(c)
    m.fit(train[f], train.role_target)
    p = m.predict_proba(test[f])[:, 1]
    test["role_prob"] = p
    auc = np.nan
    if test.role_target.nunique() == 2:
        auc = roc_auc_score(test.role_target, p)
    return test, {
        "year": year, "position": pos, "role_quantile": q, "C": c,
        "auc": auc, "brier": brier_score_loss(test.role_target, p),
        "positive_rate": float(test.role_target.mean()),
        "n_train": len(train), "n_test": len(test),
    }


def build_candidate_predictions(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run fixed v2.1.1 PPG models and candidate role models for every year."""
    rows, role_metrics = [], []
    ppg_cache = {}
    for year in EVAL_YEARS:
        for pos in POSITIONS:
            pp = fit_ppg_predict(table, year, pos)
            if len(pp):
                ppg_cache[(year, pos)] = pp[[
                    "target_year", "player_id", "player_name", "position", "predicted_ppg",
                    "actual_week_value", "target_ppg_all", "target_games_all"
                ]].copy()
            for q, c in product(ROLE_QUANTILES, ROLE_C):
                rp, met = role_predictions(table, year, pos, q, c)
                if not len(rp):
                    continue
                role_metrics.append(met)
                merged = ppg_cache[(year, pos)].merge(
                    rp[["player_id", "role_prob"]], on="player_id", how="inner"
                )
                merged["role_quantile"] = q
                merged["C"] = c
                rows.append(merged)
    return pd.concat(rows, ignore_index=True), pd.DataFrame(role_metrics)


def evaluate_grid(preds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics, ranks = [], []
    for (year, q, c, rw), g0 in preds.assign(_k=1).merge(
        pd.DataFrame({"risk_weight": RISK_WEIGHTS, "_k": 1}), on="_k"
    ).drop(columns="_k").groupby(["target_year", "role_quantile", "C", "risk_weight"]):
        g = g0.drop_duplicates("player_id").copy()
        # Partial shrinkage lets validation decide whether role risk should be a
        # mild adjustment or a full multiplier rather than assuming either.
        g["risk_multiplier"] = (1.0-rw) + rw*g["role_prob"]
        g["predicted_value"] = g["predicted_ppg"] * g["risk_multiplier"]

        if not set(POSITIONS).issubset(set(g.position)):
            continue
        _, pr = assign_starters(g[["player_id", "position", "predicted_value"]], DEFAULT_CONFIG, "predicted_value")
        actual = g[["player_id", "position", "actual_week_value"]].copy()
        _, ar = assign_starters(actual, DEFAULT_CONFIG, "actual_week_value")
        g["pred_vor"] = g.predicted_value - g.position.map(pr)
        g["actual_vor"] = g.actual_week_value - g.position.map(ar)
        g["overall_rank"] = g.pred_vor.rank(ascending=False, method="min")
        g["actual_rank"] = g.actual_vor.rank(ascending=False, method="min")

        def ov(k: int) -> float:
            k=min(k,len(g))
            return len(set(g.nlargest(k,"pred_vor").player_id) & set(g.nlargest(k,"actual_vor").player_id))/max(k,1)

        metrics.append({
            "year": int(year), "role_quantile": q, "C": c, "risk_weight": rw, "n": len(g),
            "overall_spearman": g.pred_vor.corr(g.actual_vor, method="spearman"),
            "top12_overlap": ov(12), "top24_overlap": ov(24), "top50_overlap": ov(50),
        })
        ranks.append(g)
    return pd.DataFrame(metrics), pd.concat(ranks, ignore_index=True)


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    keys=["role_quantile","C","risk_weight"]
    pre=metrics[metrics.year.lt(2023)].groupby(keys,as_index=False).agg(
        pre_spearman=("overall_spearman","mean"), pre_sd=("overall_spearman","std"),
        pre_top24=("top24_overlap","mean"), pre_top50=("top50_overlap","mean")
    )
    val=metrics[metrics.year.eq(2023)][keys+["overall_spearman","top24_overlap","top50_overlap"]].rename(columns={
        "overall_spearman":"val_spearman","top24_overlap":"val_top24","top50_overlap":"val_top50"
    })
    z=pre.merge(val,on=keys)
    z["selection_score"]=(
        .45*z.pre_spearman + .35*z.val_spearman + .075*z.pre_top24 + .075*z.val_top24
        + .025*z.pre_top50 + .025*z.val_top50 - .03*z.pre_sd.fillna(0)
    )
    return z.sort_values("selection_score",ascending=False)


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument("--data-dir",required=True)
    ap.add_argument("--players-csv")
    ap.add_argument("--out-dir",default="outputs_v22")
    args=ap.parse_args(); out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)

    raw=v21.old.load_weekly_csvs(args.data_dir)
    seasons=v21.aggregate_v21(raw)
    players=pd.read_csv(args.players_csv,low_memory=False) if args.players_csv else None
    table=v211.augment_table(seasons,players)
    table=augment_targets(table,seasons)

    preds,role_metrics=build_candidate_predictions(table)
    metrics,ranks=evaluate_grid(preds)
    summary=summarize(metrics)
    best=summary.iloc[0]
    best_ranks=ranks[
        ranks.role_quantile.eq(best.role_quantile)
        & ranks.C.eq(best.C)
        & ranks.risk_weight.eq(best.risk_weight)
    ].copy()
    best_metrics=metrics[
        metrics.role_quantile.eq(best.role_quantile)
        & metrics.C.eq(best.C)
        & metrics.risk_weight.eq(best.risk_weight)
    ].copy()

    role_metrics.to_csv(out/"v22_role_classifier_metrics.csv",index=False)
    metrics.to_csv(out/"v22_value_grid_metrics.csv",index=False)
    summary.to_csv(out/"v22_candidate_summary.csv",index=False)
    best_metrics.to_csv(out/"v22_selected_overall_metrics.csv",index=False)
    best_ranks.to_csv(out/"v22_selected_rankings.csv",index=False)

    print("=== PARIS V2.2 ROLE-SURVIVAL CANDIDATES ===")
    print(summary.head(15).to_string(index=False))
    print("\n=== SELECTED V2.2 SETTINGS ===")
    print(best.to_string())
    print("\n=== SELECTED V2.2 ROLLING RESULTS ===")
    print(best_metrics.sort_values("year").to_string(index=False))
    y=best_ranks[best_ranks.target_year.eq(2023)].sort_values("overall_rank").head(35)
    print("\n=== 2023 V2.2 TOP 35 (RETURNING PLAYERS ONLY) ===")
    print(y[["overall_rank","player_name","position","predicted_ppg","role_prob","predicted_value","actual_week_value","pred_vor","actual_vor"]].to_string(index=False))

if __name__=="__main__":
    main()
