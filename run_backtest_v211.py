from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import run_backtest_v21 as v21
from paris_v2_core import DEFAULT_CONFIG, assign_starters

POSITIONS = ["QB", "RB", "WR", "TE"]
EVAL_YEARS = [2019, 2020, 2021, 2022, 2023]
ALPHAS = [1.0, 10.0, 30.0, 100.0, 300.0]
AGE_MODES = ["linear", "quadratic", "hinge"]
OPP_MODES = ["baseline", "expanded", "share"]
ELITE_MODES = ["none", "top5", "continuous", "tier"]

# These are spline breakpoints, not assumed peaks. Ridge learns the slope on each
# side of each breakpoint. They simply give the curve enough flexibility to model
# different career phases without fitting a separate parameter to every age.
AGE_KNOTS = {
    "QB": (28.0, 32.0, 35.0),
    "RB": (23.0, 25.0, 27.0),
    "WR": (24.0, 27.0, 30.0),
    "TE": (25.0, 28.0, 31.0),
}


def augment_table(seasons: pd.DataFrame, players: pd.DataFrame | None) -> pd.DataFrame:
    table = v21.exact_lag_table(seasons, players)

    # Elite tier: top 5 = 1.0, ranks 6-10 = 0.5, outside top 10 = 0.
    # This directly tests the original Paris intuition that repeated true
    # superstars deserve a stronger signal than intermittent low-end top-10 years.
    tier = seasons[["season", "player_id", "position", "ppg"]].copy()
    tier["pos_rank"] = tier.groupby(["season", "position"])["ppg"].rank(ascending=False, method="min")
    tier["elite_tier_season"] = np.select(
        [tier.pos_rank.le(5), tier.pos_rank.le(10)], [1.0, 0.5], default=0.0
    )
    for lag, offset in [(1, 0), (2, 1), (3, 2)]:
        t = tier[["season", "player_id", "elite_tier_season"]].copy()
        t["season"] += offset
        t = t.rename(columns={"elite_tier_season": f"elite_tier_lag{lag}"})
        table = table.merge(t, on=["season", "player_id"], how="left")
    table["elite_tier"] = (
        0.50 * table["elite_tier_lag1"].fillna(0)
        + 0.30 * table["elite_tier_lag2"].fillna(0)
        + 0.20 * table["elite_tier_lag3"].fillna(0)
    )

    # Position-specific linear-spline bases for age.
    for i in range(3):
        knot = table["position"].map({p: AGE_KNOTS[p][i] for p in POSITIONS})
        table[f"age_h{i+1}"] = (table["age_next"] - knot).clip(lower=0)
    return table


def common_features(age_mode: str) -> list[str]:
    f = [
        "ppg_lag1", "ppg_lag2", "ppg_lag3",
        "efficiency_lag1", "efficiency_lag2",
        "games_lag1", "games_lag2", "ppg_trend", "draft_pick",
    ]
    if age_mode == "linear":
        return f + ["age_next"]
    if age_mode == "quadratic":
        return f + ["age_next", "age_next_sq"]
    if age_mode == "hinge":
        return f + ["age_next", "age_h1", "age_h2", "age_h3"]
    raise ValueError(age_mode)


def opportunity_features(pos: str, mode: str) -> list[str]:
    if mode == "baseline":
        return ["opportunity_lag1", "opportunity_lag2", "opportunity_lag3", "opp_trend"]
    if mode == "expanded":
        return v21.opportunity_features(pos, "expanded")
    if mode != "share":
        raise ValueError(mode)

    # Share mode deliberately removes first-down outcomes and most redundant
    # volume fields. It emphasizes repeatable role: attempts/carries/targets and
    # share of team receiving opportunity.
    if pos == "QB":
        return [
            "attempts_pg_lag1", "attempts_pg_lag2",
            "dropbacks_pg_lag1", "carries_pg_lag1", "carries_pg_lag2",
        ]
    if pos == "RB":
        return [
            "carries_pg_lag1", "carries_pg_lag2",
            "targets_pg_lag1", "targets_pg_lag2",
            "target_share_avg_lag1", "receptions_pg_lag1",
        ]
    if pos in ["WR", "TE"]:
        return [
            "targets_pg_lag1", "targets_pg_lag2",
            "target_share_avg_lag1", "target_share_avg_lag2",
            "air_yards_share_avg_lag1", "wopr_avg_lag1", "air_yards_pg_lag1",
        ]
    raise ValueError(pos)


def elite_features(mode: str) -> list[str]:
    if mode == "none": return []
    if mode == "top5": return ["elite_top5"]
    if mode == "continuous": return ["elite_cont"]
    if mode == "tier": return ["elite_tier"]
    raise ValueError(mode)


def features(pos: str, age_mode: str, opp_mode: str, elite_mode: str) -> list[str]:
    return common_features(age_mode) + opportunity_features(pos, opp_mode) + elite_features(elite_mode)


def model(alpha: float) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
        ("ridge", Ridge(alpha=alpha)),
    ])


def spearman(y: pd.Series, pred: np.ndarray) -> float:
    return float(pd.Series(y.to_numpy()).corr(pd.Series(pred), method="spearman"))


def grid(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics, predictions = [], []
    for year, pos, age_mode, opp_mode, elite_mode, alpha in product(
        EVAL_YEARS, POSITIONS, AGE_MODES, OPP_MODES, ELITE_MODES, ALPHAS
    ):
        f = features(pos, age_mode, opp_mode, elite_mode)
        tr = table[table.position.eq(pos) & table.target_year.lt(year) & table.target_ppg.notna()].copy()
        te = table[table.position.eq(pos) & table.target_year.eq(year) & table.target_ppg.notna()].copy()
        if len(tr) < 30 or len(te) < 5:
            continue
        pipe = model(alpha)
        pipe.fit(tr[f], tr.target_ppg)
        pr = pipe.predict(te[f])
        metrics.append({
            "year": year, "position": pos, "age_mode": age_mode, "opp_mode": opp_mode,
            "elite_mode": elite_mode, "alpha": alpha, "n_train": len(tr), "n_test": len(te),
            "spearman": spearman(te.target_ppg, pr), "mae": mean_absolute_error(te.target_ppg, pr),
        })
        p = te[["target_year", "player_id", "player_name", "position", "target_ppg", "target_games"]].copy()
        p["predicted_ppg"] = pr
        p["age_mode"] = age_mode; p["opp_mode"] = opp_mode; p["elite_mode"] = elite_mode; p["alpha"] = alpha
        predictions.append(p)
    return pd.DataFrame(metrics), pd.concat(predictions, ignore_index=True)


def candidate_table(metrics: pd.DataFrame) -> pd.DataFrame:
    keys = ["position", "age_mode", "opp_mode", "elite_mode", "alpha"]
    pre = metrics[metrics.year.lt(2023)].groupby(keys, as_index=False).agg(
        pre_spearman=("spearman", "mean"), pre_mae=("mae", "mean"), pre_sd=("spearman", "std")
    )
    val = metrics[metrics.year.eq(2023)][keys + ["spearman", "mae"]].rename(
        columns={"spearman": "val_spearman", "mae": "val_mae"}
    )
    z = pre.merge(val, on=keys, how="inner")
    # 2019-2022 remains the majority of the selection signal. 2023 is explicitly
    # the final tuning/validation season. A small stability penalty discourages a
    # candidate that wins only because of one unusually strong historical year.
    z["selection_score"] = 0.60*z.pre_spearman + 0.40*z.val_spearman - 0.05*z.pre_sd.fillna(0)
    z["pre_only_rank"] = z.groupby("position")["pre_spearman"].rank(ascending=False, method="first")
    return z.sort_values(["position", "selection_score"], ascending=[True, False])


def select(candidates: pd.DataFrame) -> pd.DataFrame:
    return candidates.groupby("position", as_index=False).head(1).copy()


def hybrid_predictions(preds: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for _, r in selected.iterrows():
        parts.append(preds[
            preds.position.eq(r.position)
            & preds.age_mode.eq(r.age_mode)
            & preds.opp_mode.eq(r.opp_mode)
            & preds.elite_mode.eq(r.elite_mode)
            & preds.alpha.eq(r.alpha)
        ].copy())
    return pd.concat(parts, ignore_index=True)


def overall_eval(hybrid: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, ranking_rows = [], []
    for year, g in hybrid.groupby("target_year"):
        b = g.drop_duplicates("player_id").copy()
        _, pred_repl = assign_starters(b[["player_id", "position", "predicted_ppg"]], DEFAULT_CONFIG, "predicted_ppg")
        actual = b[["player_id", "position", "target_ppg"]].rename(columns={"target_ppg": "actual_ppg"})
        _, act_repl = assign_starters(actual, DEFAULT_CONFIG, "actual_ppg")
        b["pred_vor"] = b.predicted_ppg - b.position.map(pred_repl)
        b["actual_vor"] = b.target_ppg - b.position.map(act_repl)
        b["overall_rank"] = b.pred_vor.rank(ascending=False, method="min")
        b["actual_rank"] = b.actual_vor.rank(ascending=False, method="min")
        def overlap(k: int) -> float:
            k = min(k, len(b))
            return len(set(b.nlargest(k,"pred_vor").player_id) & set(b.nlargest(k,"actual_vor").player_id))/max(k,1)
        rows.append({
            "year": int(year), "n": len(b),
            "overall_spearman": b.pred_vor.corr(b.actual_vor, method="spearman"),
            "top12_overlap": overlap(12), "top24_overlap": overlap(24), "top50_overlap": overlap(50),
        })
        ranking_rows.append(b)
    return pd.DataFrame(rows), pd.concat(ranking_rows, ignore_index=True)


def set_age_basis(frame: pd.DataFrame, pos: str, age_mode: str, age: float) -> None:
    frame["age_next"] = age
    if "age_next_sq" in frame.columns:
        frame["age_next_sq"] = age**2
    for i, knot in enumerate(AGE_KNOTS[pos], start=1):
        col = f"age_h{i}"
        if col in frame.columns:
            frame[col] = max(age-knot, 0.0)


def age_curve_points(table: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    rows = []
    age_ranges = {"QB": range(21, 41), "RB": range(20, 33), "WR": range(20, 35), "TE": range(20, 36)}
    for _, r in selected.iterrows():
        pos = r.position
        f = features(pos, r.age_mode, r.opp_mode, r.elite_mode)
        tr = table[table.position.eq(pos) & table.target_year.le(2023) & table.target_ppg.notna()].copy()
        pipe = model(float(r.alpha)); pipe.fit(tr[f], tr.target_ppg)
        template = pd.DataFrame([{c: pd.to_numeric(tr[c], errors="coerce").median() for c in f}])
        vals=[]
        for age in age_ranges[pos]:
            row=template.copy(); set_age_basis(row,pos,r.age_mode,float(age))
            vals.append((age,float(pipe.predict(row[f])[0])))
        peak=max(v for _,v in vals)
        for age,pred in vals:
            rows.append({"position":pos,"age":age,"marginal_predicted_ppg":pred,"delta_from_curve_peak":pred-peak,
                         "age_mode":r.age_mode})
    return pd.DataFrame(rows)


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument("--data-dir",required=True); ap.add_argument("--players-csv"); ap.add_argument("--out-dir",default="outputs_v211")
    args=ap.parse_args(); out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)

    raw=v21.old.load_weekly_csvs(args.data_dir)
    seasons=v21.aggregate_v21(raw)
    players=pd.read_csv(args.players_csv,low_memory=False) if args.players_csv else None
    table=augment_table(seasons,players)

    metrics,preds=grid(table)
    candidates=candidate_table(metrics)
    selected=select(candidates)
    hybrid=hybrid_predictions(preds,selected)
    overall,rankings=overall_eval(hybrid)
    curves=age_curve_points(table,selected)

    metrics.to_csv(out/"v211_grid_metrics.csv",index=False)
    candidates.to_csv(out/"v211_candidates.csv",index=False)
    selected.to_csv(out/"v211_selected_models.csv",index=False)
    overall.to_csv(out/"v211_overall_metrics.csv",index=False)
    rankings.to_csv(out/"v211_rankings.csv",index=False)
    curves.to_csv(out/"v211_age_curves.csv",index=False)

    print("=== PARIS V2.1.1 SELECTED POSITION MODELS ===")
    print(selected[["position","age_mode","opp_mode","elite_mode","alpha","pre_spearman","val_spearman","pre_sd","selection_score"]].to_string(index=False))
    print("\n=== OVERALL RESULTS ==="); print(overall.to_string(index=False))
    print("\nPre-2023 mean overall Spearman:",round(overall[overall.year<2023].overall_spearman.mean(),4))
    print("2023 validation overall Spearman:",round(float(overall.loc[overall.year==2023,"overall_spearman"].iloc[0]),4))
    print("Pre-2023 mean top-24 overlap:",round(overall[overall.year<2023].top24_overlap.mean(),4))
    print("2023 top-24 overlap:",round(float(overall.loc[overall.year==2023,"top24_overlap"].iloc[0]),4))

    y=rankings[rankings.target_year.eq(2023)].sort_values("overall_rank").head(35)
    print("\n=== 2023 VALIDATION TOP 35 (RETURNING PLAYERS ONLY) ===")
    print(y[["overall_rank","player_name","position","predicted_ppg","target_ppg","pred_vor","actual_vor"]].to_string(index=False))

if __name__=="__main__":
    main()
