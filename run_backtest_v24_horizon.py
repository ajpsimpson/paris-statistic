from __future__ import annotations

import argparse
from itertools import product
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
import run_backtest_v225 as v225
import run_backtest_v23_rookies as v23
from paris_v2_core import DEFAULT_CONFIG, assign_starters

POSITIONS = ["QB", "RB", "WR", "TE"]
FORECAST_YEARS = list(range(2016, 2024))
COMPLETE_3Y_EVAL = [2019, 2020, 2021]
PARTIAL_DIAG_YEARS = [2022, 2023]
ROOKIE_VARIANT = "capital_depth"
ROOKIE_ALPHA = 300.0
QB_Q = 0.55
OTHER_Q = 0.25
DISCOUNTS = [0.50, 0.65, 0.80, 0.90, 1.00]
ALPHAS = [10.0, 30.0, 100.0, 300.0, 1000.0]
FEATURE_VARIANTS = ["one_year", "age_curve", "age_draft", "full"]
TARGET_MODES = ["positive", "raw"]


def ridge_pipe(alpha: float) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
        ("ridge", Ridge(alpha=alpha)),
    ])


def load_week1_rosters(roster_dir: str) -> dict[int, set[str]]:
    eligible: dict[int, set[str]] = {}
    for year in FORECAST_YEARS:
        path = Path(roster_dir) / f"roster_weekly_{year}.parquet"
        df = pd.read_parquet(path)
        id_col = "gsis_id" if "gsis_id" in df.columns else "player_id"
        wk = pd.to_numeric(df["week"], errors="coerce")
        snap = df[wk.eq(1)].copy()
        if "season" in snap.columns:
            snap = snap[pd.to_numeric(snap.season, errors="coerce").eq(year)]
        if "game_type" in snap.columns:
            reg = snap.game_type.astype(str).str.upper().eq("REG")
            if reg.any():
                snap = snap[reg].copy()
        ids = snap[id_col].dropna().astype(str)
        eligible[year] = set(ids[~ids.isin(["nan", "None", ""])].unique())
    return eligible


def player_metadata(players: pd.DataFrame) -> pd.DataFrame:
    p = players.copy()
    if "gsis_id" in p.columns:
        p["player_id"] = p.gsis_id.astype(str)
    elif "player_id" not in p.columns:
        raise ValueError("players file is missing gsis_id/player_id")
    pick_col = "draft_number" if "draft_number" in p.columns else ("draft_pick" if "draft_pick" in p.columns else None)
    round_col = "draft_round" if "draft_round" in p.columns else None
    p["draft_pick_meta"] = pd.to_numeric(p[pick_col], errors="coerce") if pick_col else np.nan
    p["draft_round_meta"] = pd.to_numeric(p[round_col], errors="coerce") if round_col else np.nan
    return p[["player_id", "draft_pick_meta", "draft_round_meta"]].drop_duplicates("player_id")


def veteran_predictions_year(table: pd.DataFrame, year: int) -> pd.DataFrame:
    rows = []
    for pos in POSITIONS:
        pp = v22.fit_ppg_predict(table, year, pos)
        q = QB_Q if pos == "QB" else OTHER_Q
        rp = v225.role_predictions(table, year, pos, q)
        if len(pp) == 0 or len(rp) == 0:
            continue
        cols = [
            "target_year", "player_id", "player_name", "position",
            "predicted_ppg", "actual_week_value"
        ]
        if "age_next" in pp.columns:
            cols.append("age_next")
        m = pp[cols].merge(
            rp[["player_id", "role_prob"] + v223.DEPTH_FEATURES],
            on="player_id", how="inner"
        )
        if "age_next" not in m.columns:
            age = table[["target_year", "player_id", "age_next"]].drop_duplicates(["target_year", "player_id"])
            m = m.merge(age, on=["target_year", "player_id"], how="left")
        m["age"] = m.age_next
        m["predicted_value"] = m.predicted_ppg * m.role_prob
        m["is_rookie"] = 0
        rows.append(m)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def rookie_predictions_year(rook: pd.DataFrame, year: int) -> pd.DataFrame:
    pred = v23.fit_candidate(rook, year, ROOKIE_VARIANT, ROOKIE_ALPHA)
    if len(pred) == 0:
        return pred
    pred = pred[pred.week1_rostered.eq(1)].copy()
    pred["is_rookie"] = 1
    pred["role_prob"] = 1.0
    if "age_next" not in pred.columns:
        pred["age_next"] = pred["age"]
    return pred


def build_forecast_pool(
    year: int,
    vet_table: pd.DataFrame,
    rook: pd.DataFrame,
    eligible: dict[int, set[str]],
    meta: pd.DataFrame,
) -> pd.DataFrame:
    v = veteran_predictions_year(vet_table, year)
    if len(v):
        v = v[v.player_id.astype(str).isin(eligible[year])].copy()
    r = rookie_predictions_year(rook, year)
    rookie_ids = set(r.player_id.astype(str)) if len(r) else set()
    if len(v):
        v = v[~v.player_id.astype(str).isin(rookie_ids)].copy()

    keep = [
        "target_year", "player_id", "player_name", "position", "predicted_value",
        "age", "is_rookie", "role_prob", "depth_rank", "depth_is_first",
        "depth_is_top2", "depth_is_top3"
    ]
    parts = []
    if len(v):
        for c in keep:
            if c not in v.columns:
                v[c] = np.nan
        parts.append(v[keep])
    if len(r):
        rr = r.copy()
        for c in keep:
            if c not in rr.columns:
                rr[c] = np.nan
        parts.append(rr[keep])
    g = pd.concat(parts, ignore_index=True).drop_duplicates("player_id")
    g = g.merge(meta, on="player_id", how="left")

    if len(r):
        rd = r[["player_id", "draft_pick_model", "draft_round_model"]].drop_duplicates("player_id")
        g = g.merge(rd, on="player_id", how="left")
    else:
        g["draft_pick_model"] = np.nan
        g["draft_round_model"] = np.nan
    g["draft_pick_final"] = pd.to_numeric(g.draft_pick_model, errors="coerce").fillna(g.draft_pick_meta).fillna(300.0).clip(1, 300)
    g["draft_round_final"] = pd.to_numeric(g.draft_round_model, errors="coerce").fillna(g.draft_round_meta).fillna(8.0).clip(1, 8)

    _, repl = assign_starters(g[["player_id", "position", "predicted_value"]], DEFAULT_CONFIG, "predicted_value")
    g["pred_vor"] = g.predicted_value - g.position.map(repl)
    g["pred_vor_pos"] = g.pred_vor.clip(lower=0)
    g["forecast_year"] = year
    return g


def build_actual_annual(seasons: pd.DataFrame, eligible: dict[int, set[str]]) -> pd.DataFrame:
    rows = []
    for year in FORECAST_YEARS:
        s = seasons[seasons.season.eq(year) & seasons.position.isin(POSITIONS)].copy()
        s = s[s.player_id.astype(str).isin(eligible[year])].drop_duplicates("player_id")
        scheduled = 16.0 if year <= 2020 else 17.0
        s["actual_week_value"] = pd.to_numeric(s.paris_fp, errors="coerce").fillna(0.0) / scheduled
        _, repl = assign_starters(s[["player_id", "position", "actual_week_value"]], DEFAULT_CONFIG, "actual_week_value")
        s["actual_vor"] = s.actual_week_value - s.position.map(repl)
        s["actual_surplus"] = s.actual_vor.clip(lower=0)
        rows.append(s[["season", "player_id", "position", "actual_week_value", "actual_vor", "actual_surplus"]])
    return pd.concat(rows, ignore_index=True)


def add_horizon_targets(forecasts: pd.DataFrame, annual: pd.DataFrame) -> pd.DataFrame:
    out = forecasts.copy()
    for k in [0, 1, 2]:
        a = annual[["season", "player_id", "actual_vor", "actual_surplus"]].copy()
        a["forecast_year"] = a.season - k
        a = a.rename(columns={
            "actual_vor": f"actual_vor_y{k+1}",
            "actual_surplus": f"actual_surplus_y{k+1}",
        })
        out = out.merge(
            a[["forecast_year", "player_id", f"actual_vor_y{k+1}", f"actual_surplus_y{k+1}"]],
            on=["forecast_year", "player_id"], how="left"
        )
        out[f"actual_vor_y{k+1}"] = out[f"actual_vor_y{k+1}"].fillna(0.0)
        out[f"actual_surplus_y{k+1}"] = out[f"actual_surplus_y{k+1}"].fillna(0.0)
    return out


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["age"] = pd.to_numeric(out.age, errors="coerce")
    out["age2"] = out.age.pow(2)
    out["log_draft_pick"] = np.log1p(out.draft_pick_final)
    out["first_round"] = out.draft_round_final.eq(1).astype(float)
    out["day2"] = out.draft_round_final.isin([2, 3]).astype(float)
    out["depth_rank"] = pd.to_numeric(out.depth_rank, errors="coerce")
    for pos in POSITIONS:
        out[f"pos_{pos}"] = out.position.eq(pos).astype(float)
        out[f"pred_vor_x_{pos}"] = out.pred_vor_pos * out[f"pos_{pos}"]
        out[f"age_x_{pos}"] = out.age * out[f"pos_{pos}"]
        out[f"age2_x_{pos}"] = out.age2 * out[f"pos_{pos}"]
        out[f"rookie_x_{pos}"] = out.is_rookie * out[f"pos_{pos}"]
    return out


BASE = ["pred_vor_pos", "is_rookie"] + [f"pos_{p}" for p in POSITIONS] + [f"pred_vor_x_{p}" for p in POSITIONS]
AGE = BASE + ["age", "age2"] + [f"age_x_{p}" for p in POSITIONS] + [f"age2_x_{p}" for p in POSITIONS] + [f"rookie_x_{p}" for p in POSITIONS]
DRAFT = AGE + ["log_draft_pick", "first_round", "day2"]
FULL = DRAFT + ["predicted_value", "role_prob", "depth_rank", "depth_is_first", "depth_is_top2", "depth_is_top3"]


def feature_set(name: str) -> list[str]:
    return {"one_year": BASE, "age_curve": AGE, "age_draft": DRAFT, "full": FULL}[name]


def target_column(df: pd.DataFrame, discount: float, mode: str, horizon: int = 3) -> pd.Series:
    prefix = "actual_surplus" if mode == "positive" else "actual_vor"
    total = pd.Series(0.0, index=df.index)
    for k in range(horizon):
        total = total + (discount ** k) * df[f"{prefix}_y{k+1}"]
    return total


def overlap(df: pd.DataFrame, pred_col: str, actual_col: str, k: int) -> float:
    k = min(k, len(df))
    if k <= 0:
        return np.nan
    return len(set(df.nlargest(k, pred_col).player_id) & set(df.nlargest(k, actual_col).player_id)) / k


def fit_candidate(table: pd.DataFrame, year: int, discount: float, mode: str, variant: str, alpha: float) -> tuple[dict, pd.DataFrame]:
    train = table[table.forecast_year.add(2).lt(year)].copy()
    test = table[table.forecast_year.eq(year)].copy()
    if len(train) < 200 or len(test) < 100:
        return {}, pd.DataFrame()
    train["target"] = target_column(train, discount, mode, 3)
    test["target"] = target_column(test, discount, mode, 3)
    feats = feature_set(variant)
    model = ridge_pipe(alpha)
    model.fit(train[feats], train.target)
    test["dynasty_score"] = model.predict(test[feats])
    test["baseline_score"] = test.pred_vor_pos
    row = {
        "year": year, "discount": discount, "target_mode": mode,
        "feature_variant": variant, "alpha": alpha, "n": len(test),
        "spearman": test.dynasty_score.corr(test.target, method="spearman"),
        "baseline_spearman": test.baseline_score.corr(test.target, method="spearman"),
        "top24": overlap(test, "dynasty_score", "target", 24),
        "top50": overlap(test, "dynasty_score", "target", 50),
        "top100": overlap(test, "dynasty_score", "target", 100),
        "top150": overlap(test, "dynasty_score", "target", 150),
        "baseline_top50": overlap(test, "baseline_score", "target", 50),
        "baseline_top100": overlap(test, "baseline_score", "target", 100),
    }
    test["discount"] = discount
    test["target_mode"] = mode
    test["feature_variant"] = variant
    test["alpha"] = alpha
    return row, test


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    keys = ["discount", "target_mode", "feature_variant", "alpha"]
    pre = metrics[metrics.year.lt(2021)].groupby(keys, as_index=False).agg(
        pre_spearman=("spearman", "mean"),
        pre_baseline=("baseline_spearman", "mean"),
        pre_top50=("top50", "mean"),
        pre_top100=("top100", "mean"),
    )
    val = metrics[metrics.year.eq(2021)][keys + ["spearman", "baseline_spearman", "top50", "top100", "top150"]].rename(columns={
        "spearman": "val_spearman", "baseline_spearman": "val_baseline",
        "top50": "val_top50", "top100": "val_top100", "top150": "val_top150",
    })
    z = pre.merge(val, on=keys)
    z["pre_gain"] = z.pre_spearman - z.pre_baseline
    z["val_gain"] = z.val_spearman - z.val_baseline
    z["selection_score"] = (
        .35*z.pre_spearman + .35*z.val_spearman
        + .10*z.pre_top50 + .10*z.val_top50
        + .05*z.pre_top100 + .05*z.val_top100
    )
    return z.sort_values("selection_score", ascending=False)


def fit_selected_for_year(table: pd.DataFrame, year: int, best: pd.Series) -> pd.DataFrame:
    train = table[table.forecast_year.add(2).lt(year)].copy()
    test = table[table.forecast_year.eq(year)].copy()
    if len(train) < 200 or len(test) < 100:
        return pd.DataFrame()
    train["target"] = target_column(train, float(best.discount), str(best.target_mode), 3)
    feats = feature_set(str(best.feature_variant))
    model = ridge_pipe(float(best.alpha)); model.fit(train[feats], train.target)
    test["dynasty_score"] = model.predict(test[feats])
    test["dynasty_rank"] = test.dynasty_score.rank(ascending=False, method="min")
    return test


def partial_diagnostics(pred: pd.DataFrame, discount: float, mode: str, horizon: int) -> dict:
    x = pred.copy()
    x["partial_target"] = target_column(x, discount, mode, horizon)
    return {
        "year": int(x.forecast_year.iloc[0]), "observed_horizon": horizon, "n": len(x),
        "spearman": x.dynasty_score.corr(x.partial_target, method="spearman"),
        "top50": overlap(x, "dynasty_score", "partial_target", 50),
        "top100": overlap(x, "dynasty_score", "partial_target", 100),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--players-csv", required=True)
    ap.add_argument("--draft-parquet", required=True)
    ap.add_argument("--combine-parquet", required=True)
    ap.add_argument("--roster-dir", required=True)
    ap.add_argument("--depth-dir", required=True)
    ap.add_argument("--out-dir", default="outputs_v24_horizon")
    args = ap.parse_args(); out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    raw = v21.old.load_weekly_csvs(args.data_dir)
    seasons = v21.aggregate_v21(raw)
    players = pd.read_csv(args.players_csv, low_memory=False)
    draft = pd.read_parquet(args.draft_parquet)
    combine = pd.read_parquet(args.combine_parquet)
    meta = player_metadata(players)
    eligible = load_week1_rosters(args.roster_dir)

    rook = v23.rookie_universe(players, draft, combine)
    rook = v23.attach_preseason_role(rook, args.roster_dir, args.depth_dir)
    rook = v23.attach_outcomes(rook, seasons)

    vet_table = v211.augment_table(seasons, players)
    vet_table = v22.augment_targets(vet_table, seasons)
    depth, _ = v223.load_preseason_depth(args.depth_dir)
    vet_table = v223.attach_depth(vet_table, depth)

    forecast_rows = []
    for year in FORECAST_YEARS:
        g = build_forecast_pool(year, vet_table, rook, eligible, meta)
        forecast_rows.append(g)
    forecasts = pd.concat(forecast_rows, ignore_index=True)
    annual = build_actual_annual(seasons, eligible)
    table = engineer_features(add_horizon_targets(forecasts, annual))

    metrics, preds = [], []
    for year, discount, mode, variant, alpha in product(COMPLETE_3Y_EVAL, DISCOUNTS, TARGET_MODES, FEATURE_VARIANTS, ALPHAS):
        row, pred = fit_candidate(table, year, discount, mode, variant, alpha)
        if row:
            metrics.append(row); preds.append(pred)
    metrics_df = pd.DataFrame(metrics)
    pred_df = pd.concat(preds, ignore_index=True)
    summary = summarize(metrics_df)
    best = summary.iloc[0]

    selected = []
    for year in COMPLETE_3Y_EVAL + PARTIAL_DIAG_YEARS:
        p = fit_selected_for_year(table, year, best)
        if len(p): selected.append(p)
    selected_df = pd.concat(selected, ignore_index=True)

    complete_metrics = metrics_df[
        metrics_df.discount.eq(best.discount)
        & metrics_df.target_mode.eq(best.target_mode)
        & metrics_df.feature_variant.eq(best.feature_variant)
        & metrics_df.alpha.eq(best.alpha)
    ].sort_values("year")

    partial_rows = []
    p22 = selected_df[selected_df.forecast_year.eq(2022)].copy()
    if len(p22): partial_rows.append(partial_diagnostics(p22, float(best.discount), str(best.target_mode), 2))
    p23 = selected_df[selected_df.forecast_year.eq(2023)].copy()
    if len(p23): partial_rows.append(partial_diagnostics(p23, float(best.discount), str(best.target_mode), 1))
    partial_df = pd.DataFrame(partial_rows)

    pos_rows = []
    for year in COMPLETE_3Y_EVAL:
        p = selected_df[selected_df.forecast_year.eq(year)].copy()
        p["target"] = target_column(p, float(best.discount), str(best.target_mode), 3)
        for pos, s in p.groupby("position"):
            pos_rows.append({
                "year": year, "position": pos, "n": len(s),
                "spearman": s.dynasty_score.corr(s.target, method="spearman"),
                "baseline_spearman": s.pred_vor_pos.corr(s.target, method="spearman"),
            })
    pos_df = pd.DataFrame(pos_rows)

    table.to_csv(out/"v24_horizon_dataset.csv", index=False)
    metrics_df.to_csv(out/"v24_candidate_metrics.csv", index=False)
    summary.to_csv(out/"v24_candidate_summary.csv", index=False)
    complete_metrics.to_csv(out/"v24_selected_complete_metrics.csv", index=False)
    partial_df.to_csv(out/"v24_partial_diagnostics.csv", index=False)
    pos_df.to_csv(out/"v24_position_metrics.csv", index=False)
    selected_df.to_csv(out/"v24_selected_rankings.csv", index=False)

    print("=== PARIS V2.4 MULTI-YEAR CANDIDATES ===")
    print(summary.head(25).to_string(index=False))
    print("\n=== SELECTED V2.4 ===")
    print(best.to_string())
    print("\n=== COMPLETE 3-YEAR BACKTEST ===")
    print(complete_metrics.to_string(index=False))
    print("\n=== POSITION-SPECIFIC 3-YEAR CORRELATIONS ===")
    print(pos_df.to_string(index=False))
    print("\n=== PARTIAL LATER-YEAR DIAGNOSTICS (NOT FULL 3-YEAR TARGETS) ===")
    print(partial_df.to_string(index=False))

    y21 = selected_df[selected_df.forecast_year.eq(2021)].copy()
    y21["target"] = target_column(y21, float(best.discount), str(best.target_mode), 3)
    y21 = y21.sort_values("dynasty_rank").head(40)
    print("\n=== 2021 SELECTED DYNASTY TOP 40 VS REALIZED 2021-2023 TARGET ===")
    print(y21[["dynasty_rank", "player_name", "position", "age", "is_rookie", "pred_vor", "dynasty_score", "target"]].to_string(index=False))

    y23 = selected_df[selected_df.forecast_year.eq(2023)].sort_values("dynasty_rank").head(40)
    print("\n=== 2023 PRESEASON DYNASTY TOP 40 (FULL FUTURE TARGET NOT USED) ===")
    print(y23[["dynasty_rank", "player_name", "position", "age", "is_rookie", "pred_vor", "dynasty_score"]].to_string(index=False))


if __name__ == "__main__":
    main()
