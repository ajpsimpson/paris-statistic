from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import run_backtest_v223 as v223
import run_backtest_v222 as v222
import run_backtest_v22 as v22
import run_backtest_v211 as v211
import run_backtest_v21 as v21

EVAL_YEARS = [2019, 2020, 2021, 2022, 2023]
POSITIONS = ["QB", "RB", "WR", "TE"]
Q = 0.25
C = 0.1
QB_TRANSITION_FEATURES = [
    "qb_depth_backup",
    "qb_backup_x_ppg_lag1",
    "qb_backup_x_games_lag1",
    "qb_backup_x_opp_lag1",
    "qb_first_x_ppg_lag1",
]


def add_qb_transition_features(table: pd.DataFrame) -> pd.DataFrame:
    out = table.copy()
    is_qb = out.position.eq("QB").astype(float)
    backup = (
        out.depth_present.fillna(0).eq(1)
        & out.depth_is_first.fillna(0).eq(0)
    ).astype(float) * is_qb
    first = out.depth_is_first.fillna(0).astype(float) * is_qb
    out["qb_depth_backup"] = backup
    out["qb_backup_x_ppg_lag1"] = backup * pd.to_numeric(out.ppg_lag1, errors="coerce").fillna(0)
    out["qb_backup_x_games_lag1"] = backup * pd.to_numeric(out.games_lag1, errors="coerce").fillna(0)
    out["qb_backup_x_opp_lag1"] = backup * pd.to_numeric(out.opportunity_lag1, errors="coerce").fillna(0)
    out["qb_first_x_ppg_lag1"] = first * pd.to_numeric(out.ppg_lag1, errors="coerce").fillna(0)
    return out


def role_pipe() -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", v22.StandardScaler()),
        ("logit", LogisticRegression(C=C, class_weight="balanced", max_iter=3000)),
    ])


def role_predictions(table: pd.DataFrame, year: int, pos: str, transition: bool) -> pd.DataFrame:
    t = v22.add_role_label(table, Q)
    feats = v22.role_features(pos) + v223.DEPTH_FEATURES
    if transition and pos == "QB":
        feats += QB_TRANSITION_FEATURES
    train = t[t.position.eq(pos) & t.target_year.lt(year)].copy()
    test = t[t.position.eq(pos) & t.target_year.eq(year)].copy()
    if len(train) < 30 or len(test) < 5 or train.role_target.nunique() < 2:
        return pd.DataFrame()
    model = role_pipe()
    model.fit(train[feats], train.role_target)
    test["role_prob"] = model.predict_proba(test[feats])[:, 1]
    return test


def build_predictions(table: pd.DataFrame, transition: bool) -> pd.DataFrame:
    rows = []
    for year in EVAL_YEARS:
        for pos in POSITIONS:
            pp = v22.fit_ppg_predict(table, year, pos)
            rp = role_predictions(table, year, pos, transition)
            if len(pp) == 0 or len(rp) == 0:
                continue
            cols = ["player_id", "role_prob"] + v223.DEPTH_FEATURES
            if pos == "QB":
                cols += QB_TRANSITION_FEATURES
            cols = list(dict.fromkeys(cols))
            m = pp[[
                "target_year", "player_id", "player_name", "position", "predicted_ppg",
                "actual_week_value", "target_ppg_all", "target_games_all"
            ]].merge(rp[cols], on="player_id", how="inner")
            m["predicted_value"] = m.predicted_ppg * m.role_prob
            rows.append(m)
    return v222.attach_age_flags(pd.concat(rows, ignore_index=True), table)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--players-csv", required=True)
    ap.add_argument("--roster-dir", required=True)
    ap.add_argument("--depth-dir", required=True)
    ap.add_argument("--out-dir", default="outputs_v224")
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    raw = v21.old.load_weekly_csvs(args.data_dir)
    seasons = v21.aggregate_v21(raw)
    players = pd.read_csv(args.players_csv, low_memory=False)
    table = v211.augment_table(seasons, players)
    table = v22.augment_targets(table, seasons)
    depth, depth_diag = v223.load_preseason_depth(args.depth_dir)
    table = v223.attach_depth(table, depth)
    table = add_qb_transition_features(table)
    eligible, _, _ = v222.load_week1_rosters(args.roster_dir)

    plain = build_predictions(table, transition=False)
    transition = build_predictions(table, transition=True)
    metrics, ranks = [], []
    for year in EVAL_YEARS:
        ids = eligible[year]
        for label, pred in [
            ("depth_role", plain),
            ("depth_plus_qb_transition", transition),
        ]:
            g = pred[pred.target_year.eq(year) & pred.player_id.astype(str).isin(ids)].copy()
            row, rank = v222.evaluate_pool(g, year, label)
            metrics.append(row); ranks.append(rank)

    metrics_df = pd.DataFrame(metrics)
    ranks_df = pd.concat(ranks, ignore_index=True)
    metrics_df.to_csv(out/"v224_comparison_metrics.csv", index=False)
    ranks_df.to_csv(out/"v224_rankings.csv", index=False)

    qbs = transition[
        transition.target_year.eq(2023) & transition.position.eq("QB")
        & transition.player_id.astype(str).isin(eligible[2023])
    ].sort_values("predicted_value", ascending=False)
    qbs.to_csv(out/"v224_2023_qb_diagnostics.csv", index=False)

    print("=== V2.2.3 DEPTH VS V2.2.4 QB TRANSITION ===")
    print(metrics_df.sort_values(["year", "pool"]).to_string(index=False))
    for label in ["depth_role", "depth_plus_qb_transition"]:
        m = metrics_df[metrics_df.pool.eq(label)]
        print(f"\n{label}")
        print("2019-2022 avg overall Spearman:", round(m[m.year.lt(2023)].overall_spearman.mean(), 4))
        print("2023 overall Spearman:", round(float(m[m.year.eq(2023)].overall_spearman.iloc[0]), 4))
        print("2019-2022 avg veteran Spearman:", round(m[m.year.lt(2023)].veteran_spearman.mean(), 4))
        print("2023 veteran Spearman:", round(float(m[m.year.eq(2023)].veteran_spearman.iloc[0]), 4))
        print("2019-2022 avg Top-24 overlap:", round(m[m.year.lt(2023)].top24_overlap.mean(), 4))
        print("2023 Top-24 overlap:", round(float(m[m.year.eq(2023)].top24_overlap.iloc[0]), 4))

    print("\n=== 2023 QB DIAGNOSTIC ===")
    print(qbs.head(30)[[
        "player_name", "depth_rank", "qb_depth_backup", "predicted_ppg",
        "role_prob", "predicted_value", "actual_week_value"
    ]].to_string(index=False))

    r23 = ranks_df[
        ranks_df.pool.eq("depth_plus_qb_transition") & ranks_df.target_year.eq(2023)
    ].sort_values("overall_rank").head(35)
    print("\n=== 2023 V2.2.4 TOP 35 (RETURNERS ONLY) ===")
    print(r23[[
        "overall_rank", "player_name", "position", "predicted_ppg", "role_prob",
        "predicted_value", "actual_week_value", "pred_vor", "actual_vor"
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
