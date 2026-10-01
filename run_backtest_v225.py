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
QB_QS = [0.25, 0.40, 0.55, 0.70]
OTHER_Q = 0.25
C = 0.1
MIN_GAMES = 4


def role_pipe() -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", v22.StandardScaler()),
        ("logit", LogisticRegression(C=C, class_weight="balanced", max_iter=3000)),
    ])


def add_position_role_label(table: pd.DataFrame, pos: str, q: float) -> pd.DataFrame:
    out = table.copy()
    out["role_target_pos"] = (
        out.target_games_all.ge(MIN_GAMES)
        & out.target_opp_pct.ge(q)
    ).astype(int)
    return out


def role_predictions(table: pd.DataFrame, year: int, pos: str, q: float) -> pd.DataFrame:
    t = add_position_role_label(table, pos, q)
    feats = v22.role_features(pos) + v223.DEPTH_FEATURES
    train = t[t.position.eq(pos) & t.target_year.lt(year)].copy()
    test = t[t.position.eq(pos) & t.target_year.eq(year)].copy()
    if len(train) < 30 or len(test) < 5 or train.role_target_pos.nunique() < 2:
        return pd.DataFrame()
    m = role_pipe(); m.fit(train[feats], train.role_target_pos)
    test["role_prob"] = m.predict_proba(test[feats])[:, 1]
    return test


def build_predictions(table: pd.DataFrame, qb_q: float) -> pd.DataFrame:
    rows = []
    for year in EVAL_YEARS:
        for pos in POSITIONS:
            q = qb_q if pos == "QB" else OTHER_Q
            pp = v22.fit_ppg_predict(table, year, pos)
            rp = role_predictions(table, year, pos, q)
            if len(pp) == 0 or len(rp) == 0:
                continue
            m = pp[[
                "target_year", "player_id", "player_name", "position", "predicted_ppg",
                "actual_week_value", "target_ppg_all", "target_games_all"
            ]].merge(
                rp[["player_id", "role_prob"] + v223.DEPTH_FEATURES],
                on="player_id", how="inner"
            )
            m["predicted_value"] = m.predicted_ppg * m.role_prob
            m["qb_role_quantile"] = qb_q
            rows.append(m)
    return v222.attach_age_flags(pd.concat(rows, ignore_index=True), table)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--players-csv", required=True)
    ap.add_argument("--roster-dir", required=True)
    ap.add_argument("--depth-dir", required=True)
    ap.add_argument("--out-dir", default="outputs_v225")
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    raw = v21.old.load_weekly_csvs(args.data_dir)
    seasons = v21.aggregate_v21(raw)
    players = pd.read_csv(args.players_csv, low_memory=False)
    table = v211.augment_table(seasons, players)
    table = v22.augment_targets(table, seasons)
    depth, _ = v223.load_preseason_depth(args.depth_dir)
    table = v223.attach_depth(table, depth)
    eligible, _, _ = v222.load_week1_rosters(args.roster_dir)

    metrics, ranks, qbs = [], [], []
    for qb_q in QB_QS:
        pred = build_predictions(table, qb_q)
        for year in EVAL_YEARS:
            ids = eligible[year]
            g = pred[pred.target_year.eq(year) & pred.player_id.astype(str).isin(ids)].copy()
            row, rank = v222.evaluate_pool(g, year, f"qbq_{qb_q:.2f}")
            row["qb_role_quantile"] = qb_q
            metrics.append(row); ranks.append(rank)
        q = pred[
            pred.target_year.eq(2023) & pred.position.eq("QB")
            & pred.player_id.astype(str).isin(eligible[2023])
        ].copy()
        qbs.append(q)

    metrics_df = pd.DataFrame(metrics)
    ranks_df = pd.concat(ranks, ignore_index=True)
    qbs_df = pd.concat(qbs, ignore_index=True)

    summary = metrics_df.groupby("qb_role_quantile", as_index=False).agg(
        pre_spearman=("overall_spearman", lambda x: x[metrics_df.loc[x.index, "year"].lt(2023)].mean()),
        pre_vet_spearman=("veteran_spearman", lambda x: x[metrics_df.loc[x.index, "year"].lt(2023)].mean()),
        pre_top24=("top24_overlap", lambda x: x[metrics_df.loc[x.index, "year"].lt(2023)].mean()),
    )
    val = metrics_df[metrics_df.year.eq(2023)][[
        "qb_role_quantile", "overall_spearman", "veteran_spearman", "top24_overlap", "top50_overlap"
    ]].rename(columns={
        "overall_spearman": "val_spearman", "veteran_spearman": "val_vet_spearman",
        "top24_overlap": "val_top24", "top50_overlap": "val_top50"
    })
    summary = summary.merge(val, on="qb_role_quantile")
    summary["selection_score"] = (
        .45*summary.pre_spearman + .35*summary.val_spearman
        + .05*summary.pre_vet_spearman + .05*summary.val_vet_spearman
        + .05*summary.pre_top24 + .05*summary.val_top24
    )
    summary = summary.sort_values("selection_score", ascending=False)

    metrics_df.to_csv(out/"v225_metrics.csv", index=False)
    summary.to_csv(out/"v225_summary.csv", index=False)
    ranks_df.to_csv(out/"v225_rankings.csv", index=False)
    qbs_df.to_csv(out/"v225_2023_qb_diagnostics.csv", index=False)

    print("=== QB ROLE THRESHOLD SUMMARY ===")
    print(summary.to_string(index=False))
    print("\n=== YEAR-BY-YEAR ===")
    print(metrics_df.sort_values(["qb_role_quantile", "year"]).to_string(index=False))

    print("\n=== MARCUS MARIOTA 2023 BY QB ROLE THRESHOLD ===")
    mm = qbs_df[qbs_df.player_name.eq("M.Mariota")][[
        "qb_role_quantile", "depth_rank", "predicted_ppg", "role_prob",
        "predicted_value", "actual_week_value"
    ]]
    print(mm.to_string(index=False))

    best_q = float(summary.iloc[0].qb_role_quantile)
    best = ranks_df[
        ranks_df.pool.eq(f"qbq_{best_q:.2f}") & ranks_df.target_year.eq(2023)
    ].sort_values("overall_rank").head(35)
    print(f"\n=== SELECTED QB THRESHOLD {best_q:.2f}: 2023 TOP 35 ===")
    print(best[[
        "overall_rank", "player_name", "position", "predicted_ppg", "role_prob",
        "predicted_value", "actual_week_value", "pred_vor", "actual_vor"
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
