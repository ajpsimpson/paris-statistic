from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import run_backtest_v222 as v222
import run_backtest_v22 as v22
import run_backtest_v211 as v211
import run_backtest_v21 as v21

EVAL_YEARS = [2019, 2020, 2021, 2022, 2023]
DEPTH_YEARS = list(range(2016, 2024))
Q = 0.25
C = 0.1
POSITIONS = ["QB", "RB", "WR", "TE"]

OPENING_DATES = {
    2016: "2016-09-08", 2017: "2017-09-07", 2018: "2018-09-06",
    2019: "2019-09-05", 2020: "2020-09-10", 2021: "2021-09-09",
    2022: "2022-09-08", 2023: "2023-09-07",
}
DEPTH_FEATURES = [
    "depth_present", "depth_rank", "depth_is_first", "depth_is_top2", "depth_is_top3"
]


def load_preseason_depth(depth_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normalize historical Week-1/pre-kickoff depth charts.

    nflverse uses two schemas. Through 2024 the legacy files contain Week 1,
    `depth_team` (1/2/3), and `depth_position`. The newer ESPN-derived files use
    timestamps and `pos_rank`. We support both so historical testing is stable.

    The legacy data do not have an intraday timestamp, so for those seasons we use
    the official REG Week-1 chart and no Week-1 performance data. The depth signal
    is therefore treated as a preseason/Week-1 role feature, not an outcome.
    """
    parts: list[pd.DataFrame] = []
    diag: list[dict] = []

    for year in DEPTH_YEARS:
        path = Path(depth_dir) / f"depth_charts_{year}.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_parquet(path)
        cols = set(df.columns)

        if {"season", "week", "game_type", "depth_team", "gsis_id"}.issubset(cols):
            season_num = pd.to_numeric(df["season"], errors="coerce")
            week_num = pd.to_numeric(df["week"], errors="coerce")
            gt = df["game_type"].astype(str).str.upper()
            snap = df[
                season_num.eq(year) & week_num.eq(1) & gt.eq("REG") & df["gsis_id"].notna()
            ].copy()

            # Limit to the fantasy positions where possible. `depth_team` is a
            # rank within a legacy depth slot, so a first-team LWR and first-team
            # RWR can both correctly carry rank 1.
            if "position" in snap.columns:
                fantasy = snap["position"].astype(str).str.upper().isin(POSITIONS)
                if fantasy.any():
                    snap = snap[fantasy].copy()

            snap["player_id"] = snap["gsis_id"].astype(str)
            snap["depth_rank_num"] = pd.to_numeric(snap["depth_team"], errors="coerce")
            snap = snap[snap.depth_rank_num.notna()].copy()

            one = snap.groupby("player_id", as_index=False).agg(
                depth_rank=("depth_rank_num", "min")
            )
            one["target_year"] = year
            one["depth_snapshot"] = f"{year}_REG_week1_legacy"
            schema = "legacy_week1"
            earliest = np.nan
            latest = np.nan

        elif {"dt", "gsis_id", "pos_rank"}.issubset(cols):
            df["dt_parsed"] = pd.to_datetime(df["dt"], errors="coerce", utc=True)
            cutoff = pd.Timestamp(OPENING_DATES[year] + "T23:59:59Z")
            snap = df[
                df.dt_parsed.notna() & df.dt_parsed.le(cutoff) & df.gsis_id.notna()
            ].copy()
            if "pos_abb" in snap.columns:
                fantasy = snap["pos_abb"].astype(str).str.upper().isin(POSITIONS)
                if fantasy.any():
                    snap = snap[fantasy].copy()
            snap["player_id"] = snap["gsis_id"].astype(str)
            snap["depth_rank_num"] = pd.to_numeric(snap["pos_rank"], errors="coerce")
            snap = snap[snap.depth_rank_num.notna()].copy()

            latest_dt = snap.groupby("player_id")["dt_parsed"].transform("max")
            latest_rows = snap[snap.dt_parsed.eq(latest_dt)].copy()
            one = latest_rows.groupby("player_id", as_index=False).agg(
                depth_rank=("depth_rank_num", "min"),
                depth_snapshot=("dt_parsed", "max"),
            )
            one["target_year"] = year
            schema = "timestamp_pos_rank"
            earliest = snap.dt_parsed.min()
            latest = snap.dt_parsed.max()
        else:
            raise ValueError(f"Unrecognized depth schema in {path}: {list(df.columns)}")

        one["depth_present"] = 1.0
        one["depth_is_first"] = one.depth_rank.le(1).astype(float)
        one["depth_is_top2"] = one.depth_rank.le(2).astype(float)
        one["depth_is_top3"] = one.depth_rank.le(3).astype(float)
        parts.append(one)

        diag.append({
            "year": year,
            "schema": schema,
            "raw_rows": len(df),
            "snapshot_rows": len(snap),
            "unique_players": one.player_id.nunique(),
            "rank1_players": int(one.depth_is_first.sum()),
            "rank2_or_better_players": int(one.depth_is_top2.sum()),
            "earliest_timestamp": earliest,
            "latest_timestamp": latest,
        })

    return pd.concat(parts, ignore_index=True), pd.DataFrame(diag)


def attach_depth(table: pd.DataFrame, depth: pd.DataFrame) -> pd.DataFrame:
    out = table.merge(
        depth[["target_year", "player_id"] + DEPTH_FEATURES + ["depth_snapshot"]],
        on=["target_year", "player_id"], how="left"
    )
    out["depth_present"] = out["depth_present"].fillna(0.0)
    for c in ["depth_is_first", "depth_is_top2", "depth_is_top3"]:
        out[c] = out[c].fillna(0.0)
    return out


def depth_role_pipe() -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", v22.StandardScaler()),
        ("logit", LogisticRegression(C=C, class_weight="balanced", max_iter=3000)),
    ])


def depth_role_predictions(table: pd.DataFrame, year: int, pos: str) -> pd.DataFrame:
    t = v22.add_role_label(table, Q)
    feats = v22.role_features(pos) + DEPTH_FEATURES
    train = t[t.position.eq(pos) & t.target_year.lt(year)].copy()
    test = t[t.position.eq(pos) & t.target_year.eq(year)].copy()
    if len(train) < 30 or len(test) < 5 or train.role_target.nunique() < 2:
        return pd.DataFrame()
    m = depth_role_pipe()
    m.fit(train[feats], train.role_target)
    test["role_prob_depth"] = m.predict_proba(test[feats])[:, 1]
    return test


def build_depth_predictions(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year in EVAL_YEARS:
        for pos in POSITIONS:
            pp = v22.fit_ppg_predict(table, year, pos)
            rp = depth_role_predictions(table, year, pos)
            if len(pp) == 0 or len(rp) == 0:
                continue
            keep = ["player_id", "role_prob_depth"] + DEPTH_FEATURES
            m = pp[[
                "target_year", "player_id", "player_name", "position", "predicted_ppg",
                "actual_week_value", "target_ppg_all", "target_games_all"
            ]].merge(rp[keep], on="player_id", how="inner")
            m["role_prob"] = m["role_prob_depth"]
            m["predicted_value"] = m.predicted_ppg * m.role_prob
            rows.append(m)
    return v222.attach_age_flags(pd.concat(rows, ignore_index=True), table)


def depth_qb_diagnostics(preds: pd.DataFrame) -> pd.DataFrame:
    qbs = preds[preds.position.eq("QB")].copy()
    return qbs.sort_values(["target_year", "predicted_value"], ascending=[True, False])[[
        "target_year", "player_name", "depth_present", "depth_rank", "depth_is_first",
        "predicted_ppg", "role_prob", "predicted_value", "actual_week_value"
    ]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--players-csv", required=True)
    ap.add_argument("--roster-dir", required=True)
    ap.add_argument("--depth-dir", required=True)
    ap.add_argument("--out-dir", default="outputs_v223")
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    raw = v21.old.load_weekly_csvs(args.data_dir)
    seasons = v21.aggregate_v21(raw)
    players = pd.read_csv(args.players_csv, low_memory=False)
    table = v211.augment_table(seasons, players)
    table = v22.augment_targets(table, seasons)
    depth, depth_diag = load_preseason_depth(args.depth_dir)
    table = attach_depth(table, depth)

    eligible, roster_diag, _ = v222.load_week1_rosters(args.roster_dir)
    base = v222.selected_predictions(table)
    depth_preds = build_depth_predictions(table)

    metrics = []
    ranks = []
    for year in EVAL_YEARS:
        ids = eligible[year]
        b = base[base.target_year.eq(year) & base.player_id.astype(str).isin(ids)].copy()
        d = depth_preds[depth_preds.target_year.eq(year) & depth_preds.player_id.astype(str).isin(ids)].copy()
        br, bk = v222.evaluate_pool(b, year, "roster_base_role")
        dr, dk = v222.evaluate_pool(d, year, "roster_plus_depth_role")
        metrics.extend([br, dr]); ranks.extend([bk, dk])

    metrics_df = pd.DataFrame(metrics)
    ranks_df = pd.concat(ranks, ignore_index=True)
    depth_qbs = depth_qb_diagnostics(depth_preds)
    depth_coverage = table[table.target_year.isin(EVAL_YEARS)].groupby(
        ["target_year", "position"], as_index=False
    ).agg(
        rows=("player_id", "size"),
        depth_match_rate=("depth_present", "mean"),
        median_depth_rank=("depth_rank", "median")
    )

    metrics_df.to_csv(out/"v223_depth_comparison_metrics.csv", index=False)
    ranks_df.to_csv(out/"v223_rankings.csv", index=False)
    depth_diag.to_csv(out/"v223_depth_snapshot_diagnostics.csv", index=False)
    depth_coverage.to_csv(out/"v223_depth_coverage.csv", index=False)
    depth_qbs.to_csv(out/"v223_qb_depth_diagnostics.csv", index=False)
    roster_diag.to_csv(out/"v223_roster_diagnostics.csv", index=False)

    print("=== WEEK-1 DEPTH SNAPSHOT DIAGNOSTICS ===")
    print(depth_diag.to_string(index=False))
    print("\n=== DEPTH COVERAGE OF RETURNING PREDICTION ROWS ===")
    print(depth_coverage.to_string(index=False))
    print("\n=== V2.2.2 ROSTER-ONLY VS V2.2.3 + WEEK-1 DEPTH ===")
    print(metrics_df.sort_values(["year", "pool"]).to_string(index=False))

    for label in ["roster_base_role", "roster_plus_depth_role"]:
        m = metrics_df[metrics_df.pool.eq(label)]
        print(f"\n{label}")
        print("2019-2022 avg overall Spearman:", round(m[m.year.lt(2023)].overall_spearman.mean(), 4))
        print("2023 overall Spearman:", round(float(m[m.year.eq(2023)].overall_spearman.iloc[0]), 4))
        print("2019-2022 avg veteran Spearman:", round(m[m.year.lt(2023)].veteran_spearman.mean(), 4))
        print("2023 veteran Spearman:", round(float(m[m.year.eq(2023)].veteran_spearman.iloc[0]), 4))
        print("2019-2022 avg Top-24 overlap:", round(m[m.year.lt(2023)].top24_overlap.mean(), 4))
        print("2023 Top-24 overlap:", round(float(m[m.year.eq(2023)].top24_overlap.iloc[0]), 4))

    q = depth_qbs[depth_qbs.target_year.eq(2023)].head(25)
    print("\n=== 2023 QB ROLE DIAGNOSTIC WITH WEEK-1 DEPTH ===")
    print(q.to_string(index=False))

    d23 = ranks_df[
        ranks_df.pool.eq("roster_plus_depth_role") & ranks_df.target_year.eq(2023)
    ].sort_values("overall_rank").head(35)
    print("\n=== 2023 V2.2.3 TOP 35 (RETURNERS ONLY) ===")
    print(d23[[
        "overall_rank", "player_name", "position", "predicted_ppg", "role_prob",
        "predicted_value", "actual_week_value", "pred_vor", "actual_vor"
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
