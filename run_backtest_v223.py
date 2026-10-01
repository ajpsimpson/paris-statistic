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
RISK_WEIGHT = 1.0
POSITIONS = ["QB", "RB", "WR", "TE"]

# First regular-season game dates. Cutoff is 23:59:59 UTC on that date,
# which is still before the evening kickoff in every listed season.
OPENING_DATES = {
    2016: "2016-09-08", 2017: "2017-09-07", 2018: "2018-09-06",
    2019: "2019-09-05", 2020: "2020-09-10", 2021: "2021-09-09",
    2022: "2022-09-08", 2023: "2023-09-07",
}
DEPTH_FEATURES = [
    "depth_present", "depth_rank", "depth_is_first", "depth_is_top2", "depth_is_top3"
]


def load_preseason_depth(depth_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build one pre-kickoff depth-chart row per player-season.

    Only records timestamped before the first regular-season game are eligible.
    For a player appearing multiple times, use his latest pre-kickoff timestamp;
    if he occupies multiple slots at that timestamp, keep his best (lowest) rank.
    """
    parts = []
    diag = []
    for year in DEPTH_YEARS:
        path = Path(depth_dir) / f"depth_charts_{year}.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_parquet(path)
        required = {"dt", "gsis_id", "pos_rank"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{path} missing {sorted(missing)}; columns={list(df.columns)}")

        df["dt_parsed"] = pd.to_datetime(df["dt"], errors="coerce", utc=True)
        cutoff = pd.Timestamp(OPENING_DATES[year] + "T23:59:59Z")
        pre = df[df.dt_parsed.notna() & df.dt_parsed.le(cutoff) & df.gsis_id.notna()].copy()
        if "pos_grp" in pre.columns:
            offense = pre.pos_grp.astype(str).str.lower().str.contains("off", na=False)
            if offense.any():
                pre = pre[offense].copy()
        pre["player_id"] = pre["gsis_id"].astype(str)
        pre["pos_rank_num"] = pd.to_numeric(pre["pos_rank"], errors="coerce")

        # Latest known chart appearance for each player before opening kickoff.
        latest_dt = pre.groupby("player_id")["dt_parsed"].transform("max")
        latest = pre[pre.dt_parsed.eq(latest_dt)].copy()
        one = latest.groupby("player_id", as_index=False).agg(
            depth_rank=("pos_rank_num", "min"),
            depth_snapshot=("dt_parsed", "max"),
        )
        one["target_year"] = year
        one["depth_present"] = 1.0
        one["depth_is_first"] = one.depth_rank.le(1).astype(float)
        one["depth_is_top2"] = one.depth_rank.le(2).astype(float)
        one["depth_is_top3"] = one.depth_rank.le(3).astype(float)
        parts.append(one)

        diag.append({
            "year": year,
            "raw_rows": len(df),
            "pre_kickoff_rows": len(pre),
            "unique_players": one.player_id.nunique(),
            "earliest_pre_dt": pre.dt_parsed.min(),
            "latest_pre_dt": pre.dt_parsed.max(),
            "rank1_players": int(one.depth_is_first.sum()),
            "rank2_or_better_players": int(one.depth_is_top2.sum()),
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
    depth_coverage = table[table.target_year.isin(EVAL_YEARS)].groupby(["target_year", "position"], as_index=False).agg(
        rows=("player_id", "size"), depth_match_rate=("depth_present", "mean"), median_depth_rank=("depth_rank", "median")
    )

    metrics_df.to_csv(out/"v223_depth_comparison_metrics.csv", index=False)
    ranks_df.to_csv(out/"v223_rankings.csv", index=False)
    depth_diag.to_csv(out/"v223_depth_snapshot_diagnostics.csv", index=False)
    depth_coverage.to_csv(out/"v223_depth_coverage.csv", index=False)
    depth_qbs.to_csv(out/"v223_qb_depth_diagnostics.csv", index=False)
    roster_diag.to_csv(out/"v223_roster_diagnostics.csv", index=False)

    print("=== PRE-KICKOFF DEPTH SNAPSHOT DIAGNOSTICS ===")
    print(depth_diag.to_string(index=False))
    print("\n=== DEPTH COVERAGE OF RETURNING PREDICTION ROWS ===")
    print(depth_coverage.to_string(index=False))
    print("\n=== V2.2.2 ROSTER-ONLY VS V2.2.3 + PRESEASON DEPTH ===")
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
    print("\n=== 2023 QB ROLE DIAGNOSTIC WITH PRESEASON DEPTH ===")
    print(q.to_string(index=False))

    d23 = ranks_df[(ranks_df.pool.eq("roster_plus_depth_role")) & ranks_df.target_year.eq(2023)].sort_values("overall_rank").head(35)
    print("\n=== 2023 V2.2.3 TOP 35 (RETURNERS ONLY) ===")
    print(d23[["overall_rank","player_name","position","predicted_ppg","role_prob","predicted_value","actual_week_value","pred_vor","actual_vor"]].to_string(index=False))


if __name__ == "__main__":
    main()
