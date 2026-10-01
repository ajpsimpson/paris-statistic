from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import run_backtest_v22 as v22
import run_backtest_v211 as v211
import run_backtest_v21 as v21
from paris_v2_core import DEFAULT_CONFIG, assign_starters

EVAL_YEARS = [2019, 2020, 2021, 2022, 2023]
SELECTED_Q = 0.25
SELECTED_C = 0.1
SELECTED_RISK_WEIGHT = 1.0
POSITIONS = ["QB", "RB", "WR", "TE"]
VETERAN_AGE = {"QB": 32.0, "RB": 27.0, "WR": 29.0, "TE": 30.0}
LATE_VETERAN_AGE = {"QB": 35.0, "RB": 29.0, "WR": 31.0, "TE": 32.0}


def load_week1_rosters(roster_dir: str) -> tuple[dict[int, set[str]], pd.DataFrame, pd.DataFrame]:
    """Return Week-1 roster eligibility sets and diagnostics.

    We deliberately use roster membership only. We do not use Week-1 snaps,
    production, games played, or anything that happens after the season starts.
    All listed roster statuses are eligible, including reserve/injured statuses,
    because an injured dynasty player is still an asset. The purpose of this
    filter is to remove known retirees/free agents/non-rostered players.
    """
    eligible: dict[int, set[str]] = {}
    diagnostics = []
    status_rows = []

    for year in EVAL_YEARS:
        path = Path(roster_dir) / f"roster_weekly_{year}.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_parquet(path)
        cols = list(df.columns)
        id_col = "gsis_id" if "gsis_id" in cols else ("player_id" if "player_id" in cols else None)
        if id_col is None:
            raise ValueError(f"No gsis_id/player_id in {path}; columns={cols}")

        if "week" not in df.columns:
            raise ValueError(f"No week column in {path}; columns={cols}")
        wk = pd.to_numeric(df["week"], errors="coerce")
        snap = df[wk.eq(1)].copy()
        if "season" in snap.columns:
            snap = snap[pd.to_numeric(snap["season"], errors="coerce").eq(year)].copy()
        if "game_type" in snap.columns:
            # Weekly roster files can include multiple game types. Week 1 for our
            # forecast means the regular-season Week 1 snapshot.
            gt = snap["game_type"].astype(str).str.upper()
            reg = gt.eq("REG")
            if reg.any():
                snap = snap[reg].copy()

        snap[id_col] = snap[id_col].astype(str)
        snap = snap[snap[id_col].notna() & ~snap[id_col].isin(["nan", "None", ""])].copy()
        ids = set(snap[id_col].unique())
        eligible[year] = ids

        diagnostics.append({
            "year": year,
            "rows_week1": len(snap),
            "unique_player_ids": len(ids),
            "id_col": id_col,
            "columns": "|".join(cols),
        })
        if "status" in snap.columns:
            counts = snap["status"].fillna("MISSING").astype(str).value_counts(dropna=False)
            for status, n in counts.items():
                status_rows.append({"year": year, "status": status, "n": int(n)})

    return eligible, pd.DataFrame(diagnostics), pd.DataFrame(status_rows)


def attach_age_flags(preds: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    age = table[["target_year", "player_id", "age_next"]].drop_duplicates(["target_year", "player_id"])
    out = preds.merge(age, on=["target_year", "player_id"], how="left")
    vet_cut = out["position"].map(VETERAN_AGE)
    late_cut = out["position"].map(LATE_VETERAN_AGE)
    out["is_veteran"] = out["age_next"].ge(vet_cut).astype(int)
    out["is_late_veteran"] = out["age_next"].ge(late_cut).astype(int)
    return out


def selected_predictions(table: pd.DataFrame) -> pd.DataFrame:
    """Run the already-selected v2.2 architecture; do not retune it here."""
    rows = []
    for year in EVAL_YEARS:
        for pos in POSITIONS:
            pp = v22.fit_ppg_predict(table, year, pos)
            rp, _ = v22.role_predictions(table, year, pos, SELECTED_Q, SELECTED_C)
            if len(pp) == 0 or len(rp) == 0:
                continue
            m = pp[[
                "target_year", "player_id", "player_name", "position", "predicted_ppg",
                "actual_week_value", "target_ppg_all", "target_games_all"
            ]].merge(rp[["player_id", "role_prob"]], on="player_id", how="inner")
            m["predicted_value"] = m["predicted_ppg"] * (
                (1.0 - SELECTED_RISK_WEIGHT) + SELECTED_RISK_WEIGHT * m["role_prob"]
            )
            rows.append(m)
    return attach_age_flags(pd.concat(rows, ignore_index=True), table)


def evaluate_pool(g0: pd.DataFrame, year: int, label: str) -> tuple[dict, pd.DataFrame]:
    g = g0.drop_duplicates("player_id").copy()
    if not set(POSITIONS).issubset(set(g.position)):
        raise ValueError(f"Missing position in {year} {label}: {set(g.position)}")

    _, pred_repl = assign_starters(
        g[["player_id", "position", "predicted_value"]], DEFAULT_CONFIG, "predicted_value"
    )
    actual = g[["player_id", "position", "actual_week_value"]].copy()
    _, actual_repl = assign_starters(actual, DEFAULT_CONFIG, "actual_week_value")
    g["pred_vor"] = g["predicted_value"] - g["position"].map(pred_repl)
    g["actual_vor"] = g["actual_week_value"] - g["position"].map(actual_repl)
    g["overall_rank"] = g["pred_vor"].rank(ascending=False, method="min")
    g["actual_rank"] = g["actual_vor"].rank(ascending=False, method="min")
    g["pool"] = label

    def ov(k: int) -> float:
        k = min(k, len(g))
        return len(
            set(g.nlargest(k, "pred_vor").player_id)
            & set(g.nlargest(k, "actual_vor").player_id)
        ) / max(k, 1)

    row = {
        "year": year,
        "pool": label,
        "n": len(g),
        "overall_spearman": g.pred_vor.corr(g.actual_vor, method="spearman"),
        "top12_overlap": ov(12),
        "top24_overlap": ov(24),
        "top50_overlap": ov(50),
    }
    for subgroup, col in [("veteran", "is_veteran"), ("late_veteran", "is_late_veteran")]:
        s = g[g[col].eq(1)].copy()
        row[f"{subgroup}_n"] = len(s)
        row[f"{subgroup}_spearman"] = (
            s.pred_vor.corr(s.actual_vor, method="spearman") if len(s) >= 5 else np.nan
        )
    return row, g


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--players-csv", required=True)
    ap.add_argument("--roster-dir", required=True)
    ap.add_argument("--out-dir", default="outputs_v222")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    raw = v21.old.load_weekly_csvs(args.data_dir)
    seasons = v21.aggregate_v21(raw)
    players = pd.read_csv(args.players_csv, low_memory=False)
    table = v211.augment_table(seasons, players)
    table = v22.augment_targets(table, seasons)
    preds = selected_predictions(table)

    eligible, roster_diag, status_counts = load_week1_rosters(args.roster_dir)
    metrics = []
    rankings = []
    removal_rows = []
    coverage_rows = []

    for year in EVAL_YEARS:
        all_y = preds[preds.target_year.eq(year)].copy()
        ids = eligible[year]
        all_y["week1_rostered"] = all_y["player_id"].astype(str).isin(ids)
        filt_y = all_y[all_y.week1_rostered].copy()
        removed = all_y[~all_y.week1_rostered].copy()

        raw_row, raw_rank = evaluate_pool(all_y, year, "unfiltered_v22")
        filt_row, filt_rank = evaluate_pool(filt_y, year, "week1_roster_filtered")
        metrics.extend([raw_row, filt_row])
        rankings.append(filt_rank)

        coverage_rows.append({
            "year": year,
            "candidate_players": len(all_y),
            "week1_rostered_candidates": len(filt_y),
            "removed_candidates": len(removed),
            "candidate_match_rate": len(filt_y) / max(len(all_y), 1),
            "removed_with_zero_actual_value": int(removed.actual_week_value.eq(0).sum()),
            "removed_with_positive_actual_value": int(removed.actual_week_value.gt(0).sum()),
        })
        if len(removed):
            removed = removed[[
                "target_year", "player_id", "player_name", "position", "age_next",
                "is_veteran", "is_late_veteran", "predicted_ppg", "role_prob",
                "predicted_value", "actual_week_value", "target_games_all"
            ]].copy()
            removal_rows.append(removed)

    metrics_df = pd.DataFrame(metrics)
    rankings_df = pd.concat(rankings, ignore_index=True)
    removals_df = pd.concat(removal_rows, ignore_index=True) if removal_rows else pd.DataFrame()
    coverage_df = pd.DataFrame(coverage_rows)

    metrics_df.to_csv(out / "v222_comparison_metrics.csv", index=False)
    rankings_df.to_csv(out / "v222_filtered_rankings.csv", index=False)
    removals_df.to_csv(out / "v222_removed_players.csv", index=False)
    coverage_df.to_csv(out / "v222_roster_match_coverage.csv", index=False)
    roster_diag.to_csv(out / "v222_roster_diagnostics.csv", index=False)
    status_counts.to_csv(out / "v222_week1_status_counts.csv", index=False)

    wide = metrics_df.pivot(index="year", columns="pool", values=[
        "overall_spearman", "veteran_spearman", "late_veteran_spearman",
        "top24_overlap", "top50_overlap", "n"
    ])

    print("=== WEEK 1 ROSTER FILE DIAGNOSTICS ===")
    print(roster_diag[["year", "rows_week1", "unique_player_ids", "id_col"]].to_string(index=False))
    print("\n=== PREDICTION-POOL ROSTER MATCH COVERAGE ===")
    print(coverage_df.to_string(index=False))
    print("\n=== V2.2 VS V2.2.2 WEEK-1 ROSTER FILTER ===")
    print(metrics_df.sort_values(["year", "pool"]).to_string(index=False))

    a = metrics_df[metrics_df.pool.eq("unfiltered_v22")]
    b = metrics_df[metrics_df.pool.eq("week1_roster_filtered")]
    print("\n=== SUMMARY ===")
    print("2019-2022 avg overall Spearman, unfiltered:", round(a[a.year.lt(2023)].overall_spearman.mean(), 4))
    print("2019-2022 avg overall Spearman, filtered:  ", round(b[b.year.lt(2023)].overall_spearman.mean(), 4))
    print("2023 overall Spearman, unfiltered:", round(float(a[a.year.eq(2023)].overall_spearman.iloc[0]), 4))
    print("2023 overall Spearman, filtered:  ", round(float(b[b.year.eq(2023)].overall_spearman.iloc[0]), 4))
    print("2019-2022 avg veteran Spearman, unfiltered:", round(a[a.year.lt(2023)].veteran_spearman.mean(), 4))
    print("2019-2022 avg veteran Spearman, filtered:  ", round(b[b.year.lt(2023)].veteran_spearman.mean(), 4))
    print("2023 veteran Spearman, unfiltered:", round(float(a[a.year.eq(2023)].veteran_spearman.iloc[0]), 4))
    print("2023 veteran Spearman, filtered:  ", round(float(b[b.year.eq(2023)].veteran_spearman.iloc[0]), 4))

    if len(removals_df):
        print("\n=== HIGHEST-PROJECTED PLAYERS REMOVED BY PRESEASON ROSTER FILTER ===")
        print(removals_df.sort_values(["target_year", "predicted_value"], ascending=[True, False]).groupby("target_year").head(12).to_string(index=False))

    y = rankings_df[rankings_df.target_year.eq(2023)].sort_values("overall_rank").head(35)
    print("\n=== 2023 V2.2.2 TOP 35 (RETURNING PLAYERS ONLY; ROOKIES NOT ADDED YET) ===")
    print(y[[
        "overall_rank", "player_name", "position", "predicted_ppg", "role_prob",
        "predicted_value", "actual_week_value", "pred_vor", "actual_vor"
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
