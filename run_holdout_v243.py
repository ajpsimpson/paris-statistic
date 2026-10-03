from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import run_backtest_v21 as v21
import run_backtest_v211 as v211
import run_backtest_v22 as v22
import run_backtest_v223 as v223
import run_backtest_v23_rookies as v23
import run_backtest_v24_horizon as v24
import run_backtest_v241_retention as v241
import run_backtest_v243_top150 as v243
from paris_long_td_bonus import apply_long_td_bonus, load_long_td_bonus


FROZEN_ARCHITECTURE_COMMIT = "f139dbec901c1d8a516bff6948b19893752a1483"
HOLDOUT_YEARS = [2024, 2025]
ALL_ACTUAL_YEARS = list(range(2012, 2026))
ALL_FORECAST_YEARS = list(range(2016, 2026))
PBP_YEARS = list(range(2012, 2026))
DISCOUNT = 0.50
SHRINK = 5.0
CAP = 1.00
TAIL_MODE = "signed"

OPENING_DATES_EXT = {
    2024: "2024-09-05",
    2025: "2025-09-04",
}


def rookie_universe_extended(
    players: pd.DataFrame,
    draft: pd.DataFrame,
    combine: pd.DataFrame,
    end_year: int = 2025,
) -> pd.DataFrame:
    """Exact v2.3 rookie feature construction, extended only through the holdout years."""
    p = players.copy()
    p["position"] = v23.normalize_position(p["position"])
    p["target_year"] = pd.to_numeric(p["rookie_season"], errors="coerce")
    p = p[p.position.isin(v23.POSITIONS) & p.target_year.between(v23.TRAIN_START, end_year)].copy()
    p = p[p.gsis_id.notna()].copy()
    p["player_id"] = p.gsis_id.astype(str)
    p["player_name"] = p.display_name.astype(str)
    p["target_year"] = p.target_year.astype(int)

    birth = pd.to_datetime(p.birth_date, errors="coerce")
    cutoff = pd.to_datetime(p.target_year.astype(str) + "-09-01", errors="coerce")
    p["age"] = (cutoff - birth).dt.days / 365.25

    for c in ["draft_year", "draft_round", "draft_pick", "weight", "height"]:
        p[c] = pd.to_numeric(p[c], errors="coerce")

    d = draft[["season", "round", "pick", "gsis_id", "pfr_player_id", "position", "age"]].copy()
    d["season"] = pd.to_numeric(d.season, errors="coerce")
    d = d[d.season.between(v23.TRAIN_START, end_year)].copy()
    d["player_id"] = d.gsis_id.astype(str)
    d = d.rename(columns={
        "season": "target_year",
        "round": "draft_round_dp",
        "pick": "draft_pick_dp",
        "pfr_player_id": "pfr_id_dp",
        "age": "draft_age_dp",
    })
    p = p.merge(
        d[["target_year", "player_id", "draft_round_dp", "draft_pick_dp", "pfr_id_dp", "draft_age_dp"]],
        on=["target_year", "player_id"],
        how="left",
    )
    p["draft_round_final"] = p.draft_round.fillna(p.draft_round_dp)
    p["draft_pick_final"] = p.draft_pick.fillna(p.draft_pick_dp)
    p["pfr_join"] = p.pfr_id.fillna(p.pfr_id_dp)
    p["age"] = p.age.fillna(p.draft_age_dp)
    p["drafted"] = p.draft_pick_final.notna().astype(float)
    p["draft_pick_model"] = p.draft_pick_final.fillna(300.0).clip(1, 300)
    p["draft_round_model"] = p.draft_round_final.fillna(8.0).clip(1, 8)
    p["log_draft_pick"] = np.log1p(p.draft_pick_model)
    p["inv_sqrt_pick"] = 1.0 / np.sqrt(p.draft_pick_model)
    p["first_round"] = p.draft_round_model.eq(1).astype(float)
    p["day2"] = p.draft_round_model.isin([2, 3]).astype(float)

    c = combine.copy()
    c["target_year"] = pd.to_numeric(c.season, errors="coerce")
    c = c[c.target_year.between(v23.TRAIN_START, end_year)].copy()
    c = c.rename(columns={"pfr_id": "pfr_join"})
    combine_cols = ["forty", "bench", "vertical", "broad_jump", "cone", "shuttle", "wt"]
    for col in combine_cols:
        c[col] = pd.to_numeric(c[col], errors="coerce")
    c = c[["target_year", "pfr_join"] + combine_cols].drop_duplicates(["target_year", "pfr_join"])
    p = p.merge(c, on=["target_year", "pfr_join"], how="left")
    p["combine_weight"] = p.wt.fillna(p.weight)
    p["speed_score"] = p.combine_weight * (200.0 / p.forty.replace(0, np.nan)) ** 4

    for pos in v23.POSITIONS:
        p[f"pos_{pos}"] = p.position.eq(pos).astype(float)
        p[f"log_pick_x_{pos}"] = p.log_draft_pick * p[f"pos_{pos}"]
        p[f"age_x_{pos}"] = p.age * p[f"pos_{pos}"]

    return p


def configure_extended_years() -> None:
    # Architecture/hyperparameters are untouched. These patches only let the
    # already-frozen data loaders see the two holdout seasons.
    v223.DEPTH_YEARS = list(range(2016, 2026))
    v223.OPENING_DATES.update(OPENING_DATES_EXT)
    v23.ALL_YEARS = list(range(v23.TRAIN_START, 2026))
    v241.ACTUAL_YEARS = ALL_ACTUAL_YEARS
    v241.FORECAST_YEARS = ALL_FORECAST_YEARS


def overlap(g: pd.DataFrame, pred: str, actual: str, k: int) -> float:
    kk = min(k, len(g))
    if kk <= 0:
        return np.nan
    a = set(g.nlargest(kk, pred).player_id)
    b = set(g.nlargest(kk, actual).player_id)
    return len(a & b) / kk


def build_holdout_table(args):
    configure_extended_years()

    raw = v21.old.load_weekly_csvs(args.data_dir)
    seasons_base = v21.aggregate_v21(raw)
    bonus_counts, bonus_diag = load_long_td_bonus(args.pbp_dir, PBP_YEARS, threshold=50)
    seasons = apply_long_td_bonus(seasons_base, bonus_counts)

    players = pd.read_csv(args.players_csv, low_memory=False)
    draft = pd.read_parquet(args.draft_parquet)
    combine = pd.read_parquet(args.combine_parquet)

    ages = v241.age_metadata(players)
    eligible_all = v241.load_week1_sets(args.roster_dir, ALL_ACTUAL_YEARS)
    eligible_forecast = {y: eligible_all[y] for y in ALL_FORECAST_YEARS}

    rook = rookie_universe_extended(players, draft, combine, 2025)
    rook = v23.attach_preseason_role(rook, args.roster_dir, args.depth_dir)
    rook = v23.attach_outcomes(rook, seasons)

    vet_table = v211.augment_table(seasons, players)
    vet_table = v22.augment_targets(vet_table, seasons)
    depth, depth_diag = v223.load_preseason_depth(args.depth_dir)
    vet_table = v223.attach_depth(vet_table, depth)
    meta = v24.player_metadata(players)

    forecasts = []
    for year in HOLDOUT_YEARS:
        forecasts.append(v24.build_forecast_pool(year, vet_table, rook, eligible_forecast, meta))
    forecasts = pd.concat(forecasts, ignore_index=True)

    annual = v241.build_actual_annual(seasons, eligible_all, ages)
    table = v24.engineer_features(v24.add_horizon_targets(forecasts, annual))
    return table, annual, bonus_diag, depth_diag, rook


def add_frozen_dynasty_scores(table: pd.DataFrame, annual: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for year in HOLDOUT_YEARS:
        g = table[table.forecast_year.eq(year)].copy().reset_index(drop=True)
        curves, pos = v241.retention_table(annual, year, SHRINK, CAP)
        r1 = v241.lookup_retention(g, curves, pos, 0)
        r2 = v241.lookup_retention(g, curves, pos, 1)
        g["retention_y2"] = r1.to_numpy()
        g["retention_y3_conditional"] = r2.to_numpy()
        g["horizon_multiplier"] = 1.0 + DISCOUNT * r1 + (DISCOUNT ** 2) * r1 * r2
        g["dynasty_signal"] = v243.dynasty_signal(g.pred_vor, g.horizon_multiplier, TAIL_MODE)
        g["overall_rank"] = g.dynasty_signal.rank(ascending=False, method="min")
        g["position_rank"] = g.groupby("position").dynasty_signal.rank(ascending=False, method="min")
        g["paris_score"] = v243.top150_scale(g.dynasty_signal, 150)
        pieces.append(g)
    return pd.concat(pieces, ignore_index=True)


def evaluate_holdout(selected: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    overall_rows = []
    position_rows = []
    target_rows = []

    for year in HOLDOUT_YEARS:
        g = selected[selected.forecast_year.eq(year)].copy()

        # The one-year test is fully observed for both holdout seasons.
        g["one_year_actual_vor"] = g["actual_vor_y1"]
        one_s = g.pred_vor.corr(g.one_year_actual_vor, method="spearman")

        # Full 3-year dynasty outcomes are not yet observable. 2024 has two
        # completed seasons (2024-25); 2025 has one completed season.
        horizon = 2 if year == 2024 else 1
        g["observed_dynasty_target"] = 0.0
        for k in range(horizon):
            g["observed_dynasty_target"] += (DISCOUNT ** k) * g[f"actual_surplus_y{k+1}"]

        overall_rows.append({
            "forecast_year": year,
            "n": len(g),
            "training_information_through": year - 1,
            "one_year_spearman": one_s,
            "one_year_top24": overlap(g, "pred_vor", "one_year_actual_vor", 24),
            "one_year_top50": overlap(g, "pred_vor", "one_year_actual_vor", 50),
            "one_year_top100": overlap(g, "pred_vor", "one_year_actual_vor", 100),
            "observed_dynasty_horizon_years": horizon,
            "dynasty_spearman": g.dynasty_signal.corr(g.observed_dynasty_target, method="spearman"),
            "dynasty_top50": overlap(g, "dynasty_signal", "observed_dynasty_target", 50),
            "dynasty_top100": overlap(g, "dynasty_signal", "observed_dynasty_target", 100),
            "dynasty_top150": overlap(g, "dynasty_signal", "observed_dynasty_target", 150),
        })

        for pos, p in g.groupby("position"):
            position_rows.append({
                "forecast_year": year,
                "position": pos,
                "n": len(p),
                "one_year_spearman": p.pred_vor.corr(p.one_year_actual_vor, method="spearman"),
                "dynasty_spearman": p.dynasty_signal.corr(p.observed_dynasty_target, method="spearman"),
            })

        target_rows.append(g)

    return pd.DataFrame(overall_rows), pd.DataFrame(position_rows), pd.concat(target_rows, ignore_index=True)


def rookie_holdout_metrics(rook: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year in HOLDOUT_YEARS:
        pred = v23.fit_candidate(rook, year, "capital_depth", 300.0)
        pred = pred[pred.week1_rostered.eq(1)].copy()
        if len(pred) == 0:
            continue
        rows.append({
            "forecast_year": year,
            "n_rookies": len(pred),
            "spearman": pred.predicted_value.corr(pred.actual_week_value, method="spearman"),
            "top10_overlap": v23.overlap(pred, 10),
            "top20_overlap": v23.overlap(pred, 20),
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--players-csv", required=True)
    ap.add_argument("--draft-parquet", required=True)
    ap.add_argument("--combine-parquet", required=True)
    ap.add_argument("--roster-dir", required=True)
    ap.add_argument("--depth-dir", required=True)
    ap.add_argument("--pbp-dir", required=True)
    ap.add_argument("--out-dir", default="outputs_v243_holdout")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    table, annual, bonus_diag, depth_diag, rook = build_holdout_table(args)
    selected = add_frozen_dynasty_scores(table, annual)
    metrics, pos_metrics, evaluated = evaluate_holdout(selected)
    rookie_metrics = rookie_holdout_metrics(rook)

    top150 = evaluated[evaluated.overall_rank.le(150)].copy().sort_values(["forecast_year", "overall_rank"])
    top25 = top150[top150.overall_rank.le(25)].copy()

    metrics.to_csv(out / "v243_holdout_metrics.csv", index=False)
    pos_metrics.to_csv(out / "v243_holdout_position_metrics.csv", index=False)
    rookie_metrics.to_csv(out / "v243_holdout_rookie_metrics.csv", index=False)
    evaluated.to_csv(out / "v243_holdout_all_rankings.csv", index=False)
    top150.to_csv(out / "v243_holdout_top150.csv", index=False)
    bonus_diag.to_csv(out / "v243_holdout_long_td_diagnostics.csv", index=False)
    depth_diag.to_csv(out / "v243_holdout_depth_diagnostics.csv", index=False)

    manifest = {
        "version": "Paris v2.4.3",
        "evaluation": "one-shot 2024-2025 holdout",
        "frozen_architecture_commit": FROZEN_ARCHITECTURE_COMMIT,
        "architecture_changed_after_opening_holdout": False,
        "rolling_origin_rule": {
            "2024_forecast_uses_information_through": 2023,
            "2025_forecast_uses_information_through": 2024,
            "explanation": "2024 is legitimately available before a 2025 preseason forecast; no 2025 outcomes are used to predict 2025.",
        },
        "frozen_settings": {
            "returning_players": "v2.2.5",
            "rookies": "v2.3 capital_depth Ridge alpha 300",
            "longevity": {"discount": DISCOUNT, "shrink": SHRINK, "cap": CAP},
            "tail": TAIL_MODE,
            "long_td_bonus": "+1 for 50+ yard TD",
            "public_scale": "Top 150 maps monotonically to 1..1000",
        },
        "outcome_availability": {
            "2024": "two completed seasons available: 2024 and 2025",
            "2025": "one completed season available: 2025",
            "full_three_year_dynasty_holdout": "not yet observable",
        },
    }
    (out / "V243_HOLDOUT_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("=== PARIS v2.4.3: 2024-2025 HOLDOUT OPENED ===")
    print("Frozen architecture commit:", FROZEN_ARCHITECTURE_COMMIT)
    print("No model/tail/hyperparameter selection was performed on 2024 or 2025 outcomes.")
    print("\n=== HOLDOUT METRICS ===")
    print(metrics.to_string(index=False))
    print("\n=== HOLDOUT POSITION METRICS ===")
    print(pos_metrics.to_string(index=False))
    print("\n=== HOLDOUT ROOKIE METRICS ===")
    print(rookie_metrics.to_string(index=False))
    print("\n=== PRESEASON TOP 25s FROM THE FROZEN MODEL ===")
    cols = [
        "forecast_year", "overall_rank", "player_name", "position", "position_rank",
        "is_rookie", "age", "pred_vor", "horizon_multiplier", "dynasty_signal",
        "paris_score", "one_year_actual_vor", "observed_dynasty_target"
    ]
    print(top25[cols].to_string(index=False))


if __name__ == "__main__":
    main()
