from __future__ import annotations

import argparse
import glob
import os
import re
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

POSITIONS = ["QB", "RB", "WR", "TE"]


def _col(df: pd.DataFrame, name: str, default=0.0) -> pd.Series:
    if name in df:
        return pd.to_numeric(df[name], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype=float)


def load_weekly_csvs(data_dir: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(data_dir, "player_stats_*.csv")))
    if not files:
        raise FileNotFoundError(f"No player_stats_*.csv files found in {data_dir}")
    frames = []
    for f in files:
        if "kicking" in os.path.basename(f):
            continue
        df = pd.read_csv(f, low_memory=False)
        if "season" not in df.columns:
            m = re.search(r"(20\d{2}|19\d{2})", os.path.basename(f))
            if not m:
                raise ValueError(f"Could not infer season from {f}")
            df["season"] = int(m.group(1))
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rename = {}
    if "player_display_name" in out and "player_name" not in out:
        rename["player_display_name"] = "player_name"
    if "recent_team" in out and "team" not in out:
        rename["recent_team"] = "team"
    out = out.rename(columns=rename)
    if "player_id" not in out.columns:
        if "player_name" not in out.columns:
            raise ValueError("Need player_id or player_name")
        out["player_id"] = out["player_name"].astype(str)
    if "position" not in out.columns:
        raise ValueError("Dataset needs position")
    out["position"] = out["position"].astype(str).str.upper()
    return out[out["position"].isin(POSITIONS)].copy()


def add_custom_fantasy_points(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["paris_fp"] = (
        _col(out, "passing_yards") / 25.0
        + _col(out, "passing_tds") * 5.0
        - _col(out, "interceptions") * 2.0
        + _col(out, "rushing_yards") / 10.0
        + _col(out, "rushing_tds") * 6.0
        + _col(out, "receiving_yards") / 10.0
        + _col(out, "receiving_tds") * 6.0
        + _col(out, "receptions")
        - _col(out, "fumbles_lost") * 2.0
        + _col(out, "long_td_50_count")
    )
    return out


def aggregate_season(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(add_custom_fantasy_points(normalize_columns(df)))
    group_cols = ["season", "player_id", "player_name", "position"]
    if "team" in df.columns:
        group_cols.append("team")
    sum_cols = [
        "paris_fp", "passing_yards", "passing_tds", "interceptions", "attempts",
        "rushing_yards", "rushing_tds", "carries", "receiving_yards",
        "receiving_tds", "receptions", "targets", "fumbles_lost"
    ]
    for c in sum_cols:
        if c not in df.columns:
            df[c] = 0.0
    agg = df.groupby(group_cols, dropna=False).agg(
        games=("paris_fp", "size"),
        **{c: (c, "sum") for c in sum_cols}
    ).reset_index()
    agg["ppg"] = agg["paris_fp"] / agg["games"].clip(lower=1)
    agg["opportunity"] = np.select(
        [agg.position.eq("QB"), agg.position.eq("RB"), agg.position.isin(["WR", "TE"])],
        [agg["attempts"] + 0.75 * agg["carries"],
         agg["carries"] + 1.5 * agg["targets"],
         agg["targets"]], default=0.0,
    ) / agg["games"].clip(lower=1)
    touches = (agg["carries"] + agg["targets"]).replace(0, np.nan)
    pass_att = agg["attempts"].replace(0, np.nan)
    agg["efficiency"] = np.select(
        [agg.position.eq("QB"), agg.position.eq("RB"), agg.position.isin(["WR", "TE"])],
        [((agg["passing_yards"] + agg["rushing_yards"]) / pass_att).fillna(0),
         ((agg["rushing_yards"] + agg["receiving_yards"]) / touches).fillna(0),
         (agg["receiving_yards"] / agg["targets"].replace(0, np.nan)).fillna(0)], default=0.0,
    )
    agg["pos_pct"] = agg.groupby(["season", "position"])["ppg"].rank(pct=True, method="average")
    agg["top5"] = (
        agg.groupby(["season", "position"])["ppg"].rank(ascending=False, method="min") <= 5
    ).astype(int)
    if "team" in agg.columns:
        team = agg.groupby(["season", "team"]).agg(team_fp=("paris_fp", "sum"), team_games=("games", "max")).reset_index()
        team["team_fp_pg"] = team["team_fp"] / team["team_games"].clip(lower=1)
        agg = agg.merge(team[["season", "team", "team_fp_pg"]], on=["season", "team"], how="left")
    else:
        agg["team_fp_pg"] = np.nan
    return agg


def attach_player_metadata(season_df: pd.DataFrame, players: pd.DataFrame | None) -> pd.DataFrame:
    out = season_df.copy()
    if players is None:
        out["age"] = np.nan
        out["draft_pick"] = np.nan
        return out
    p = players.copy()
    if "gsis_id" in p.columns and "player_id" not in p.columns:
        p = p.rename(columns={"gsis_id": "player_id"})
    keep = [c for c in ["player_id", "birth_date", "draft_number", "draft_pick"] if c in p.columns]
    if "player_id" not in keep:
        out["age"] = np.nan
        out["draft_pick"] = np.nan
        return out
    p = p[keep].drop_duplicates("player_id")
    if "draft_number" in p and "draft_pick" not in p:
        p = p.rename(columns={"draft_number": "draft_pick"})
    out = out.merge(p, on="player_id", how="left")
    if "birth_date" in out:
        birth = pd.to_datetime(out["birth_date"], errors="coerce")
        season_date = pd.to_datetime(out["season"].astype(str) + "-09-01", errors="coerce")
        out["age"] = (season_date - birth).dt.days / 365.25
    else:
        out["age"] = np.nan
    if "draft_pick" not in out:
        out["draft_pick"] = np.nan
    return out


def build_model_table(season_df: pd.DataFrame, players: pd.DataFrame | None = None) -> pd.DataFrame:
    df = attach_player_metadata(season_df, players)
    df = df.sort_values(["player_id", "season"]).copy()
    lag_cols = ["ppg", "pos_pct", "top5", "opportunity", "efficiency", "games", "team_fp_pg"]
    for lag in [1, 2, 3]:
        for c in lag_cols:
            df[f"{c}_lag{lag}"] = df.groupby("player_id")[c].shift(lag)
    df["ppg_trend"] = df["ppg_lag1"] - df["ppg_lag2"]
    df["opp_trend"] = df["opportunity_lag1"] - df["opportunity_lag2"]
    df["elite_top5"] = 0.50*df["top5_lag1"].fillna(0) + 0.30*df["top5_lag2"].fillna(0) + 0.20*df["top5_lag3"].fillna(0)
    df["elite_cont"] = (
        0.50*df["pos_pct_lag1"].fillna(0).pow(3)
        + 0.30*df["pos_pct_lag2"].fillna(0).pow(3)
        + 0.20*df["pos_pct_lag3"].fillna(0).pow(3)
    )
    df["target_year"] = df["season"] + 1
    df["target_ppg"] = df.groupby("player_id")["ppg"].shift(-1)
    return df


BASE_FEATURES = [
    "ppg_lag1", "ppg_lag2", "ppg_lag3",
    "opportunity_lag1", "opportunity_lag2", "opportunity_lag3",
    "efficiency_lag1", "efficiency_lag2", "efficiency_lag3",
    "games_lag1", "games_lag2", "games_lag3",
    "team_fp_pg_lag1", "ppg_trend", "opp_trend", "age", "draft_pick",
]


def features_for_variant(variant: str) -> list[str]:
    if variant == "none":
        return BASE_FEATURES
    if variant == "top5":
        return BASE_FEATURES + ["elite_top5"]
    if variant == "continuous":
        return BASE_FEATURES + ["elite_cont"]
    raise ValueError(variant)


def make_ridge(alpha: float = 10.0) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
        ("model", Ridge(alpha=alpha)),
    ])


def spearman(y_true: pd.Series, y_pred: np.ndarray) -> float:
    return float(pd.Series(y_true).corr(pd.Series(y_pred, index=y_true.index), method="spearman"))


def backtest_one_year(table: pd.DataFrame, prediction_year: int, variant: str, alpha: float):
    features = features_for_variant(variant)
    all_preds, metrics = [], []
    for pos in POSITIONS:
        train = table[(table["position"] == pos) & (table["target_year"] < prediction_year) & table["target_ppg"].notna()].copy()
        test = table[(table["position"] == pos) & (table["target_year"] == prediction_year) & table["target_ppg"].notna()].copy()
        if len(train) < 30 or len(test) < 5:
            continue
        model = make_ridge(alpha)
        model.fit(train[features], train["target_ppg"])
        pred = model.predict(test[features])
        test["predicted_ppg"] = pred
        test["variant"] = variant
        test["alpha"] = alpha
        all_preds.append(test[["target_year", "player_id", "player_name", "position", "target_ppg", "predicted_ppg", "variant", "alpha"]])
        metrics.append({
            "year": prediction_year, "position": pos, "variant": variant, "alpha": alpha,
            "n_train": len(train), "n_test": len(test),
            "mae": mean_absolute_error(test["target_ppg"], pred),
            "spearman": spearman(test["target_ppg"], pred),
        })
    return (pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame(), pd.DataFrame(metrics))


def grid_backtest(table: pd.DataFrame, years: Iterable[int]):
    metrics, preds = [], []
    for year in years:
        for variant in ["none", "top5", "continuous"]:
            for alpha in [0.1, 1.0, 10.0, 30.0, 100.0]:
                p, m = backtest_one_year(table, year, variant, alpha)
                if len(p): preds.append(p)
                if len(m): metrics.append(m)
    return pd.concat(metrics, ignore_index=True), pd.concat(preds, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--players-csv")
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--eval-years", nargs="+", type=int, default=[2019, 2020, 2021, 2022, 2023])
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    weekly = load_weekly_csvs(args.data_dir)
    seasons = aggregate_season(weekly)
    players = pd.read_csv(args.players_csv, low_memory=False) if args.players_csv else None
    table = build_model_table(seasons, players)
    metrics, preds = grid_backtest(table, args.eval_years)
    metrics.to_csv(os.path.join(args.out_dir, "model_grid_metrics.csv"), index=False)
    preds.to_csv(os.path.join(args.out_dir, "rolling_predictions.csv"), index=False)
    pre = metrics[metrics.year < 2023].groupby(["variant", "alpha", "position"], as_index=False).agg(
        mean_spearman=("spearman", "mean"), mean_mae=("mae", "mean")
    )
    pre.to_csv(os.path.join(args.out_dir, "pre2023_summary.csv"), index=False)
    print(metrics.sort_values(["year", "position", "spearman"], ascending=[True, True, False]).to_string(index=False))


if __name__ == "__main__":
    main()
