from __future__ import annotations

import argparse
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import paris_v2_backtest as old
from paris_v2_core import DEFAULT_CONFIG, assign_starters

POSITIONS = ["QB", "RB", "WR", "TE"]
EVAL_YEARS = [2019, 2020, 2021, 2022, 2023]
ALPHAS = [1.0, 10.0, 30.0, 100.0, 300.0]
AGE_MODES = ["linear", "quadratic"]
OPP_MODES = ["baseline", "expanded"]
ELITE_MODES = ["none", "top5", "continuous"]
MIN_TARGET_GAMES = 4


def num(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype=float)


def mean_numeric(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def aggregate_v21(raw: pd.DataFrame) -> pd.DataFrame:
    """Create one regular-season row per player-season.

    Unlike v2.0, this keeps separate opportunity components instead of collapsing
    all volume into one hand-weighted number.
    """
    df = old.normalize_columns(raw)
    if "season_type" in df.columns:
        df = df[df["season_type"].astype(str).str.upper().eq("REG")].copy()

    if "fumbles_lost" in df.columns:
        lost = num(df, "fumbles_lost")
    else:
        lost = (
            num(df, "sack_fumbles_lost")
            + num(df, "rushing_fumbles_lost")
            + num(df, "receiving_fumbles_lost")
        )

    df["paris_fp"] = (
        num(df, "passing_yards") / 25.0
        + 5.0 * num(df, "passing_tds")
        - 2.0 * num(df, "interceptions")
        + num(df, "rushing_yards") / 10.0
        + 6.0 * num(df, "rushing_tds")
        + num(df, "receiving_yards") / 10.0
        + 6.0 * num(df, "receiving_tds")
        + num(df, "receptions")
        - 2.0 * lost
    )
    df["paris_fumbles_lost"] = lost

    sum_cols = [
        "paris_fp", "passing_yards", "passing_tds", "interceptions", "attempts",
        "completions", "sacks", "passing_first_downs", "carries", "rushing_yards",
        "rushing_tds", "rushing_first_downs", "targets", "receptions",
        "receiving_yards", "receiving_tds", "receiving_first_downs",
        "receiving_air_yards", "paris_fumbles_lost",
    ]
    for c in sum_cols:
        if c not in df.columns:
            df[c] = 0.0

    # Weekly shares are already normalized by team volume, so average them over
    # the player's observed regular-season weeks rather than summing them.
    for c in ["target_share", "air_yards_share", "wopr"]:
        if c not in df.columns:
            df[c] = np.nan
        else:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    group_cols = ["season", "player_id", "player_name", "position"]
    named = {c: (c, "sum") for c in sum_cols}
    named.update({
        "target_share_avg": ("target_share", "mean"),
        "air_yards_share_avg": ("air_yards_share", "mean"),
        "wopr_avg": ("wopr", "mean"),
    })
    if "week" in df.columns:
        named["games"] = ("week", "nunique")
    else:
        named["games"] = ("paris_fp", "size")

    s = df.groupby(group_cols, dropna=False).agg(**named).reset_index()
    g = s["games"].clip(lower=1)
    s["ppg"] = s["paris_fp"] / g

    # Separate role components. These are the key opportunity upgrade in v2.1.
    s["attempts_pg"] = s["attempts"] / g
    s["completions_pg"] = s["completions"] / g
    s["dropbacks_pg"] = (s["attempts"] + s["sacks"]) / g
    s["carries_pg"] = s["carries"] / g
    s["targets_pg"] = s["targets"] / g
    s["receptions_pg"] = s["receptions"] / g
    s["air_yards_pg"] = s["receiving_air_yards"] / g
    s["pass_fd_pg"] = s["passing_first_downs"] / g
    s["rush_fd_pg"] = s["rushing_first_downs"] / g
    s["rec_fd_pg"] = s["receiving_first_downs"] / g

    # Retain the v2.0 opportunity feature so expanded opportunity can be tested
    # against, rather than assumed better than, the old baseline.
    s["opportunity"] = np.select(
        [s.position.eq("QB"), s.position.eq("RB"), s.position.isin(["WR", "TE"])],
        [s["attempts_pg"] + 0.75 * s["carries_pg"],
         s["carries_pg"] + 1.5 * s["targets_pg"],
         s["targets_pg"]],
        default=0.0,
    )

    touches = (s["carries"] + s["targets"]).replace(0, np.nan)
    pass_att = s["attempts"].replace(0, np.nan)
    s["efficiency"] = np.select(
        [s.position.eq("QB"), s.position.eq("RB"), s.position.isin(["WR", "TE"])],
        [((s["passing_yards"] + s["rushing_yards"]) / pass_att).fillna(0),
         ((s["rushing_yards"] + s["receiving_yards"]) / touches).fillna(0),
         (s["receiving_yards"] / s["targets"].replace(0, np.nan)).fillna(0)],
        default=0.0,
    )

    s["pos_pct"] = s.groupby(["season", "position"])["ppg"].rank(pct=True, method="average")
    s["top5"] = (
        s.groupby(["season", "position"])["ppg"].rank(ascending=False, method="min") <= 5
    ).astype(int)
    return s


def add_metadata(seasons: pd.DataFrame, players: pd.DataFrame | None) -> pd.DataFrame:
    out = seasons.copy()
    if players is None:
        out["age"] = np.nan
        out["draft_pick"] = 300.0
        return out

    p = players.copy()
    if "gsis_id" in p.columns and "player_id" not in p.columns:
        p = p.rename(columns={"gsis_id": "player_id"})
    if "draft_number" in p.columns and "draft_pick" not in p.columns:
        p = p.rename(columns={"draft_number": "draft_pick"})

    keep = [c for c in ["player_id", "birth_date", "draft_pick"] if c in p.columns]
    if "player_id" not in keep:
        out["age"] = np.nan
        out["draft_pick"] = 300.0
        return out

    out = out.merge(p[keep].drop_duplicates("player_id"), on="player_id", how="left")
    if "birth_date" in out.columns:
        birth = pd.to_datetime(out["birth_date"], errors="coerce")
        season_day = pd.to_datetime(out["season"].astype(str) + "-09-01", errors="coerce")
        out["age"] = (season_day - birth).dt.days / 365.25
    else:
        out["age"] = np.nan

    out["draft_pick"] = pd.to_numeric(out.get("draft_pick", 300.0), errors="coerce").fillna(300.0).clip(1, 300)
    return out


HISTORY_COLS = [
    "ppg", "pos_pct", "top5", "opportunity", "efficiency", "games",
    "attempts_pg", "completions_pg", "dropbacks_pg", "carries_pg", "targets_pg",
    "receptions_pg", "air_yards_pg", "pass_fd_pg", "rush_fd_pg", "rec_fd_pg",
    "target_share_avg", "air_yards_share_avg", "wopr_avg",
]


def exact_lag_table(seasons: pd.DataFrame, players: pd.DataFrame | None) -> pd.DataFrame:
    """Build prediction rows with exact calendar-year lags.

    A player's 2018 row can only use 2018/2017/2016 information and can only be
    scored against 2019. Missing seasons remain missing instead of silently jumping
    across a year.
    """
    hist = add_metadata(seasons, players)
    base = hist[["season", "player_id", "player_name", "position", "age", "draft_pick"]].copy()
    base["target_year"] = base["season"] + 1
    base["age_next"] = base["age"] + 1.0
    base["age_next_sq"] = base["age_next"] ** 2

    for lag, offset in [(1, 0), (2, 1), (3, 2)]:
        cols = ["season", "player_id"] + HISTORY_COLS
        h = hist[cols].copy()
        h["season"] = h["season"] + offset
        h = h.rename(columns={c: f"{c}_lag{lag}" for c in HISTORY_COLS})
        base = base.merge(h, on=["season", "player_id"], how="left")

    base["ppg_trend"] = base["ppg_lag1"] - base["ppg_lag2"]
    base["opp_trend"] = base["opportunity_lag1"] - base["opportunity_lag2"]
    base["elite_top5"] = (
        0.50 * base["top5_lag1"].fillna(0)
        + 0.30 * base["top5_lag2"].fillna(0)
        + 0.20 * base["top5_lag3"].fillna(0)
    )
    base["elite_cont"] = (
        0.50 * base["pos_pct_lag1"].fillna(0).pow(3)
        + 0.30 * base["pos_pct_lag2"].fillna(0).pow(3)
        + 0.20 * base["pos_pct_lag3"].fillna(0).pow(3)
    )

    # Exact next-season outcome join: no groupby.shift survivorship/misalignment bug.
    target = hist[["season", "player_id", "ppg", "games"]].copy()
    target = target.rename(columns={"season": "target_year", "ppg": "target_ppg", "games": "target_games"})
    base = base.merge(target, on=["target_year", "player_id"], how="left")
    base.loc[base["target_games"].fillna(0) < MIN_TARGET_GAMES, "target_ppg"] = np.nan
    return base


def common_features(age_mode: str) -> list[str]:
    f = [
        "ppg_lag1", "ppg_lag2", "ppg_lag3",
        "efficiency_lag1", "efficiency_lag2",
        "games_lag1", "games_lag2",
        "ppg_trend", "draft_pick",
    ]
    if age_mode == "linear":
        f += ["age_next"]
    elif age_mode == "quadratic":
        f += ["age_next", "age_next_sq"]
    else:
        raise ValueError(age_mode)
    return f


def opportunity_features(pos: str, mode: str) -> list[str]:
    if mode == "baseline":
        return ["opportunity_lag1", "opportunity_lag2", "opportunity_lag3", "opp_trend"]
    if mode != "expanded":
        raise ValueError(mode)

    if pos == "QB":
        return [
            "attempts_pg_lag1", "attempts_pg_lag2",
            "carries_pg_lag1", "carries_pg_lag2",
            "dropbacks_pg_lag1", "pass_fd_pg_lag1",
        ]
    if pos == "RB":
        return [
            "carries_pg_lag1", "carries_pg_lag2",
            "targets_pg_lag1", "targets_pg_lag2",
            "receptions_pg_lag1", "target_share_avg_lag1",
            "rush_fd_pg_lag1", "rec_fd_pg_lag1",
        ]
    if pos in ["WR", "TE"]:
        return [
            "targets_pg_lag1", "targets_pg_lag2",
            "receptions_pg_lag1", "receptions_pg_lag2",
            "target_share_avg_lag1", "air_yards_share_avg_lag1",
            "wopr_avg_lag1", "air_yards_pg_lag1", "rec_fd_pg_lag1",
        ]
    raise ValueError(pos)


def elite_features(mode: str) -> list[str]:
    if mode == "none":
        return []
    if mode == "top5":
        return ["elite_top5"]
    if mode == "continuous":
        return ["elite_cont"]
    raise ValueError(mode)


def feature_list(pos: str, age_mode: str, opp_mode: str, elite_mode: str) -> list[str]:
    return common_features(age_mode) + opportunity_features(pos, opp_mode) + elite_features(elite_mode)


def make_model(alpha: float) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
        ("ridge", Ridge(alpha=alpha)),
    ])


def spear(y: pd.Series, pred: np.ndarray) -> float:
    return float(pd.Series(y.to_numpy()).corr(pd.Series(pred), method="spearman"))


def run_grid(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict] = []
    pred_frames: list[pd.DataFrame] = []

    for year, pos, age_mode, opp_mode, elite_mode, alpha in product(
        EVAL_YEARS, POSITIONS, AGE_MODES, OPP_MODES, ELITE_MODES, ALPHAS
    ):
        feats = feature_list(pos, age_mode, opp_mode, elite_mode)
        train = table[
            table["position"].eq(pos)
            & table["target_year"].lt(year)
            & table["target_ppg"].notna()
        ].copy()
        test = table[
            table["position"].eq(pos)
            & table["target_year"].eq(year)
            & table["target_ppg"].notna()
        ].copy()
        if len(train) < 30 or len(test) < 5:
            continue

        model = make_model(alpha)
        model.fit(train[feats], train["target_ppg"])
        pred = model.predict(test[feats])
        metric_rows.append({
            "year": year, "position": pos, "age_mode": age_mode,
            "opp_mode": opp_mode, "elite_mode": elite_mode, "alpha": alpha,
            "n_train": len(train), "n_test": len(test),
            "spearman": spear(test["target_ppg"], pred),
            "mae": mean_absolute_error(test["target_ppg"], pred),
        })
        p = test[["target_year", "player_id", "player_name", "position", "target_ppg", "target_games"]].copy()
        p["predicted_ppg"] = pred
        p["age_mode"] = age_mode
        p["opp_mode"] = opp_mode
        p["elite_mode"] = elite_mode
        p["alpha"] = alpha
        pred_frames.append(p)

    return pd.DataFrame(metric_rows), pd.concat(pred_frames, ignore_index=True)


def select_position_models(metrics: pd.DataFrame) -> pd.DataFrame:
    """Select without letting 2023 freely search the whole grid.

    2019-2022 identify the three strongest candidates at each position. 2023 then
    acts as final validation/tuning only among those three.
    """
    keys = ["position", "age_mode", "opp_mode", "elite_mode", "alpha"]
    pre = metrics[metrics.year.lt(2023)].groupby(keys, as_index=False).agg(
        pre_spearman=("spearman", "mean"),
        pre_mae=("mae", "mean"),
        pre_sd=("spearman", "std"),
    )
    val = metrics[metrics.year.eq(2023)][keys + ["spearman", "mae"]].rename(
        columns={"spearman": "val_spearman", "mae": "val_mae"}
    )
    z = pre.merge(val, on=keys, how="inner")
    z["pre_rank"] = z.groupby("position")["pre_spearman"].rank(ascending=False, method="first")
    finalists = z[z.pre_rank.le(3)].copy()
    finalists = finalists.sort_values(
        ["position", "val_spearman", "pre_spearman", "val_mae"],
        ascending=[True, False, False, True],
    )
    selected = finalists.groupby("position", as_index=False).head(1).copy()
    return selected, z.sort_values(["position", "pre_rank"])


def hybrid_predictions(preds: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for _, r in selected.iterrows():
        p = preds[
            preds.position.eq(r.position)
            & preds.age_mode.eq(r.age_mode)
            & preds.opp_mode.eq(r.opp_mode)
            & preds.elite_mode.eq(r.elite_mode)
            & preds.alpha.eq(r.alpha)
        ].copy()
        pieces.append(p)
    return pd.concat(pieces, ignore_index=True)


def overall_eval(hybrid: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, ranks = [], []
    for year, g in hybrid.groupby("target_year"):
        b = g.drop_duplicates("player_id").copy()
        if not set(POSITIONS).issubset(set(b.position)):
            continue
        _, pred_repl = assign_starters(
            b[["player_id", "position", "predicted_ppg"]], DEFAULT_CONFIG, "predicted_ppg"
        )
        actual = b[["player_id", "position", "target_ppg"]].rename(columns={"target_ppg": "actual_ppg"})
        _, actual_repl = assign_starters(actual, DEFAULT_CONFIG, "actual_ppg")
        b["pred_replacement"] = b.position.map(pred_repl)
        b["actual_replacement"] = b.position.map(actual_repl)
        b["pred_vor"] = b.predicted_ppg - b.pred_replacement
        b["actual_vor"] = b.target_ppg - b.actual_replacement
        b["overall_rank"] = b.pred_vor.rank(ascending=False, method="min")
        b["actual_rank"] = b.actual_vor.rank(ascending=False, method="min")

        def overlap(k: int) -> float:
            k = min(k, len(b))
            a = set(b.nlargest(k, "pred_vor").player_id)
            c = set(b.nlargest(k, "actual_vor").player_id)
            return len(a & c) / max(k, 1)

        rows.append({
            "year": int(year), "n": len(b),
            "overall_spearman": b.pred_vor.corr(b.actual_vor, method="spearman"),
            "top12_overlap": overlap(12), "top24_overlap": overlap(24), "top50_overlap": overlap(50),
        })
        ranks.append(b)
    return pd.DataFrame(rows), pd.concat(ranks, ignore_index=True)


def age_curve_coefficients(table: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    """Fit selected 2023 models and expose learned age coefficients for inspection."""
    rows = []
    for _, r in selected.iterrows():
        pos = r.position
        feats = feature_list(pos, r.age_mode, r.opp_mode, r.elite_mode)
        train = table[
            table.position.eq(pos) & table.target_year.le(2023) & table.target_ppg.notna()
        ].copy()
        model = make_model(float(r.alpha))
        model.fit(train[feats], train.target_ppg)
        names = list(model.named_steps["impute"].get_feature_names_out(feats))
        coefs = model.named_steps["ridge"].coef_
        for nme, coef in zip(names, coefs):
            if nme in ["age_next", "age_next_sq"]:
                rows.append({"position": pos, "age_mode": r.age_mode, "feature": nme, "standardized_coef": coef})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--players-csv")
    ap.add_argument("--out-dir", default="outputs_v21")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    raw = old.load_weekly_csvs(args.data_dir)
    seasons = aggregate_v21(raw)
    players = pd.read_csv(args.players_csv, low_memory=False) if args.players_csv else None
    table = exact_lag_table(seasons, players)

    metrics, preds = run_grid(table)
    selected, all_candidates = select_position_models(metrics)
    hybrid = hybrid_predictions(preds, selected)
    overall, rankings = overall_eval(hybrid)
    age_coefs = age_curve_coefficients(table, selected)

    metrics.to_csv(out / "v21_grid_metrics.csv", index=False)
    preds.to_csv(out / "v21_grid_predictions.csv", index=False)
    all_candidates.to_csv(out / "v21_position_candidates.csv", index=False)
    selected.to_csv(out / "v21_selected_position_models.csv", index=False)
    overall.to_csv(out / "v21_overall_metrics.csv", index=False)
    rankings.to_csv(out / "v21_hybrid_rankings.csv", index=False)
    age_coefs.to_csv(out / "v21_age_coefficients.csv", index=False)

    print("=== PARIS V2.1 SELECTED POSITION MODELS ===")
    print(selected[["position", "age_mode", "opp_mode", "elite_mode", "alpha", "pre_spearman", "val_spearman", "pre_mae", "val_mae"]].to_string(index=False))
    print("\n=== PARIS V2.1 OVERALL ROLLING RESULTS ===")
    print(overall.to_string(index=False))
    print("\nPre-2023 mean overall Spearman:", round(overall[overall.year < 2023].overall_spearman.mean(), 4))
    if (overall.year == 2023).any():
        print("2023 validation overall Spearman:", round(float(overall.loc[overall.year == 2023, "overall_spearman"].iloc[0]), 4))

    y = rankings[rankings.target_year.eq(2023)].sort_values("overall_rank").head(35)
    print("\n=== 2023 VALIDATION TOP 35 (RETURNING PLAYERS ONLY) ===")
    print(y[["overall_rank", "player_name", "position", "predicted_ppg", "target_ppg", "pred_vor", "actual_vor"]].to_string(index=False))


if __name__ == "__main__":
    main()
