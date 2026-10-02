from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


PBP_COLUMNS = [
    "season_type",
    "yards_gained",
    "pass_touchdown",
    "rush_touchdown",
    "passer_player_id",
    "receiver_player_id",
    "rusher_player_id",
]


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(0.0, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").fillna(0.0)


def _credits(ids: pd.Series, year: int, credit_type: str) -> pd.DataFrame:
    out = pd.DataFrame({"player_id": ids})
    out = out[out.player_id.notna()].copy()
    out["player_id"] = out.player_id.astype(str)
    out = out[~out.player_id.isin(["", "nan", "None"])].copy()
    out["season"] = int(year)
    out["credit_type"] = credit_type
    out["bonus_count"] = 1
    return out


def long_td_credits_for_year(path: str | Path, year: int, threshold: int = 50) -> pd.DataFrame:
    """Return one row per fantasy-player credit on a 50+ yard offensive TD.

    A 50+ yard passing touchdown credits both the passer and the receiver because
    both fantasy players score a touchdown on the play. A 50+ yard rushing
    touchdown credits the rusher. Return/defensive touchdowns are intentionally
    excluded because the Paris league scoring specification does not score them.
    """
    path = Path(path)
    schema = set(pq.ParquetFile(path).schema.names)
    missing = {"yards_gained", "pass_touchdown", "rush_touchdown"} - schema
    if missing:
        raise ValueError(f"PBP file {path} missing required columns: {sorted(missing)}")

    cols = [c for c in PBP_COLUMNS if c in schema]
    df = pd.read_parquet(path, columns=cols)
    if "season_type" in df.columns:
        df = df[df.season_type.astype(str).str.upper().eq("REG")].copy()

    yards = _num(df, "yards_gained")
    pass_long = _num(df, "pass_touchdown").eq(1) & yards.ge(threshold)
    rush_long = _num(df, "rush_touchdown").eq(1) & yards.ge(threshold)

    pieces: list[pd.DataFrame] = []
    if "passer_player_id" in df.columns:
        pieces.append(_credits(df.loc[pass_long, "passer_player_id"], year, "passer"))
    if "receiver_player_id" in df.columns:
        pieces.append(_credits(df.loc[pass_long, "receiver_player_id"], year, "receiver"))
    if "rusher_player_id" in df.columns:
        pieces.append(_credits(df.loc[rush_long, "rusher_player_id"], year, "rusher"))

    if not pieces:
        return pd.DataFrame(columns=["season", "player_id", "credit_type", "bonus_count"])
    return pd.concat(pieces, ignore_index=True)


def load_long_td_bonus(pbp_dir: str | Path, years: list[int], threshold: int = 50) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load PBP files and return player-season bonus counts plus diagnostics."""
    all_credits: list[pd.DataFrame] = []
    for year in years:
        path = Path(pbp_dir) / f"play_by_play_{year}.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        all_credits.append(long_td_credits_for_year(path, year, threshold))

    credits = pd.concat(all_credits, ignore_index=True) if all_credits else pd.DataFrame()
    if len(credits) == 0:
        counts = pd.DataFrame(columns=["season", "player_id", "long_td_50_count"])
        diag = pd.DataFrame(columns=["season", "fantasy_bonus_credits", "unique_players"])
        return counts, diag

    counts = credits.groupby(["season", "player_id"], as_index=False).bonus_count.sum()
    counts = counts.rename(columns={"bonus_count": "long_td_50_count"})

    by_type = credits.groupby(["season", "credit_type"], as_index=False).bonus_count.sum()
    pivot = by_type.pivot(index="season", columns="credit_type", values="bonus_count").fillna(0).reset_index()
    for c in ["passer", "receiver", "rusher"]:
        if c not in pivot.columns:
            pivot[c] = 0
    totals = counts.groupby("season", as_index=False).agg(
        fantasy_bonus_credits=("long_td_50_count", "sum"),
        unique_players=("player_id", "nunique"),
        max_player_credits=("long_td_50_count", "max"),
    )
    diag = totals.merge(pivot[["season", "passer", "receiver", "rusher"]], on="season", how="left")
    return counts, diag


def apply_long_td_bonus(seasons: pd.DataFrame, counts: pd.DataFrame) -> pd.DataFrame:
    """Add +1 per long-TD credit and recompute production ranks used by Paris."""
    out = seasons.copy()
    out["player_id"] = out.player_id.astype(str)
    c = counts.copy()
    c["player_id"] = c.player_id.astype(str)
    out = out.merge(c, on=["season", "player_id"], how="left")
    out["long_td_50_count"] = pd.to_numeric(out.long_td_50_count, errors="coerce").fillna(0.0)
    out["paris_fp"] = pd.to_numeric(out.paris_fp, errors="coerce").fillna(0.0) + out.long_td_50_count
    games = pd.to_numeric(out.games, errors="coerce").fillna(0.0).clip(lower=1.0)
    out["ppg"] = out.paris_fp / games
    out["pos_pct"] = out.groupby(["season", "position"])["ppg"].rank(pct=True, method="average")
    out["top5"] = (
        out.groupby(["season", "position"])["ppg"].rank(ascending=False, method="min") <= 5
    ).astype(int)
    return out


def value_scale_1000(values: pd.Series) -> pd.Series:
    """Linear 0-1000 value index that preserves the exact ordering of raw value.

    Zero/below-replacement value maps to 0. The highest positive dynasty value in
    the pool maps to 1000. Everyone else is proportional to that top value.
    """
    x = pd.to_numeric(values, errors="coerce").fillna(0.0).clip(lower=0.0)
    top = float(x.max()) if len(x) else 0.0
    if not np.isfinite(top) or top <= 0:
        return pd.Series(0.0, index=values.index)
    return 1000.0 * x / top
