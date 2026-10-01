from __future__ import annotations

from dataclasses import dataclass
import math
import pandas as pd


@dataclass(frozen=True)
class LeagueConfig:
    teams: int = 12
    qb: int = 1
    rb: int = 2
    wr: int = 3
    te: int = 2
    flex: int = 2
    superflex: int = 1
    ppr: float = 1.0
    pass_yards_per_point: float = 25.0
    pass_td: float = 5.0
    interception: float = -2.0
    rush_rec_yards_per_point: float = 10.0
    rush_rec_td: float = 6.0
    fumble_lost: float = -2.0
    long_td_bonus: float = 1.0
    long_td_threshold: int = 50


DEFAULT_CONFIG = LeagueConfig()


def fantasy_points(row: pd.Series | dict, config: LeagueConfig = DEFAULT_CONFIG) -> float:
    """Score a player stat row under the Paris league rules.

    The optional long_td_50_count field should count 50+ yard TDs credited to the
    player. For a passing TD, this currently assumes the QB receives the bonus as
    well as the scoring receiver; that convention can be changed in PBP preprocessing.
    """
    def g(name: str) -> float:
        v = row.get(name, 0.0)
        if pd.isna(v):
            return 0.0
        return float(v)

    return (
        g("passing_yards") / config.pass_yards_per_point
        + g("passing_tds") * config.pass_td
        + g("interceptions") * config.interception
        + g("rushing_yards") / config.rush_rec_yards_per_point
        + g("rushing_tds") * config.rush_rec_td
        + g("receiving_yards") / config.rush_rec_yards_per_point
        + g("receiving_tds") * config.rush_rec_td
        + g("receptions") * config.ppr
        + g("fumbles_lost") * config.fumble_lost
        + g("long_td_50_count") * config.long_td_bonus
    )


def elite_persistence(top5_lag1: int, top5_lag2: int, top5_lag3: int) -> float:
    """Transparent candidate superstar feature: 0 to 1."""
    return 0.50 * top5_lag1 + 0.30 * top5_lag2 + 0.20 * top5_lag3


def continuous_elite_persistence(
    pct_lag1: float, pct_lag2: float, pct_lag3: float, exponent: float = 3.0
) -> float:
    """Smooth superstar feature.

    Inputs are positional percentiles on 0..1, where 1 is best. Cubing makes the
    very top of the distribution matter much more without an arbitrary rank cliff.
    """
    vals = [max(0.0, min(1.0, float(x))) for x in (pct_lag1, pct_lag2, pct_lag3)]
    weights = (0.50, 0.30, 0.20)
    return sum(w * (v ** exponent) for w, v in zip(weights, vals))


def stud_adjusted_vor(vor: float, gamma: float = 1.0) -> float:
    """Convex value-over-replacement transformation.

    gamma=1 is no stud premium. gamma>1 gives disproportionate value to players
    far above replacement. Gamma must be selected on validation data, not by feel.
    """
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    return math.copysign(abs(float(vor)) ** gamma, float(vor))


def assign_starters(
    projections: pd.DataFrame,
    config: LeagueConfig = DEFAULT_CONFIG,
    value_col: str = "projected_ppg",
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Fill a theoretical league starting lineup and estimate replacement PPG."""
    required = {"player_id", "position", value_col}
    missing = required - set(projections.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")

    df = projections.copy()
    df = df[df["position"].isin(["QB", "RB", "WR", "TE"])].copy()
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=[value_col]).sort_values(value_col, ascending=False)
    df["starter_slot"] = None

    mandatory = {
        "QB": config.teams * config.qb,
        "RB": config.teams * config.rb,
        "WR": config.teams * config.wr,
        "TE": config.teams * config.te,
    }

    chosen: set = set()
    for pos, count in mandatory.items():
        idx = df[(df["position"] == pos) & (~df.index.isin(chosen))].head(count).index
        df.loc[idx, "starter_slot"] = pos
        chosen.update(idx)

    flex_eligible = df[(df["position"].isin(["RB", "WR", "TE"])) & (~df.index.isin(chosen))]
    flex_idx = flex_eligible.head(config.teams * config.flex).index
    df.loc[flex_idx, "starter_slot"] = "FLEX"
    chosen.update(flex_idx)

    sf_eligible = df[~df.index.isin(chosen)]
    sf_idx = sf_eligible.head(config.teams * config.superflex).index
    df.loc[sf_idx, "starter_slot"] = "SUPERFLEX"
    chosen.update(sf_idx)

    replacement: dict[str, float] = {}
    for pos in ["QB", "RB", "WR", "TE"]:
        bench = df[(df["position"] == pos) & (~df.index.isin(chosen))]
        replacement[pos] = float(bench.iloc[0][value_col]) if len(bench) else 0.0

    return df, replacement


def add_vor(
    projections: pd.DataFrame,
    replacement: dict[str, float],
    value_col: str = "projected_ppg",
    gamma: float = 1.0,
) -> pd.DataFrame:
    out = projections.copy()
    out["replacement_ppg"] = out["position"].map(replacement).fillna(0.0)
    out["vor"] = out[value_col] - out["replacement_ppg"]
    out["stud_vor"] = out["vor"].map(lambda x: stud_adjusted_vor(x, gamma))
    return out


def rank_to_paris_score(values: pd.Series, floor: float = 0.0, ceiling: float = 1000.0) -> pd.Series:
    """Map a continuous dynasty-value signal to a transparent 0-1000 percentile scale."""
    pct = values.rank(method="average", pct=True)
    return floor + (ceiling - floor) * pct


def spearman_rank_correlation(rank_a: pd.Series, rank_b: pd.Series) -> float:
    """Spearman correlation from two aligned rank/value series."""
    return float(rank_a.corr(rank_b, method="spearman"))
