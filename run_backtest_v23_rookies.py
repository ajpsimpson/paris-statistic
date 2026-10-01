from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import run_backtest_v225 as v225
import run_backtest_v223 as v223
import run_backtest_v222 as v222
import run_backtest_v22 as v22
import run_backtest_v211 as v211
import run_backtest_v21 as v21
from paris_v2_core import DEFAULT_CONFIG, assign_starters

TRAIN_START = 2015
EVAL_YEARS = [2019, 2020, 2021, 2022, 2023]
ALL_YEARS = list(range(TRAIN_START, 2024))
POSITIONS = ["QB", "RB", "WR", "TE"]
ALPHAS = [1.0, 10.0, 30.0, 100.0, 300.0]
VARIANTS = ["capital", "capital_combine", "capital_depth", "full"]
SELECTED_QB_Q = 0.55


def ridge_pipe(alpha: float) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
        ("ridge", Ridge(alpha=alpha)),
    ])


def load_roster_features(roster_dir: str) -> pd.DataFrame:
    rows = []
    for year in ALL_YEARS:
        path = Path(roster_dir) / f"roster_weekly_{year}.parquet"
        df = pd.read_parquet(path)
        id_col = "gsis_id" if "gsis_id" in df.columns else "player_id"
        wk = pd.to_numeric(df["week"], errors="coerce")
        snap = df[wk.eq(1)].copy()
        if "season" in snap.columns:
            snap = snap[pd.to_numeric(snap.season, errors="coerce").eq(year)]
        if "game_type" in snap.columns:
            reg = snap.game_type.astype(str).str.upper().eq("REG")
            if reg.any():
                snap = snap[reg].copy()
        ids = snap[id_col].dropna().astype(str).unique()
        rows.extend({"target_year": year, "player_id": x, "week1_rostered": 1.0} for x in ids)
    return pd.DataFrame(rows).drop_duplicates(["target_year", "player_id"])


def normalize_position(x: pd.Series) -> pd.Series:
    s = x.astype(str).str.upper()
    return s.replace({"HB": "RB", "FB": "RB"})


def rookie_universe(players: pd.DataFrame, draft: pd.DataFrame, combine: pd.DataFrame) -> pd.DataFrame:
    p = players.copy()
    p["position"] = normalize_position(p["position"])
    p["target_year"] = pd.to_numeric(p["rookie_season"], errors="coerce")
    p = p[p.position.isin(POSITIONS) & p.target_year.between(TRAIN_START, 2023)].copy()
    p = p[p.gsis_id.notna()].copy()
    p["player_id"] = p.gsis_id.astype(str)
    p["player_name"] = p.display_name.astype(str)
    p["target_year"] = p.target_year.astype(int)

    birth = pd.to_datetime(p.birth_date, errors="coerce")
    cutoff = pd.to_datetime(p.target_year.astype(str) + "-09-01", errors="coerce")
    p["age"] = (cutoff - birth).dt.days / 365.25

    for c in ["draft_year", "draft_round", "draft_pick", "weight", "height"]:
        p[c] = pd.to_numeric(p[c], errors="coerce")

    # Use immutable draft facts only. Do not import PFR career outcome fields.
    d = draft[["season", "round", "pick", "gsis_id", "pfr_player_id", "position", "age"]].copy()
    d["season"] = pd.to_numeric(d.season, errors="coerce")
    d = d[d.season.between(TRAIN_START, 2023)].copy()
    d["player_id"] = d.gsis_id.astype(str)
    d = d.rename(columns={
        "season": "target_year", "round": "draft_round_dp", "pick": "draft_pick_dp",
        "pfr_player_id": "pfr_id_dp", "age": "draft_age_dp"
    })
    p = p.merge(
        d[["target_year", "player_id", "draft_round_dp", "draft_pick_dp", "pfr_id_dp", "draft_age_dp"]],
        on=["target_year", "player_id"], how="left"
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
    c = c[c.target_year.between(TRAIN_START, 2023)].copy()
    c = c.rename(columns={"pfr_id": "pfr_join"})
    combine_cols = ["forty", "bench", "vertical", "broad_jump", "cone", "shuttle", "wt"]
    for col in combine_cols:
        c[col] = pd.to_numeric(c[col], errors="coerce")
    c = c[["target_year", "pfr_join"] + combine_cols].drop_duplicates(["target_year", "pfr_join"])
    p = p.merge(c, on=["target_year", "pfr_join"], how="left")
    p["combine_weight"] = p.wt.fillna(p.weight)
    p["speed_score"] = p.combine_weight * (200.0 / p.forty.replace(0, np.nan)) ** 4

    for pos in POSITIONS:
        p[f"pos_{pos}"] = p.position.eq(pos).astype(float)
        p[f"log_pick_x_{pos}"] = p.log_draft_pick * p[f"pos_{pos}"]
        p[f"age_x_{pos}"] = p.age * p[f"pos_{pos}"]

    return p


def attach_preseason_role(rookies: pd.DataFrame, roster_dir: str, depth_dir: str) -> pd.DataFrame:
    r = load_roster_features(roster_dir)
    out = rookies.merge(r, on=["target_year", "player_id"], how="left")
    out["week1_rostered"] = out.week1_rostered.fillna(0.0)

    depth, _ = v223.load_preseason_depth(depth_dir)
    out = out.merge(
        depth[["target_year", "player_id"] + v223.DEPTH_FEATURES],
        on=["target_year", "player_id"], how="left"
    )
    out["depth_present"] = out.depth_present.fillna(0.0)
    for c in ["depth_is_first", "depth_is_top2", "depth_is_top3"]:
        out[c] = out[c].fillna(0.0)
    return out


def attach_outcomes(rookies: pd.DataFrame, seasons: pd.DataFrame) -> pd.DataFrame:
    s = seasons[["season", "player_id", "position", "paris_fp", "ppg", "games", "opportunity"]].copy()
    s = s.rename(columns={
        "season": "target_year", "paris_fp": "actual_fp", "ppg": "actual_ppg",
        "games": "actual_games", "opportunity": "actual_opportunity"
    })
    out = rookies.merge(
        s[["target_year", "player_id", "actual_fp", "actual_ppg", "actual_games", "actual_opportunity"]],
        on=["target_year", "player_id"], how="left"
    )
    for c in ["actual_fp", "actual_ppg", "actual_games", "actual_opportunity"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
    scheduled = np.where(out.target_year.le(2020), 16.0, 17.0)
    out["actual_week_value"] = out.actual_fp / scheduled
    out["rookie_role"] = (out.actual_games.ge(4) & out.actual_opportunity.gt(0)).astype(int)
    return out


CAPITAL = [
    "drafted", "log_draft_pick", "inv_sqrt_pick", "draft_round_model", "first_round", "day2", "age",
    "pos_QB", "pos_RB", "pos_WR", "pos_TE",
    "log_pick_x_QB", "log_pick_x_RB", "log_pick_x_WR", "log_pick_x_TE",
    "age_x_QB", "age_x_RB", "age_x_WR", "age_x_TE",
]
COMBINE = ["combine_weight", "forty", "bench", "vertical", "broad_jump", "cone", "shuttle", "speed_score"]
DEPTH = ["week1_rostered", "depth_present", "depth_rank", "depth_is_first", "depth_is_top2", "depth_is_top3"]


def feature_set(variant: str) -> list[str]:
    if variant == "capital": return CAPITAL
    if variant == "capital_combine": return CAPITAL + COMBINE
    if variant == "capital_depth": return CAPITAL + DEPTH
    if variant == "full": return CAPITAL + COMBINE + DEPTH
    raise ValueError(variant)


def fit_candidate(table: pd.DataFrame, year: int, variant: str, alpha: float) -> pd.DataFrame:
    feats = feature_set(variant)
    train = table[table.target_year.lt(year)].copy()
    test = table[table.target_year.eq(year)].copy()
    if len(train) < 80 or len(test) < 15:
        return pd.DataFrame()
    model = ridge_pipe(alpha)
    model.fit(train[feats], train.actual_week_value)
    test["predicted_value"] = np.maximum(0.0, model.predict(test[feats]))
    test["variant"] = variant
    test["alpha"] = alpha
    return test


def overlap(g: pd.DataFrame, k: int) -> float:
    k = min(k, len(g))
    if k == 0: return np.nan
    return len(
        set(g.nlargest(k, "predicted_value").player_id)
        & set(g.nlargest(k, "actual_week_value").player_id)
    ) / k


def rookie_metrics(pred: pd.DataFrame) -> dict:
    rostered = pred[pred.week1_rostered.eq(1)].drop_duplicates("player_id").copy()
    if len(rostered) < 5:
        return {}
    drafted = rostered[rostered.drafted.eq(1)]
    return {
        "year": int(rostered.target_year.iloc[0]),
        "variant": rostered.variant.iloc[0],
        "alpha": float(rostered.alpha.iloc[0]),
        "n_rookies": len(rostered),
        "rookie_spearman": rostered.predicted_value.corr(rostered.actual_week_value, method="spearman"),
        "rookie_mae": mean_absolute_error(rostered.actual_week_value, rostered.predicted_value),
        "top10_overlap": overlap(rostered, 10),
        "top20_overlap": overlap(rostered, 20),
        "drafted_spearman": drafted.predicted_value.corr(drafted.actual_week_value, method="spearman") if len(drafted) >= 5 else np.nan,
    }


def selected_veterans(vet_table: pd.DataFrame, eligible: dict[int, set[str]], year: int) -> pd.DataFrame:
    pred = v225.build_predictions(vet_table, SELECTED_QB_Q)
    g = pred[pred.target_year.eq(year) & pred.player_id.astype(str).isin(eligible[year])].copy()
    return g[[
        "target_year", "player_id", "player_name", "position", "predicted_value", "actual_week_value"
    ]].drop_duplicates("player_id")


def merged_metrics(vets: pd.DataFrame, rookies: pd.DataFrame, year: int, variant: str, alpha: float) -> tuple[dict, pd.DataFrame]:
    r = rookies[rookies.week1_rostered.eq(1)][[
        "target_year", "player_id", "player_name", "position", "predicted_value", "actual_week_value"
    ]].drop_duplicates("player_id")
    v = vets[~vets.player_id.isin(set(r.player_id))].copy()
    g = pd.concat([v, r], ignore_index=True)
    _, pr = assign_starters(g[["player_id", "position", "predicted_value"]], DEFAULT_CONFIG, "predicted_value")
    actual = g[["player_id", "position", "actual_week_value"]].copy()
    _, ar = assign_starters(actual, DEFAULT_CONFIG, "actual_week_value")
    g["pred_vor"] = g.predicted_value - g.position.map(pr)
    g["actual_vor"] = g.actual_week_value - g.position.map(ar)
    g["overall_rank"] = g.pred_vor.rank(ascending=False, method="min")
    g["actual_rank"] = g.actual_vor.rank(ascending=False, method="min")
    g["is_rookie"] = g.player_id.isin(set(r.player_id)).astype(int)
    def ov(k: int) -> float:
        k=min(k,len(g)); return len(set(g.nlargest(k,"pred_vor").player_id)&set(g.nlargest(k,"actual_vor").player_id))/max(k,1)
    return {
        "year": year, "variant": variant, "alpha": alpha, "n": len(g), "n_rookies": len(r),
        "merged_spearman": g.pred_vor.corr(g.actual_vor, method="spearman"),
        "top24_overlap": ov(24), "top50_overlap": ov(50), "top100_overlap": ov(100),
    }, g


def summarize(rookie_m: pd.DataFrame, merged_m: pd.DataFrame) -> pd.DataFrame:
    keys = ["variant", "alpha"]
    rp = rookie_m[rookie_m.year.lt(2023)].groupby(keys, as_index=False).agg(
        pre_rookie_spearman=("rookie_spearman","mean"), pre_top10=("top10_overlap","mean"), pre_top20=("top20_overlap","mean")
    )
    rv = rookie_m[rookie_m.year.eq(2023)][keys+["rookie_spearman","top10_overlap","top20_overlap"]].rename(columns={
        "rookie_spearman":"val_rookie_spearman","top10_overlap":"val_top10","top20_overlap":"val_top20"})
    mp = merged_m[merged_m.year.lt(2023)].groupby(keys, as_index=False).agg(
        pre_merged_spearman=("merged_spearman","mean"), pre_top50=("top50_overlap","mean")
    )
    mv = merged_m[merged_m.year.eq(2023)][keys+["merged_spearman","top50_overlap"]].rename(columns={
        "merged_spearman":"val_merged_spearman","top50_overlap":"val_top50"})
    z=rp.merge(rv,on=keys).merge(mp,on=keys).merge(mv,on=keys)
    z["selection_score"]=(
        .25*z.pre_rookie_spearman + .20*z.val_rookie_spearman
        + .20*z.pre_merged_spearman + .20*z.val_merged_spearman
        + .05*z.pre_top10 + .05*z.val_top10 + .025*z.pre_top50 + .025*z.val_top50
    )
    return z.sort_values("selection_score", ascending=False)


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--players-csv", required=True)
    ap.add_argument("--draft-parquet", required=True)
    ap.add_argument("--combine-parquet", required=True)
    ap.add_argument("--roster-dir", required=True)
    ap.add_argument("--depth-dir", required=True)
    ap.add_argument("--out-dir", default="outputs_v23_rookies")
    args=ap.parse_args(); out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)

    raw=v21.old.load_weekly_csvs(args.data_dir)
    seasons=v21.aggregate_v21(raw)
    players=pd.read_csv(args.players_csv,low_memory=False)
    draft=pd.read_parquet(args.draft_parquet)
    combine=pd.read_parquet(args.combine_parquet)

    rook=rookie_universe(players,draft,combine)
    rook=attach_preseason_role(rook,args.roster_dir,args.depth_dir)
    rook=attach_outcomes(rook,seasons)

    # Returning-player table stays frozen at v2.2.5.
    vet_table=v211.augment_table(seasons,players)
    vet_table=v22.augment_targets(vet_table,seasons)
    depth,_=v223.load_preseason_depth(args.depth_dir)
    vet_table=v223.attach_depth(vet_table,depth)
    eligible,_,_=v222.load_week1_rosters(args.roster_dir)
    vet_cache={year:selected_veterans(vet_table,eligible,year) for year in EVAL_YEARS}

    rookie_metric_rows=[]; merged_metric_rows=[]; pred_rows=[]; merged_rows=[]
    for year,variant,alpha in product(EVAL_YEARS,VARIANTS,ALPHAS):
        pred=fit_candidate(rook,year,variant,alpha)
        if len(pred)==0: continue
        rm=rookie_metrics(pred)
        if rm: rookie_metric_rows.append(rm)
        mm,mg=merged_metrics(vet_cache[year],pred,year,variant,alpha)
        merged_metric_rows.append(mm)
        pred_rows.append(pred)
        mg["variant"]=variant; mg["alpha"]=alpha; mg["target_year"]=year
        merged_rows.append(mg)

    rookie_m=pd.DataFrame(rookie_metric_rows)
    merged_m=pd.DataFrame(merged_metric_rows)
    predictions=pd.concat(pred_rows,ignore_index=True)
    merged_rankings=pd.concat(merged_rows,ignore_index=True)
    summary=summarize(rookie_m,merged_m)
    best=summary.iloc[0]

    selected_pred=predictions[(predictions.variant.eq(best.variant))&(predictions.alpha.eq(best.alpha))].copy()
    selected_merged=merged_rankings[(merged_rankings.variant.eq(best.variant))&(merged_rankings.alpha.eq(best.alpha))].copy()

    coverage=rook.groupby(["target_year","position"],as_index=False).agg(
        rookies=("player_id","size"), roster_rate=("week1_rostered","mean"),
        depth_rate=("depth_present","mean"), drafted_rate=("drafted","mean"),
        combine_40_rate=("forty",lambda x: x.notna().mean())
    )

    coverage.to_csv(out/"v23_rookie_coverage.csv",index=False)
    rookie_m.to_csv(out/"v23_rookie_metrics.csv",index=False)
    merged_m.to_csv(out/"v23_merged_metrics.csv",index=False)
    summary.to_csv(out/"v23_candidate_summary.csv",index=False)
    selected_pred.to_csv(out/"v23_selected_rookie_predictions.csv",index=False)
    selected_merged.to_csv(out/"v23_selected_merged_rankings.csv",index=False)

    print("=== ROOKIE DATA COVERAGE ===")
    print(coverage[coverage.target_year.isin(EVAL_YEARS)].to_string(index=False))
    print("\n=== PARIS V2.3 ROOKIE CANDIDATES ===")
    print(summary.head(20).to_string(index=False))
    print("\n=== SELECTED ROOKIE MODEL ===")
    print(best.to_string())

    rmet=rookie_m[(rookie_m.variant.eq(best.variant))&(rookie_m.alpha.eq(best.alpha))].sort_values("year")
    mmet=merged_m[(merged_m.variant.eq(best.variant))&(merged_m.alpha.eq(best.alpha))].sort_values("year")
    print("\n=== SELECTED ROOKIE-ONLY METRICS ===")
    print(rmet.to_string(index=False))
    print("\n=== SELECTED MERGED VETERAN + ROOKIE METRICS ===")
    print(mmet.to_string(index=False))

    y=selected_pred[selected_pred.target_year.eq(2023)&selected_pred.week1_rostered.eq(1)].sort_values("predicted_value",ascending=False).head(30)
    print("\n=== 2023 ROOKIE TOP 30, SELECTED MODEL ===")
    print(y[["player_name","position","draft_pick_model","age","depth_rank","predicted_value","actual_week_value"]].to_string(index=False))

    all23=selected_merged[selected_merged.target_year.eq(2023)].sort_values("overall_rank").head(50)
    print("\n=== 2023 MERGED TOP 50 ===")
    print(all23[["overall_rank","player_name","position","is_rookie","predicted_value","actual_week_value","pred_vor","actual_vor"]].to_string(index=False))


if __name__=="__main__":
    main()
