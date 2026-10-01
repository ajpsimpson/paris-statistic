from __future__ import annotations

import argparse
import pandas as pd


def show(name: str, df: pd.DataFrame) -> None:
    print(f"\n=== {name} ===")
    print("rows:", len(df))
    print("columns:", list(df.columns))
    print(df.head(8).to_string(index=False))
    for c in ["season", "draft_year", "year", "position", "pos", "round", "pick", "draft_number", "gsis_id", "pfr_player_id"]:
        if c in df.columns:
            print(f"{c} sample:", df[c].dropna().astype(str).head(12).tolist())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--players", required=True)
    ap.add_argument("--draft", required=True)
    ap.add_argument("--combine", required=True)
    args = ap.parse_args()

    players = pd.read_csv(args.players, low_memory=False)
    draft = pd.read_parquet(args.draft)
    combine = pd.read_parquet(args.combine)
    show("PLAYERS", players)
    show("DRAFT PICKS", draft)
    show("COMBINE", combine)


if __name__ == "__main__":
    main()
