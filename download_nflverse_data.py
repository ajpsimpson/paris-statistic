from __future__ import annotations

from pathlib import Path
import requests

START_YEAR = 2015
END_YEAR = 2023  # Stop at validation year. 2024-2025 remain holdout data.
BASE = "https://github.com/nflverse/nflverse-data/releases/download/player_stats/player_stats_{year}.csv"
OUT = Path("data/player_stats")


def download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        print(f"skip {path}")
        return
    print(f"download {url}")
    with requests.get(url, stream=True, timeout=90) as r:
        r.raise_for_status()
        with path.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def main() -> None:
    for year in range(START_YEAR, END_YEAR + 1):
        download(BASE.format(year=year), OUT / f"player_stats_{year}.csv")
    print("Historical training/validation files are ready.")


if __name__ == "__main__":
    main()
