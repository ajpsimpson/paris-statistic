import math
import pandas as pd

from paris_v2_core import (
    add_vor,
    assign_starters,
    continuous_elite_persistence,
    elite_persistence,
    fantasy_points,
    stud_adjusted_vor,
)


def test_scoring():
    row = {
        "passing_yards": 300,
        "passing_tds": 2,
        "interceptions": 1,
        "rushing_yards": 20,
        "rushing_tds": 1,
        "receiving_yards": 0,
        "receiving_tds": 0,
        "receptions": 0,
        "fumbles_lost": 1,
        "long_td_50_count": 1,
    }
    # 12 + 10 - 2 + 2 + 6 - 2 + 1 = 27
    assert math.isclose(fantasy_points(row), 27.0)


def test_elite_features():
    assert math.isclose(elite_persistence(1, 1, 1), 1.0)
    assert elite_persistence(1, 0, 0) > elite_persistence(0, 0, 1)
    assert continuous_elite_persistence(.99, .98, .97) > continuous_elite_persistence(.90, .89, .88)


def test_stud_vor():
    assert stud_adjusted_vor(5, 1.0) == 5
    assert stud_adjusted_vor(5, 1.2) > 5
    assert stud_adjusted_vor(-5, 1.2) < -5


def test_replacement_logic():
    rows = []
    for pos, n, start in [("QB", 30, 30), ("RB", 70, 25), ("WR", 90, 24), ("TE", 60, 18)]:
        for i in range(n):
            rows.append({"player_id": f"{pos}{i}", "position": pos, "projected_ppg": start - i * 0.1})
    df = pd.DataFrame(rows)
    starters, repl = assign_starters(df)
    assert starters["starter_slot"].notna().sum() == 12 * (1 + 2 + 3 + 2 + 2 + 1)
    assert set(repl) == {"QB", "RB", "WR", "TE"}
    assert all(math.isfinite(x) for x in repl.values())


def test_add_vor():
    df = pd.DataFrame([
        {"player_id": "a", "position": "QB", "projected_ppg": 20.0},
        {"player_id": "b", "position": "RB", "projected_ppg": 15.0},
    ])
    out = add_vor(df, {"QB": 15, "RB": 12, "WR": 10, "TE": 9}, gamma=1.0)
    assert list(out["vor"]) == [5.0, 3.0]


if __name__ == "__main__":
    tests = [test_scoring, test_elite_features, test_stud_vor, test_replacement_logic, test_add_vor]
    for t in tests:
        t()
        print(f"PASS: {t.__name__}")
    print("All Paris v2 core tests passed.")
