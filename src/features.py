"""Builds per-driver-per-race feature rows from raw FastF1 results: recent
form, track history, and team pace-trend (car development proxy) features.
"""

import numpy as np
import pandas as pd

from src.fastf1_data import fetch_all_results

ROLLING_WINDOW = 5  # races


def _add_rolling(df: pd.DataFrame, group_col: str, value_col: str, window: int, out_col: str) -> pd.DataFrame:
    """Rolling mean of `value_col` over the driver's/team's previous `window`
    races, shifted by one so the current race's own result never leaks in."""
    df = df.sort_values(["year", "round"])
    df[out_col] = (
        df.groupby(group_col)[value_col]
        .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
    )
    return df


def build_race_features(session_type: str = "R") -> pd.DataFrame:
    """Returns one row per (year, round, driver) for the given session type
    (R=race, S=sprint), with leakage-safe rolling features computed only from
    *prior* races."""
    results = fetch_all_results()
    df = results[results["session_type"] == session_type].copy()

    df["Position"] = pd.to_numeric(df["Position"], errors="coerce")
    df["GridPosition"] = pd.to_numeric(df["GridPosition"], errors="coerce")
    df["Points"] = pd.to_numeric(df["Points"], errors="coerce").fillna(0)
    df["finish_position"] = df["Position"].fillna(21)  # DNF/DNS -> worse than last

    df = df.sort_values(["year", "round"]).reset_index(drop=True)

    # Driver-level recent form
    df = _add_rolling(df, "Abbreviation", "finish_position", ROLLING_WINDOW, "driver_recent_finish")
    df = _add_rolling(df, "Abbreviation", "Points", ROLLING_WINDOW, "driver_recent_points")
    df = _add_rolling(df, "Abbreviation", "GridPosition", ROLLING_WINDOW, "driver_recent_grid")

    # Team-level recent form (car development / pace-trend proxy)
    df = _add_rolling(df, "TeamName", "finish_position", ROLLING_WINDOW, "team_recent_finish")
    df = _add_rolling(df, "TeamName", "Points", ROLLING_WINDOW, "team_recent_points")
    df = _add_rolling(df, "TeamName", "GridPosition", ROLLING_WINDOW, "team_recent_grid")

    # Team trend: last-3-race avg finish vs the 3 before that (negative = improving)
    def _trend(s: pd.Series) -> pd.Series:
        prior = s.shift(1)
        recent3 = prior.rolling(3, min_periods=1).mean()
        prev3 = prior.shift(3).rolling(3, min_periods=1).mean()
        return recent3 - prev3

    df["team_trend"] = df.groupby("TeamName")["finish_position"].transform(_trend)

    # Driver-track history: average finish at this specific event, from all
    # *prior* years only (leakage-safe: only rows with year < current year)
    track_hist = []
    for (driver, event), group in df.groupby(["Abbreviation", "event_name"]):
        group = group.sort_values("year")
        prior_avg = group["finish_position"].expanding().mean().shift(1)
        track_hist.append(pd.Series(prior_avg.values, index=group.index))
    df["driver_track_history"] = pd.concat(track_hist).sort_index()

    # Fill missing rolling/history features (rookies, new teams, first race at
    # a track) with sensible neutral defaults rather than NaN
    df["driver_recent_finish"] = df["driver_recent_finish"].fillna(df["finish_position"].mean())
    df["driver_recent_points"] = df["driver_recent_points"].fillna(0)
    df["driver_recent_grid"] = df["driver_recent_grid"].fillna(df["GridPosition"].mean())
    df["team_recent_finish"] = df["team_recent_finish"].fillna(df["finish_position"].mean())
    df["team_recent_points"] = df["team_recent_points"].fillna(0)
    df["team_recent_grid"] = df["team_recent_grid"].fillna(df["GridPosition"].mean())
    df["team_trend"] = df["team_trend"].fillna(0)
    df["driver_track_history"] = df["driver_track_history"].fillna(df["finish_position"].mean())

    return df


FEATURE_COLUMNS = [
    "GridPosition",
    "driver_recent_finish",
    "driver_recent_points",
    "driver_recent_grid",
    "team_recent_finish",
    "team_recent_points",
    "team_recent_grid",
    "team_trend",
    "driver_track_history",
]

TARGET_COLUMN = "finish_position"


if __name__ == "__main__":
    features = build_race_features("R")
    print(f"{len(features):,} race-result rows")
    print(features[["year", "round", "Abbreviation", "TeamName"] + FEATURE_COLUMNS + [TARGET_COLUMN]].tail(20))
