"""Builds per-driver-per-race feature rows from raw FastF1 results: recent
form, track history (driver and team), and team pace-trend (car development
proxy) features.
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


def _expanding_track_history(df: pd.DataFrame, group_col: str, event_col: str = "event_name") -> pd.Series:
    """Leakage-safe expanding average finish at each (group, event) pair,
    using only *prior* years -- shared logic for both driver- and
    team-level track history."""
    parts = []
    for _, group in df.groupby([group_col, event_col]):
        group = group.sort_values("year")
        prior_avg = group["finish_position"].expanding().mean().shift(1)
        parts.append(pd.Series(prior_avg.values, index=group.index))
    return pd.concat(parts).sort_index()


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

    # Track history: average finish at this specific event, from prior years
    # only. Both driver- and team-level: a driver's personal skill at a track
    # (e.g. Monaco specialists) and a team's car characteristics suiting a
    # track (e.g. high-downforce cars at Hungary/Monaco vs. low-drag at
    # Monza) are separate, real signals -- team-level especially matters
    # since driver-level alone is a weak (low feature-importance) signal that
    # current-form features tend to drown out when predicting future races.
    df["driver_track_history"] = _expanding_track_history(df, "Abbreviation")
    df["team_track_history"] = _expanding_track_history(df, "TeamName")

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
    df["team_track_history"] = df["team_track_history"].fillna(df["finish_position"].mean())

    return df


def compute_track_averages(session_type: str = "R") -> tuple[pd.DataFrame, pd.DataFrame]:
    """All-time average finish position per (driver, event_name) and per
    (team, event_name), using every actual result to date. Safe to use for
    *future* races (unlike the leakage-safe version baked into
    build_race_features, which only looks at prior years for each historical
    row) -- this is specifically for looking up "how has this driver/team
    historically done at this specific upcoming track", not for training."""
    results = fetch_all_results()
    df = results[results["session_type"] == session_type].copy()
    df["Position"] = pd.to_numeric(df["Position"], errors="coerce")
    df["finish_position"] = df["Position"].fillna(21)

    driver_avg = df.groupby(["Abbreviation", "event_name"])["finish_position"].mean().reset_index()
    driver_avg.columns = ["Abbreviation", "event_name", "driver_track_history"]

    team_avg = df.groupby(["TeamName", "event_name"])["finish_position"].mean().reset_index()
    team_avg.columns = ["TeamName", "event_name", "team_track_history"]

    return driver_avg, team_avg


def get_event_features(
    base_snapshot: pd.DataFrame,
    driver_track_avg: pd.DataFrame,
    team_track_avg: pd.DataFrame,
    event_name: str,
) -> pd.DataFrame:
    """Overrides `driver_track_history` and `team_track_history` on a base
    feature snapshot with actual history at `event_name` specifically.
    Without this, every upcoming race prediction reuses whatever track the
    driver/team last actually raced at, so predictions barely differ from
    one upcoming track to the next."""
    snap = base_snapshot.copy()

    driver_event_avg = driver_track_avg[driver_track_avg["event_name"] == event_name][
        ["Abbreviation", "driver_track_history"]
    ]
    snap = snap.merge(driver_event_avg, on="Abbreviation", how="left")
    driver_overall_avg = driver_track_avg.groupby("Abbreviation")["driver_track_history"].mean()
    snap["driver_track_history"] = snap["driver_track_history"].fillna(snap["Abbreviation"].map(driver_overall_avg))
    snap["driver_track_history"] = snap["driver_track_history"].fillna(snap["driver_track_history"].mean())

    team_event_avg = team_track_avg[team_track_avg["event_name"] == event_name][["TeamName", "team_track_history"]]
    snap = snap.merge(team_event_avg, on="TeamName", how="left")
    team_overall_avg = team_track_avg.groupby("TeamName")["team_track_history"].mean()
    snap["team_track_history"] = snap["team_track_history"].fillna(snap["TeamName"].map(team_overall_avg))
    snap["team_track_history"] = snap["team_track_history"].fillna(snap["team_track_history"].mean())

    return snap


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
    "team_track_history",
]

TARGET_COLUMN = "finish_position"


if __name__ == "__main__":
    features = build_race_features("R")
    print(f"{len(features):,} race-result rows")
    print(features[["year", "round", "Abbreviation", "TeamName"] + FEATURE_COLUMNS + [TARGET_COLUMN]].tail(20))
