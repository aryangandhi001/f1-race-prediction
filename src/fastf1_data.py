"""Pulls and caches F1 session data via FastF1: race/qualifying/sprint results
and lap-time summaries, across recent seasons plus the current 2026 season.
"""

from datetime import datetime
from pathlib import Path

import fastf1
import numpy as np
import pandas as pd

CACHE_DIR = Path("fastf1_cache")
DATA_DIR = Path("data")
RESULTS_PATH = DATA_DIR / "results.parquet"
LAPS_SUMMARY_PATH = DATA_DIR / "laps_summary.parquet"
SCHEDULE_PATH = DATA_DIR / "schedule.parquet"

YEARS = [2026, 2025, 2024, 2023, 2022]  # most recent/relevant first -- FastF1's public API
# rate-limits at 500 calls/hour, so a full 5-season pull can get interrupted
# partway through; fetching newest-first means a partial run still captures
# the most valuable (current-season) data before older data.
SESSION_TYPES = ["R", "Q", "S"]  # Race, Qualifying, Sprint


def _init_cache():
    CACHE_DIR.mkdir(exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE_DIR))


def get_full_schedule(years: list[int] = YEARS, force_refresh: bool = False) -> pd.DataFrame:
    if SCHEDULE_PATH.exists() and not force_refresh:
        cached = pd.read_parquet(SCHEDULE_PATH)
        if set(years).issubset(set(cached["year"].unique())):
            return cached[cached["year"].isin(years)].copy()

    _init_cache()
    frames = []
    for year in years:
        sched = fastf1.get_event_schedule(year, include_testing=False)
        sched = sched.copy()
        sched["year"] = year
        frames.append(sched)
    schedule = pd.concat(frames, ignore_index=True)
    DATA_DIR.mkdir(exist_ok=True)
    schedule.to_parquet(SCHEDULE_PATH)
    return schedule


def _session_results(year: int, round_number: int, session_type: str) -> pd.DataFrame | None:
    try:
        session = fastf1.get_session(year, round_number, session_type)
        session.load(laps=False, telemetry=False, weather=False, messages=False)
    except Exception as e:
        print(f"  [skip] {year} round {round_number} {session_type}: {e}")
        return None

    results = session.results
    if results is None or len(results) == 0:
        return None

    df = results.copy()
    df["year"] = year
    df["round"] = round_number
    df["session_type"] = session_type
    df["event_name"] = session.event["EventName"]
    return df


def fetch_all_results(years: list[int] = YEARS, force_refresh: bool = False) -> pd.DataFrame:
    if RESULTS_PATH.exists() and not force_refresh:
        return pd.read_parquet(RESULTS_PATH)

    _init_cache()
    schedule = get_full_schedule(years)
    today = pd.Timestamp(datetime.now())

    all_results = []
    for _, event in schedule.iterrows():
        event_date = event.get("EventDate") or event.get("Session5Date")
        if pd.isna(event_date) or pd.Timestamp(event_date) > today:
            continue  # hasn't happened yet

        year, round_number = int(event["year"]), int(event["RoundNumber"])
        is_sprint_weekend = "sprint" in str(event.get("EventFormat", "")).lower()
        session_types = SESSION_TYPES if is_sprint_weekend else ["R", "Q"]

        for st in session_types:
            df = _session_results(year, round_number, st)
            if df is not None:
                all_results.append(df)
        print(f"  done: {year} round {round_number} ({event['EventName']})")

    results = pd.concat(all_results, ignore_index=True)
    DATA_DIR.mkdir(exist_ok=True)
    results.to_parquet(RESULTS_PATH)
    print(f"Cached {len(results):,} result rows to {RESULTS_PATH}")
    return results


def fetch_lap_summaries(years: list[int] = YEARS, force_refresh: bool = False) -> pd.DataFrame:
    """Per driver/session median + best lap time, as a pace-trend proxy for
    car development -- FastF1 lap loading is much slower than results, so
    this is kept separate and optional."""
    if LAPS_SUMMARY_PATH.exists() and not force_refresh:
        return pd.read_parquet(LAPS_SUMMARY_PATH)

    _init_cache()
    schedule = get_full_schedule(years)
    today = pd.Timestamp(datetime.now())

    summaries = []
    for _, event in schedule.iterrows():
        event_date = event.get("EventDate") or event.get("Session5Date")
        if pd.isna(event_date) or pd.Timestamp(event_date) > today:
            continue

        year, round_number = int(event["year"]), int(event["RoundNumber"])
        is_sprint_weekend = "sprint" in str(event.get("EventFormat", "")).lower()
        session_types = ["R", "Q", "S"] if is_sprint_weekend else ["R", "Q"]

        for st in session_types:
            try:
                session = fastf1.get_session(year, round_number, st)
                session.load(laps=True, telemetry=False, weather=False, messages=False)
                laps = session.laps
                if laps is None or len(laps) == 0:
                    continue
                valid = laps[laps["LapTime"].notna()].copy()
                valid["LapSeconds"] = valid["LapTime"].dt.total_seconds()
                agg = valid.groupby("Driver")["LapSeconds"].agg(["median", "min"]).reset_index()
                agg.columns = ["driver", "median_lap_s", "best_lap_s"]
                agg["year"] = year
                agg["round"] = round_number
                agg["session_type"] = st
                summaries.append(agg)
            except Exception as e:
                print(f"  [skip laps] {year} round {round_number} {st}: {e}")
        print(f"  laps done: {year} round {round_number}")

    laps_summary = pd.concat(summaries, ignore_index=True)
    DATA_DIR.mkdir(exist_ok=True)
    laps_summary.to_parquet(LAPS_SUMMARY_PATH)
    print(f"Cached {len(laps_summary):,} lap-summary rows to {LAPS_SUMMARY_PATH}")
    return laps_summary


def get_remaining_2026_events() -> pd.DataFrame:
    schedule = get_full_schedule([2026])
    today = pd.Timestamp(datetime.now())
    remaining = schedule[pd.to_datetime(schedule["EventDate"]) > today].copy()
    return remaining.sort_values("RoundNumber")


if __name__ == "__main__":
    results = fetch_all_results()
    print(results[["year", "round", "session_type", "event_name"]].drop_duplicates().tail(20))

    remaining = get_remaining_2026_events()
    print(f"\nRemaining 2026 events: {len(remaining)}")
    print(remaining[["RoundNumber", "EventName", "EventDate", "EventFormat"]])
