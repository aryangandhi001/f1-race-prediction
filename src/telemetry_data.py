"""Loads full car + positional telemetry for a single session -- much
heavier than fastf1_data.py's results-only fetch (session.load(laps=False,
telemetry=False, ...)), which doesn't include per-car Speed/Throttle/DRS or
X/Y position needed for the replay, strategy, and telemetry-comparison
features. Kept as a separate module since the win-probability/championship
pipeline never needs this heavier data.
"""

from datetime import datetime

import fastf1
import pandas as pd

from src.fastf1_data import CACHE_DIR, get_full_schedule

# Only the single most-recently loaded session is kept in memory. A full
# session's car + positional telemetry for ~20 drivers is large enough
# (tens of thousands of rows per driver per channel) that caching more than
# one at a time risks real memory pressure on a small deployment -- FastF1's
# own on-disk cache (see _init_cache) already makes repeat loads of the same
# session fast, so an in-memory LRU beyond size 1 buys little.
_loaded_session: dict | None = None  # {"key": (year, round, type), "session": Session}


def _init_cache():
    CACHE_DIR.mkdir(exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE_DIR))


def get_finished_2026_events() -> pd.DataFrame:
    """2026 events whose date has already passed, most recent first --
    the only ones FastF1 actually has session data for."""
    schedule = get_full_schedule([2026])
    today = pd.Timestamp(datetime.now())
    finished = schedule[pd.to_datetime(schedule["EventDate"]) <= today].copy()
    return finished.sort_values("RoundNumber", ascending=False)


def load_session(year: int, round_number: int, session_type: str = "R"):
    """Loads one session's laps + car telemetry + positional data +
    weather, keeping only this single session in memory (see module
    docstring). A repeat call for the same session within one process is
    free; a different session evicts the previous one before loading."""
    global _loaded_session
    key = (year, round_number, session_type)
    if _loaded_session is not None and _loaded_session["key"] == key:
        return _loaded_session["session"]

    _init_cache()
    session = fastf1.get_session(year, round_number, session_type)
    session.load(laps=True, telemetry=True, weather=True, messages=False)
    _loaded_session = {"key": key, "session": session}
    return session


if __name__ == "__main__":
    events = get_finished_2026_events()
    print(events[["RoundNumber", "EventName", "EventDate"]].to_string())

    session = load_session(int(events.iloc[0]["year"]), int(events.iloc[0]["RoundNumber"]), "R")
    print(f"\nLoaded: {session.event['EventName']}, drivers: {session.drivers}")
