"""Pit stop and tyre strategy analysis: which compound each driver ran on
each stint, when they pitted, and how long each stop cost -- the actual
strategic picture a race engineer looks at, not just the finishing order.
"""

import matplotlib.pyplot as plt
import pandas as pd

_COMPOUND_COLORS = {
    "SOFT": "#e21313",
    "MEDIUM": "#f0d43a",
    "HARD": "#e8e8e8",
    "INTERMEDIATE": "#3aa832",
    "WET": "#3a7bd5",
    "UNKNOWN": "#999999",
}


def build_strategy_chart(session) -> plt.Figure:
    laps = session.laps.copy()
    laps["Compound"] = laps["Compound"].fillna("UNKNOWN")

    results = session.results.sort_values("Position")
    driver_order = [d for d in results["Abbreviation"] if d in laps["Driver"].unique()]

    stints = (
        laps.groupby(["Driver", "Stint", "Compound"])["LapNumber"]
        .agg(["min", "max"]).reset_index()
        .rename(columns={"min": "lap_start", "max": "lap_end"})
    )

    fig, ax = plt.subplots(figsize=(11, max(6, 0.35 * len(driver_order))))
    for i, drv in enumerate(driver_order):
        drv_stints = stints[stints["Driver"] == drv].sort_values("lap_start")
        for _, s in drv_stints.iterrows():
            width = s["lap_end"] - s["lap_start"] + 1
            ax.barh(
                i, width, left=s["lap_start"] - 0.5, height=0.7,
                color=_COMPOUND_COLORS.get(s["Compound"], "#999999"),
                edgecolor="black", linewidth=0.5,
            )

    ax.set_yticks(range(len(driver_order)))
    ax.set_yticklabels(driver_order)
    ax.invert_yaxis()
    ax.set_xlabel("Lap")
    ax.set_title(f"{session.event['EventName']} {session.event.year} — Tyre strategy (ordered by finish)")

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in _COMPOUND_COLORS.values()]
    ax.legend(handles, _COMPOUND_COLORS.keys(), loc="upper center",
              bbox_to_anchor=(0.5, -0.08), ncol=len(_COMPOUND_COLORS), frameon=False)
    fig.tight_layout()
    return fig


def pit_stop_summary(session) -> pd.DataFrame:
    """One row per actual pit visit: driver, lap, and pit-lane time (the
    delta from crossing the pit-in line to crossing the pit-out line).
    Note this is pit-*lane* time, not the official sub-3-second pit-*box*
    stationary time TV graphics show -- FastF1 doesn't expose box-only
    timing, only lap-level in/out timestamps, so this necessarily includes
    the drive through the pit lane itself."""
    laps = session.laps.copy()
    pits = laps[laps["PitInTime"].notna()][["Driver", "LapNumber", "PitInTime"]].copy()
    outs = laps[laps["PitOutTime"].notna()][["Driver", "LapNumber", "PitOutTime"]].copy()

    rows = []
    for _, pit in pits.iterrows():
        candidates = outs[
            (outs["Driver"] == pit["Driver"]) & (outs["LapNumber"] >= pit["LapNumber"])
        ].sort_values("LapNumber")
        if len(candidates) == 0:
            continue
        out_row = candidates.iloc[0]
        pit_lane_time = (out_row["PitOutTime"] - pit["PitInTime"]).total_seconds()
        rows.append({
            "Driver": pit["Driver"],
            "Lap": int(pit["LapNumber"]),
            "PitLaneTime_s": round(pit_lane_time, 1),
        })
    return pd.DataFrame(rows).sort_values(["Lap", "Driver"]).reset_index(drop=True)


if __name__ == "__main__":
    from src.telemetry_data import get_finished_2026_events, load_session

    events = get_finished_2026_events()
    row = events.iloc[0]
    session = load_session(int(row["year"]), int(row["RoundNumber"]), "R")

    fig = build_strategy_chart(session)
    fig.savefig("strategy_preview.png", dpi=100)
    print("Wrote strategy_preview.png")

    summary = pit_stop_summary(session)
    print(summary.to_string())
