"""Per-lap telemetry comparison between two drivers -- the classic
distance-aligned speed/throttle/brake/DRS trace overlay real race engineers
use to see exactly where one driver gains or loses time on track -- plus a
straight-line top-speed trend across the season, which lets a claim like
"car X gained N km/h top speed after an update" be checked against real
telemetry instead of taken on faith from a news report.
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.telemetry_data import load_session

# FastF1's DRS channel is a raw integer status code (0/1 closed-ish,
# 8 = detected zone, 10/12/14 = open) rather than a plain boolean --
# collapsing to "open" (>=10) is the standard interpretation used for
# analysis, since the exact closed-substates aren't meaningfully different
# for a driver comparison.
_DRS_OPEN_THRESHOLD = 10


def _pick_lap(session, driver: str, lap_selector: str = "fastest"):
    driver_laps = session.laps.pick_drivers(driver)
    if lap_selector == "fastest":
        return driver_laps.pick_fastest()
    return driver_laps[driver_laps["LapNumber"] == int(lap_selector)].iloc[0]


def compare_lap_telemetry(session, driver1: str, driver2: str, lap_selector: str = "fastest") -> go.Figure:
    lap1 = _pick_lap(session, driver1, lap_selector)
    lap2 = _pick_lap(session, driver2, lap_selector)

    tel1 = lap1.get_car_data().add_distance()
    tel2 = lap2.get_car_data().add_distance()

    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05,
        row_heights=[0.4, 0.2, 0.2, 0.2],
        subplot_titles=["Speed (km/h)", "Throttle (%)", "Brake", "DRS open"],
    )
    for tel, drv, lap, color in [
        (tel1, driver1, lap1, "#1976d2"),
        (tel2, driver2, lap2, "#e63946"),
    ]:
        lap_time = lap["LapTime"]
        legend_name = f"{drv} ({lap_time})" if pd.notna(lap_time) else drv
        fig.add_trace(go.Scatter(x=tel["Distance"], y=tel["Speed"], name=legend_name,
                                  line=dict(color=color)), row=1, col=1)
        fig.add_trace(go.Scatter(x=tel["Distance"], y=tel["Throttle"], name=legend_name,
                                  line=dict(color=color), showlegend=False), row=2, col=1)
        fig.add_trace(go.Scatter(x=tel["Distance"], y=tel["Brake"].astype(int), name=legend_name,
                                  line=dict(color=color, shape="hv"), showlegend=False), row=3, col=1)
        drs_open = (tel["DRS"] >= _DRS_OPEN_THRESHOLD).astype(int)
        fig.add_trace(go.Scatter(x=tel["Distance"], y=drs_open, name=legend_name,
                                  line=dict(color=color, shape="hv"), showlegend=False), row=4, col=1)

    fig.update_layout(
        height=800,
        title=f"{session.event['EventName']} {session.event.year} — {driver1} vs {driver2} ({lap_selector} lap)",
        legend=dict(orientation="h", y=1.08),
        margin=dict(t=90),
    )
    fig.update_xaxes(title_text="Distance (m)", row=4, col=1)
    return fig


def driver_top_speed_by_race(driver_abbreviation: str, events: list[tuple], session_type: str = "R") -> pd.DataFrame:
    """Max recorded car speed for one driver in each of the given
    (year, round_number) events -- a genuine, checkable proxy for
    straight-line-speed / low-drag setup changes (e.g. a reported rear-wing
    change), computed directly from real telemetry rather than asserted
    from a news report."""
    rows = []
    for year, round_number in events:
        session = load_session(year, round_number, session_type)
        results = session.results
        matches = results[results["Abbreviation"] == driver_abbreviation]
        if len(matches) == 0:
            continue
        drv_num = matches.index[0]
        car = session.car_data.get(drv_num)
        if car is None or len(car) == 0:
            continue
        rows.append({
            "year": year,
            "round": round_number,
            "event_name": session.event["EventName"],
            "top_speed_kmh": round(float(car["Speed"].max()), 1),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    from src.telemetry_data import get_finished_2026_events

    events = get_finished_2026_events()
    row = events.iloc[0]
    session = load_session(int(row["year"]), int(row["RoundNumber"]), "R")

    drivers = session.results.sort_values("Position")["Abbreviation"].tolist()[:2]
    fig = compare_lap_telemetry(session, drivers[0], drivers[1])
    fig.write_html("telemetry_compare_preview.html")
    print(f"Compared {drivers[0]} vs {drivers[1]}, wrote telemetry_compare_preview.html")

    event_list = [(int(r["year"]), int(r["RoundNumber"])) for _, r in events.iterrows()]
    top_speed_df = driver_top_speed_by_race(drivers[0], event_list)
    print(top_speed_df.to_string())
