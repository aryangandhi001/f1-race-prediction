"""Builds an interactive 2D race replay from FastF1's real positional
telemetry (X/Y car coordinates over time) -- an actual reconstruction of
where each car was on track, not an illustrative animation.

Rendered via a plain Gradio Slider + Plot (matplotlib), not a Plotly
animation embedded in gr.Plot/gr.HTML. Two earlier attempts at shipping a
client-side-animated Plotly figure both failed in the browser: gr.Plot
renders a Plotly figure's data/layout as a static snapshot and drops the
frames/play-button JS entirely, and embedding the full animated HTML via
gr.HTML (and then via an iframe data-URI) still didn't render, consistent
with Gradio sanitizing embedded <script>/<iframe> content out of HTML
component output. Rather than keep fighting that, replay frames are
rendered server-side on demand (one matplotlib figure per requested time
index) and driven by Gradio's own native Slider/Button reactivity, which
doesn't depend on any embedded third-party JS executing at all.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import fastf1.plotting as fp

# Caps how many distinct time steps the replay has. A full race is ~90 real
# minutes of telemetry sampled at ~4Hz -- there's no reason to keep more
# steps than a scrubber slider can usefully distinguish, and each one only
# costs a cheap on-demand matplotlib render (not a large upfront payload
# like the old Plotly-animation approach), so this can stay generous.
MAX_FRAMES = 300
MIN_FRAME_SPACING_S = 2.0


def _track_outline(session) -> tuple[np.ndarray, np.ndarray]:
    """Track outline traced from one clean, complete lap's position data --
    any full green-flag lap works, since the physical track doesn't move."""
    fastest = session.laps.pick_fastest()
    pos = fastest.get_pos_data()
    return pos["X"].to_numpy(), pos["Y"].to_numpy()


def _resample_driver_position(pos_df: pd.DataFrame, time_grid: np.ndarray) -> pd.DataFrame:
    """Aligns one driver's raw position samples (irregular, ~4Hz) onto a
    shared replay time grid via as-of backward matching: each replay frame
    shows the driver's most recent actual telemetry sample at or before
    that instant. `tolerance` means a driver with no telemetry near a given
    frame (not yet on track, retired, DNS) simply has no marker there,
    rather than a fabricated interpolated position."""
    pos_df = pos_df.sort_values("SessionTimeSeconds")
    grid_df = pd.DataFrame({"SessionTimeSeconds": time_grid})
    return pd.merge_asof(
        grid_df, pos_df, on="SessionTimeSeconds",
        direction="backward", tolerance=5.0,
    )


def prepare_replay_data(session, max_frames: int = MAX_FRAMES) -> dict:
    """Precomputes everything needed to render any single replay time step
    on demand: the static track outline, each driver's position aligned onto
    a shared time grid, and driver labels/colors. Rendering one frame is
    then just a cheap matplotlib plot indexed into these precomputed
    arrays -- the actual (X, Y) telemetry lookup and team-color resolution
    only happen once here, not on every slider drag."""
    track_x, track_y = _track_outline(session)
    results = session.results

    driver_list = [drv for drv in session.drivers if session.pos_data.get(drv) is not None and len(session.pos_data[drv]) > 0]

    t_min = min(session.pos_data[drv]["SessionTime"].min().total_seconds() for drv in driver_list)
    t_max = max(session.pos_data[drv]["SessionTime"].max().total_seconds() for drv in driver_list)
    n_frames = int(min(max_frames, (t_max - t_min) / MIN_FRAME_SPACING_S))
    n_frames = max(n_frames, 30)
    time_grid = np.linspace(t_min, t_max, n_frames)

    aligned = {}
    for drv in driver_list:
        pos = session.pos_data[drv][["X", "Y", "SessionTime"]].copy()
        pos["SessionTimeSeconds"] = pos["SessionTime"].dt.total_seconds()
        aligned[drv] = _resample_driver_position(pos, time_grid)
        del pos

    def _label(drv):
        return results.loc[drv, "Abbreviation"] if drv in results.index else drv

    def _color(drv):
        if drv not in results.index:
            return "#888888"
        try:
            return fp.get_team_color(results.loc[drv, "TeamName"], session=session)
        except Exception:
            return "#888888"

    return {
        "track_x": track_x,
        "track_y": track_y,
        "driver_list": driver_list,
        "aligned": aligned,
        "labels": {drv: _label(drv) for drv in driver_list},
        "colors": {drv: _color(drv) for drv in driver_list},
        "time_grid": time_grid,
        "t_min": t_min,
        "n_frames": n_frames,
        "event_name": session.event["EventName"],
        "year": session.event.year,
    }


def render_replay_frame(data: dict, frame_idx: int):
    """Renders one time step as a static matplotlib figure: the track
    outline plus every driver's position at that step (drivers with no
    telemetry near this instant -- not yet on track, retired, DNS -- are
    simply omitted, not shown at a fabricated position)."""
    frame_idx = max(0, min(int(frame_idx), data["n_frames"] - 1))
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(data["track_x"], data["track_y"], "-", color="#aaaaaa", linewidth=1.5)

    for drv in data["driver_list"]:
        row = data["aligned"][drv].iloc[frame_idx]
        if pd.isna(row["X"]):
            continue
        ax.plot(row["X"], row["Y"], "o", color=data["colors"][drv], markersize=9, markeredgecolor="black", markeredgewidth=0.5)
        ax.annotate(data["labels"][drv], (row["X"], row["Y"]), fontsize=8, xytext=(4, 4), textcoords="offset points")

    ax.set_aspect("equal")
    ax.axis("off")
    elapsed = int(data["time_grid"][frame_idx] - data["t_min"])
    ax.set_title(f"{data['event_name']} {data['year']} — t={elapsed}s")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    from src.telemetry_data import get_finished_2026_events, load_session

    events = get_finished_2026_events()
    row = events.iloc[0]
    session = load_session(int(row["year"]), int(row["RoundNumber"]), "R")
    data = prepare_replay_data(session)
    print(f"Prepared {data['n_frames']} frames for {len(data['driver_list'])} drivers")
    fig = render_replay_frame(data, data["n_frames"] // 2)
    fig.savefig("replay_frame_check.png", dpi=100)
    print("Wrote replay_frame_check.png")
