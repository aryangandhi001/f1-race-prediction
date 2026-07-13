"""Builds an interactive 2D race replay from FastF1's real positional
telemetry (X/Y car coordinates over time) -- an actual reconstruction of
where each car was on track, not an illustrative animation. Rendered as a
Plotly figure with animation frames + a scrubber slider, embeddable
directly in Gradio via gr.Plot.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import fastf1.plotting as fp

# Caps the number of animation frames sent to the browser. A full race is
# ~90 real minutes of telemetry sampled at ~4Hz -- animating every sample
# would mean tens of thousands of frames, which is both too large a
# payload to ship to a browser and too slow to be watchable as a replay
# anyway. Frames are instead spaced evenly across the race so the whole
# thing plays back in well under a minute.
MAX_FRAMES = 400
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


def build_race_replay(session, max_frames: int = MAX_FRAMES) -> go.Figure:
    track_x, track_y = _track_outline(session)
    results = session.results

    driver_positions = {}
    for drv in session.drivers:
        pos = session.pos_data.get(drv)
        if pos is None or len(pos) == 0:
            continue
        pos = pos.copy()
        pos["SessionTimeSeconds"] = pos["SessionTime"].dt.total_seconds()
        driver_positions[drv] = pos

    t_min = min(p["SessionTimeSeconds"].min() for p in driver_positions.values())
    t_max = max(p["SessionTimeSeconds"].max() for p in driver_positions.values())
    n_frames = int(min(max_frames, (t_max - t_min) / MIN_FRAME_SPACING_S))
    n_frames = max(n_frames, 30)
    time_grid = np.linspace(t_min, t_max, n_frames)

    driver_list = list(driver_positions.keys())
    aligned = {drv: _resample_driver_position(driver_positions[drv], time_grid) for drv in driver_list}

    def _label(drv):
        return results.loc[drv, "Abbreviation"] if drv in results.index else drv

    def _color(drv):
        if drv not in results.index:
            return "#888888"
        try:
            return fp.get_team_color(results.loc[drv, "TeamName"], session=session)
        except Exception:
            return "#888888"

    colors = {drv: _color(drv) for drv in driver_list}
    labels = {drv: _label(drv) for drv in driver_list}

    def _frame_traces(i):
        traces = []
        for drv in driver_list:
            row = aligned[drv].iloc[i]
            x = [row["X"]] if pd.notna(row["X"]) else [None]
            y = [row["Y"]] if pd.notna(row["Y"]) else [None]
            traces.append(go.Scatter(
                x=x, y=y, mode="markers+text",
                marker=dict(size=11, color=colors[drv], line=dict(width=1, color="black")),
                text=[labels[drv]], textposition="top center",
                textfont=dict(size=9),
                name=labels[drv], hoverinfo="name",
            ))
        return traces

    frames = [
        go.Frame(data=_frame_traces(i), name=str(i), traces=list(range(1, len(driver_list) + 1)))
        for i in range(n_frames)
    ]

    track_trace = go.Scatter(
        x=track_x, y=track_y, mode="lines",
        line=dict(color="#aaaaaa", width=2), showlegend=False, hoverinfo="skip",
    )

    fig = go.Figure(data=[track_trace] + list(frames[0].data), frames=frames)
    fig.update_layout(
        title=f"{session.event['EventName']} {session.event.year} — Race Replay",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, scaleanchor="x", scaleratio=1),
        showlegend=False,
        height=700,
        margin=dict(l=10, r=10, t=50, b=10),
        updatemenus=[dict(
            type="buttons", showactive=False, x=0.05, y=0.02,
            buttons=[
                dict(label="▶ Play", method="animate",
                     args=[None, {"frame": {"duration": 60, "redraw": True}, "fromcurrent": True, "transition": {"duration": 0}}]),
                dict(label="⏸ Pause", method="animate",
                     args=[[None], {"frame": {"duration": 0}, "mode": "immediate"}]),
            ],
        )],
        sliders=[dict(
            x=0.1, len=0.85, y=0,
            steps=[
                dict(method="animate", args=[[str(i)], {"frame": {"duration": 0}, "mode": "immediate"}],
                     label=f"{int(time_grid[i] - t_min)}s")
                for i in range(0, n_frames, max(1, n_frames // 40))
            ],
        )],
    )
    return fig


if __name__ == "__main__":
    from src.telemetry_data import get_finished_2026_events, load_session

    events = get_finished_2026_events()
    row = events.iloc[0]
    session = load_session(int(row["year"]), int(row["RoundNumber"]), "R")
    fig = build_race_replay(session)
    print(f"Built replay with {len(fig.frames)} frames, {len(fig.data)} traces")
    fig.write_html("replay_preview.html")
    print("Wrote replay_preview.html")
