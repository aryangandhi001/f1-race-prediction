"""Gradio demo: race replay, pit strategy, telemetry comparison, and
top-speed trend -- deployed as a SEPARATE service from app.py's prediction
demo.

Why separate: loading a full FastF1 session's car + positional telemetry is
memory-heavy enough that adding it to app.py's always-on process (which
also holds the LightGBM models, matplotlib, and the win-probability
pipeline) got that process OOM-killed on a 512MB Render instance -- and
because it's one process, the crash took down the working prediction demo
along with it. Splitting this into its own service means a memory spike
here can't affect the (much lighter, always-should-be-up) prediction demo.
See src/telemetry_data.py and ANALYSIS.md for the full incident.
"""

import os

import gradio as gr
import matplotlib.pyplot as plt

from src.fastf1_data import fetch_all_results
from src.replay import build_race_replay
from src.strategy import build_strategy_chart, pit_stop_summary
from src.telemetry_analysis import compare_lap_telemetry, driver_top_speed_by_race
from src.telemetry_data import get_finished_2026_events, load_session

finished_events = get_finished_2026_events()
FINISHED_EVENT_NAMES = finished_events["EventName"].tolist()
_event_round_lookup = dict(zip(finished_events["EventName"], finished_events["RoundNumber"]))
_finished_rounds_ascending = sorted(int(r) for r in finished_events["RoundNumber"])

# Cheap, already-fetched race results (no telemetry) -- used only to
# populate driver-name dropdowns without triggering a slow full-telemetry
# session load just to list who was in a given race.
_results_2026 = fetch_all_results()
_results_2026 = _results_2026[(_results_2026["year"] == 2026) & (_results_2026["session_type"] == "R")]
ALL_DRIVERS_2026 = sorted(_results_2026["Abbreviation"].dropna().unique().tolist())


def _drivers_for_event(event_name: str) -> list[str]:
    round_number = _event_round_lookup.get(event_name)
    subset = _results_2026[_results_2026["round"] == round_number]
    return sorted(subset["Abbreviation"].dropna().unique().tolist())


def update_driver_choices(event_name: str):
    drivers = _drivers_for_event(event_name)
    d1 = drivers[0] if drivers else None
    d2 = drivers[1] if len(drivers) > 1 else None
    return gr.update(choices=drivers, value=d1), gr.update(choices=drivers, value=d2)


def build_replay_tab(event_name: str, progress=gr.Progress()):
    round_number = int(_event_round_lookup[event_name])
    progress(0.1, desc="Loading full session telemetry (slow on first load, cached after)...")
    session = load_session(2026, round_number, "R")
    progress(0.7, desc="Building replay frames...")
    fig = build_race_replay(session)
    progress(0.95, desc="Rendering...")
    # gr.Plot renders Plotly figures as a static data/layout snapshot and
    # drops the frames/play-button/slider JS wiring -- confirmed directly:
    # the replay showed up as a single still frame with a non-functional
    # Play button. Rendering as raw HTML (fig.to_html) ships the actual
    # Plotly.js animation wiring, the same code path a standalone
    # fig.write_html() export uses, so Play/the slider genuinely work.
    html = fig.to_html(include_plotlyjs="cdn", full_html=False)
    progress(1.0)
    return html


def build_strategy_tab(event_name: str, progress=gr.Progress()):
    round_number = int(_event_round_lookup[event_name])
    progress(0.2, desc="Loading session data...")
    session = load_session(2026, round_number, "R")
    progress(0.7, desc="Building strategy chart...")
    fig = build_strategy_chart(session)
    table = pit_stop_summary(session)
    progress(1.0)
    return fig, table


def build_telemetry_comparison_tab(event_name: str, driver1: str, driver2: str, lap_selector: str, progress=gr.Progress()):
    round_number = int(_event_round_lookup[event_name])
    progress(0.2, desc="Loading session telemetry...")
    session = load_session(2026, round_number, "R")
    progress(0.7, desc="Building comparison...")
    fig = compare_lap_telemetry(session, driver1, driver2, lap_selector.strip() or "fastest")
    progress(1.0)
    return fig


def _line_chart(df, x_col, y_col, title):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(df[x_col], df[y_col], marker="o", color="#1976d2")
    ax.set_title(title)
    ax.set_ylabel(y_col.replace("_", " ").title())
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    return fig


def build_top_speed_trend_tab(driver: str, progress=gr.Progress()):
    events = [(2026, r) for r in _finished_rounds_ascending]
    progress(0.1, desc=f"Loading telemetry across {len(events)} races (slow the first time, cached after)...")
    df = driver_top_speed_by_race(driver, events)
    progress(0.9, desc="Building chart...")
    fig = _line_chart(df, "event_name", "top_speed_kmh", f"{driver} — max recorded speed by race (km/h)")
    progress(1.0)
    return fig, df


with gr.Blocks(title="F1 2026 Telemetry") as demo:
    gr.Markdown(
        "# F1 2026 Telemetry & Race Engineering Tools\n"
        "Real 2D race replay, pit strategy, per-lap telemetry comparison, and "
        "straight-line top-speed trends -- all built directly from FastF1's raw "
        "car + positional telemetry, not the win-probability model. "
        "**This is a separate service from the main predictor** because loading "
        "full session telemetry is memory-heavy; see ANALYSIS.md for why. "
        "**First load of any race is slow** (real download from FastF1); cached after."
    )

    with gr.Tab("Race Replay"):
        replay_event_input = gr.Dropdown(
            choices=FINISHED_EVENT_NAMES,
            value=FINISHED_EVENT_NAMES[0] if FINISHED_EVENT_NAMES else None,
            label="Finished 2026 race",
        )
        replay_btn = gr.Button("Build replay", variant="primary")
        replay_plot = gr.HTML()
        replay_btn.click(build_replay_tab, inputs=replay_event_input, outputs=replay_plot)

    with gr.Tab("Pit Strategy"):
        gr.Markdown(
            "Tyre stint strategy per driver, plus pit-lane time for each stop. "
            "Pit-lane time is pit-in-to-pit-out, not the official sub-3s box-only "
            "stationary time TV graphics show -- FastF1 doesn't expose box-only timing."
        )
        strategy_event_input = gr.Dropdown(
            choices=FINISHED_EVENT_NAMES,
            value=FINISHED_EVENT_NAMES[0] if FINISHED_EVENT_NAMES else None,
            label="Finished 2026 race",
        )
        strategy_btn = gr.Button("Build strategy chart", variant="primary")
        strategy_plot = gr.Plot()
        strategy_table = gr.Dataframe(label="Pit stops")
        strategy_btn.click(build_strategy_tab, inputs=strategy_event_input, outputs=[strategy_plot, strategy_table])

    with gr.Tab("Telemetry Comparison"):
        gr.Markdown(
            "Distance-aligned speed/throttle/brake/DRS trace comparison between two "
            "drivers on the same lap -- where one driver gains or loses time on track."
        )
        tel_event_input = gr.Dropdown(
            choices=FINISHED_EVENT_NAMES,
            value=FINISHED_EVENT_NAMES[0] if FINISHED_EVENT_NAMES else None,
            label="Finished 2026 race",
        )
        _initial_drivers = _drivers_for_event(FINISHED_EVENT_NAMES[0]) if FINISHED_EVENT_NAMES else []
        with gr.Row():
            driver1_input = gr.Dropdown(
                choices=_initial_drivers, value=_initial_drivers[0] if _initial_drivers else None, label="Driver 1"
            )
            driver2_input = gr.Dropdown(
                choices=_initial_drivers,
                value=_initial_drivers[1] if len(_initial_drivers) > 1 else None,
                label="Driver 2",
            )
        lap_selector_input = gr.Textbox(label="Lap ('fastest', or a specific lap number)", value="fastest")
        tel_event_input.change(update_driver_choices, inputs=tel_event_input, outputs=[driver1_input, driver2_input])
        tel_btn = gr.Button("Compare", variant="primary")
        tel_plot = gr.Plot()
        tel_btn.click(
            build_telemetry_comparison_tab,
            inputs=[tel_event_input, driver1_input, driver2_input, lap_selector_input],
            outputs=tel_plot,
        )

    with gr.Tab("Top Speed Trend"):
        gr.Markdown(
            "Max recorded car speed per race across the season -- a real, checkable "
            "way to see whether a driver's straight-line speed genuinely changed "
            "(e.g. after a reported rear-wing/aero setup change), instead of taking "
            "a news claim on faith."
        )
        speed_driver_input = gr.Dropdown(
            choices=ALL_DRIVERS_2026, value=ALL_DRIVERS_2026[0] if ALL_DRIVERS_2026 else None, label="Driver"
        )
        speed_btn = gr.Button("Show top speed trend", variant="primary")
        speed_plot = gr.Plot()
        speed_table = gr.Dataframe()
        speed_btn.click(build_top_speed_trend_tab, inputs=speed_driver_input, outputs=[speed_plot, speed_table])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
