"""Gradio demo: predicted win probabilities for the next upcoming 2026 race
(and sprint, if applicable), championship win-probability estimates, and
real official F1.com news headlines (never fabricated -- see src/news.py)."""

import os

import gradio as gr
import joblib
import matplotlib.pyplot as plt

from src.championship_sim import get_driver_snapshot, simulate_championship
from src.fastf1_data import fetch_all_results, get_remaining_2026_events
from src.features import compute_track_averages, get_event_features
from src.news import fetch_latest_news
from src.replay import build_race_replay
from src.strategy import build_strategy_chart, pit_stop_summary
from src.telemetry_analysis import compare_lap_telemetry, driver_top_speed_by_race
from src.telemetry_data import get_finished_2026_events, load_session
from src.train_model import MODEL_DIR, predict_race_probabilities

race_model = joblib.load(MODEL_DIR / "model_R.joblib")
sprint_model = joblib.load(MODEL_DIR / "model_S.joblib")
race_base = get_driver_snapshot("R").drop(columns=["driver_track_history", "team_track_history"])
race_driver_avg, race_team_avg = compute_track_averages("R")
sprint_base = get_driver_snapshot("S").drop(columns=["driver_track_history", "team_track_history"])
sprint_driver_avg, sprint_team_avg = compute_track_averages("S")
remaining_events = get_remaining_2026_events()

TEAMS = sorted(race_base["TeamName"].dropna().unique().tolist())

# --- Telemetry-driven tabs: replay, strategy, per-lap comparison, top speed trend ---
# These use FastF1's heavier car+positional telemetry (src/telemetry_data.py),
# not the lightweight results-only fetch the prediction tabs above use.
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
    progress(1.0)
    return fig


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


def _bar_chart(df, value_col, label_col, title):
    fig, ax = plt.subplots(figsize=(7, 5))
    top = df.head(10).iloc[::-1]
    ax.barh(top[label_col], top[value_col], color="#1976d2")
    ax.set_xlabel(value_col.replace("_", " ").title())
    ax.set_title(title)
    fig.tight_layout()
    return fig


def _no_sprint_fig(event_name: str):
    """A blank sprint plot with no explanation reads as 'broken' -- most
    events don't have a sprint (only 6/24 do), so this makes that explicit
    instead of silently leaving the chart empty."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.text(
        0.5, 0.5, f"{event_name} has no sprint race\n(only select 2026 weekends do)",
        ha="center", va="center", fontsize=12, color="#666666", wrap=True,
    )
    ax.axis("off")
    return fig


def predict_next_race(event_name: str):
    event_row = remaining_events[remaining_events["EventName"] == event_name].iloc[0]
    is_sprint = "sprint" in str(event_row.get("EventFormat", "")).lower()

    race_features = get_event_features(race_base, race_driver_avg, race_team_avg, event_name)
    race_probs = predict_race_probabilities(race_model, race_features)
    race_fig = _bar_chart(race_probs, "win_probability", "Abbreviation", f"{event_name} — Race win probability")

    if is_sprint:
        sprint_features = get_event_features(sprint_base, sprint_driver_avg, sprint_team_avg, event_name)
        sprint_probs = predict_race_probabilities(sprint_model, sprint_features)
        sprint_fig = _bar_chart(sprint_probs, "win_probability", "Abbreviation", f"{event_name} — Sprint win probability")
        return race_fig, sprint_fig
    return race_fig, _no_sprint_fig(event_name)


def show_championship():
    summary = simulate_championship()
    summary = summary.reset_index().rename(columns={"index": "Abbreviation"})
    fig = _bar_chart(summary, "championship_probability", "Abbreviation", "2026 Championship win probability")
    table = summary[["Abbreviation", "current_points", "mean_final_points", "championship_probability"]].round(3)
    return fig, table


def show_news(team: str):
    items = fetch_latest_news(team if team != "All teams" else None, limit=8)
    if not items:
        return "No matching articles found in the current feed."
    if "error" in items[0]:
        return f"⚠️ {items[0]['error']}"

    lines = [f"**Latest official F1.com news{f' — {team}' if team != 'All teams' else ''}:**\n"]
    for item in items:
        date = item["pub_date"].split(" +")[0] if item["pub_date"] else ""
        lines.append(f"- [{item['title']}]({item['link']})  \n  _{date}_")
    return "\n".join(lines)


with gr.Blocks(title="F1 2026 Predictor") as demo:
    gr.Markdown(
        "# F1 2026 Race & Championship Predictor\n"
        "Trained on real FastF1 timing data (2022-2026). **Note:** 2026 is a major "
        "regulation-reset year, so predictions lean on this season's actual results "
        "as they accumulate rather than assuming past team competitiveness carries over. "
        "See `ANALYSIS.md`/README for the full methodology and honest limitations."
    )

    with gr.Tab("Next race prediction"):
        event_input = gr.Dropdown(
            choices=remaining_events["EventName"].tolist(),
            value=remaining_events["EventName"].iloc[0] if len(remaining_events) else None,
            label="Upcoming event",
        )
        predict_btn = gr.Button("Predict", variant="primary")
        race_plot = gr.Plot(label="Race")
        sprint_plot = gr.Plot(label="Sprint")
        predict_btn.click(predict_next_race, inputs=event_input, outputs=[race_plot, sprint_plot])

    with gr.Tab("Championship simulation"):
        gr.Markdown(
            f"Monte Carlo simulation ({2000:,} runs) over all remaining 2026 races and "
            "sprints, combined with actual current standings."
        )
        champ_btn = gr.Button("Run simulation", variant="primary")
        champ_plot = gr.Plot()
        champ_table = gr.Dataframe()
        champ_btn.click(show_championship, outputs=[champ_plot, champ_table])

    with gr.Tab("Official news"):
        gr.Markdown(
            "Real headlines pulled live from the official Formula1.com RSS feed — "
            "shown verbatim, never summarized or generated. Useful context "
            "(e.g. upgrade announcements) that isn't part of the numeric model, "
            "since there's no reliable structured data feed for that."
        )
        news_team_input = gr.Dropdown(choices=["All teams"] + TEAMS, value="All teams", label="Filter by team")
        news_btn = gr.Button("Get latest news", variant="primary")
        news_output = gr.Markdown()
        news_btn.click(show_news, inputs=news_team_input, outputs=news_output)

    with gr.Tab("Race Replay"):
        gr.Markdown(
            "A real 2D reconstruction of the race from FastF1's actual positional "
            "telemetry (car X/Y coordinates over time) -- not an illustration. "
            "**First load of a given race is slow** (full telemetry download); "
            "it's cached after that."
        )
        replay_event_input = gr.Dropdown(
            choices=FINISHED_EVENT_NAMES,
            value=FINISHED_EVENT_NAMES[0] if FINISHED_EVENT_NAMES else None,
            label="Finished 2026 race",
        )
        replay_btn = gr.Button("Build replay", variant="primary")
        replay_plot = gr.Plot()
        replay_btn.click(build_replay_tab, inputs=replay_event_input, outputs=replay_plot)

    with gr.Tab("Pit Strategy"):
        gr.Markdown(
            "Tyre stint strategy per driver, plus pit-lane time for each stop, from "
            "real FastF1 timing data. Pit-lane time is pit-in-to-pit-out (not the "
            "official sub-3s box-only stationary time TV graphics show -- FastF1 "
            "doesn't expose that at box-only granularity)."
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
            "drivers on the same lap -- the real engineering view of exactly where "
            "one driver gains or loses time on track."
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
            "a news claim on faith. Loads telemetry across every finished 2026 race, "
            "so the first run is slow; cached after."
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
