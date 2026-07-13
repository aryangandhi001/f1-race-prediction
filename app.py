"""Gradio demo: predicted win probabilities for the next upcoming 2026 race
(and sprint, if applicable), championship win-probability estimates, and
real official F1.com news headlines (never fabricated -- see src/news.py)."""

import os

import gradio as gr
import joblib
import matplotlib.pyplot as plt

from src.championship_sim import get_driver_snapshot, simulate_championship
from src.fastf1_data import get_remaining_2026_events
from src.features import compute_track_averages, get_event_features
from src.news import fetch_latest_news
from src.train_model import MODEL_DIR, predict_race_probabilities

race_model = joblib.load(MODEL_DIR / "model_R.joblib")
sprint_model = joblib.load(MODEL_DIR / "model_S.joblib")
race_base = get_driver_snapshot("R").drop(columns=["driver_track_history", "team_track_history"])
race_driver_avg, race_team_avg = compute_track_averages("R")
sprint_base = get_driver_snapshot("S").drop(columns=["driver_track_history", "team_track_history"])
sprint_driver_avg, sprint_team_avg = compute_track_averages("S")
remaining_events = get_remaining_2026_events()

TEAMS = sorted(race_base["TeamName"].dropna().unique().tolist())


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

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
