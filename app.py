"""Gradio demo: predicted win probabilities for the next upcoming 2026 race
(and sprint, if applicable), plus championship win-probability estimates."""

import os

import gradio as gr
import joblib
import matplotlib.pyplot as plt

from src.championship_sim import get_driver_snapshot, simulate_championship
from src.fastf1_data import get_remaining_2026_events
from src.train_model import MODEL_DIR, predict_race_probabilities

race_model = joblib.load(MODEL_DIR / "model_R.joblib")
sprint_model = joblib.load(MODEL_DIR / "model_S.joblib")
race_snapshot = get_driver_snapshot("R")
sprint_snapshot = get_driver_snapshot("S")
remaining_events = get_remaining_2026_events()


def _bar_chart(df, value_col, label_col, title):
    fig, ax = plt.subplots(figsize=(7, 5))
    top = df.head(10).iloc[::-1]
    ax.barh(top[label_col], top[value_col], color="#1976d2")
    ax.set_xlabel(value_col.replace("_", " ").title())
    ax.set_title(title)
    fig.tight_layout()
    return fig


def predict_next_race(event_name: str):
    event_row = remaining_events[remaining_events["EventName"] == event_name].iloc[0]
    is_sprint = "sprint" in str(event_row.get("EventFormat", "")).lower()

    race_probs = predict_race_probabilities(race_model, race_snapshot)
    race_fig = _bar_chart(race_probs, "win_probability", "Abbreviation", f"{event_name} — Race win probability")

    if is_sprint:
        sprint_probs = predict_race_probabilities(sprint_model, sprint_snapshot)
        sprint_fig = _bar_chart(sprint_probs, "win_probability", "Abbreviation", f"{event_name} — Sprint win probability")
        return race_fig, sprint_fig
    return race_fig, None


def show_championship():
    summary = simulate_championship()
    summary = summary.reset_index().rename(columns={"index": "Abbreviation"})
    fig = _bar_chart(summary, "championship_probability", "Abbreviation", "2026 Championship win probability")
    table = summary[["Abbreviation", "current_points", "mean_final_points", "championship_probability"]].round(3)
    return fig, table


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
            "Monte Carlo simulation (5,000 runs) over all remaining 2026 races and "
            "sprints, combined with actual current standings."
        )
        champ_btn = gr.Button("Run simulation", variant="primary")
        champ_plot = gr.Plot()
        champ_table = gr.Dataframe()
        champ_btn.click(show_championship, outputs=[champ_plot, champ_table])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
