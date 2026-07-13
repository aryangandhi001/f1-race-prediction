"""Gradio demo: predicted win probabilities for the next upcoming 2026 race
(and sprint, if applicable), championship win-probability estimates, and
real official F1.com news headlines (never fabricated -- see src/news.py)."""

import os

import gradio as gr
import joblib
import matplotlib.pyplot as plt

from src.calibration import calibration_report
from src.championship_sim import get_driver_snapshot, simulate_championship
from src.elo_model import current_ratings_table, evaluate_elo_vs_lightgbm, train_elo_ratings
from src.explain import explain_field_summary, explain_single_prediction
from src.fastf1_data import fetch_all_results, get_remaining_2026_events
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
CURRENT_DRIVERS = sorted(race_base["Abbreviation"].dropna().unique().tolist())


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


def explain_next_race(event_name: str, driver: str, progress=gr.Progress()):
    progress(0.2, desc="Building race features...")
    race_features = get_event_features(race_base, race_driver_avg, race_team_avg, event_name)
    progress(0.5, desc="Computing SHAP values for the field...")
    field_fig = explain_field_summary(race_model, race_features)
    progress(0.8, desc=f"Explaining {driver}'s prediction...")
    driver_row = race_features[race_features["Abbreviation"] == driver].iloc[0]
    driver_fig = explain_single_prediction(race_model, driver_row)
    progress(1.0)
    return driver_fig, field_fig


def show_model_comparison(progress=gr.Progress()):
    progress(0.1, desc="Training Elo ratings on race history...")
    results = fetch_all_results()
    team_elo, driver_elo, _ = train_elo_ratings(results)
    teams_table, drivers_table = current_ratings_table(team_elo, driver_elo)

    progress(0.5, desc="Evaluating Elo vs LightGBM on held-out races...")
    comparison = evaluate_elo_vs_lightgbm(race_model)

    progress(0.8, desc="Checking LightGBM probability calibration...")
    calib = calibration_report(race_model)

    summary = (
        "### Elo vs. LightGBM on the identical held-out races\n\n"
        f"| Model | Brier score (lower=better) | Log loss (lower=better) |\n"
        f"|---|---|---|\n"
        f"| LightGBM | {comparison['lightgbm']['brier_score']:.4f} | {comparison['lightgbm']['log_loss']:.4f} |\n"
        f"| Elo (team + teammate-relative driver) | {comparison['elo']['brier_score']:.4f} | {comparison['elo']['log_loss']:.4f} |\n\n"
        f"_n={comparison['lightgbm']['n_predictions']} driver-race predictions on races never seen in training._\n\n"
        "LightGBM wins clearly here — expected, since it uses much richer per-race "
        "features (recent form, track history, grid position) than Elo's rating-only "
        "view. Elo's real value is the ratings themselves as an interpretable, "
        "independent signal (see tables below), not as a better predictor.\n\n"
        "### LightGBM probability calibration\n\n"
        f"Brier score: **{calib['brier_score']:.4f}** &nbsp;|&nbsp; Log loss: **{calib['log_loss']:.4f}** "
        f"&nbsp;|&nbsp; n={calib['n_predictions']}\n\n"
        "_Caveat: the held-out set is only ~6 races (~130 driver-race rows) -- enough "
        "to catch gross overconfidence, too small for precise per-bin calibration claims._"
    )
    return summary, teams_table.round(1), drivers_table.round(1), calib["reliability_curve"].round(3)


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

    with gr.Tab("Why this prediction (SHAP)"):
        gr.Markdown(
            "Which features actually pushed a driver's predicted finishing position "
            "up or down, using real SHAP (TreeSHAP) values on the LightGBM model — "
            "not just a static feature-importance list. **Lower predicted position is "
            "better**, so a feature pushing the prediction *down* is good for that driver."
        )
        explain_event_input = gr.Dropdown(
            choices=remaining_events["EventName"].tolist(),
            value=remaining_events["EventName"].iloc[0] if len(remaining_events) else None,
            label="Upcoming event",
        )
        explain_driver_input = gr.Dropdown(choices=CURRENT_DRIVERS, value=CURRENT_DRIVERS[0] if CURRENT_DRIVERS else None, label="Driver")
        explain_btn = gr.Button("Explain", variant="primary")
        explain_driver_plot = gr.Plot(label="This driver's prediction, broken down")
        explain_field_plot = gr.Plot(label="Whole field: feature impact summary")
        explain_btn.click(
            explain_next_race, inputs=[explain_event_input, explain_driver_input],
            outputs=[explain_driver_plot, explain_field_plot],
        )

    with gr.Tab("Model comparison"):
        gr.Markdown(
            "A second model — Elo-style ratings (separate team Elo for car/PU "
            "competitiveness, and driver Elo isolated via teammate-only comparisons, "
            "the standard way analysts separate driver skill from car performance) — "
            "benchmarked head-to-head against LightGBM on the identical held-out races, "
            "plus a calibration check on LightGBM's probabilities."
        )
        compare_btn = gr.Button("Run comparison", variant="primary")
        compare_summary = gr.Markdown()
        with gr.Row():
            elo_teams_table = gr.Dataframe(label="Team Elo (car/PU competitiveness)")
            elo_drivers_table = gr.Dataframe(label="Driver Elo (teammate-relative, car-isolated)")
        reliability_table = gr.Dataframe(label="Calibration: predicted vs. actual win rate by probability bin")
        compare_btn.click(
            show_model_comparison,
            outputs=[compare_summary, elo_teams_table, elo_drivers_table, reliability_table],
        )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
