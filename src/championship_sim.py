"""Monte Carlo championship simulation: samples a full finishing order for
each remaining 2026 race/sprint from the model's predicted probabilities
(Plackett-Luce sampling), awards real F1 points, and combines with actual
current standings to estimate each driver's championship win probability.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.fastf1_data import fetch_all_results, get_remaining_2026_events
from src.features import FEATURE_COLUMNS, build_race_features
from src.train_model import MODEL_DIR, predict_race_probabilities

RACE_POINTS = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
SPRINT_POINTS = {1: 8, 2: 7, 3: 6, 4: 5, 5: 4, 6: 3, 7: 2, 8: 1}
N_SIMULATIONS = 5000


def get_current_standings() -> pd.Series:
    """Actual points scored so far in the real 2026 season, race + sprint."""
    results = fetch_all_results()
    season = results[(results["year"] == 2026) & (results["session_type"].isin(["R", "S"]))]
    points = pd.to_numeric(season["Points"], errors="coerce").fillna(0)
    return season.assign(Points=points).groupby("Abbreviation")["Points"].sum()


def get_driver_snapshot(session_type: str = "R") -> pd.DataFrame:
    """Each driver's most recent rolling-feature row -- the best available
    estimate of current form/car competitiveness, used as the feature vector
    for all future (not-yet-run) races under a stationarity assumption."""
    features = build_race_features(session_type)
    latest = features.sort_values(["year", "round"]).groupby("Abbreviation").tail(1).copy()
    # Grid position isn't known for future races -- use recent average as a proxy.
    latest["GridPosition"] = latest["driver_recent_grid"]
    return latest


def _sample_order(probs: pd.Series, rng: np.random.Generator) -> list[str]:
    """Plackett-Luce sampling: repeatedly draw a winner from the remaining
    probability mass, renormalize, repeat -- produces one plausible full
    finishing order per call."""
    items = probs.index.tolist()
    weights = probs.values.astype(float).copy()
    order = []
    while items:
        p = weights / weights.sum()
        idx = rng.choice(len(items), p=p)
        order.append(items[idx])
        del items[idx]
        weights = np.delete(weights, idx)
    return order


def simulate_championship(n_simulations: int = N_SIMULATIONS, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    current_points = get_current_standings()

    race_model = joblib.load(MODEL_DIR / "model_R.joblib")
    sprint_model = joblib.load(MODEL_DIR / "model_S.joblib")
    race_snapshot = get_driver_snapshot("R")
    sprint_snapshot = get_driver_snapshot("S")

    remaining = get_remaining_2026_events()

    race_probs = predict_race_probabilities(race_model, race_snapshot).set_index("Abbreviation")["win_probability"]
    sprint_probs = None
    if len(sprint_snapshot) > 0:
        sprint_probs = predict_race_probabilities(sprint_model, sprint_snapshot).set_index("Abbreviation")["win_probability"]

    drivers = current_points.index.tolist()
    final_points = np.tile(current_points.reindex(drivers, fill_value=0).values, (n_simulations, 1)).astype(float)
    driver_idx = {d: i for i, d in enumerate(drivers)}

    for _, event in remaining.iterrows():
        is_sprint = "sprint" in str(event.get("EventFormat", "")).lower()
        for sim in range(n_simulations):
            order = _sample_order(race_probs, rng)
            for pos, driver in enumerate(order, start=1):
                if driver in driver_idx and pos in RACE_POINTS:
                    final_points[sim, driver_idx[driver]] += RACE_POINTS[pos]
            if is_sprint and sprint_probs is not None:
                sprint_order = _sample_order(sprint_probs, rng)
                for pos, driver in enumerate(sprint_order, start=1):
                    if driver in driver_idx and pos in SPRINT_POINTS:
                        final_points[sim, driver_idx[driver]] += SPRINT_POINTS[pos]

    champion_idx = final_points.argmax(axis=1)
    champion_counts = pd.Series(champion_idx).value_counts(normalize=True)
    championship_prob = pd.Series(0.0, index=drivers)
    for idx, prob in champion_counts.items():
        championship_prob.iloc[idx] = prob

    summary = pd.DataFrame({
        "current_points": current_points.reindex(drivers, fill_value=0),
        "mean_final_points": final_points.mean(axis=0),
        "championship_probability": championship_prob.values,
    }, index=drivers)
    return summary.sort_values("championship_probability", ascending=False)


if __name__ == "__main__":
    summary = simulate_championship()
    print(summary.to_string())
