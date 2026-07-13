"""Monte Carlo championship simulation: samples a full finishing order for
each remaining 2026 race/sprint from the model's predicted probabilities,
awards real F1 points, and combines with actual current standings to
estimate each driver's championship win probability.

Sampling uses the Gumbel-max trick (add Gumbel noise to log-weights, sort) --
mathematically equivalent to sequential Plackett-Luce sampling (draw a
winner, remove it, renormalize, repeat), but fully vectorized across all
simulations at once instead of a nested Python loop. The naive loop version
did ~1.3M individual Python-level sampling steps for a full run (5,000 sims
x 13 events x ~20 drivers) and was slow enough to look hung in the demo.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.fastf1_data import fetch_all_results, get_remaining_2026_events
from src.features import FEATURE_COLUMNS, compute_track_averages, get_event_features
from src.train_model import MODEL_DIR, predict_race_probabilities

RACE_POINTS = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
SPRINT_POINTS = {1: 8, 2: 7, 3: 6, 4: 5, 5: 4, 6: 3, 7: 2, 8: 1}
N_SIMULATIONS = 2000


def get_current_standings() -> pd.Series:
    """Actual points scored so far in the real 2026 season, race + sprint."""
    results = fetch_all_results()
    season = results[(results["year"] == 2026) & (results["session_type"].isin(["R", "S"]))]
    points = pd.to_numeric(season["Points"], errors="coerce").fillna(0)
    return season.assign(Points=points).groupby("Abbreviation")["Points"].sum()


def get_driver_snapshot(session_type: str = "R") -> pd.DataFrame:
    """Each driver's most recent rolling-feature row -- the best available
    estimate of current form/car competitiveness, used as the base feature
    vector for all future (not-yet-run) races under a stationarity
    assumption. Track-history columns on this base row reflect whichever
    track was last actually raced -- callers predicting a specific upcoming
    event should override via `get_event_features`/`get_features_for_event`."""
    from src.features import build_race_features

    features = build_race_features(session_type)
    latest = features.sort_values(["year", "round"]).groupby("Abbreviation").tail(1).copy()
    # Grid position isn't known for future races -- use recent average as a proxy.
    latest["GridPosition"] = latest["driver_recent_grid"]
    return latest


def get_features_for_event(session_type: str, event_name: str) -> pd.DataFrame:
    """Convenience wrapper: driver snapshot with track history overridden
    for a specific upcoming event. For repeated calls across many events
    (e.g. the championship simulation), prefer computing the base snapshot
    and track averages once and calling `get_event_features` directly."""
    base = get_driver_snapshot(session_type).drop(columns=["driver_track_history", "team_track_history"])
    driver_avg, team_avg = compute_track_averages(session_type)
    return get_event_features(base, driver_avg, team_avg, event_name)


def _sample_orders_batch(probs: pd.Series, n_simulations: int, rng: np.random.Generator) -> np.ndarray:
    """Vectorized Plackett-Luce sampling via the Gumbel-max trick. Returns an
    (n_simulations, n_drivers) array where each row is a full finishing
    order (driver labels) for one simulated race."""
    items = probs.index.to_numpy()
    log_weights = np.log(probs.values.astype(float) + 1e-12)
    gumbel_noise = -np.log(-np.log(rng.uniform(1e-12, 1.0, size=(n_simulations, len(items)))))
    scores = log_weights[None, :] + gumbel_noise
    order_idx = np.argsort(-scores, axis=1)
    return items[order_idx]


def _award_points(final_points: np.ndarray, orders: np.ndarray, driver_idx: dict, points_map: dict):
    """Vectorized points assignment: for each scoring position, look up
    which driver landed there in every simulation at once."""
    n_positions = min(orders.shape[1], max(points_map.keys()))
    for pos in range(1, n_positions + 1):
        if pos not in points_map:
            continue
        drivers_at_pos = orders[:, pos - 1]
        col_idx = np.array([driver_idx.get(d, -1) for d in drivers_at_pos])
        valid = col_idx >= 0
        final_points[np.nonzero(valid)[0], col_idx[valid]] += points_map[pos]


def simulate_championship(n_simulations: int = N_SIMULATIONS, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    current_points = get_current_standings()

    race_model = joblib.load(MODEL_DIR / "model_R.joblib")
    sprint_model = joblib.load(MODEL_DIR / "model_S.joblib")

    # Compute base snapshots + track-history tables once; per-event feature
    # rows below are then a cheap merge instead of rebuilding all of
    # build_race_features() for every remaining event.
    race_base = get_driver_snapshot("R").drop(columns=["driver_track_history", "team_track_history"])
    race_driver_avg, race_team_avg = compute_track_averages("R")
    sprint_base = get_driver_snapshot("S").drop(columns=["driver_track_history", "team_track_history"])
    sprint_driver_avg, sprint_team_avg = compute_track_averages("S")

    remaining = get_remaining_2026_events()

    drivers = current_points.index.tolist()
    final_points = np.tile(current_points.reindex(drivers, fill_value=0).values, (n_simulations, 1)).astype(float)
    driver_idx = {d: i for i, d in enumerate(drivers)}

    for _, event in remaining.iterrows():
        event_name = event["EventName"]
        is_sprint = "sprint" in str(event.get("EventFormat", "")).lower()

        race_features = get_event_features(race_base, race_driver_avg, race_team_avg, event_name)
        race_probs = predict_race_probabilities(race_model, race_features).set_index("Abbreviation")["win_probability"]
        race_orders = _sample_orders_batch(race_probs, n_simulations, rng)
        _award_points(final_points, race_orders, driver_idx, RACE_POINTS)

        if is_sprint:
            sprint_features = get_event_features(sprint_base, sprint_driver_avg, sprint_team_avg, event_name)
            if len(sprint_features) > 0:
                sprint_probs = predict_race_probabilities(sprint_model, sprint_features).set_index("Abbreviation")["win_probability"]
                sprint_orders = _sample_orders_batch(sprint_probs, n_simulations, rng)
                _award_points(final_points, sprint_orders, driver_idx, SPRINT_POINTS)

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
    import time
    t0 = time.perf_counter()
    summary = simulate_championship()
    print(summary.to_string())
    print(f"\nSimulation took {time.perf_counter()-t0:.1f}s")
