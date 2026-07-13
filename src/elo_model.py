"""An Elo-style alternative to the LightGBM regressor, evaluated on the
exact same held-out races for a genuine apples-to-apples comparison --
the "compare multiple modeling approaches" idea, done as one well-scoped
second model rather than a half-built lineup of several.

Design: team strength and driver skill are tracked as SEPARATE Elo
ratings, updated from different comparisons each race:

  - Team Elo updates from team-vs-team relative finishing (each team's
    best-classified car that race) -- captures car/power-unit
    competitiveness.
  - Driver Elo updates ONLY from teammate-vs-teammate results (who beat
    their own teammate that race) -- the standard way motorsport analysts
    isolate driver skill from car performance, since teammates share
    identical equipment. This deliberately keeps driver skill measured
    independently of the car, which matters specifically because 2026 is
    a regulation reset (the central framing of this whole project): team
    Elo can and should swing a lot as car competitiveness resets, while a
    properly car-isolated driver Elo shouldn't have to.

A driver's predicted race strength is team_elo[team] + driver_elo[driver],
converted to win probabilities via a softmax on rating (scale=175, chosen
to match classical Elo's own convention that a 400-point gap implies
10:1 odds -- 400/ln(10) = 173.7 -- rather than an arbitrarily tuned
constant).
"""

import numpy as np
import pandas as pd

BASE_RATING = 1500.0
TEAM_K = 24.0
DRIVER_K = 16.0
ELO_SCALE = 175.0  # 400/ln(10): matches classical Elo's 400-points-per-decade-of-odds convention


def _pairwise_elo_update(ratings: dict, ordered_ids: list, k: float) -> None:
    """Standard multiplayer Elo via pairwise decomposition: every pair in
    `ordered_ids` (already sorted best-to-worst) counts as one pairwise
    comparison; each entity's rating moves by K times its average surprise
    across all opponents that race (averaging, rather than summing, keeps
    the rating swing on a consistent scale regardless of field size)."""
    n = len(ordered_ids)
    if n < 2:
        return
    current = {rid: ratings.get(rid, BASE_RATING) for rid in ordered_ids}
    deltas = {rid: 0.0 for rid in ordered_ids}
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            ri, rj = current[ordered_ids[i]], current[ordered_ids[j]]
            expected_i = 1.0 / (1.0 + 10 ** ((rj - ri) / 400.0))
            actual_i = 1.0 if i < j else 0.0  # lower index = better finishing position
            deltas[ordered_ids[i]] += actual_i - expected_i
    n_opponents = n - 1
    for rid in ordered_ids:
        ratings[rid] = current[rid] + k * deltas[rid] / n_opponents


def train_elo_ratings(results_df: pd.DataFrame, session_type: str = "R") -> tuple[dict, dict, pd.DataFrame]:
    """Replays every race in strict chronological order, updating team and
    driver Elo after each one. Returns the final ratings plus a full
    history of every driver's PRE-race rating snapshot for every race --
    required to evaluate held-out predictions without leakage, since a
    held-out race's prediction must only ever use ratings as they stood
    immediately before that race, not after."""
    df = results_df[results_df["session_type"] == session_type].copy()
    df["Position"] = pd.to_numeric(df["Position"], errors="coerce")
    df = df.sort_values(["year", "round"])

    team_elo: dict[str, float] = {}
    driver_elo: dict[str, float] = {}
    history_rows = []

    for (year, rnd), race in df.groupby(["year", "round"], sort=False):
        classified = race[race["Position"].notna()].sort_values("Position")
        if len(classified) == 0:
            continue

        for _, row in classified.iterrows():
            history_rows.append({
                "year": year,
                "round": rnd,
                "Abbreviation": row["Abbreviation"],
                "TeamName": row["TeamName"],
                "team_elo_pre": team_elo.get(row["TeamName"], BASE_RATING),
                "driver_elo_pre": driver_elo.get(row["Abbreviation"], BASE_RATING),
            })

        team_order = classified.groupby("TeamName")["Position"].min().sort_values().index.tolist()
        _pairwise_elo_update(team_elo, team_order, TEAM_K)

        for _, group in classified.groupby("TeamName"):
            if len(group) < 2:
                continue  # no valid teammate comparison this race (DNS/DNF/single entry)
            ordered_drivers = group.sort_values("Position")["Abbreviation"].tolist()
            _pairwise_elo_update(driver_elo, ordered_drivers, DRIVER_K)

    return team_elo, driver_elo, pd.DataFrame(history_rows)


def elo_win_probabilities(pre_race_ratings: pd.DataFrame, scale: float = ELO_SCALE) -> np.ndarray:
    """Converts a race's field of (team_elo_pre + driver_elo_pre) combined
    ratings into a win-probability distribution via softmax -- the Elo
    equivalent of train_model.predict_race_probabilities's softmax on
    predicted position."""
    combined = (pre_race_ratings["team_elo_pre"] + pre_race_ratings["driver_elo_pre"]).to_numpy()
    scores = combined / scale
    probs = np.exp(scores - scores.max())
    return probs / probs.sum()


def current_ratings_table(team_elo: dict, driver_elo: dict) -> pd.DataFrame:
    """A readable snapshot of final ratings, sorted descending -- useful
    for sanity-checking the Elo model actually learned something sensible
    (e.g. do the top teams/drivers match the real current standings)."""
    teams = pd.DataFrame(sorted(team_elo.items(), key=lambda x: -x[1]), columns=["TeamName", "team_elo"])
    drivers = pd.DataFrame(sorted(driver_elo.items(), key=lambda x: -x[1]), columns=["Abbreviation", "driver_elo"])
    return teams, drivers


def evaluate_elo_vs_lightgbm(model, session_type: str = "R") -> dict:
    """Head-to-head comparison of the Elo model against the LightGBM
    regressor on the *exact same* held-out races (train_model's
    time_based_split) -- the genuine apples-to-apples comparison, not two
    separately-chosen evaluation sets."""
    from src.calibration import _race_win_probabilities, brier_score, log_loss_score
    from src.fastf1_data import fetch_all_results
    from src.features import build_race_features
    from src.train_model import time_based_split

    features_df = build_race_features(session_type)
    _, val_df = time_based_split(features_df)
    val_keys = set(zip(val_df["year"], val_df["round"]))

    results_df = fetch_all_results()
    _, _, history = train_elo_ratings(results_df, session_type)
    val_history = history[history.apply(lambda r: (r["year"], r["round"]) in val_keys, axis=1)]

    elo_rows = []
    for (year, rnd), group in val_history.groupby(["year", "round"]):
        probs = elo_win_probabilities(group)
        actual = val_df[(val_df["year"] == year) & (val_df["round"] == rnd)].set_index("Abbreviation")["finish_position"]
        for abbr, prob in zip(group["Abbreviation"], probs):
            elo_rows.append({"predicted_win_prob": prob, "actually_won": actual.get(abbr, 99) == 1})
    elo_predictions = pd.DataFrame(elo_rows)

    lgbm_predictions = _race_win_probabilities(model, val_df)

    def _score(preds):
        p = preds["predicted_win_prob"].to_numpy()
        y = preds["actually_won"].to_numpy().astype(float)
        return {
            "brier_score": brier_score(p, y),
            "log_loss": log_loss_score(p, y),
            "n_predictions": len(preds),
        }

    return {"elo": _score(elo_predictions), "lightgbm": _score(lgbm_predictions)}


if __name__ == "__main__":
    from src.fastf1_data import fetch_all_results

    results = fetch_all_results()
    team_elo, driver_elo, history = train_elo_ratings(results)
    teams, drivers = current_ratings_table(team_elo, driver_elo)
    print("=== Team Elo (car/PU competitiveness) ===")
    print(teams.round(1).to_string())
    print("\n=== Driver Elo (teammate-relative, car-isolated) ===")
    print(drivers.round(1).to_string())
