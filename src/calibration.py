"""Checks whether the model's predicted win probabilities are actually
trustworthy as probabilities -- of the races where a driver was given
~30% win probability, do they actually win about 30% of the time? A model
can have a good "pick the winner" accuracy while still being badly
miscalibrated (e.g. systematically overconfident), and the championship
Monte Carlo simulation (src/championship_sim.py) feeds these probabilities
directly into thousands of simulated seasons -- if they're miscalibrated,
the simulation inherits that bias silently.
"""

import numpy as np
import pandas as pd

from src.features import FEATURE_COLUMNS, build_race_features
from src.train_model import time_based_split


def _race_win_probabilities(model, val_df: pd.DataFrame, temperature: float = 2.0) -> pd.DataFrame:
    """Per-race softmax win probability for every driver in the held-out
    set, plus whether they actually won -- the raw (predicted_prob,
    actual_outcome) pairs calibration is measured from."""
    rows = []
    for (year, rnd), group in val_df.groupby(["year", "round"]):
        preds = model.predict(group[FEATURE_COLUMNS])
        scores = -preds / temperature
        probs = np.exp(scores - scores.max())
        probs = probs / probs.sum()
        actual_winner = group["finish_position"] == 1
        rows.append(pd.DataFrame({
            "year": year, "round": rnd,
            "predicted_win_prob": probs,
            "actually_won": actual_winner.values,
        }))
    return pd.concat(rows, ignore_index=True)


def brier_score(predicted_probs: np.ndarray, actual_outcomes: np.ndarray) -> float:
    """Mean squared error between predicted probability and the 0/1
    outcome -- the standard proper scoring rule for probabilistic
    forecasts (lower is better; 0 is perfect, 0.25 is what a constant
    50/50 guess scores against a 50/50-balanced outcome)."""
    return float(np.mean((predicted_probs - actual_outcomes) ** 2))


def log_loss_score(predicted_probs: np.ndarray, actual_outcomes: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(predicted_probs, eps, 1 - eps)
    return float(-np.mean(actual_outcomes * np.log(p) + (1 - actual_outcomes) * np.log(1 - p)))


def reliability_curve(predicted_probs: np.ndarray, actual_outcomes: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Bins predictions by predicted probability and compares the mean
    predicted probability in each bin to the actual fraction of winners in
    that bin -- a perfectly calibrated model has these two columns equal
    (the classic reliability-diagram data)."""
    bins = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.clip(np.digitize(predicted_probs, bins) - 1, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() == 0:
            continue
        rows.append({
            "bin_low": bins[b],
            "bin_high": bins[b + 1],
            "n_predictions": int(mask.sum()),
            "mean_predicted_prob": float(predicted_probs[mask].mean()),
            "actual_win_rate": float(actual_outcomes[mask].mean()),
        })
    return pd.DataFrame(rows)


def calibration_report(model, session_type: str = "R") -> dict:
    """Full calibration report on the same held-out (most recent N races)
    set used for validation everywhere else in this project -- never a
    different or re-shuffled split, so this reflects genuine out-of-sample
    calibration, not a re-fit view.

    Honest caveat, stated here because it matters for how much to trust
    this: the held-out set is only ~6 races (~120-140 driver-race rows).
    That's enough to sanity-check gross overconfidence but too small for
    fine-grained per-bin calibration claims -- individual bins in the
    reliability curve can have single-digit sample counts. Treat this as a
    coarse sanity check, not a precise calibration certificate."""
    df = build_race_features(session_type)
    _, val_df = time_based_split(df)
    predictions = _race_win_probabilities(model, val_df)

    probs = predictions["predicted_win_prob"].to_numpy()
    outcomes = predictions["actually_won"].to_numpy().astype(float)

    return {
        "brier_score": brier_score(probs, outcomes),
        "log_loss": log_loss_score(probs, outcomes),
        "n_predictions": len(predictions),
        "reliability_curve": reliability_curve(probs, outcomes),
    }


if __name__ == "__main__":
    import joblib
    from src.train_model import MODEL_DIR

    model = joblib.load(MODEL_DIR / "model_R.joblib")
    report = calibration_report(model)
    print(f"Brier score: {report['brier_score']:.4f} (0=perfect, 0.25=uninformative on balanced outcomes)")
    print(f"Log loss: {report['log_loss']:.4f}")
    print(f"n predictions: {report['n_predictions']}")
    print(report["reliability_curve"].to_string())
