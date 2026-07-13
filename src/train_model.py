"""Trains a LightGBM model predicting finishing position from race features,
validated with a time-based split (never random k-fold -- this is a time
series problem, and random splits would leak future race information into
training via the rolling features).
"""

from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from src.features import FEATURE_COLUMNS, TARGET_COLUMN, build_race_features

MODEL_DIR = Path("models")
N_VALIDATION_RACES = 6  # most recent N (year, round) pairs held out for validation


def time_based_split(df: pd.DataFrame, n_val_races: int = N_VALIDATION_RACES):
    race_keys = df[["year", "round"]].drop_duplicates().sort_values(["year", "round"])
    val_keys = race_keys.tail(n_val_races)
    is_val = df.set_index(["year", "round"]).index.isin(val_keys.set_index(["year", "round"]).index)
    return df[~is_val].copy(), df[is_val].copy()


def train(session_type: str = "R"):
    df = build_race_features(session_type)
    train_df, val_df = time_based_split(df)

    X_train, y_train = train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN]
    X_val, y_val = val_df[FEATURE_COLUMNS], val_df[TARGET_COLUMN]

    model = lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=10,
        random_state=42,
        verbose=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )

    val_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, val_pred)
    print(f"[{session_type}] Validation MAE (finishing position): {mae:.2f}")

    # Win-prediction accuracy: for each held-out race, does the model's #1
    # predicted finisher match the actual winner?
    val_df = val_df.copy()
    val_df["pred_position"] = val_pred
    correct, total = 0, 0
    for (year, rnd), group in val_df.groupby(["year", "round"]):
        predicted_winner = group.loc[group["pred_position"].idxmin(), "Abbreviation"]
        actual_winner_rows = group[group["finish_position"] == 1]
        if len(actual_winner_rows) == 0:
            continue
        actual_winner = actual_winner_rows.iloc[0]["Abbreviation"]
        correct += int(predicted_winner == actual_winner)
        total += 1
    print(f"[{session_type}] Winner prediction accuracy on held-out races: {correct}/{total}")

    MODEL_DIR.mkdir(exist_ok=True)
    model_path = MODEL_DIR / f"model_{session_type}.joblib"
    joblib.dump(model, model_path)
    print(f"Saved model to {model_path}")

    importances = sorted(zip(FEATURE_COLUMNS, model.feature_importances_), key=lambda x: -x[1])
    print("Feature importances:")
    for name, imp in importances:
        print(f"  {name}: {imp}")

    return model, mae


def predict_race_probabilities(model, race_features: pd.DataFrame, temperature: float = 2.0) -> pd.DataFrame:
    """Converts predicted finishing positions into a win-probability
    distribution over the drivers in a single race via a softmax on negative
    predicted position (lower predicted position -> higher probability)."""
    preds = model.predict(race_features[FEATURE_COLUMNS])
    scores = -preds / temperature
    probs = np.exp(scores - scores.max())
    probs = probs / probs.sum()
    out = race_features[["Abbreviation", "TeamName"]].copy()
    out["predicted_position"] = preds
    out["win_probability"] = probs
    return out.sort_values("win_probability", ascending=False)


if __name__ == "__main__":
    train("R")
    train("S")
