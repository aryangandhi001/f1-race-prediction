"""SHAP-based explainability for the LightGBM race-position model: which
features actually pushed a specific driver's predicted finishing position
up or down, and which features matter most overall -- real, per-prediction
explanations rather than just a global feature-importance list.
"""

import matplotlib.pyplot as plt
import pandas as pd
import shap

from src.features import FEATURE_COLUMNS


def build_explainer(model) -> shap.TreeExplainer:
    """shap.TreeExplainer computes exact SHAP values for tree ensembles
    (LightGBM included) via the fast, model-specific TreeSHAP algorithm --
    no sampling/approximation needed, unlike the model-agnostic KernelSHAP
    path, and cheap enough to rebuild per request rather than caching."""
    return shap.TreeExplainer(model)


def explain_single_prediction(model, feature_row: pd.Series) -> plt.Figure:
    """Waterfall plot: how each feature pushed this one driver's predicted
    finishing position away from the model's average (base value) output --
    e.g. 'grid position pushed the prediction 2 places better, but poor
    recent team form pushed it 1.5 places worse'. Note lower predicted
    position is better in this model, so a SHAP value that DECREASES the
    predicted position is a good sign for that driver, not bad."""
    explainer = build_explainer(model)
    X = feature_row[FEATURE_COLUMNS].to_frame().T.astype(float)
    explanation = explainer(X)

    fig = plt.figure(figsize=(8, 5))
    shap.plots.waterfall(explanation[0], show=False)
    fig = plt.gcf()
    fig.tight_layout()
    return fig


def explain_field_summary(model, features_df: pd.DataFrame) -> plt.Figure:
    """Global summary plot (beeswarm) across every driver in a given race
    (or any set of rows): which features have the most impact on
    predictions overall, and in which direction, across the whole field --
    not just one driver's breakdown."""
    explainer = build_explainer(model)
    X = features_df[FEATURE_COLUMNS].astype(float)
    explanation = explainer(X)

    plt.figure(figsize=(8, 6))
    shap.plots.beeswarm(explanation, show=False)
    fig = plt.gcf()
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    import joblib

    from src.championship_sim import get_driver_snapshot
    from src.train_model import MODEL_DIR

    model = joblib.load(MODEL_DIR / "model_R.joblib")
    snapshot = get_driver_snapshot("R")

    fig = explain_field_summary(model, snapshot)
    fig.savefig("shap_summary_check.png", dpi=100)
    print("Wrote shap_summary_check.png")

    row = snapshot.iloc[0]
    print(f"Explaining prediction for {row['Abbreviation']}")
    fig2 = explain_single_prediction(model, row)
    fig2.savefig("shap_waterfall_check.png", dpi=100)
    print("Wrote shap_waterfall_check.png")
