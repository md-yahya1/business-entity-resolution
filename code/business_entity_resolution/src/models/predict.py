"""
Inference script and module for Entity Resolution matching model.
"""
import os
import json
import joblib
import pandas as pd
import numpy as np
from ..features import extract_features_dataframe, FEATURE_NAMES


def load_model_and_config(model_path: str, config_path: str = None, meta_path: str = None):
    """
    Load trained model object, feature config, and decision threshold.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    model = joblib.load(model_path)

    # Derive directory of model if config/meta paths not explicitly given
    model_dir = os.path.dirname(model_path)
    if config_path is None:
        config_path = os.path.join(model_dir, "feature_config.json")
    if meta_path is None:
        meta_path = os.path.join(model_dir, "model_metadata.json")

    threshold = 0.5
    feature_names = FEATURE_NAMES

    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
            threshold = meta.get("decision_threshold", 0.5)
            feature_names = meta.get("feature_names", FEATURE_NAMES)

    elif os.path.exists(config_path):
        with open(config_path, "r") as f:
            cfg = json.load(f)
            feature_names = cfg.get("feature_names", FEATURE_NAMES)

    return model, feature_names, threshold


def predict_candidate_pairs(
    df: pd.DataFrame,
    model_path: str,
    output_path: str = None
) -> pd.DataFrame:
    """
    Run inference on candidate pair DataFrame.
    Calculates features if not present and returns predicted probabilities and match decisions.
    """
    model, expected_features, threshold = load_model_and_config(model_path)

    df_out = df.copy()

    # Check if feature columns are already present
    has_features = all(col in df.columns for col in expected_features)

    if has_features:
        X = df[expected_features].values
    else:
        # Extract features from pair entity attribute columns
        features_df = extract_features_dataframe(df)
        X = features_df[expected_features].values
        for col in expected_features:
            df_out[col] = features_df[col]

    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= threshold).astype(int)

    df_out["match_probability"] = probs
    df_out["is_match"] = preds

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_out.to_csv(output_path, sep="\t", index=False)

    return df_out
