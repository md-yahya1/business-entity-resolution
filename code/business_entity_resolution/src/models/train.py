"""
Model training module for Business Entity Resolution.
"""
import os
import json
import joblib
from datetime import datetime
from typing import Dict, Any, Tuple, List
import pandas as pd
import numpy as np

from sklearn.model_selection import GroupShuffleSplit

from ..features import extract_features_dataframe, FEATURE_NAMES
from ..evaluation.metrics import (
    evaluate_predictions,
    find_optimal_threshold,
    find_optimal_entity_threshold,
    entity_level_f0_5,
    model_selection_score,
)
from .ensemble import build_candidate_models, is_weight_tuned_ensemble


def split_data_by_group(
    df: pd.DataFrame,
    features_df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Perform leakage-safe group split based on entity_group_id.
    Ensures candidate pairs sharing entity records stay in the same split.
    """
    groups = df["entity_group_id"].values
    y = df["label"].values
    X = features_df.values

    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_val_idx, test_idx = next(gss_test.split(X, y, groups=groups))

    X_train_val, y_train_val, groups_train_val = X[train_val_idx], y[train_val_idx], groups[train_val_idx]
    X_test, y_test, groups_test = X[test_idx], y[test_idx], groups[test_idx]

    relative_val_size = val_size / (1.0 - test_size)
    gss_val = GroupShuffleSplit(n_splits=1, test_size=relative_val_size, random_state=random_state)
    train_idx, val_idx = next(gss_val.split(X_train_val, y_train_val, groups=groups_train_val))

    X_train, y_train = X_train_val[train_idx], y_train_val[train_idx]
    X_val, y_val = X_train_val[val_idx], y_train_val[val_idx]
    entity_ids = df["entity_id_1"].astype(str).values

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "entity_ids_train": entity_ids[train_val_idx][train_idx],
        "entity_ids_val": entity_ids[train_val_idx][val_idx],
        "entity_ids_test": entity_ids[test_idx],
        "feature_names": FEATURE_NAMES,
    }


def _fit_model(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    random_seed: int,
) -> None:
    model.fit(X_train, y_train)
    if is_weight_tuned_ensemble(model):
        model.tune_weights(X_val, y_val, target_metric="f0_5", random_state=random_seed)


def _json_safe_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(item) for item in value]
    if isinstance(value, dict):
        return {str(k): _json_safe_value(v) for k, v in value.items()}
    if hasattr(value, "get_params"):
        return {
            "class": type(value).__name__,
            "params": _json_safe_value(value.get_params(deep=False)),
        }
    return repr(value)


def get_model_hyperparams(model: Any) -> Dict[str, Any]:
    params = _json_safe_value(model.get_params(deep=False))
    if is_weight_tuned_ensemble(model) and hasattr(model, "weights_"):
        params["tuned_weights"] = model.weights_.tolist()
    return params


def train_and_evaluate_models(
    split_data: Dict[str, Any],
    random_seed: int = 42,
    tuning_profile: str = "fast",
) -> Dict[str, Any]:
    """
    Train and compare base learners plus ensemble compositions; select by validation F0.5 composite score.
    """
    X_train, y_train = split_data["X_train"], split_data["y_train"]
    X_val, y_val = split_data["X_val"], split_data["y_val"]
    X_test, y_test = split_data["X_test"], split_data["y_test"]
    entity_ids_val = split_data["entity_ids_val"]
    entity_ids_test = split_data["entity_ids_test"]

    candidate_models = build_candidate_models(random_seed, profile=tuning_profile)

    model_results = {}
    best_model_name = None
    best_val_score = -1.0
    best_model_obj = None

    for name, model in candidate_models.items():
        _fit_model(model, X_train, y_train, X_val, y_val, random_seed)

        val_probs = model.predict_proba(X_val)[:, 1]
        val_preds_default = (val_probs >= 0.5).astype(int)
        val_metrics_default = evaluate_predictions(y_val, val_preds_default, val_probs)

        opt_thresh, opt_metrics = find_optimal_entity_threshold(
            entity_ids_val, y_val, val_probs, fine_refine=True
        )
        selection_score = model_selection_score(opt_metrics, y_val, val_probs)

        model_results[name] = {
            "model_object": model,
            "val_default_metrics": val_metrics_default,
            "optimal_threshold": opt_thresh,
            "val_optimal_metrics": opt_metrics,
            "selection_score": selection_score,
        }

        if selection_score > best_val_score:
            best_val_score = selection_score
            best_model_name = name
            best_model_obj = model

    best_result = model_results[best_model_name]
    best_thresh = best_result["optimal_threshold"]

    test_probs = best_model_obj.predict_proba(X_test)[:, 1]
    test_preds = (test_probs >= best_thresh).astype(int)
    test_metrics = evaluate_predictions(y_test, test_preds, test_probs)
    test_metrics["entity_f0_5"] = entity_level_f0_5(
        entity_ids_test, y_test, test_probs, best_thresh
    )

    return {
        "all_model_results": model_results,
        "best_model_name": best_model_name,
        "best_model": best_model_obj,
        "optimal_threshold": best_thresh,
        "test_metrics": test_metrics,
        "test_probs": test_probs,
        "test_preds": test_preds,
    }


def save_model_artifacts(
    model_obj: Any,
    model_name: str,
    feature_names: List[str],
    hyperparams: Dict[str, Any],
    metrics: Dict[str, Any],
    decision_thresh: float,
    output_dir: str = "models",
    dataset_version: str = "1.0",
    random_seed: int = 42
) -> Tuple[str, str, str]:
    """
    Save trained model, feature configuration, and model metadata to disk.
    """
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(output_dir, "entity_resolution_model.joblib")
    config_path = os.path.join(output_dir, "feature_config.json")
    meta_path = os.path.join(output_dir, "model_metadata.json")

    joblib.dump(model_obj, model_path)

    feature_config = {
        "feature_names": feature_names,
        "num_features": len(feature_names),
        "description": "Comparison features computed on business_name, business_address, and country"
    }

    with open(config_path, "w") as f:
        json.dump(feature_config, f, indent=2)

    import sklearn
    import pandas
    import numpy

    metadata = {
        "model_type": model_name,
        "feature_names": feature_names,
        "training_date": datetime.now().isoformat(),
        "dataset_version": dataset_version,
        "random_seed": random_seed,
        "hyperparameters": hyperparams,
        "metrics": metrics,
        "decision_threshold": decision_thresh,
        "software_versions": {
            "scikit-learn": sklearn.__version__,
            "pandas": pandas.__version__,
            "numpy": numpy.__version__,
            "joblib": joblib.__version__,
        }
    }

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return model_path, config_path, meta_path
