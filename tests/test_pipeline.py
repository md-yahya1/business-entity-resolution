import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "code")))

from business_entity_resolution.src.features import FEATURE_NAMES
from business_entity_resolution.src.models import (
    split_data_by_group,
    train_and_evaluate_models,
    save_model_artifacts,
    get_model_hyperparams,
    build_candidate_models,
)


def test_end_to_end_training_and_saving(tmp_path):
    np.random.seed(42)
    n_samples = 100
    df = pd.DataFrame({
        "entity_group_id": np.repeat(np.arange(20), 5),
        "label": np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3]),
    })

    # Generate synthetic feature dataframe
    feature_data = {feat: np.random.rand(n_samples) for feat in FEATURE_NAMES}
    features_df = pd.DataFrame(feature_data)

    split_data = split_data_by_group(df, features_df, test_size=0.2, val_size=0.2, random_state=42)

    assert len(split_data["X_train"]) > 0
    assert len(split_data["X_val"]) > 0
    assert len(split_data["X_test"]) > 0

    eval_results = train_and_evaluate_models(
        split_data, random_seed=42, tuning_profile="minimal"
    )

    assert eval_results["best_model"] is not None
    assert 0.0 <= eval_results["optimal_threshold"] <= 1.0
    assert "precision" in eval_results["test_metrics"]

    # Test saving artifacts
    out_dir = str(tmp_path / "models")
    m_path, c_path, meta_path = save_model_artifacts(
        model_obj=eval_results["best_model"],
        model_name=eval_results["best_model_name"],
        feature_names=FEATURE_NAMES,
        hyperparams=get_model_hyperparams(eval_results["best_model"]),
        metrics=eval_results["test_metrics"],
        decision_thresh=eval_results["optimal_threshold"],
        output_dir=out_dir
    )

    assert os.path.exists(m_path)
    assert os.path.exists(c_path)
    assert os.path.exists(meta_path)
