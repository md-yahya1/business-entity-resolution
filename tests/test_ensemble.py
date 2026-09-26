import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "code")))

from sklearn.tree import DecisionTreeClassifier

from business_entity_resolution.src.models.ensemble import (
    build_candidate_models,
    get_base_learner_specs,
    WeightedVotingEnsemble,
)


def test_base_learner_catalog_is_stable():
    names = [spec.name for spec in get_base_learner_specs()]
    assert names == ["hist", "extra", "rf", "gb", "lr", "ada"]


def test_candidate_models_include_ensembles():
    fast_models = build_candidate_models(42, profile="fast")
    assert "SoftVotingEnsemble" in fast_models
    assert "WeightedVotingEnsemble" in fast_models
    assert "StackingEnsemble_HGB" in fast_models
    assert "StackingEnsemble_LR" not in fast_models

    full_models = build_candidate_models(42, profile="full")
    assert "StackingEnsemble_LR" in full_models
    assert "WeightedExtendedVotingEnsemble" in full_models
    assert "HistGradientBoosting" in fast_models


def test_weighted_voting_tunes_on_validation():
    rng = np.random.RandomState(0)
    X_train = rng.rand(80, 5)
    y_train = (X_train[:, 0] + X_train[:, 1] > 1.0).astype(int)
    X_val = rng.rand(30, 5)
    y_val = (X_val[:, 0] + X_val[:, 1] > 1.0).astype(int)

    base = [
        ("a", DecisionTreeClassifier(max_depth=3, random_state=0)),
        ("b", DecisionTreeClassifier(max_depth=3, random_state=1)),
    ]
    model = WeightedVotingEnsemble(estimators=base)
    model.fit(X_train, y_train)
    weights = model.tune_weights(X_val, y_val, n_random_restarts=2)

    assert weights.shape == (len(base),)
    assert np.isclose(weights.sum(), 1.0)
    assert (weights >= 0).all()
    probs = model.predict_proba(X_val)
    assert probs.shape == (len(X_val), 2)
