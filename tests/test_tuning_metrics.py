import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "code")))

from business_entity_resolution.src.evaluation.metrics import (
    find_optimal_threshold,
    pairwise_f_beta,
    model_selection_score,
)
from business_entity_resolution.src.models.tuning import optimize_soft_weights


def test_pairwise_f_beta_prefers_precision():
    high_prec = pairwise_f_beta(0.99, 0.80, beta=0.5)
    low_prec = pairwise_f_beta(0.70, 0.99, beta=0.5)
    assert high_prec > low_prec


def test_find_optimal_threshold_f0_5_refinement():
    rng = np.random.RandomState(1)
    y = np.array([0, 0, 0, 1, 1, 1, 1, 1])
    scores = rng.rand(len(y))
    scores[y == 1] += 0.45
    thresh, metrics = find_optimal_threshold(y, scores, target_metric="f0_5", fine_refine=True)
    assert 0.0 < thresh < 1.0
    assert metrics["f0_5"] >= metrics["f1_score"] * 0.5


def test_optimize_soft_weights_sums_to_one():
    rng = np.random.RandomState(2)
    y = rng.randint(0, 2, size=60)
    p1 = rng.rand(60)
    p2 = np.clip(p1 + rng.normal(0, 0.05, size=60), 0, 1)
    cols = np.column_stack([p1, p2])
    weights = optimize_soft_weights(cols, y, target_metric="f0_5", n_random_restarts=4)
    assert weights.shape == (2,)
    assert np.isclose(weights.sum(), 1.0)
    assert (weights >= 0).all()


def test_model_selection_score_in_valid_range():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    metrics = {"f0_5": 0.9, "f1_score": 0.88}
    score = model_selection_score(metrics, y, p)
    assert 0.0 <= score <= 1.0
