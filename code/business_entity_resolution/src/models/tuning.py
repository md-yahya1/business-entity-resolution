"""
Hyperparameter-free tuning helpers (thresholds, ensemble weights).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from ..evaluation.metrics import find_optimal_threshold


def _weight_objective(
    weights: np.ndarray,
    proba_columns: np.ndarray,
    y_true: np.ndarray,
    target_metric: str,
) -> float:
    w = np.clip(weights, 0.0, None)
    total = w.sum()
    if total <= 1e-12:
        return 1.0
    w = w / total
    blended = proba_columns @ w
    _, metrics = find_optimal_threshold(
        y_true, blended, target_metric=target_metric, fine_refine=False
    )
    if target_metric == "composite":
        return -metrics["objective"]
    key = "f0_5" if target_metric == "f0_5" else "f1_score"
    return -metrics[key]


def optimize_soft_weights(
    proba_columns: np.ndarray,
    y_true: np.ndarray,
    *,
    target_metric: str = "f0_5",
    n_random_restarts: int = 10,
    random_state: int = 42,
) -> np.ndarray:
    """
    Find non-negative ensemble weights (sum to 1) maximizing validation metric.
    Combines Dirichlet random restarts with sparse and uniform seeds.
    """
    n_models = proba_columns.shape[1]
    if n_models == 1:
        return np.array([1.0], dtype=float)

    rng = np.random.RandomState(random_state)
    seeds = [np.full(n_models, 1.0 / n_models)]

    for idx in range(n_models):
        corner = np.zeros(n_models, dtype=float)
        corner[idx] = 1.0
        seeds.append(corner)

    for _ in range(n_random_restarts):
        seeds.append(rng.dirichlet(np.ones(n_models)))

    best_weights = seeds[0]
    best_objective = np.inf

    constraints = {"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}
    bounds = [(0.0, 1.0)] * n_models

    for seed in seeds:
        result = minimize(
            _weight_objective,
            seed,
            args=(proba_columns, y_true, target_metric),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 120, "ftol": 1e-8},
        )
        candidate = result.x if result.success else seed
        objective = _weight_objective(candidate, proba_columns, y_true, target_metric)
        if objective < best_objective:
            best_objective = objective
            best_weights = candidate

    best_weights = np.clip(best_weights, 0.0, None)
    return best_weights / best_weights.sum()


def tune_soft_voting_weights(
    val_proba_columns: np.ndarray,
    y_val: np.ndarray,
    *,
    target_metric: str = "f0_5",
    random_state: int = 42,
) -> np.ndarray:
    """Alias for ensemble weight search used by voting wrappers."""
    return optimize_soft_weights(
        val_proba_columns,
        y_val,
        target_metric=target_metric,
        random_state=random_state,
    )
