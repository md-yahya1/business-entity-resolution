"""
Evaluation metrics module for Entity Resolution model training.
"""
from typing import Dict, Any, Tuple
import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
)


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray = None
) -> Dict[str, Any]:
    """
    Compute comprehensive metrics for entity resolution matching model.
    """
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    acc = float(accuracy_score(y_true, y_pred))

    cm = confusion_matrix(y_true, y_pred).tolist()

    metrics = {
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "accuracy": acc,
        "confusion_matrix": cm,
    }

    if y_prob is not None and len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
    else:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None

    return metrics


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_metric: str = "f1"
) -> Tuple[float, Dict[str, float]]:
    """
    Find optimal decision threshold on validation set to maximize F1-score or target metric.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)

    best_thresh = 0.5
    best_score = -1.0
    best_metrics = {}

    for thresh in np.linspace(0.05, 0.95, 91):
        preds = (y_prob >= thresh).astype(int)
        score_f1 = f1_score(y_true, preds, zero_division=0)
        score_p = precision_score(y_true, preds, zero_division=0)
        score_r = recall_score(y_true, preds, zero_division=0)

        current_score = score_f1 if target_metric == "f1" else score_p
        if current_score > best_score:
            best_score = current_score
            best_thresh = float(thresh)
            best_metrics = {
                "threshold": float(thresh),
                "precision": float(score_p),
                "recall": float(score_r),
                "f1_score": float(score_f1),
            }

    return best_thresh, best_metrics
