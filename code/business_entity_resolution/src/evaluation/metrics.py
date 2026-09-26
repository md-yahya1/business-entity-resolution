"""
Evaluation metrics module for Entity Resolution model training.
"""
from typing import Dict, Any, Tuple, Iterable
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


def pairwise_f_beta(precision: float, recall: float, beta: float = 0.5) -> float:
    """Pairwise F-beta with beta < 1 emphasizing precision (competition uses 0.5)."""
    beta_sq = beta * beta
    denom = beta_sq * precision + recall
    if denom == 0.0:
        return 0.0
    return (1.0 + beta_sq) * precision * recall / denom


def _threshold_candidates(y_true: np.ndarray, y_prob: np.ndarray, center: float = None) -> np.ndarray:
    coarse = np.linspace(0.02, 0.98, 193)
    candidates = set(float(t) for t in coarse)
    _, _, pr_thresholds = precision_recall_curve(y_true, y_prob)
    candidates.update(float(t) for t in pr_thresholds if 0.01 <= float(t) <= 0.99)

    if center is not None:
        fine = np.linspace(max(0.005, center - 0.08), min(0.995, center + 0.08), 161)
        candidates.update(float(t) for t in fine)

    return np.array(sorted(candidates), dtype=float)


def _score_at_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresh: float,
    target_metric: str,
) -> Tuple[float, Dict[str, float]]:
    preds = (y_prob >= thresh).astype(int)
    score_p = float(precision_score(y_true, preds, zero_division=0))
    score_r = float(recall_score(y_true, preds, zero_division=0))
    score_f1 = float(f1_score(y_true, preds, zero_division=0))
    score_f05 = float(pairwise_f_beta(score_p, score_r, beta=0.5))

    if target_metric == "f0_5":
        current_score = score_f05
    elif target_metric == "precision":
        current_score = score_p
    elif target_metric == "composite":
        current_score = 0.55 * score_f05 + 0.45 * score_f1
    else:
        current_score = score_f1

    metrics = {
        "threshold": float(thresh),
        "precision": score_p,
        "recall": score_r,
        "f1_score": score_f1,
        "f0_5": score_f05,
        "objective": float(current_score),
    }
    return current_score, metrics


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
        "f0_5": float(pairwise_f_beta(prec, rec, beta=0.5)),
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
    target_metric: str = "f0_5",
    fine_refine: bool = True,
) -> Tuple[float, Dict[str, float]]:
    """
    Find decision threshold maximizing pairwise F0.5, F1, precision, or a composite.
    Uses PR-curve thresholds plus a two-pass fine grid around the best point.
    """
    best_thresh = 0.5
    best_score = -1.0
    best_metrics: Dict[str, float] = {}

    for thresh in _threshold_candidates(y_true, y_prob):
        current_score, metrics = _score_at_threshold(y_true, y_prob, thresh, target_metric)
        if current_score > best_score:
            best_score = current_score
            best_thresh = float(thresh)
            best_metrics = metrics

    if fine_refine:
        for thresh in _threshold_candidates(y_true, y_prob, center=best_thresh):
            current_score, metrics = _score_at_threshold(y_true, y_prob, thresh, target_metric)
            if current_score > best_score:
                best_score = current_score
                best_thresh = float(thresh)
                best_metrics = metrics

    return best_thresh, best_metrics


def model_selection_score(
    val_threshold_metrics: Dict[str, float],
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> float:
    """
    Rank models on validation: prioritize F0.5, then F1 and ranking quality (ROC-AUC).
    """
    roc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.0
    f05 = val_threshold_metrics.get("f0_5", val_threshold_metrics["f1_score"])
    f1 = val_threshold_metrics["f1_score"]
    return 0.50 * f05 + 0.30 * f1 + 0.20 * roc


def entity_level_f0_5(
    entity_ids: np.ndarray,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> float:
    """Competition-style macro F0.5: compute F0.5 separately per S1 entity."""
    frame = {}
    for eid, truth, prob in zip(entity_ids, y_true, y_prob):
        key = str(eid)
        bucket = frame.setdefault(key, {"true": set(), "pred": set()})
        if int(truth) == 1:
            bucket["true"].add(str(eid))
        if float(prob) >= threshold:
            bucket["pred"].add(str(eid))

    # Pair rows only carry a binary label, so recover per-entity counts directly.
    # For each S1 entity: TP = predicted positive rows that are true, FP = predicted
    # positive rows that are false, FN = true positive rows below threshold.
    by_entity = {}
    for eid, truth, prob in zip(entity_ids, y_true, y_prob):
        key = str(eid)
        stats = by_entity.setdefault(key, [0, 0, 0])
        pred = float(prob) >= threshold
        truth = int(truth) == 1
        if pred and truth:
            stats[0] += 1
        elif pred and not truth:
            stats[1] += 1
        elif truth and not pred:
            stats[2] += 1

    scores = []
    for tp, fp, fn in by_entity.values():
        precision = tp / (tp + fp) if (tp + fp) else 1.0
        recall = tp / (tp + fn) if (tp + fn) else 1.0
        scores.append(pairwise_f_beta(precision, recall, beta=0.5))
    return float(np.mean(scores)) if scores else 0.0


def find_optimal_entity_threshold(
    entity_ids: np.ndarray,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    fine_refine: bool = True,
) -> Tuple[float, Dict[str, float]]:
    """Optimize the competition-style macro-average F0.5 over S1 entities."""
    candidates = _threshold_candidates(y_true, y_prob)
    best_thresh = 0.5
    best_score = -1.0
    best_metrics = {}

    for thresh in candidates:
        score = entity_level_f0_5(entity_ids, y_true, y_prob, float(thresh))
        if score > best_score:
            best_score = score
            best_thresh = float(thresh)

    if fine_refine:
        fine = np.linspace(max(0.005, best_thresh - 0.05), min(0.995, best_thresh + 0.05), 101)
        for thresh in fine:
            score = entity_level_f0_5(entity_ids, y_true, y_prob, float(thresh))
            if score > best_score:
                best_score = score
                best_thresh = float(thresh)

    preds = (y_prob >= best_thresh).astype(int)
    pair_metrics = evaluate_predictions(y_true, preds, y_prob)
    pair_metrics["entity_f0_5"] = float(best_score)
    pair_metrics["threshold"] = float(best_thresh)
    return best_thresh, pair_metrics
