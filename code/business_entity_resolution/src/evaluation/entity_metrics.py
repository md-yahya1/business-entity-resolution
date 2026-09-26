"""
Entity-level evaluation for Business Entity Resolution.

IMPORTANT: this is NOT the same thing as evaluation/metrics.py.

metrics.py scores the model on the *pairwise* classification task (is this
one candidate pair a match or not?) using plain precision/recall/F1.

This module scores it the way the actual leaderboard does: per Source-1
entity, comparing the SET of predicted matches against the SET of true
matches, using F_0.5, then macro-averaging across every Source-1 entity
(singletons included). A model can have great pairwise F1 and still score
badly here if its errors are concentrated on a few entities, or vice versa
-- so this is the number to optimize the decision threshold against, and
the number to report as your real expected leaderboard score.

Formula (from the challenge spec):
    F_0.5 = (1.25 * P * R) / (0.25 * P + R)
    - Singleton with correctly predicted empty match list -> 1.0
    - Singleton with any predicted match (false merge)    -> 0.0
    - Entity with true matches but empty prediction        -> 0.0 (recall=0)
"""
from typing import Dict, Set, Iterable, Tuple
import pandas as pd


def entity_f0_5(true_matches: Set[str], pred_matches: Set[str]) -> float:
    """F_0.5 for a single Source-1 entity given its true and predicted match sets."""
    if not true_matches and not pred_matches:
        return 1.0
    if not true_matches or not pred_matches:
        # either a false merge on a singleton, or a missed entity entirely
        return 0.0

    tp = len(true_matches & pred_matches)
    precision = tp / len(pred_matches)
    recall = tp / len(true_matches)

    denom = 0.25 * precision + recall
    if denom == 0:
        return 0.0
    return (1.25 * precision * recall) / denom


def macro_f0_5(
    true_by_entity: Dict[str, Set[str]],
    pred_by_entity: Dict[str, Set[str]],
    entity_ids: Iterable[str] = None,
) -> Tuple[float, pd.DataFrame]:
    """
    Macro-averaged F_0.5 across Source-1 entities.

    true_by_entity / pred_by_entity: source1_entity_id -> set of matched ids.
    An entity missing from either dict is treated as having an empty set
    (this is what happens for correctly-predicted singletons).
    entity_ids: the full universe of Source-1 ids to score. If None, uses
    the union of keys from both dicts -- but you should almost always pass
    the actual list of Source-1 ids in your eval split, since a Source-1
    entity absent from your predictions dict must still be scored as an
    (incorrect) empty prediction.

    Returns (macro_score, per_entity_breakdown_df) -- the breakdown is
    handy for error analysis (sort by f0_5 to find your worst offenders).
    """
    if entity_ids is None:
        entity_ids = set(true_by_entity) | set(pred_by_entity)

    rows = []
    for eid in entity_ids:
        true_set = true_by_entity.get(eid, set())
        pred_set = pred_by_entity.get(eid, set())
        score = entity_f0_5(true_set, pred_set)
        rows.append({
            "source1_entity_id": eid,
            "n_true": len(true_set),
            "n_pred": len(pred_set),
            "n_correct": len(true_set & pred_set),
            "f0_5": score,
        })

    breakdown = pd.DataFrame(rows)
    macro_score = breakdown["f0_5"].mean() if len(breakdown) else 0.0
    return float(macro_score), breakdown


def ground_truth_to_dict(gt_df: pd.DataFrame) -> Dict[str, Set[str]]:
    """Parse a train_ground_truth.tsv-shaped dataframe into source1_id -> set(matched_ids)."""
    result = {}
    for row in gt_df.itertuples(index=False):
        matches = row.matched_entity_ids
        if pd.isna(matches) or not str(matches).strip():
            result[row.source1_entity_id] = set()
        else:
            result[row.source1_entity_id] = {m.strip() for m in str(matches).split(",") if m.strip()}
    return result


def pair_predictions_to_dict(
    pred_pairs_df: pd.DataFrame,
    entity_id_1_col: str = "entity_id_1",
    entity_id_2_col: str = "entity_id_2",
    is_match_col: str = "is_match",
) -> Dict[str, Set[str]]:
    """
    Collapse pairwise predictions (one row per S1-candidate pair, with a
    binary is_match column) into source1_id -> set(matched_ids), keeping
    only the S2/S3 side of each pair regardless of which column it landed in.
    """
    result: Dict[str, Set[str]] = {}
    matched = pred_pairs_df[pred_pairs_df[is_match_col] == 1]
    for row in matched.itertuples(index=False):
        e1 = getattr(row, entity_id_1_col)
        e2 = getattr(row, entity_id_2_col)
        s1_id, other_id = (e1, e2) if str(e1).startswith("S1-") else (e2, e1)
        result.setdefault(s1_id, set()).add(other_id)
    return result


def tune_threshold_for_macro_f0_5(
    pred_pairs_df: pd.DataFrame,
    true_by_entity: Dict[str, Set[str]],
    entity_ids: Iterable[str],
    prob_col: str = "match_probability",
    thresholds=None,
):
    """
    Sweep decision thresholds and pick the one maximizing macro-averaged
    entity-level F_0.5 -- NOT pairwise F1. Use this instead of (or in
    addition to) evaluation/metrics.py's find_optimal_threshold, since the
    two objectives can disagree, especially near the precision/recall
    trade-off F_0.5 cares about.

    pred_pairs_df must have entity_id_1, entity_id_2, and prob_col columns
    for every candidate pair in your validation split.
    """
    if thresholds is None:
        thresholds = [round(0.05 * i, 2) for i in range(1, 20)]

    entity_ids = list(entity_ids)
    best_threshold, best_score, best_breakdown = None, -1.0, None
    history = []

    for t in thresholds:
        df = pred_pairs_df.copy()
        df["is_match"] = (df[prob_col] >= t).astype(int)
        pred_by_entity = pair_predictions_to_dict(df)
        score, breakdown = macro_f0_5(true_by_entity, pred_by_entity, entity_ids)
        history.append({"threshold": t, "macro_f0_5": score})
        if score > best_score:
            best_threshold, best_score, best_breakdown = t, score, breakdown

    return best_threshold, best_score, pd.DataFrame(history), best_breakdown
