"""
Entity-level match selection for macro F0.5 (precision-heavy).

Pairwise thresholding alone often over-predicts on singletons and ambiguous
top-1 candidates. These rules gate predictions per Source-1 entity.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple


def select_entity_matches(
    candidate_ids: Sequence[str],
    probabilities: Sequence[float],
    *,
    match_threshold: float,
    no_match_max_prob: float = 0.42,
    min_single_match_margin: float = 0.06,
    min_confident_single_match: float = 0.88,
) -> List[str]:
    """
    Choose matched candidate IDs for one Source-1 entity.

    - If the best score is below ``no_match_max_prob``, return no matches
      (confident singleton / no-link).
    - Otherwise keep candidates with prob >= ``match_threshold``.
    - If exactly one match survives and a close runner-up exists, drop the
      lone match unless it is high-confidence or clearly separated by margin.
    """
    if not candidate_ids or not probabilities:
        return []

    pairs: List[Tuple[float, str]] = sorted(
        zip(probabilities, candidate_ids),
        key=lambda x: (-x[0], x[1]),
    )
    best_prob = pairs[0][0]

    if best_prob < no_match_max_prob:
        return []

    selected = [cid for prob, cid in pairs if prob >= match_threshold]
    if not selected:
        return []

    if len(selected) == 1 and len(pairs) > 1:
        second_prob = pairs[1][0]
        if (
            best_prob - second_prob < min_single_match_margin
            and best_prob < min_confident_single_match
        ):
            return []

    return list(dict.fromkeys(selected))


def tune_entity_decision_on_validation(
    pairs_by_entity: dict,
    true_by_entity: dict,
    entity_ids: Sequence[str],
    match_thresholds=None,
    no_match_values=None,
    margin_values=None,
):
    """
    Grid search entity decision knobs using precomputed (candidate_id, prob) lists.
    Returns best settings and macro F0.5.
    """
    from .entity_metrics import macro_f0_5

    if match_thresholds is None:
        match_thresholds = [round(0.05 * i, 2) for i in range(8, 20)]
    if no_match_values is None:
        no_match_values = [0.30, 0.35, 0.40, 0.45, 0.50]
    if margin_values is None:
        margin_values = [0.0, 0.04, 0.06, 0.08, 0.10]

    entity_ids = list(entity_ids)
    best = (-1.0, None)
    history = []

    for t in match_thresholds:
        for nm in no_match_values:
            for mg in margin_values:
                pred_by_entity = {}
                for eid in entity_ids:
                    pairs = pairs_by_entity.get(eid, [])
                    cids = [p[0] for p in pairs]
                    probs = [p[1] for p in pairs]
                    pred_by_entity[eid] = set(
                        select_entity_matches(
                            cids,
                            probs,
                            match_threshold=t,
                            no_match_max_prob=nm,
                            min_single_match_margin=mg,
                        )
                    )
                score, _ = macro_f0_5(true_by_entity, pred_by_entity, entity_ids)
                history.append(
                    {
                        "match_threshold": t,
                        "no_match_max_prob": nm,
                        "min_single_match_margin": mg,
                        "macro_f0_5": score,
                    }
                )
                if score > best[0]:
                    best = (score, (t, nm, mg))

    best_score, best_params = best
    return best_score, best_params, history
