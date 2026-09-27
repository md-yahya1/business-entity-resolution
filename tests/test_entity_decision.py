import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "code")))

from business_entity_resolution.src.evaluation.entity_decision import select_entity_matches


def test_singleton_no_match_when_best_prob_low():
    ids = ["S2-a", "S2-b"]
    probs = [0.25, 0.20]
    out = select_entity_matches(ids, probs, match_threshold=0.2, no_match_max_prob=0.42)
    assert out == []


def test_margin_drops_ambiguous_single_match():
    ids = ["S2-a", "S2-b"]
    probs = [0.72, 0.69]
    out = select_entity_matches(
        ids,
        probs,
        match_threshold=0.65,
        no_match_max_prob=0.40,
        min_single_match_margin=0.06,
        min_confident_single_match=0.88,
    )
    assert out == []


def test_keeps_multiple_matches_above_threshold():
    ids = ["S2-a", "S2-b", "S2-c"]
    probs = [0.91, 0.87, 0.40]
    out = select_entity_matches(ids, probs, match_threshold=0.85, no_match_max_prob=0.42)
    assert set(out) == {"S2-a", "S2-b"}
