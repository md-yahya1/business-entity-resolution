"""
Feature engineering module for candidate pair comparison in Business Entity Resolution.
"""
from typing import Dict, Any, List
import pandas as pd
import numpy as np
import rapidfuzz.fuzz as fuzz
import rapidfuzz.distance.JaroWinkler as jw
from ..preprocessing import normalize_text, normalize_business_name, normalize_address, normalize_country


FEATURE_NAMES = [
    "name_ratio",
    "name_partial_ratio",
    "name_token_sort_ratio",
    "name_token_set_ratio",
    "name_jw_similarity",
    "address_ratio",
    "address_partial_ratio",
    "address_token_set_ratio",
    "address_jw_similarity",
    "country_exact_match",
    "country_missing",
    "name_exact_match",
    "address_exact_match",
    "name_token_overlap",
    "address_token_overlap",
    "name_char_len_diff",
    "address_char_len_diff",
    "name_missing",
    "address_missing",
]


def _jaccard_overlap(s1: str, s2: str) -> float:
    """Calculate token Jaccard similarity between two normalized strings."""
    tokens1 = set(s1.split())
    tokens2 = set(s2.split())
    if not tokens1 or not tokens2:
        return 0.0
    intersection = tokens1.intersection(tokens2)
    union = tokens1.union(tokens2)
    return len(intersection) / len(union)


def compute_pair_features(
    name1: str,
    addr1: str,
    country1: str,
    name2: str,
    addr2: str,
    country2: str
) -> Dict[str, float]:
    """
    Compute similarity and distance features for a single pair of entity records.
    """
    n1 = normalize_business_name(name1)
    n2 = normalize_business_name(name2)
    a1 = normalize_address(addr1)
    a2 = normalize_address(addr2)
    c1 = normalize_country(country1)
    c2 = normalize_country(country2)

    name_missing = 1.0 if not n1 or not n2 else 0.0
    addr_missing = 1.0 if not a1 or not a2 else 0.0
    country_missing = 1.0 if not c1 or not c2 else 0.0

    return {
        "name_ratio": fuzz.ratio(n1, n2) / 100.0,
        "name_partial_ratio": fuzz.partial_ratio(n1, n2) / 100.0,
        "name_token_sort_ratio": fuzz.token_sort_ratio(n1, n2) / 100.0,
        "name_token_set_ratio": fuzz.token_set_ratio(n1, n2) / 100.0,
        "name_jw_similarity": float(jw.similarity(n1, n2)),
        "address_ratio": fuzz.ratio(a1, a2) / 100.0,
        "address_partial_ratio": fuzz.partial_ratio(a1, a2) / 100.0,
        "address_token_set_ratio": fuzz.token_set_ratio(a1, a2) / 100.0,
        "address_jw_similarity": float(jw.similarity(a1, a2)),
        "country_exact_match": 1.0 if c1 and c2 and c1 == c2 else 0.0,
        "country_missing": country_missing,
        "name_exact_match": 1.0 if n1 and n2 and n1 == n2 else 0.0,
        "address_exact_match": 1.0 if a1 and a2 and a1 == a2 else 0.0,
        "name_token_overlap": _jaccard_overlap(n1, n2),
        "address_token_overlap": _jaccard_overlap(a1, a2),
        "name_char_len_diff": float(abs(len(n1) - len(n2))),
        "address_char_len_diff": float(abs(len(a1) - len(a2))),
        "name_missing": name_missing,
        "address_missing": addr_missing,
    }


def extract_features_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract features for a DataFrame containing pair entity attributes.
    Expected columns:
    ['business_name_1', 'business_address_1', 'country_1',
     'business_name_2', 'business_address_2', 'country_2']
    """
    features_list = []
    for row in df.itertuples(index=False):
        f = compute_pair_features(
            getattr(row, 'business_name_1', ''),
            getattr(row, 'business_address_1', ''),
            getattr(row, 'country_1', ''),
            getattr(row, 'business_name_2', ''),
            getattr(row, 'business_address_2', ''),
            getattr(row, 'country_2', '')
        )
        features_list.append(f)
    return pd.DataFrame(features_list)[FEATURE_NAMES]
