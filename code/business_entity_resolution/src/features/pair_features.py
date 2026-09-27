"""
Feature engineering module for candidate pair comparison in Business Entity Resolution.
"""
from typing import Dict, List, Sequence
import pandas as pd
import numpy as np
import rapidfuzz.fuzz as fuzz
import rapidfuzz.distance.JaroWinkler as jw
import rapidfuzz.distance.Levenshtein as lev
from ..preprocessing import (
    normalize_business_name,
    normalize_address,
    normalize_country,
    extract_address_hints,
)


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
    # Extended similarity / geography features (retrain model after adding)
    "name_levenshtein_sim",
    "address_token_sort_ratio",
    "address_levenshtein_sim",
    "name_bag_cosine_sim",
    "address_bag_cosine_sim",
    "city_exact_match",
    "postal_exact_match",
    "house_number_exact_match",
    "state_exact_match",
]


def _jaccard_overlap(s1: str, s2: str) -> float:
    tokens1 = set(s1.split())
    tokens2 = set(s2.split())
    if not tokens1 or not tokens2:
        return 0.0
    intersection = tokens1.intersection(tokens2)
    union = tokens1.union(tokens2)
    return len(intersection) / len(union)


def _bag_cosine(s1: str, s2: str) -> float:
    tokens1 = s1.split()
    tokens2 = s2.split()
    if not tokens1 or not tokens2:
        return 0.0
    counts1: Dict[str, int] = {}
    counts2: Dict[str, int] = {}
    for t in tokens1:
        counts1[t] = counts1.get(t, 0) + 1
    for t in tokens2:
        counts2[t] = counts2.get(t, 0) + 1
    shared = set(counts1) | set(counts2)
    dot = sum(counts1.get(t, 0) * counts2.get(t, 0) for t in shared)
    norm1 = sum(v * v for v in counts1.values()) ** 0.5
    norm2 = sum(v * v for v in counts2.values()) ** 0.5
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return float(dot / (norm1 * norm2))


def _exact_hint_match(h1: str, h2: str) -> float:
    return 1.0 if h1 and h2 and h1 == h2 else 0.0


def _compute_feature_vector(
    n1: str,
    a1: str,
    c1: str,
    n2: str,
    a2: str,
    c2: str,
) -> np.ndarray:
    out = np.empty(len(FEATURE_NAMES), dtype=np.float32)

    name_miss = 1.0 if not n1 or not n2 else 0.0
    addr_miss = 1.0 if not a1 or not a2 else 0.0
    country_miss = 1.0 if not c1 or not c2 else 0.0

    hints1 = extract_address_hints(a1)
    hints2 = extract_address_hints(a2)

    out[0] = fuzz.ratio(n1, n2) / 100.0
    out[1] = fuzz.partial_ratio(n1, n2) / 100.0
    out[2] = fuzz.token_sort_ratio(n1, n2) / 100.0
    out[3] = fuzz.token_set_ratio(n1, n2) / 100.0
    out[4] = float(jw.similarity(n1, n2))
    out[5] = fuzz.ratio(a1, a2) / 100.0
    out[6] = fuzz.partial_ratio(a1, a2) / 100.0
    out[7] = fuzz.token_set_ratio(a1, a2) / 100.0
    out[8] = float(jw.similarity(a1, a2))
    out[9] = 1.0 if c1 and c2 and c1 == c2 else 0.0
    out[10] = country_miss
    out[11] = 1.0 if n1 and n2 and n1 == n2 else 0.0
    out[12] = 1.0 if a1 and a2 and a1 == a2 else 0.0
    out[13] = _jaccard_overlap(n1, n2)
    out[14] = _jaccard_overlap(a1, a2)
    out[15] = float(abs(len(n1) - len(n2)))
    out[16] = float(abs(len(a1) - len(a2)))
    out[17] = name_miss
    out[18] = addr_miss
    out[19] = float(lev.normalized_similarity(n1, n2))
    out[20] = fuzz.token_sort_ratio(a1, a2) / 100.0
    out[21] = float(lev.normalized_similarity(a1, a2))
    out[22] = _bag_cosine(n1, n2)
    out[23] = _bag_cosine(a1, a2)
    out[24] = _exact_hint_match(hints1["city"], hints2["city"])
    out[25] = _exact_hint_match(hints1["postal_code"], hints2["postal_code"])
    out[26] = _exact_hint_match(hints1["house_number"], hints2["house_number"])
    out[27] = _exact_hint_match(hints1["state"], hints2["state"])

    return out


def compute_pair_features(
    name1: str,
    addr1: str,
    country1: str,
    name2: str,
    addr2: str,
    country2: str,
) -> Dict[str, float]:
    n1 = normalize_business_name(name1)
    n2 = normalize_business_name(name2)
    a1 = normalize_address(addr1)
    a2 = normalize_address(addr2)
    c1 = normalize_country(country1)
    c2 = normalize_country(country2)

    vec = _compute_feature_vector(n1, a1, c1, n2, a2, c2)
    return {name: float(vec[i]) for i, name in enumerate(FEATURE_NAMES)}


def compute_features_batch_fast(
    n1_col: Sequence[str],
    a1_col: Sequence[str],
    c1_col: Sequence[str],
    n2_col: Sequence[str],
    a2_col: Sequence[str],
    c2_col: Sequence[str],
    are_pre_normalized: bool = False,
) -> np.ndarray:
    """Returns np.ndarray of shape (n_samples, len(FEATURE_NAMES)) in float32."""
    n_samples = len(n1_col)
    out = np.empty((n_samples, len(FEATURE_NAMES)), dtype=np.float32)

    for i in range(n_samples):
        n1 = n1_col[i]
        a1 = a1_col[i]
        c1 = c1_col[i]
        n2 = n2_col[i]
        a2 = a2_col[i]
        c2 = c2_col[i]

        if not are_pre_normalized:
            n1 = normalize_business_name(n1)
            n2 = normalize_business_name(n2)
            a1 = normalize_address(a1)
            a2 = normalize_address(a2)
            c1 = normalize_country(c1)
            c2 = normalize_country(c2)

        out[i] = _compute_feature_vector(n1, a1, c1, n2, a2, c2)

    return out


def align_feature_matrix(feat_matrix: np.ndarray, model_feature_names: List[str]) -> np.ndarray:
    """Select model-expected columns (supports legacy 19-feature models)."""
    if feat_matrix.shape[1] == len(model_feature_names):
        return feat_matrix
    indices = [FEATURE_NAMES.index(name) for name in model_feature_names]
    return feat_matrix[:, indices]


def extract_features_dataframe(df: pd.DataFrame, batch_size: int = 50000) -> pd.DataFrame:
    n1_col = df["business_name_1"].fillna("").astype(str).tolist() if "business_name_1" in df.columns else [""] * len(df)
    a1_col = df["business_address_1"].fillna("").astype(str).tolist() if "business_address_1" in df.columns else [""] * len(df)
    c1_col = df["country_1"].fillna("").astype(str).tolist() if "country_1" in df.columns else [""] * len(df)
    n2_col = df["business_name_2"].fillna("").astype(str).tolist() if "business_name_2" in df.columns else [""] * len(df)
    a2_col = df["business_address_2"].fillna("").astype(str).tolist() if "business_address_2" in df.columns else [""] * len(df)
    c2_col = df["country_2"].fillna("").astype(str).tolist() if "country_2" in df.columns else [""] * len(df)

    feat_matrix = compute_features_batch_fast(n1_col, a1_col, c1_col, n2_col, a2_col, c2_col, are_pre_normalized=False)
    return pd.DataFrame(feat_matrix, columns=FEATURE_NAMES)
