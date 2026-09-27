"""Minimal 13-feature inference/training feature set."""
from __future__ import annotations

import numpy as np
from rapidfuzz import fuzz, process

FEATURE_NAMES_FINAL = [
    "name_ratio",
    "name_token_set_ratio",
    "address_ratio",
    "address_token_set_ratio",
    "country_exact_match",
    "name_exact_match",
    "address_exact_match",
    "city_exact_match",
    "postal_exact_match",
    "house_number_exact_match",
    "state_exact_match",
    "name_length_difference",
    "address_length_difference",
]

def compute_final_features(
    n1, a1, c1, p1, h1, city1, state1,
    n2, a2, c2, p2, h2, city2, state2,
) -> np.ndarray:
    n = len(n1)
    out = np.empty((n, len(FEATURE_NAMES_FINAL)), dtype=np.float32)

    # RapidFuzz C-API computes the four expensive similarities in bulk.
    out[:, 0] = process.cpdist(n1, n2, scorer=fuzz.ratio, workers=-1, dtype=np.float32) / 100.0
    out[:, 1] = process.cpdist(n1, n2, scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32) / 100.0
    out[:, 2] = process.cpdist(a1, a2, scorer=fuzz.ratio, workers=-1, dtype=np.float32) / 100.0
    out[:, 3] = process.cpdist(a1, a2, scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32) / 100.0

    out[:, 4] = np.asarray(c1) == np.asarray(c2)
    out[:, 5] = np.asarray(n1) == np.asarray(n2)
    out[:, 6] = np.asarray(a1) == np.asarray(a2)
    out[:, 7] = np.asarray(city1) == np.asarray(city2)
    out[:, 8] = np.asarray(p1) == np.asarray(p2)
    out[:, 9] = np.asarray(h1) == np.asarray(h2)
    out[:, 10] = np.asarray(state1) == np.asarray(state2)
    out[:, 11] = np.abs(
        np.fromiter((len(x) for x in n1), dtype=np.int16, count=n) -
        np.fromiter((len(x) for x in n2), dtype=np.int16, count=n)
    )
    out[:, 12] = np.abs(
        np.fromiter((len(x) for x in a1), dtype=np.int32, count=n) -
        np.fromiter((len(x) for x in a2), dtype=np.int32, count=n)
    )
    return out
