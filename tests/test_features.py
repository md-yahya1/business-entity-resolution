import pytest
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "code")))

from business_entity_resolution.src.features import (
    compute_pair_features,
    extract_features_dataframe,
    FEATURE_NAMES,
)


def test_compute_pair_features_exact_match():
    feats = compute_pair_features(
        name1="Acme Corporation",
        addr1="123 Main Street",
        country1="US",
        name2="Acme Corporation",
        addr2="123 Main Street",
        country2="US"
    )
    assert feats["name_exact_match"] == 1.0
    assert feats["address_exact_match"] == 1.0
    assert feats["country_exact_match"] == 1.0
    assert feats["name_ratio"] == 1.0
    assert feats["name_missing"] == 0.0


def test_compute_pair_features_different_records():
    feats = compute_pair_features(
        name1="Acme Corp",
        addr1="123 Main St",
        country1="US",
        name2="Beta LLC",
        addr2="456 Oak Ave",
        country2="India"
    )
    assert feats["name_exact_match"] == 0.0
    assert feats["country_exact_match"] == 0.0
    assert feats["name_ratio"] < 0.5


def test_extract_features_dataframe():
    df = pd.DataFrame([{
        "business_name_1": "Google LLC",
        "business_address_1": "1600 Amphitheatre Pkwy",
        "country_1": "US",
        "business_name_2": "Google Inc",
        "business_address_2": "1600 Amphitheatre Parkway",
        "country_2": "US",
    }])
    feats_df = extract_features_dataframe(df)
    assert list(feats_df.columns) == FEATURE_NAMES
    assert len(feats_df) == 1
    assert feats_df.loc[0, "country_exact_match"] == 1.0
