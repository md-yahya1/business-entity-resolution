import pytest
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "code")))

from business_entity_resolution.src.preprocessing import (
    normalize_text,
    normalize_business_name,
    normalize_address,
    normalize_country,
    preprocess_dataframe,
)


def test_normalize_text_null_and_accents():
    assert normalize_text(None) == ""
    assert normalize_text(float("nan")) == ""
    assert normalize_text("  Cafe   Regence!  ") == "cafe regence"


def test_normalize_business_name():
    assert normalize_business_name("ABC Corp, Inc.") == "abc corp inc"
    assert normalize_business_name("  Google LLC  ") == "google llc"


def test_normalize_address():
    assert normalize_address("123 Main St, Suite 400;") == "123 main st suite 400"
    assert normalize_address(None) == ""


def test_normalize_country():
    assert normalize_country("  US  ") == "us"
    assert normalize_country("India") == "india"


def test_preprocess_dataframe():
    df = pd.DataFrame({
        "entity_id": ["E1"],
        "business_name": ["Acme Corp."],
        "business_address": ["100 Market St."],
        "country": ["US"],
    })
    res = preprocess_dataframe(df)
    assert "business_name_normalized" in res.columns
    assert res.loc[0, "business_name_normalized"] == "acme corp"
    assert res.loc[0, "business_address_normalized"] == "100 market st"
