import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "code")))

from business_entity_resolution.src.blocking.laptop_final import build_index


def test_laptop_blocking_recovers_changed_name_via_address_token():
    s1 = pd.DataFrame([{
        "entity_id": "S1-1",
        "business_name": "Completely Different Name",
        "business_address": "15 Harbor Street New York NY 10001",
        "country": "US",
    }])
    s2 = pd.DataFrame([
        {
            "entity_id": "S2-decoy",
            "business_name": "Harbor Cafe",
            "business_address": "99 Far Road Boston MA 02110",
            "country": "US",
        },
        {
            "entity_id": "S2-match",
            "business_name": "Acme Holdings",
            "business_address": "15 Harbor Street New York NY 10001",
            "country": "US",
        },
    ])
    s3 = pd.DataFrame(columns=["entity_id", "business_name", "business_address", "country"])

    index = build_index(s2, s3)
    q = s1.iloc[0]

    from business_entity_resolution.src.preprocessing import (
        extract_address_hints,
        preprocess_dataframe,
    )

    processed = preprocess_dataframe(s1).iloc[0]
    hints = extract_address_hints(processed["business_address_normalized"])
    candidates = index.candidates(
        processed["country_normalized"],
        processed["business_name_normalized"],
        processed["business_address_normalized"],
        hints["postal_code"],
        hints["house_number"],
        hints["city"],
        hints["state"],
        limit=1,
    )

    assert index.ids[candidates[0]] == "S2-match"


def test_laptop_blocking_prefers_exact_address_over_name_prefix_decoy():
    s2 = pd.DataFrame([
        {
            "entity_id": "S2-decoy",
            "business_name": "Acme Harbor Services",
            "business_address": "999 Other Road Boston MA 02110",
            "country": "US",
        },
        {
            "entity_id": "S2-match",
            "business_name": "Different Legal Name",
            "business_address": "15 Harbor Street New York NY 10001",
            "country": "US",
        },
    ])
    s3 = pd.DataFrame(columns=["entity_id", "business_name", "business_address", "country"])
    s1 = pd.DataFrame([{
        "entity_id": "S1-1",
        "business_name": "Acme Holdings",
        "business_address": "15 Harbor Street New York NY 10001",
        "country": "US",
    }])

    index = build_index(s2, s3)
    processed = __import__(
        "business_entity_resolution.src.preprocessing",
        fromlist=["preprocess_dataframe"],
    ).preprocess_dataframe(s1).iloc[0]

    from business_entity_resolution.src.preprocessing import extract_address_hints

    hints = extract_address_hints(processed["business_address_normalized"])
    candidates = index.candidates(
        processed["country_normalized"],
        processed["business_name_normalized"],
        processed["business_address_normalized"],
        hints["postal_code"],
        hints["house_number"],
        hints["city"],
        hints["state"],
        limit=1,
    )

    assert index.ids[candidates[0]] == "S2-match"
