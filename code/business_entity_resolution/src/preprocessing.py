import re
import unicodedata
import pandas as pd
from typing import Dict


REQUIRED_COLUMNS = [
    "entity_id",
    "business_name",
    "business_address",
    "country",
]

_US_ZIP_RE = re.compile(r"\b(\d{5})(?:\s*-\s*(\d{4}))?\b")
_UK_POSTAL_RE = re.compile(r"\b([a-z]{1,2}\d[a-z\d]?\s*\d[a-z]{2})\b", re.IGNORECASE)


_COUNTRY_ALIASES = {
    "us": "us", "usa": "us", "u s": "us",
    "united states": "us", "united states of america": "us",
    "in": "india", "ind": "india", "india": "india",
    "fr": "france", "fra": "france", "france": "france",
    "gb": "uk", "gbr": "uk", "uk": "uk", "united kingdom": "uk",
}

_US_STATE_CODES = {
    "al", "ak", "az", "ar", "ca", "co", "ct", "de", "fl", "ga",
    "hi", "id", "il", "in", "ia", "ks", "ky", "la", "me", "md",
    "ma", "mi", "mn", "ms", "mo", "mt", "ne", "nv", "nh", "nj",
    "nm", "ny", "nc", "nd", "oh", "ok", "or", "pa", "ri", "sc",
    "sd", "tn", "tx", "ut", "vt", "va", "wa", "wv", "wi", "wy", "dc",
}


def normalize_text(value: object) -> str:
    """
    General text normalization.

    Steps:
    - Handle missing values
    - Unicode normalization
    - Convert to lowercase
    - Remove punctuation
    - Normalize whitespace
    """
    if pd.isna(value):
        return ""

    value = str(value)

    value = unicodedata.normalize("NFKC", value)
    value = value.lower()
    value = re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE)
    value = re.sub(r"\s+", " ", value)

    return value.strip()


def normalize_business_name(value: object) -> str:
    """Normalize a business name while preserving meaningful words."""
    return normalize_text(value)


def normalize_address(value: object) -> str:
    """Normalize a business address."""
    if pd.isna(value):
        return ""

    value = str(value)
    value = unicodedata.normalize("NFKC", value)
    value = value.lower()
    value = re.sub(r"[,;/]+", " ", value)
    value = re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE)
    value = re.sub(r"\s+", " ", value)

    return value.strip()


def normalize_country(value: object) -> str:
    """Normalize country values."""
    if pd.isna(value):
        return ""

    value = str(value)
    value = unicodedata.normalize("NFKC", value)
    value = re.sub(r"\s+", " ", value.strip().lower())
    return _COUNTRY_ALIASES.get(value, value)


def extract_address_hints(normalized_address: str) -> Dict[str, str]:
    """
    Extract lightweight structured hints from a normalized address.

    Postal codes are removed before state/city detection so a trailing
    ZIP/UK postcode cannot hide a state token.
    """
    addr = normalized_address or ""
    hints = {"house_number": "", "postal_code": "", "city": "", "state": ""}
    tokens = addr.split()

    postal_match = _US_ZIP_RE.search(addr)
    if postal_match:
        hints["postal_code"] = postal_match.group(1)
        start_token = len(addr[:postal_match.start()].split())
        end_token = len(addr[:postal_match.end()].split())
        del tokens[start_token:end_token]
    else:
        uk = _UK_POSTAL_RE.search(addr)
        if uk:
            hints["postal_code"] = uk.group(1).replace(" ", "")
            start_token = len(addr[:uk.start()].split())
            end_token = len(addr[:uk.end()].split())
            del tokens[start_token:end_token]

    if tokens and re.match(r"^\d+[a-z]?$", tokens[0]):
        hints["house_number"] = tokens[0]

    state_index = None
    for idx in range(len(tokens) - 1, -1, -1):
        if tokens[idx] in _US_STATE_CODES:
            state_index = idx
            hints["state"] = tokens[idx]
            break

    city_tokens = tokens[:state_index] if state_index is not None else tokens
    city_tokens = [
        t for t in city_tokens
        if t and not t.isdigit()
        and t not in {"usa", "us", "uk", "in", "india"}
    ]
    if city_tokens:
        hints["city"] = city_tokens[-1]

    return hints


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Missing text values are converted to empty strings."""
    df = df.copy()

    text_columns = [
        "business_name",
        "business_address",
        "country",
    ]

    for column in text_columns:
        if column in df.columns:
            df[column] = df[column].fillna("")

    return df


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Complete preprocessing pipeline for a business entity dataset."""
    df = df.copy()

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    df = handle_missing_values(df)

    df["business_name_normalized"] = (
        df["business_name"]
        .apply(normalize_business_name)
    )

    df["business_address_normalized"] = (
        df["business_address"]
        .apply(normalize_address)
    )

    df["country_normalized"] = (
        df["country"]
        .apply(normalize_country)
    )

    return df
