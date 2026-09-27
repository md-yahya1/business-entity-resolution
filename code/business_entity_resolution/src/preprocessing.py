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
    value = value.strip().lower()

    return value


def extract_address_hints(normalized_address: str) -> Dict[str, str]:
    """
    Lightweight address parsing for blocking and match features.
    Works on normalized address strings (lowercase, punctuation stripped).
    """
    addr = normalized_address or ""
    hints = {"house_number": "", "postal_code": "", "city": "", "state": ""}

    zip_match = _US_ZIP_RE.search(addr)
    if zip_match:
        hints["postal_code"] = zip_match.group(1)
    else:
        uk = _UK_POSTAL_RE.search(addr)
        if uk:
            hints["postal_code"] = uk.group(1).replace(" ", "")

    tokens = addr.split()
    if tokens and re.match(r"^\d+[a-z]?$", tokens[0]):
        hints["house_number"] = tokens[0]

    if len(tokens) >= 2 and re.match(r"^[a-z]{2}$", tokens[-1]):
        hints["state"] = tokens[-1]
        city_tokens = tokens[-3:-1] if len(tokens) >= 3 else tokens[-2:-1]
    else:
        city_tokens = tokens[-2:] if len(tokens) >= 2 else tokens[-1:]

    city_tokens = [
        t for t in city_tokens
        if t and not t.isdigit() and t not in {"usa", "us", "uk", "in", "india"}
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
