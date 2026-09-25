import re
import unicodedata
import pandas as pd


REQUIRED_COLUMNS = [
    "entity_id",
    "business_name",
    "business_address",
    "country",
]


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

    # Unicode normalization
    value = unicodedata.normalize("NFKC", value)

    # Lowercase
    value = value.lower()

    # Replace punctuation with spaces
    value = re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE)

    # Normalize whitespace
    value = re.sub(r"\s+", " ", value)

    return value.strip()


def normalize_business_name(value: object) -> str:
    """
    Normalize a business name while preserving meaningful words.
    """
    return normalize_text(value)


def normalize_address(value: object) -> str:
    """
    Normalize a business address.
    """
    if pd.isna(value):
        return ""

    value = str(value)

    # Unicode normalization
    value = unicodedata.normalize("NFKC", value)

    # Lowercase
    value = value.lower()

    # Convert common address separators to spaces
    value = re.sub(r"[,;/]+", " ", value)

    # Remove remaining punctuation
    value = re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE)

    # Normalize whitespace
    value = re.sub(r"\s+", " ", value)

    return value.strip()


def normalize_country(value: object) -> str:
    """
    Normalize country values.
    """
    if pd.isna(value):
        return ""

    value = str(value)

    value = unicodedata.normalize("NFKC", value)
    value = value.strip().lower()

    return value


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values in text columns.

    Missing text values are converted to empty strings.
    """
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
    """
    Complete preprocessing pipeline for a business entity dataset.
    """

    df = df.copy()

    # Validate expected schema
    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Handle missing values
    df = handle_missing_values(df)

    # Create normalized columns
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