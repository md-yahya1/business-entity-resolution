# Data Preprocessing

## Purpose

Normalize business names, addresses, and countries before blocking and pair-feature extraction. The implementation is `code/business_entity_resolution/src/preprocessing.py`.

## Inputs and Schema

`preprocess_dataframe` accepts a pandas DataFrame and requires:

| Column | Meaning |
|---|---|
| `entity_id` | Record identifier, retained unchanged |
| `business_name` | Raw business name |
| `business_address` | Raw business address |
| `country` | Raw country value |

The repository's source TSVs are `train_source1.tsv`, `train_source2.tsv`, and `train_source3.tsv`; the checked workspace places them under `datasets/train/`. Test inputs, when available, use the same names under `datasets/test/`. Ground truth is kept separate from record features.

Other input columns are retained but are not transformed. There are no city, state, postal-code, phone, email, or URL transformations.

## Pipeline and Transformations

The function copies the DataFrame, raises `ValueError` if a required column is absent, fills null name/address/country values with empty strings, and appends normalized fields. It does not remove rows or columns.

| Output column | Transformation |
|---|---|
| `business_name_normalized` | String conversion, Unicode NFKC, lowercase, punctuation-to-space, whitespace collapse, trim |
| `business_address_normalized` | String conversion, Unicode NFKC, lowercase, comma/semicolon/slash-to-space, remaining punctuation-to-space, whitespace collapse, trim |
| `country_normalized` | String conversion, Unicode NFKC, lowercase, trim |

Country aliases such as `US` and `USA` are not unified. Business suffixes are not expanded or removed. These conservative rules avoid unsupported equivalence assumptions.

## Missing Values and Duplicates

Null values in the three text columns become `""` in both the raw and normalized columns. Missing identifiers are not repaired. Duplicate rows and duplicate IDs are neither removed nor reported; callers are responsible for identifier uniqueness.

## Data Quality and Leakage

No general data-quality report is generated. The code validates required-column presence only; it does not report row counts, missingness, malformed values, or duplicate counts. Normalization is deterministic and has no fitted state, so it does not learn from labels or evaluation-set statistics. Keep ground-truth columns outside model features.

## Output

The function returns the original DataFrame columns plus the three normalized columns listed above. It does not write a file.

## Execution and Tests

There is no `scripts/preprocess.py` in this repository. Call the function from Python; training and inference modules also call it internally:

```python
from business_entity_resolution.src.preprocessing import preprocess_dataframe

processed = preprocess_dataframe(records)
```

Run `pytest tests/test_preprocessing.py` to validate normalization and required-column behavior.

## Limitations

- Only name, address, and country are normalized.
- Country aliases, legal suffixes, and contact fields are not standardized.
- Duplicate handling, a standalone CLI, and a data-quality report are not implemented.
- Some scripts default to `dataset/`, while the checked workspace data directory is `datasets/`.
