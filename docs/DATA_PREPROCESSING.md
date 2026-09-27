# Data Preprocessing

## Purpose

Normalize the fields used by candidate generation and pairwise matching. The implementation is:

`code/business_entity_resolution/src/preprocessing.py`

## Required schema

Each source record requires:

| Column | Meaning |
|---|---|
| `entity_id` | Source-specific record ID |
| `business_name` | Raw business name |
| `business_address` | Raw business address |
| `country` | Raw country |

Additional columns are preserved but are not used by the final laptop inference pipeline.

## Normalization

### Business name

- missing values -> empty string
- Unicode NFKC normalization
- lowercase
- punctuation -> spaces
- repeated whitespace collapsed
- surrounding whitespace removed

### Business address

- missing values -> empty string
- Unicode NFKC normalization
- lowercase
- comma, semicolon and slash -> spaces
- remaining punctuation -> spaces
- repeated whitespace collapsed
- surrounding whitespace removed

### Country

- missing values -> empty string
- Unicode NFKC normalization
- lowercase
- repeated whitespace collapsed
- common aliases canonicalized

Country canonicalization currently includes US/USA/United States, India/IN, France/FR, and UK/GB/United Kingdom.

## Address hint extraction

The normalized address is parsed into lightweight hints:

- `house_number`
- `postal_code`
- `state`
- `city`

Postal extraction supports US ZIP codes and UK-style postcodes. The parser removes the postal-code tokens before scanning for a trailing US state code.

The city contract intentionally remains a single-token city hint because the saved 13-feature model was trained with this representation.

## Missing values

Missing name, address and country values are converted to `""`. Missing entity IDs are not repaired.

## Duplicates

Rows and entity IDs are not automatically deduplicated. The inference pipeline assumes source IDs are suitable for submission.

## Leakage

Preprocessing is deterministic and does not fit statistics from the labels. Ground truth remains outside the feature matrix.

## Tests

```powershell
pytest tests/test_preprocessing.py -q
```

The regression tests verify:

- Unicode/null normalization
- country alias canonicalization
- ZIP extraction
- state extraction when ZIP follows the state
- city extraction
- required-column validation

## Limitations

This stage does not:

- infer missing business names
- infer missing addresses
- resolve duplicate entity IDs
- standardize every country spelling/code in existence
- create phone/email/domain features
