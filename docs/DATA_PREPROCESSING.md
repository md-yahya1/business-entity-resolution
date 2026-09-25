You are responsible for designing and implementing the DATA PREPROCESSING stage of a machine-learning Entity Resolution project.

PROJECT:
Business Entity Resolution

OBJECTIVE:
We have raw business/entity records that may contain duplicate or slightly different representations of the same real-world business. The downstream pipeline will perform candidate generation, feature engineering, and model-based matching.

Your task is ONLY to handle the data preprocessing stage.

IMPORTANT:
Do not implement candidate generation or model training unless required to make the preprocessing pipeline testable.

==================================================
GOALS
==================================================

1. Inspect the available raw dataset and understand:
   - File formats
   - Columns
   - Data types
   - Missing values
   - Duplicate records
   - Unique identifiers
   - Text fields
   - Numerical fields
   - Categorical fields
   - Address/contact/business information
   - Training labels, if present

2. Determine which columns are:
   - Entity identifiers
   - Business names
   - Addresses
   - Cities
   - States
   - Countries
   - Postal codes
   - Phone numbers
   - Email addresses
   - URLs
   - Other useful entity attributes

3. Design a reproducible preprocessing pipeline.

==================================================
PREPROCESSING REQUIREMENTS
==================================================

Implement preprocessing for relevant fields.

For textual fields consider:

- Unicode normalization
- Lowercasing
- Whitespace normalization
- Punctuation normalization
- Removal of unnecessary special characters
- Standardization of common separators
- Handling of accents/diacritics
- Safe handling of null values
- Consistent string conversion

For business names consider:

- Case normalization
- Legal suffix normalization where appropriate
- Common abbreviation handling where justified
- Removal of unnecessary punctuation
- Whitespace normalization

Examples of potentially equivalent forms:

"ABC Pvt. Ltd."
"ABC PRIVATE LIMITED"
"abc pvt ltd"

However, DO NOT aggressively normalize information if doing so can merge genuinely different entities.

For addresses consider:

- Case normalization
- Unicode normalization
- Whitespace normalization
- Standardization of punctuation
- Component-wise processing where possible

For phone numbers:

- Remove formatting characters where appropriate
- Normalize country codes if the dataset permits
- Preserve enough information for downstream matching

For email:

- Lowercase where appropriate
- Trim whitespace
- Preserve the semantic structure
- Do not make provider-specific assumptions unless justified

For URLs:

- Normalize scheme/case where appropriate
- Remove irrelevant formatting differences
- Preserve meaningful domain information

==================================================
DATA QUALITY
==================================================

Create a data-quality analysis that reports:

- Number of rows
- Number of columns
- Missing values per column
- Duplicate rows
- Unique values
- Data types
- Invalid/malformed values
- Potentially suspicious records

Do NOT silently delete records.

Any row filtering must be explicitly documented and justified.

==================================================
DATA LEAKAGE
==================================================

Pay special attention to preventing data leakage.

Do not use:

- Target labels as input features
- Future information
- Test-set information when creating training transformations
- Information derived from the evaluation set to modify training preprocessing

If preprocessing requires fitted transformations, fit them only on the appropriate training data and reuse them during inference.

==================================================
OUTPUT
==================================================

Create a deterministic preprocessing script.

Prefer:

scripts/preprocess.py

The script should:

1. Load raw data
2. Validate required columns
3. Normalize data
4. Handle missing values
5. Perform required transformations
6. Save processed data
7. Produce a preprocessing/data-quality report
8. Fail with a clear error if required input data is missing

Avoid hardcoded absolute paths such as:

C:\Users\...

Use configurable relative paths or command-line arguments.

Example:

python scripts/preprocess.py \
    --input dataset/raw/train.tsv \
    --output artifacts/processed/train_processed.parquet

==================================================
REPRODUCIBILITY
==================================================

The preprocessing process must be deterministic.

Document:

- Input files
- Input schema
- Transformations
- Output schema
- Removed/retained columns
- Missing-value handling
- Normalization rules
- Any assumptions

==================================================
TESTING
==================================================

Create tests for important preprocessing behavior.

At minimum test:

- Null values
- Unicode normalization
- Case normalization
- Whitespace normalization
- Duplicate handling
- Phone normalization
- Email normalization
- Address normalization
- Required-column validation

==================================================
DOCUMENTATION
==================================================

Create:

docs/DATA_PREPROCESSING.md

The documentation must be written so that:

1. A new developer can understand the preprocessing stage without reading the entire repository.
2. An LLM can easily parse the document and understand the pipeline.
3. Every input and output is explicitly documented.
4. Every transformation has a reason.
5. The document contains no vague statements.

Use clear Markdown headings.

Required structure:

# Data Preprocessing

## 1. Purpose

## 2. Input Data

## 3. Input Schema

## 4. Data Quality Analysis

## 5. Preprocessing Pipeline

## 6. Field-Level Transformations

## 7. Missing Value Handling

## 8. Duplicate Handling

## 9. Data Leakage Prevention

## 10. Output Data

## 11. Output Schema

## 12. Reproducibility

## 13. Execution

## 14. Validation and Tests

## 15. Known Limitations

## 16. Handoff to Candidate Generation

At the end, explicitly state:

INPUT:
<files>

PROCESS:
<steps>

OUTPUT:
<files>

NEXT STAGE:
Candidate Generation

Do not invent dataset columns. Inspect the actual dataset and document the real schema.