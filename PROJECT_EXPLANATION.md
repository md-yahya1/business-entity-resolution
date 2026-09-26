# Business Entity Resolution: Project Overview

## Goal

Link records that describe the same real-world business across three data sources. Source 1 records are matched to possible records from Sources 2 and 3. The classifier predicts a match probability for each candidate pair; the submission stage groups accepted IDs under every Source 1 ID.

## Pipeline

```text
Raw TSV records
  -> deterministic field normalization
  -> candidate retrieval
  -> 19 pair-comparison features
  -> trained binary classifier
  -> grouped candidate and match files
  -> entity-level macro F0.5 evaluation
```

Training pair sampling and inference candidate retrieval are distinct. Training uses ground-truth matches and sampled negatives. Inference has no labels and only forms Source 1-to-Source 2/3 pairs.

## Source Data and Schema

Each entity record requires `entity_id`, `business_name`, `business_address`, and `country`. The workspace data folders are `datasets/train/` and `datasets/test/`; source files use names such as `train_source1.tsv` and `test_source1.tsv`. The test folder is currently empty in this checkout. Training labels are separate: `train_ground_truth.tsv` maps `source1_entity_id` to a comma-separated `matched_entity_ids` list; empty values represent entities with no known matches.

## Preprocessing

`preprocess_dataframe` in `code/business_entity_resolution/src/preprocessing.py` validates required columns, fills null name/address/country values with empty strings, and adds normalized columns while preserving raw fields. Names and addresses use Unicode NFKC, lowercase, punctuation-to-space, whitespace collapse, and trim; country uses NFKC, lowercase, and trim. The implementation does not expand legal suffixes, map country aliases, remove duplicate rows, or create a data-quality report. See [Data Preprocessing](docs/DATA_PREPROCESSING.md).

## Candidate Retrieval

`generate_inference_candidates` indexes Source 2 and Source 3 by normalized country plus the first business-name character. It augments that block with up to two selective same-country token postings from name or address, deduplicates candidates, and ranks them by the stronger of name/address token-set similarity. Defaults are `top_k=25` and a token/fallback posting cap of 500. This cap controls work but can exclude true matches.

`scripts/generate_submission.py` expands candidate lists into feature rows, applies the model threshold, and writes:

- `candidate_pairs.tsv` with `source1_entity_id` and `candidate_entity_ids`
- `matching_results.tsv` with `source1_entity_id` and `matched_entity_ids`

Both files include every Source 1 ID, even when the list is empty. See [Candidate Generation](docs/CANDIDATE_GENERATION.md).

## Pair Features and Model

The feature extractor compares normalized names, addresses, and countries with fuzzy-string similarities, token overlap, exact-match flags, missingness flags, and string-length differences. The ordered list of 19 features is stored in `models/model_metadata.json` and `models/feature_config.json`.

`scripts/train.py` compares Logistic Regression, Random Forest, Gradient Boosting, Extra Trees, and HistGradientBoosting. It chooses the model using validation pairwise F1 and stores a threshold tuned for pairwise F1. Current GitHub `main` metadata identifies HistGradientBoosting, threshold `0.62`, pairwise test F1 `0.9987`, and pairwise precision `0.9989`. These metrics do not represent the entity-level challenge score. See [Model Training](docs/MODEL_TRAINING.md).

## Challenge Score

For each Source 1 ID, compare the set of predicted matches with its true match set and compute $F_{0.5}$. Average the per-entity values over the complete evaluation set, including singletons. A correctly predicted empty list for a singleton scores 1; any false match for a singleton scores 0. Precision has greater weight than recall.

The last user-reported overall score is **0.056**. It has not been recalculated as macro $F_{0.5}$ from this checkout: `datasets/test` is empty and the checked-in matching output is partial. A prior bounded diagnostic found 95.1% candidate recall on a training-data slice, but candidate recall and pairwise F1 are not the challenge score.

## Running the Pipeline

Install dependencies and run tests:

```bash
pip install -r requirements.txt
pytest tests/
```

Generate submission files when `test_source1.tsv`, `test_source2.tsv`, and `test_source3.tsv` are available:

```bash
python scripts/generate_submission.py --data-dir datasets/test --model models/entity_resolution_model.joblib --output-dir output --top-k 25
```

The training and pairwise prediction scripts include legacy fallback paths using `dataset/` singular. The checked workspace uses `datasets/`; use an existing feature table or align the data path before invoking those fallbacks.
