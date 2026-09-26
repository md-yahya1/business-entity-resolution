# Model Training

## Purpose

Train a binary classifier that estimates whether two records in a generated pair refer to the same business. The training implementation is in `code/business_entity_resolution/src/models/train.py`; the CLI is `scripts/train.py`.

## Inputs

Training consumes a candidate-pair table with source attributes, `label`, and `entity_group_id`. If the configured feature file exists, the CLI reads it as Parquet or TSV. Otherwise it tries to build pairs from `dataset/train/train_source{1,2,3}.tsv` and `dataset/train/train_ground_truth.tsv` (note the singular legacy path). The source files in this workspace are under `datasets/train/`.

Default command:

```bash
python scripts/train.py --input artifacts/candidates/training_features.parquet --output models/entity_resolution_model.joblib
```

When the feature file is missing, the script samples up to 10,000 positive ground-truth anchors and 30,000 negatives by default, with random seed 42. It extracts features and writes the generated table before splitting. The fallback data directory is hard-coded; the script does not currently expose a `--train-dir` option.

## Feature Input

The model consumes these 19 comparison features, in the order recorded by `models/model_metadata.json`:

`name_ratio`, `name_partial_ratio`, `name_token_sort_ratio`, `name_token_set_ratio`, `name_jw_similarity`, `address_ratio`, `address_partial_ratio`, `address_token_set_ratio`, `address_jw_similarity`, `country_exact_match`, `country_missing`, `name_exact_match`, `address_exact_match`, `name_token_overlap`, `address_token_overlap`, `name_char_len_diff`, `address_char_len_diff`, `name_missing`, `address_missing`.

Pair IDs, source strings, labels, and group IDs are not model features.

## Split and Model Selection

`split_data_by_group` uses `GroupShuffleSplit` on `entity_group_id` to create train, validation, and test portions. The training code compares Logistic Regression, Random Forest, and Gradient Boosting. It tunes a threshold on validation pairwise predictions for F1, chooses the model with highest validation F1, then reports pairwise metrics on the test portion.

This objective differs from the entity-level competition score. A strong pairwise result does not imply a strong per-Source-1 macro score, particularly for singletons or records with multiple true matches.

## Checked-In Model Snapshot

Current GitHub `main` metadata identifies a `HistGradientBoosting` model with `max_iter=250`, `max_depth=8`, `learning_rate=0.08`, and `random_state=42`. Its stored decision threshold is `0.62`. The current trainer compares HistGradientBoosting, Extra Trees, Random Forest, Gradient Boosting, and Logistic Regression, choosing by validation pairwise F1.

The metadata reports these pairwise test metrics:

| Metric | Value |
|---|---:|
| Precision | 0.9989 |
| Recall | 0.9984 |
| F1 | 0.9987 |
| Accuracy | 0.9978 |
| ROC-AUC | 0.9999 |
| PR-AUC | 1.0000 |

These values are from the saved model metadata; they are pairwise metrics, not macro entity-level $F_{0.5}$ and not a newly computed leaderboard result.

## Competition Metric and Score Status

For each Source 1 entity, compare the set of predicted IDs to its ground-truth set. Compute precision and recall for that entity, then:

$$F_{0.5} = \frac{1.25 \times P \times R}{0.25 \times P + R}$$

The final score is the arithmetic mean over every Source 1 entity. A true singleton with no predicted matches scores 1; a singleton with any predicted match scores 0. The metric is precision-weighted because false merges are costly.

The last user-reported overall score is **0.056**. The checked-in `output/matching_results.tsv` contains only a partial set of IDs, and the evaluation source files and labels are not present under `datasets/test`; therefore no new overall macro $F_{0.5}$ score can be computed from this checkout. The local 95.1% candidate recall measurement and the pairwise F1 above are not final scores.

`scripts/tune_threshold_f0_5.py` can sweep thresholds against the entity-level metric, but its default procedure rebuilds candidate pairs from the supplied training directory. It should be treated as exploratory unless the evaluated entities are held out from model training and threshold selection.

## Artifacts

Training writes the following files under the output model directory:

- `entity_resolution_model.joblib`: serialized classifier
- `feature_config.json`: ordered feature names and count
- `model_metadata.json`: model name, hyperparameters, pairwise metrics, threshold, seed, and package versions

## Inference and Tests

For a precomputed feature table, `scripts/predict.py` writes detailed pair predictions. For grouped submission files covering every Source 1 ID, use `scripts/generate_submission.py`; see [Candidate Generation](CANDIDATE_GENERATION.md).

Run `pytest tests/test_pipeline.py` for training, grouped split, threshold, and artifact tests. Run the full suite with `pytest tests/`.

## Limitations

- Model selection tunes pairwise F1 rather than macro entity-level $F_{0.5}$.
- The training fallback uses a singular `dataset/` path, while the workspace data directory is `datasets/`.
- Reproducible leaderboard scoring requires the official evaluation labels and full evaluation source files, which are not present in this checkout.
