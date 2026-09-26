# Business Entity Resolution

This repository matches Source 1 business records to candidate records in Sources 2 and 3. It contains text normalization, pairwise feature extraction, model training and inference, plus submission-file generation.

## Pipeline

```text
Source TSVs -> normalize fields -> generate candidates -> compute pair features
            -> train or load classifier -> emit matched IDs for every Source 1 ID
```

Training pair generation and inference candidate generation are separate paths. Training uses labeled ground truth to create positive examples and sampled negatives. Inference does not use labels: it blocks Source 1 records against Sources 2 and 3, scores candidate pairs, and writes grouped output files.

## Setup

Install dependencies and run tests:

```bash
pip install -r requirements.txt
pytest tests/
```

The source data in this workspace is under `datasets/train/` and `datasets/test/`. The training and prediction scripts currently use some `dataset/...` singular defaults; provide existing feature files explicitly or align the data directory before relying on those fallback paths.

## Training

Train from an existing candidate-feature file, or let the script attempt to build one from its configured raw-data path:

```bash
python scripts/train.py --input artifacts/candidates/training_features.parquet --output models/entity_resolution_model.joblib
```

Training compares Logistic Regression, Random Forest, Gradient Boosting, Extra Trees, and HistGradientBoosting using validation pairwise F1. Current GitHub `main` metadata describes a HistGradientBoosting model with a pairwise-F1 decision threshold of `0.62`. Its saved pairwise test F1 is `0.9987`; this is not the entity-level leaderboard score.

## Pairwise Inference

For a file containing candidate pairs or precomputed features:

```bash
python scripts/predict.py --model models/entity_resolution_model.joblib --input artifacts/candidates/test_features.parquet --output output/predictions.tsv
```

This writes detailed pair predictions and a basic grouped match file. For submission-format files covering every Source 1 entity, use the submission generator instead.

## Submission Generation

With `test_source1.tsv`, `test_source2.tsv`, and `test_source3.tsv` present in the data directory:

```bash
python scripts/generate_submission.py --data-dir datasets/test --model models/entity_resolution_model.joblib --output-dir output --top-k 25
```

The script writes `matching_results.tsv` and `candidate_pairs.tsv`. It includes one row for every Source 1 entity, including entities with empty candidate or match lists, and performs basic output validation.

## Evaluation Metric

The challenge score is macro-averaged entity-level $F_{0.5}$: compute per-entity precision and recall from the predicted and true match sets, score each Source 1 entity, then average across the full evaluation set. Correctly predicting an empty list for a singleton scores `1`; predicting any match for a singleton scores `0`.

The last user-reported overall score is **0.056**; it has not been recalculated as macro $F_{0.5}$ from the current checkout. The evaluation split is not present under `datasets/test`, so no new official macro score can be verified locally. The bounded training-sample candidate-recall measurement and the saved pairwise F1 are different metrics and must not be reported as the final score.

## Documentation

- [Model training](docs/MODEL_TRAINING.md)
- [Data preprocessing](docs/DATA_PREPROCESSING.md)
- [Candidate generation](docs/CANDIDATE_GENERATION.md)
- [Pipeline overview](PROJECT_EXPLANATION.md)
