# Business Entity Resolution

This repository resolves Source-1 business records against Source-2 and Source-3 records at laptop scale.

## Current production pipeline

Source TSVs -> deterministic preprocessing -> multi-pass compact blocking -> <=10 candidates per Source-1 entity -> 13 pairwise features -> saved HistGradientBoosting model -> entity-level decision rules -> matching_results.tsv + candidate_pairs.tsv.

The current production path is the tune/laptop-final-inference branch and is documented in docs/FINAL_INFERENCE.md.

## Setup

Windows PowerShell:

    python -m venv .venv
    .venv\Scripts\Activate.ps1
    pip install -r requirements.txt

Run tests:

    pytest tests/ -q

## Final saved model

models/laptop_final/

The saved classifier is a HistGradientBoostingClassifier using 13 features:

    name_ratio
    name_token_set_ratio
    address_ratio
    address_token_set_ratio
    country_exact_match
    name_exact_match
    address_exact_match
    city_exact_match
    postal_exact_match
    house_number_exact_match
    state_exact_match
    name_char_len_diff
    address_char_len_diff

Saved pairwise validation metrics are precision 99.6730%, recall 98.4657%, F0.5 99.4292%, F1 99.0657%, and accuracy 98.9800%. These are pairwise metrics, not the final entity-level challenge score.

## Candidate generation

The final blocker uses cheap, same-country posting indexes for exact name, name prefix, selective name token, postal code, house number, city, and selective address token.

Candidates are unioned, deduplicated, and cheaply ranked. Fuzzy similarity is deliberately not used during blocking because the full dataset must remain practical on a laptop.

Default retrieval is 10 internal candidates and 10 output candidates per Source-1 entity.

## Benchmark before full inference

Never spend the full test-run time before checking the exact production pipeline on 5,000 training entities.

    python scripts/generate_submission_laptop_final.py --data-dir dataset/train --source1-file train_source1.tsv --source2-file train_source2.tsv --source3-file train_source3.tsv --limit 5000 --chunk-size 5000 --retrieval-limit 10 --output-candidate-limit 10 --output-dir output/final_fast_benchmark

Then evaluate the generated files against dataset/train/train_ground_truth.tsv.

Remember:

    pairwise model quality != candidate recall != final entity-level macro F0.5

A missed candidate cannot be recovered by the classifier.

## Full inference

After the benchmark has been reviewed:

    python scripts/generate_submission_laptop_final.py --data-dir dataset/test --source1-file test_source1.tsv --source2-file test_source2.tsv --source3-file test_source3.tsv --chunk-size 50000 --retrieval-limit 10 --output-candidate-limit 10 --output-dir output/laptop_final_submission

Output:

    output/laptop_final_submission/matching_results.tsv
    output/laptop_final_submission/candidate_pairs.tsv

Every Source-1 entity should have one output row.

## Documentation

- docs/DATA_PREPROCESSING.md
- docs/CANDIDATE_GENERATION.md
- docs/MODEL_TRAINING.md
- docs/FINAL_INFERENCE.md
- PROJECT_EXPLANATION.md
