# Candidate Generation

## Purpose

Candidate generation reduces the number of record comparisons passed to feature extraction and the classifier. It does not decide final matches.

## Position in Pipeline

```text
Source TSVs -> preprocessing -> candidates -> pair features -> classifier
            -> matching_results.tsv and candidate_pairs.tsv
```

There are separate training and inference implementations: `candidate_generator.generate_candidate_pairs` and `inference_blocking.generate_inference_candidates`. Current `main` also contains `candidate_generator.generate_test_candidates`, a helper that is not called by either inference CLI.

## Inputs

Each source DataFrame requires `entity_id`, `business_name`, `business_address`, and `country`. Ground truth is optional for training and has `source1_entity_id` plus comma-separated `matched_entity_ids`. The checked workspace stores source files under `datasets/train/` and `datasets/test/`; some scripts retain a legacy `dataset/` default.

## Training Pair Generation

`scripts/train.py` calls `generate_candidate_pairs` when its configured training-feature file is absent. Ground-truth matches provide positive pairs, capped by `--max-positives` (default 10,000). The function samples up to 60,000 records for negative-pair blocking, groups them by normalized country and first character of normalized name, samples up to 40 records per block, and creates a limited set of nearby index pairs until `--max-negatives` (default 30,000) is reached. The random seed defaults to 42. Pairs include raw source attributes, `label`, and `entity_group_id` for training.

This is a training sampler, not the inference candidate strategy. Negative pairs may be drawn from any source combination. Pair deduplication uses the unordered pair of IDs; positive pairs are added from ground truth even if the names would not share a blocking key.

`generate_test_candidates` is an additional helper on current `main`. It blocks on exact normalized country and first full name token, then emits up to 15 pairs per Source 1 record by default. It does not use address tokens and currently sets `label=0` and `entity_group_id=0` on output rows. It is not the path used by `scripts/generate_submission.py`; do not treat its placeholder label as a ground-truth prediction.

## Inference Blocking Strategy

`generate_inference_candidates` preprocesses each source and returns one row for every Source 1 ID. Only Source 2 and Source 3 IDs can be candidates.

For each Source 1 record, the implementation:

1. Adds records sharing normalized country and the first character of normalized business name.
2. Builds separate same-country postings for each normalized name token and address token of at least three characters. It selects up to two rare postings whose size is at most `country_fallback_cap` (default 500), sorted by posting size, field, then token.
3. Unions these IDs and removes duplicates while preserving insertion order.
4. If the union is empty, takes the first up to 500 records from the same-country index.
5. Ranks the resulting pool by the larger of name and address `token_set_ratio` and retains `top_k` (default 25).

The bounded candidate set is a recall/compute tradeoff. Prefix, country, posting-size, fallback-order, and top-k limits can exclude true matches. An unseen country produces no candidates; an empty candidate list is allowed and is needed to represent a singleton.

## Inference Output and Submission Files

`generate_inference_candidates` returns `source1_entity_id` and a list-valued `candidate_entity_ids`. `expand_candidates_to_pairs` expands those lists into Source 1/Source 2-or-3 raw attribute pairs for feature extraction.

`scripts/generate_submission.py` writes:

- `candidate_pairs.tsv`: `source1_entity_id`, `candidate_entity_ids`
- `matching_results.tsv`: `source1_entity_id`, `matched_entity_ids`

Both files contain a row for every Source 1 ID, including empty lists. Run it with `--data-dir`, `--model`, `--output-dir`, and optionally `--top-k` or `--threshold`. The threshold defaults to model metadata. The script checks ID validity, duplicate IDs in lists, and that every predicted match is in that entity's candidate list.

## Recall and Evaluation Status

The bounded training-data diagnostic previously run on the first 100,000 rows of each source and first 250,000 ground-truth rows contained 741 true pairs; the current blocker retrieved 705 at `top_k=25` (95.1% candidate recall on that slice). This is not an unbiased full-data estimate and is not the competition score.

The competition metric is macro entity-level $F_{0.5}$ over every Source 1 entity, including singletons. The last user-reported overall score is 0.056, but the evaluation/test split is absent from this checkout, so a new official macro $F_{0.5}$ score cannot be calculated here. Candidate recall must not be substituted for that score.

## Validation and Limitations

Run `pytest tests/test_blocking.py`. The test covers recovery of a differently named record through address tokens. The full suite is `pytest tests/`. The current index does not use phonetic or character n-gram retrieval; common name/address tokens can be excluded by the posting cap, and fallback selection can depend on input row order.

## Handoff

The submission generator expands candidates, computes the 19 pair-comparison features, and applies the saved classifier and threshold.

```text
INPUT: Source 1, Source 2, and Source 3 TSV files
PROCESS: normalize -> prefix and selective-token blocks -> rank/cap -> feature extraction -> classifier
OUTPUT: output/candidate_pairs.tsv and output/matching_results.tsv
NEXT STAGE: Feature extraction and model-based match prediction
```
