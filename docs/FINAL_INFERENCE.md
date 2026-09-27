# Final Laptop Inference

## Purpose

This is the exact production path for the current `tune/laptop-final-inference` branch.

The goal is to improve candidate recall and correctness without reintroducing the expensive fuzzy retrieval stage that previously made the 5,000-row benchmark too slow.

## Production files

- Blocker: `code/business_entity_resolution/src/blocking/laptop_final.py`
- Inference CLI: `scripts/generate_submission_laptop_final.py`
- Decision rules: `code/business_entity_resolution/src/evaluation/entity_decision.py`
- Model: `models/laptop_final/`
- Benchmark evaluator: `scripts/evaluate_laptop_benchmark.py`

## Pipeline

```text
S1/S2/S3 TSVs
    |
    v
deterministic preprocessing
    |
    v
multi-pass compact blocking
    |
    +--> exact-name fast path
    |
    v
up to 10 candidates/entity
    |
    v
13 pairwise features
    |
    v
saved HistGradientBoosting model
    |
    v
entity-level decision rules
    |
    +--> matching_results.tsv
    +--> candidate_pairs.tsv
```

## Runtime-preserving improvements

### Address-token block

Adds a cheap same-country address-token index. This creates another route to retrieve a true match when the business name is substantially different.

### Exact-address ranking

Exact normalized address receives a strong deterministic ranking weight. This improves candidate ordering without running fuzzy similarity during blocking.

### Removed repeated prefix hashing

The query prefix hash is already computed once per chunk. The blocker reuses that hash instead of constructing a one-row pandas DataFrame for every entity during pre-ranking.

### Address parsing fix

A trailing ZIP code previously could prevent a preceding US state token from being detected. Postal tokens are now removed before state detection.

### Country aliases

Common country aliases are canonicalized before blocking so equivalent country values do not create separate hard blocks.

## Benchmark gate

Do not start the 1,732,544-row test run until the exact final pipeline passes the 5,000-row benchmark.

Run:

```powershell
python scripts/generate_submission_laptop_final.py `
  --data-dir dataset/train `
  --source1-file train_source1.tsv `
  --source2-file train_source2.tsv `
  --source3-file train_source3.tsv `
  --limit 5000 `
  --chunk-size 5000 `
  --retrieval-limit 10 `
  --output-candidate-limit 10 `
  --output-dir output/final_fast_benchmark
```

Then evaluate the generated files against the training ground truth.

## Full test run

Only after the benchmark is reviewed:

```powershell
python scripts/generate_submission_laptop_final.py `
  --data-dir dataset/test `
  --source1-file test_source1.tsv `
  --source2-file test_source2.tsv `
  --source3-file test_source3.tsv `
  --chunk-size 50000 `
  --retrieval-limit 10 `
  --output-candidate-limit 10 `
  --output-dir output/laptop_final_submission
```

Known test sizes are:

- Source 1: 1,732,544
- Source 2: 4,887,273
- Source 3: 5,082,316

The script always prints the actual loaded counts.

## Output contract

`matching_results.tsv`:

```text
source1_entity_id    matched_entity_ids
```

`candidate_pairs.tsv`:

```text
source1_entity_id    candidate_entity_ids
```

Every Source-1 entity receives one row, including entities with no candidates or no matches.

## Runtime rule

The target is approximately one hour for the full test run.

Do not increase `retrieval-limit`, `posting-cap`, or fuzzy retrieval without benchmarking first. The downstream 13-feature model already performs fuzzy comparisons after blocking; the blocker must remain cheap.

## Validation rule

Pairwise model metrics and end-to-end entity metrics are different.

Before submission inspect:

1. candidate recall
2. entity-level precision
3. entity-level recall
4. macro F0.5
5. benchmark runtime
6. full-run throughput/ETA
7. output row count and duplicate-ID validity

Never call the final entity-resolution system 99% accurate unless that exact entity-level metric has actually been measured.
