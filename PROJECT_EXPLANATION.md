# Business Entity Resolution — Project Explanation

## Goal

Match each Source-1 business entity to the corresponding entity IDs in Sources 2 and 3.

## Final architecture

S1 / S2 / S3
-> preprocessing
-> country canonicalization + name/address normalization + address hints
-> multi-pass blocking
-> cheap deterministic ranking
-> <=10 candidates/entity
-> 13 trained pair features
-> HistGradientBoostingClassifier
-> entity decision rules
-> matching_results.tsv + candidate_pairs.tsv

## Preprocessing

Implementation: code/business_entity_resolution/src/preprocessing.py

Required fields are entity_id, business_name, business_address, and country.

Names and addresses use Unicode NFKC, lowercase conversion, punctuation normalization, whitespace normalization, and trimming.

Countries are canonicalized for common aliases such as US/USA/United States and India/IN.

Address hints extract postal code, house number, city, and state. Postal tokens are removed before state detection so a trailing ZIP cannot hide the state.

## Blocking

Implementation: code/business_entity_resolution/src/blocking/laptop_final.py

Source 2 and Source 3 are combined into one compact posting index scoped by normalized country.

The active passes are exact name, name prefix, selective name token, postal code, house number, city, and selective address token.

The blocker is deterministic and cheap. It does not run RapidFuzz similarity during retrieval.

When the block union is larger than the retrieval limit, candidates are ranked using exact name, exact address, exact postal code, exact house number, exact city, exact address token, exact state, name prefix, and length differences.

The query prefix hash is computed once per chunk and reused.

## Model

Implementation: models/laptop_final/

The saved model is a HistGradientBoostingClassifier with 13 features:

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

Saved pairwise validation metrics are precision 99.6730%, recall 98.4657%, F0.5 99.4292%, F1 99.0657%, and accuracy 98.9800%.

These are pairwise metrics, not the final entity-level challenge score.

## Entity decision

Implementation: code/business_entity_resolution/src/evaluation/entity_decision.py

For each Source-1 entity:

- reject when the best probability is below 0.42
- retain probabilities at or above the model threshold
- reject an ambiguous lone match when the top-two margin is below 0.06 unless the best probability is at least 0.88
- deduplicate selected IDs

A unique exact normalized-name candidate has a fast direct path in production inference.

## Evaluation

The challenge metric is macro entity-level F0.5:

    F0.5 = 1.25PR / (0.25P + R)

Three measurements must not be confused:

1. candidate recall
2. pairwise model metrics
3. final entity-level macro F0.5

The classifier cannot recover a true match that blocking discarded.

## Historical baseline

Before the current blocker fixes, a 5,000-entity diagnostic measured approximately:

- candidate recall: 32.91%
- final precision: 48.05%
- final recall: 28.97%
- macro F0.5: 39.87%

These numbers are historical only. The current branch must be benchmarked again.

## Runtime target

The full test set contains approximately 1.73M Source-1, 4.89M Source-2, and 5.08M Source-3 records.

The laptop target is approximately one hour.

The final design therefore avoids fuzzy retrieval for every query, large candidate lists, expensive ANN/embedding infrastructure, and retraining during inference.

See docs/FINAL_INFERENCE.md for exact commands.
