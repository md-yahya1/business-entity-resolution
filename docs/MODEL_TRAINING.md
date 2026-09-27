# Model Training

## Purpose

The final laptop inference pipeline uses an already-trained binary classifier to score candidate pairs. Training is a separate stage; changing the inference blocker does not retrain the model.

The saved production artifact is:

`models/laptop_final/entity_resolution_model.joblib`

Metadata:

`models/laptop_final/model_metadata.json`

## Saved model

Model type:

`HistGradientBoostingClassifier`

The saved model consumes 13 features:

1. `name_ratio`
2. `name_token_set_ratio`
3. `address_ratio`
4. `address_token_set_ratio`
5. `country_exact_match`
6. `name_exact_match`
7. `address_exact_match`
8. `city_exact_match`
9. `postal_exact_match`
10. `house_number_exact_match`
11. `state_exact_match`
12. `name_char_len_diff`
13. `address_char_len_diff`

Feature order must remain identical to `model_metadata.json`.

## Saved validation metrics

The current saved model metadata reports:

| Metric | Value |
|---|---:|
| Precision | 99.6730% |
| Recall | 98.4657% |
| F0.5 | 99.4292% |
| F1 | 99.0657% |
| Accuracy | 98.9800% |

These are **pairwise validation metrics** for candidate pairs. They are not the final entity-level challenge score.

The saved decision threshold is:

`0.8900000000000003`

## Inference decision

For each Source-1 entity:

1. retrieve up to 10 candidates
2. bypass ML only for a unique exact normalized-name candidate
3. compute the 13 trained features for the remaining candidates
4. obtain match probabilities from the saved classifier
5. apply entity-level decision rules
6. write the accepted IDs

Current entity decision parameters:

- match threshold: saved model threshold
- no-match maximum probability: 0.42
- minimum single-match margin: 0.06
- minimum confident single match: 0.88

These rules are inference-time behavior. They do not modify model weights.

## Important metric distinction

A pairwise precision near 99% does **not** imply a 99% end-to-end entity-resolution score.

The final entity score depends on:

`blocking recall -> candidate ranking -> pairwise classifier -> entity decision`

If blocking misses a true match, the classifier never sees it. Therefore candidate recall must be measured separately.

## No retraining in the final tuning pass

The current optimization work is inference-only:

- improve deterministic candidate recall
- fix parsing/indexing bugs
- preserve the 13-feature model interface
- preserve the saved model
- keep retrieval small enough for the laptop runtime target

Retraining is not required for these changes.

## Tests

```powershell
pytest tests/test_pipeline.py -q
pytest tests/ -q
```

For the actual final pipeline, use the training benchmark described in [Final Inference](FINAL_INFERENCE.md).
