# Model Training

## 1. Purpose
The purpose of the Model Training stage is to construct, evaluate, and save a binary classification machine learning model that determines whether a candidate pair of business entity records represents the exact same real-world business entity.

## 2. Position in Pipeline
The Model Training stage sits downstream of Candidate Generation & Feature Engineering, and upstream of Entity Matching:

```
Raw Data
   ↓
Preprocessing
   ↓
Candidate Generation
   ↓
Candidate Pairs
   ↓
Feature Engineering
   ↓
Model Training [THIS STAGE]
   ↓
Trained Model
   ↓
Entity Matching
```

## 3. Input Data
The model training process consumes:
1. **Candidate-pair dataset & features artifact**: `artifacts/candidates/training_features.parquet` (or `artifacts/candidates/training_features.tsv`).
2. **Ground-truth labels**: Derived from `dataset/train/train_ground_truth.tsv`.
3. **Entity record sources**: `dataset/train/train_source1.tsv`, `dataset/train/train_source2.tsv`, and `dataset/train/train_source3.tsv`.

## 4. Input Schema
Each candidate record pair contains the raw and normalized attributes along with engineered comparison features:

| Column Name | Type | Description |
|---|---|---|
| `entity_id_1` | string | Unique identifier for entity record 1 |
| `business_name_1` | string | Business name for record 1 |
| `business_address_1` | string | Business address for record 1 |
| `country_1` | string | Country code for record 1 |
| `entity_id_2` | string | Unique identifier for entity record 2 |
| `business_name_2` | string | Business name for record 2 |
| `business_address_2` | string | Business address for record 2 |
| `country_2` | string | Country code for record 2 |
| `label` | integer | Ground-truth binary label (1 = match, 0 = non-match) |
| `entity_group_id` | integer | Entity cluster ID used for leakage-safe grouping |

## 5. Feature Engineering
Comparison features measure string similarity, token overlap, exact attribute equality, and missingness across business entity attributes (`business_name`, `business_address`, `country`). All string comparisons utilize normalized fields produced by the preprocessing module.

## 6. Feature Definitions
The 19 engineered features used by the model are:

1. `name_ratio`: RapidFuzz normalized Levenshtein ratio on business names [0.0, 1.0]
2. `name_partial_ratio`: RapidFuzz partial substring similarity on business names [0.0, 1.0]
3. `name_token_sort_ratio`: RapidFuzz token-sorted ratio handling word order permutations [0.0, 1.0]
4. `name_token_set_ratio`: RapidFuzz token-set similarity handling duplicated/extra words [0.0, 1.0]
5. `name_jw_similarity`: Jaro-Winkler similarity on business names [0.0, 1.0]
6. `address_ratio`: RapidFuzz ratio on normalized addresses [0.0, 1.0]
7. `address_partial_ratio`: RapidFuzz partial substring ratio on addresses [0.0, 1.0]
8. `address_token_set_ratio`: RapidFuzz token-set ratio on addresses [0.0, 1.0]
9. `address_jw_similarity`: Jaro-Winkler similarity on addresses [0.0, 1.0]
10. `country_exact_match`: Binary indicator (1.0 if both non-empty country strings match, else 0.0)
11. `country_missing`: Binary indicator (1.0 if either country is missing/empty, else 0.0)
12. `name_exact_match`: Binary indicator (1.0 if normalized business names match exactly, else 0.0)
13. `address_exact_match`: Binary indicator (1.0 if normalized addresses match exactly, else 0.0)
14. `name_token_overlap`: Token Jaccard similarity index on business name words [0.0, 1.0]
15. `address_token_overlap`: Token Jaccard similarity index on address words [0.0, 1.0]
16. `name_char_len_diff`: Absolute difference in character length of normalized business names
17. `address_char_len_diff`: Absolute difference in character length of normalized addresses
18. `name_missing`: Binary indicator (1.0 if either business name is missing, else 0.0)
19. `address_missing`: Binary indicator (1.0 if either address is missing, else 0.0)

## 7. Label Definition
- `1` = Positive match (both candidate records refer to the same real-world business entity).
- `0` = Negative match (candidate records represent different business entities).

Ground-truth labels were assigned by mapping `source1_entity_id` to its matched IDs in `train_ground_truth.tsv`. Negative pairs were generated via multi-pass blocking (same country + prefix/token key) between non-matching records.

## 8. Dataset Split
To avoid data leakage (where records belonging to the same underlying entity cluster appear in both training and test splits), a **Group-Based Split** (`GroupShuffleSplit`) on `entity_group_id` was executed:

- **Split Ratio**: 70% Train, 15% Validation, 15% Test
- **Random Seed**: `42`
- **Train Size**: 22,537 candidate pairs
- **Validation Size**: 4,695 candidate pairs
- **Test Size**: 4,888 candidate pairs

## 9. Class Distribution
The overall dataset contains 32,120 candidate pairs:
- **Positive Examples (label = 1)**: 18,331 (57.07%)
- **Negative Examples (label = 0)**: 13,789 (42.93%)

Distribution across splits:
- **Train Set**: 22,537 pairs
- **Validation Set**: 4,695 pairs
- **Test Set**: 4,888 pairs

## 10. Models Evaluated
Three baseline models were evaluated:
1. **Logistic Regression**: Baseline linear model with balanced class weighting.
2. **Random Forest Classifier**: Non-linear ensemble model (100 estimators, max depth 12).
3. **Gradient Boosting Classifier**: Sequential boosting tree model (100 estimators, max depth 5, learning rate 0.1).

## 11. Evaluation Metrics
Models were compared using:
- Precision
- Recall
- F1-Score
- Accuracy
- ROC-AUC
- PR-AUC
- Confusion Matrix

## 12. Model Selection Criteria
The primary metric for model selection was **Validation F1-Score** after threshold tuning on the validation set, ensuring an optimal balance between precision and recall.

## 13. Final Model
**Gradient Boosting Classifier** was selected as the final model.

Validation Performance Comparison:
- **Logistic Regression**: Default F1 = 0.9868 | Optimal (thresh=0.45) F1 = 0.9878
- **Random Forest**: Default F1 = 0.9956 | Optimal (thresh=0.68) F1 = 0.9961
- **Gradient Boosting**: Default F1 = 0.9957 | Optimal (thresh=0.72) F1 = **0.9965**

Test Set Performance (Final Model):
- **Accuracy**: 0.9939
- **Precision**: 0.9964
- **Recall**: 0.9929
- **F1-Score**: 0.9946
- **ROC-AUC**: 0.9998
- **PR-AUC**: 0.9999
- **Confusion Matrix**: `[[2074, 10], [20, 2784]]`

## 14. Hyperparameters
Final Gradient Boosting Classifier parameters:
- `n_estimators`: 100
- `max_depth`: 5
- `learning_rate`: 0.1
- `subsample`: 1.0
- `random_state`: 42

## 15. Decision Threshold
- **Default Threshold**: 0.50
- **Optimal Decision Threshold**: **0.72**
- **Methodology**: Evaluated decision thresholds from 0.05 to 0.95 on the **Validation Set** to maximize F1-score. The test set was NOT used for threshold tuning.

## 16. Model Artifact
Model artifacts are saved under `models/`:
- `models/entity_resolution_model.joblib`: Serialized Gradient Boosting model binary
- `models/feature_config.json`: Feature list and count configuration
- `models/model_metadata.json`: Model type, hyperparameters, evaluation metrics, decision threshold, and software dependency versions

## 17. Inference
Inference can be executed independently using `scripts/predict.py`:
```bash
python scripts/predict.py \
    --model models/entity_resolution_model.joblib \
    --input artifacts/candidates/test_features.parquet \
    --output output/predictions.tsv
```

## 18. Reproducibility
- **Python Version**: 3.11.9
- **Random Seed**: 42
- **Key Dependencies**: `scikit-learn==1.8.0`, `pandas==2.2.3`, `numpy==1.26.4`, `rapidfuzz==3.14.6`, `joblib==1.5.3`

## 19. Validation
The pipeline was verified through unit tests in `tests/`:
- `tests/test_preprocessing.py`: Validates string normalization
- `tests/test_features.py`: Validates feature engineering calculations
- `tests/test_pipeline.py`: Validates group splitting, model training, threshold tuning, and artifact serialization

All tests passed cleanly (`pytest tests/`).

## 20. Known Limitations
- Candidate generation hard negatives rely on country and name character prefix blocking.
- Missing values in address or business name reduce token overlap features to zero, relying more heavily on exact country matches and missing indicators.

## 21. Handoff / Deployment
The trained model artifact `models/entity_resolution_model.joblib` and feature configuration `models/feature_config.json` can be loaded independently in production. Incoming candidate pairs must pass through `extract_features_dataframe` to ensure identical feature ordering before calling `predict_proba`.

---

INPUT:
artifacts/candidates/training_features.parquet

FEATURES:
name_ratio, name_partial_ratio, name_token_sort_ratio, name_token_set_ratio, name_jw_similarity, address_ratio, address_partial_ratio, address_token_set_ratio, address_jw_similarity, country_exact_match, country_missing, name_exact_match, address_exact_match, name_token_overlap, address_token_overlap, name_char_len_diff, address_char_len_diff, name_missing, address_missing

MODEL:
GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)

OUTPUT:
models/entity_resolution_model.joblib

INFERENCE:
python scripts/predict.py --model models/entity_resolution_model.joblib --input artifacts/candidates/test_features.parquet --output output/predictions.tsv