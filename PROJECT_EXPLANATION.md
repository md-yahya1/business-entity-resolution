# Complete Beginner's Guide to Business Entity Resolution Pipeline

---

## 1. Introduction: What is Business Entity Resolution? (Problem Statement)

Imagine Amazon receives business records from three different supplier databases. 

In Database A, a seller is registered as:
- **Name:** `Walmart Inc.`
- **Address:** `123 Main St, Seattle, WA`
- **Country:** `USA`

In Database B, the exact same seller is registered as:
- **Name:** `Wal-Mart Stores Co.`
- **Address:** `123 Main Street, Suite 100, Seattle`
- **Country:** `US`

### The Problem
Are these two records describing the **same real-world business** or two completely different businesses?
To a human, it is obvious that both refer to Walmart. However, a computer looking at raw text sees two different strings (`"Walmart Inc." != "Wal-Mart Stores Co."`).

When dealing with millions of records across multiple global data sources, manual verification is impossible. Simple database lookups (exact string match) fail because of:
1. **Punctuation & Typos:** `"Wal-Mart"` vs `"Walmart"`
2. **Abbreviations:** `"St"` vs `"Street"`, `"Co."` vs `"Company"`, `"Inc"` vs `"Incorporated"`
3. **Country Variations:** `"USA"` vs `"US"` vs `"United States"`
4. **Address Variations:** Extra suite numbers or missing zip codes.

### The Goal of this Project
**Business Entity Resolution** is an automated Machine Learning pipeline that takes pairs of records from different databases, compares their attributes (Name, Address, Country), computes numerical similarity scores, and feeds them into a trained Machine Learning classifier to output:
- **`is_match`**: `1` (Same business entity) or `0` (Different business entities).
- **`match_probability`**: A confidence score between `0.0` (0% match confidence) and `1.0` (100% match confidence).

---

## 2. High-Level Architecture & Pipeline Workflow

Our machine learning pipeline follows a 5-stage modular architecture:

```
+------------------+     +------------------+     +------------------+
|   1. Raw Data    | --> |  2. Preprocessing| --> |  3. Candidate    |
|   (Source TSVs)  |     |  (Clean text)    |     |  Gen / Blocking  |
+------------------+     +------------------+     +------------------+
                                                           |
                                                           v
+------------------+     +------------------+     +------------------+
|  5. Prediction / | <-- | 4. ML Model      | <-- | 3. Feature       |
|  Inference       |     |    Training      |     |    Engineering   |
+------------------+     +------------------+     +------------------+
```

1. **Preprocessing (`preprocessing.py`):** Clean and normalize text strings (lowercase, remove special characters, trim whitespace).
2. **Candidate Generation & Blocking (`candidate_generator.py`):** Instead of comparing *every* record against *every* other record (which would require billions of comparisons), we use **blocking** (grouping records by country and first letter) to quickly filter candidates.
3. **Feature Engineering (`pair_features.py`):** Convert pair of text strings into 19 mathematical similarity metrics (fuzzy ratios, Jaro-Winkler distance, token overlaps, exact matches).
4. **Model Training & Threshold Tuning (`models/train.py` & `scripts/train.py`):** Train classifiers (Logistic Regression, Random Forest, Gradient Boosting), evaluate them using leakage-safe group splitting, tune the decision threshold on validation set, and select the best model.
5. **Inference & Prediction (`models/predict.py` & `scripts/predict.py`):** Load saved model artifacts and predict matches on new target dataset pairs.

---

## 3. Step-by-Step Code Walkthrough & Explanations

### Step 1: Preprocessing Module
**File:** [`code/business_entity_resolution/src/preprocessing.py`](file:///c:/Users/Mohammed%20yahya/OneDrive/Desktop/Amazon/business-entity-resolution/code/business_entity_resolution/src/preprocessing.py)

Before comparing any names or addresses, we must clean them so minor formatting variations do not spoil string comparison.

```python
def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    value = str(value)
    value = unicodedata.normalize("NFKC", value)  # Standardize unicode characters
    value = value.lower()                          # Convert to lowercase
    value = re.sub(r"[^\w\s]", " ", value)        # Replace punctuation with spaces
    value = re.sub(r"\s+", " ", value)             # Collapse multiple spaces into single space
    return value.strip()
```

- **Line 31:** `unicodedata.normalize("NFKC", value)` handles fancy accented letters or unicode symbols (e.g. converting `é` or `ñ` into clean standard unicode).
- **Line 34:** `.lower()` ensures `"WALMART"` and `"walmart"` match.
- **Line 37:** `re.sub(r"[^\w\s]", " ", value)` converts commas, hyphens, and dots to spaces (so `"wal-mart"` becomes `"wal mart"`).
- **Line 115-152 (`preprocess_dataframe`):** Applies normalization across `business_name`, `business_address`, and `country` columns, adding normalized helper columns to the Pandas DataFrame.

---

### Step 2: Candidate Generation & Blocking
**File:** [`code/business_entity_resolution/src/blocking/candidate_generator.py`](file:///c:/Users/Mohammed%20yahya/OneDrive/Desktop/Amazon/business-entity-resolution/code/business_entity_resolution/src/blocking/candidate_generator.py)

If we have 100,000 records in Source 1 and 100,000 records in Source 2, comparing every pair requires $100,000 \times 100,000 = 10,000,000,000$ (10 Billion) pairs! This is computationally impossible.

**Blocking solution:** We place records into "blocks" using a fast rule. We only compare records within the same block.

```python
# Create blocking key using normalized country + first letter of business name
first_char = name_norm[0] if len(name_norm) > 0 else ""
key = f"{c}_{first_char}"  # Example key: "usa_w"
```

- **Lines 77-86:** If Record A is in `"usa"` and starts with `"w"`, its block key is `"usa_w"`. Only other records in `"usa_w"` will be compared against it.
- **Leakage Prevention (`entity_group_id`):** Lines 55-64 assign an `entity_group_id` to connected entities. When splitting dataset into Train/Validation/Test splits later, all pairs belonging to the same entity group stay together. This prevents **data leakage** (where the model accidentally memorizes a business name seen in training).

---

### Step 3: Feature Engineering
**File:** [`code/business_entity_resolution/src/features/pair_features.py`](file:///c:/Users/Mohammed%20yahya/OneDrive/Desktop/Amazon/business-entity-resolution/code/business_entity_resolution/src/features/pair_features.py)

Machine Learning models operate on numbers, not raw text strings. The feature extractor converts a candidate pair into 19 numerical similarity features (`FEATURE_NAMES`):

1. **Fuzzy Ratio Features (`RapidFuzz`):**
   - `name_ratio`: Basic Levenshtein character similarity score ($0.0$ to $1.0$).
   - `name_partial_ratio`: Finds highest similarity for substring matches (useful if one record has extra words like `"Walmart"` vs `"Walmart Supercenter"`).
   - `name_token_sort_ratio`: Sorts words alphabetically before comparing (`"Inc Walmart"` vs `"Walmart Inc"` scores $1.0$).
   - `name_token_set_ratio`: Ignores duplicate words between strings.
2. **Phonetic & Distance Similarity:**
   - `name_jw_similarity`, `address_jw_similarity`: **Jaro-Winkler similarity**, which penalizes errors more at the beginning of words than at the end (great for name matching).
3. **Exact & Flag Features:**
   - `country_exact_match`, `name_exact_match`, `address_exact_match`: $1.0$ if normalized strings are identical, else $0.0$.
   - `name_missing`, `address_missing`, `country_missing`: Flags if data was absent.
4. **Token Overlap (Jaccard Similarity):**
   - `_jaccard_overlap` (Lines 35-43): Measures word intersection divided by word union ($\frac{|A \cap B|}{|A \cup B|}$).
5. **Length Differences:**
   - `name_char_len_diff`, `address_char_len_diff`: Absolute character length differences.

---

### Step 4: Model Training, Evaluation, and Artifact Saving
**Files:** [`code/business_entity_resolution/src/models/train.py`](file:///c:/Users/Mohammed%20yahya/OneDrive/Desktop/Amazon/business-entity-resolution/code/business_entity_resolution/src/models/train.py) & [`scripts/train.py`](file:///c:/Users/Mohammed%20yahya/OneDrive/Desktop/Amazon/business-entity-resolution/scripts/train.py)

#### Group-based Data Splitting
```python
gss_test = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
```
In `split_data_by_group()` (Lines 20-56), we split data into **70% Train, 15% Validation, 15% Test** using `GroupShuffleSplit` on `entity_group_id`.

#### Baseline Model Comparison
In `train_and_evaluate_models()` (Lines 69-79), 3 classifiers are trained and compared:
1. **Logistic Regression:** Simple linear baseline.
2. **Random Forest Classifier:** Ensemble of decision trees (`n_estimators=100`, `max_depth=12`).
3. **Gradient Boosting Classifier:** Sequential boosted decision trees (`n_estimators=100`, `max_depth=5`, `learning_rate=0.1`).

#### Decision Threshold Tuning
Normally, classifiers use a default decision threshold of $0.5$ ($P(\text{match}) \ge 0.5 \Rightarrow \text{Match}$). However, entity matching datasets are highly imbalanced (far more non-matches than matches).
Using `find_optimal_threshold()` in [`metrics.py`](file:///c:/Users/Mohammed%20yahya/OneDrive/Desktop/Amazon/business-entity-resolution/code/business_entity_resolution/src/evaluation/metrics.py#L51-L82), we scan thresholds from $0.05$ to $0.95$ on the **validation set** to find the exact threshold that maximizes the **F1-Score**:

$$\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

#### Artifact Saving
`save_model_artifacts()` (Lines 128-184) exports three critical files:
- `models/entity_resolution_model.joblib`: Serialized trained model weights.
- `models/feature_config.json`: Ordered list of expected input features.
- `models/model_metadata.json`: Model architecture name, hyperparameters, software package versions, evaluation metrics, and optimal decision threshold.

---

### Step 5: Model Inference & Prediction
**Files:** [`code/business_entity_resolution/src/models/predict.py`](file:///c:/Users/Mohammed%20yahya/OneDrive/Desktop/Amazon/business-entity-resolution/code/business_entity_resolution/src/models/predict.py) & [`scripts/predict.py`](file:///c:/Users/Mohammed%20yahya/OneDrive/Desktop/Amazon/business-entity-resolution/scripts/predict.py)

During inference on unlabelled data:
1. Candidate pairs are passed to `predict_candidate_pairs()`.
2. Feature extraction is automatically run if features are not already calculated.
3. The saved model predicts match probabilities $P(\text{match})$.
4. The stored optimal threshold is applied:
   ```python
   preds = (probs >= threshold).astype(int)
   ```
5. Results are written to a tab-separated TSV file (`output/predictions.tsv`) with columns `match_probability` and `is_match`.

---

## 4. Ideas to Further Tune & Improve the Model

To push model performance from baseline to state-of-the-art, consider the following enhancements:

### A. Advanced Blocking (Recall & Efficiency)
- **Multi-Pass Blocking:** Currently, blocking uses `country + first_character`. If a business name has a typo in the first letter (`"Vandals Inc"` vs `"Wandals Inc"`), blocking misses it. Multi-pass blocking combines multiple rules (e.g. `country + zip_code`, `soundex(name)`, `TF-IDF top word`).
- **Dense Vector / Embedding Blocking:** Use lightweight sentence embeddings (`all-MiniLM-L6-v2`) with FAISS or Annoy vector indices to retrieve top-$k$ nearest neighbors for each business entity.

### B. Richer Feature Engineering
- **TF-IDF & Cosine Similarity:** Compute character n-gram TF-IDF similarity (captures shared rare words like unique corporate names).
- **Transformer Embeddings Similarity:** Extract semantic embeddings for business names using domain-specific pretrained models and compute cosine similarity.
- **Legal Entity Term Extraction:** Automatically detect and strip corporate designators (`Inc`, `LLC`, `GmbH`, `Corp`, `Ltd`, `S.A.`) into a separate categorical feature to compare entity legal status.
- **Geographic Distance:** Convert addresses/zip codes to latitude & longitude and compute Haversine spatial distance.

### C. Advanced Machine Learning Algorithms
- **Gradient Boosted Trees (XGBoost / LightGBM / CatBoost):** Replace standard scikit-learn GradientBoosting with `XGBClassifier` or `LGBMClassifier`, which handle tabular similarity features faster and support GPU acceleration.
- **Hyperparameter Optimization with Optuna:** Automate tuning of `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, and `min_child_weight`.
- **Deep Learning Cross-Encoders:** Fine-tune a DeBERTa-v3 or RoBERTa Cross-Encoder directly on pairs of raw text:
  `"[CLS] Name1 Address1 Country1 [SEP] Name2 Address2 Country2 [SEP]"`.

---

## 5. How to Run Training, Inference & Submit the Model

### 1. Execute Model Training
Run the training pipeline script. It will generate candidate features (if missing), execute group splitting, compare models, tune decision thresholds, and output performance metrics:

```bash
python scripts/train.py --input artifacts/candidates/training_features.parquet --output models/entity_resolution_model.joblib
```

**Expected console output:**
```
==================================================
      BUSINESS ENTITY RESOLUTION MODEL TRAINING   
==================================================
Executing leakage-safe group split based on entity_group_id...
Train size: 28000
Validation size: 6000
Test size: 6000

Training and evaluating baseline models...
SELECTED FINAL MODEL: RandomForest (or GradientBoosting)
Optimal Decision Threshold: 0.4200

--- TEST SET PERFORMANCE ---
Accuracy:  0.9650
Precision: 0.9420
Recall:    0.9310
F1-Score:  0.9365
ROC-AUC:   0.9880

Model artifacts successfully saved:
  Model:   models/entity_resolution_model.joblib
  Config:  models/feature_config.json
  Metadata:models/model_metadata.json
```

---

### 2. Run Inference / Prediction
Run the prediction script on test set candidate pairs:

```bash
python scripts/predict.py --model models/entity_resolution_model.joblib --input artifacts/candidates/test_features.parquet --output output/predictions.tsv
```

---

### 3. How to Submit the Model & Predictions

When submitting your model solution to a competition, client, or production repository:

1. **Output Predictions File:**
   - Verify `output/predictions.tsv` contains the required columns (typically `entity_id_1`, `entity_id_2`, `match_probability`, `is_match`).
   - Check that there are no missing values or NaN rows.

2. **Model Bundle / Artifacts:**
   Include the complete `models/` directory:
   - `entity_resolution_model.joblib` (Trained model parameters)
   - `feature_config.json` (Feature names & order)
   - `model_metadata.json` (Decision threshold, metrics, dependencies)

3. **Reproducibility Verification:**
   Run tests to ensure everything functions properly before final submission:
   ```bash
   pytest tests/
   ```
