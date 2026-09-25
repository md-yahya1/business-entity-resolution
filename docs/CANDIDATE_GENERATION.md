You are responsible for documenting and implementing the CANDIDATE GENERATION stage of a machine-learning Entity Resolution project.

PROJECT:
Business Entity Resolution

PIPELINE:

Raw Dataset
    ↓
Data Preprocessing
    ↓
Processed Entity Records
    ↓
Candidate Generation
    ↓
Candidate Pairs
    ↓
Feature Engineering
    ↓
Model Training
    ↓
Entity Matching

Your responsibility is the candidate-generation stage.

==================================================
OBJECTIVE
==================================================

Candidate generation should reduce the number of possible entity pairs that need expensive comparison.

Instead of comparing every record against every other record:

N × M

generate a smaller set of plausible candidate pairs.

The candidate-generation stage must prioritize high recall while controlling the number of generated pairs.

Do NOT make final entity-match decisions here unless explicitly required.

==================================================
INPUT
==================================================

Read the actual output produced by DATA_PREPROCESSING.md.

Do not invent filenames or columns.

Document:

- Input file
- Input schema
- Entity identifiers
- Available normalized fields
- Record counts

==================================================
CANDIDATE GENERATION METHODS
==================================================

Inspect the dataset and determine appropriate blocking/candidate-generation strategies.

Potential methods include:

- Exact blocking
- Prefix blocking
- Token-based blocking
- Phonetic blocking
- Character n-gram blocking
- Postal-code blocking
- City/state blocking
- Domain blocking
- Multi-pass blocking

Do not blindly implement every method.

Choose methods based on the actual dataset.

Explain:

- Why each blocking key is used
- Expected recall
- Expected candidate reduction
- Risks of false negatives

If multiple blocking strategies are used, union their candidates.

==================================================
IMPORTANT PRINCIPLE
==================================================

Candidate generation should generally favor RECALL.

It is acceptable to generate additional candidate pairs.

It is NOT acceptable to aggressively filter candidates in a way that causes true matches to disappear.

Document this tradeoff.

==================================================
OUTPUT
==================================================

Generate a candidate-pair dataset containing at minimum:

- left/entity record ID
- right/entity record ID
- blocking method(s)
- optional blocking key
- optional metadata useful for downstream feature engineering

Do not include the target label unless it is explicitly available and needed for training.

Example:

candidate_pairs.parquet

Schema:

left_id
right_id
blocking_method
blocking_key

Use the actual dataset schema rather than blindly copying this example.

==================================================
DEDUPLICATION
==================================================

Ensure:

- (A, B) and (B, A) are not duplicated when entity matching is symmetric.
- Self-pairs are removed.
- Duplicate candidate pairs from multiple blocking strategies are deduplicated.

==================================================
EVALUATION
==================================================

If labeled training data is available, measure:

- Candidate recall
- Number of generated pairs
- Candidate reduction ratio
- Average candidates per entity
- Distribution of candidate counts

Candidate Recall:

true matching pairs retained
-----------------------------
true matching pairs available

Report this clearly.

==================================================
IMPLEMENTATION
==================================================

Create:

scripts/generate_candidates.py

Example interface:

python scripts/generate_candidates.py \
    --input artifacts/processed/train_processed.parquet \
    --output artifacts/candidates/candidate_pairs.parquet

Do not hardcode Windows absolute paths.

Use configurable paths.

==================================================
DOCUMENTATION
==================================================

Create:

docs/CANDIDATE_GENERATION.md

Required structure:

# Candidate Generation

## 1. Purpose

## 2. Position in Pipeline

## 3. Input Data

## 4. Input Schema

## 5. Candidate Generation Strategy

## 6. Blocking Keys

## 7. Blocking Algorithms

## 8. Candidate Deduplication

## 9. Self-Match Removal

## 10. Candidate Recall

## 11. Candidate Reduction

## 12. Output Data

## 13. Output Schema

## 14. Execution

## 15. Validation

## 16. Known Limitations

## 17. Handoff to Feature Engineering / Model Training

At the end provide:

INPUT:
<exact file>

PROCESS:
<exact candidate generation stages>

OUTPUT:
<exact file>

NEXT STAGE:
Feature Engineering / Model Training

Do not invent statistics. Calculate them from the actual data.