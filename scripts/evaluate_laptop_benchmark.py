import pandas as pd

GROUND_TRUTH = "dataset/train/train_ground_truth.tsv"
CANDIDATES = "output/laptop_train_benchmark/candidate_pairs.tsv"
PREDICTIONS = "output/laptop_train_benchmark/matching_results.tsv"


def parse_ids(value):
    if pd.isna(value) or not str(value).strip():
        return set()

    return {
        x.strip()
        for x in str(value).split(",")
        if x.strip()
    }


print("Loading files...")

gt = pd.read_csv(GROUND_TRUTH, sep="\t")
candidates = pd.read_csv(CANDIDATES, sep="\t")
predictions = pd.read_csv(PREDICTIONS, sep="\t")

# Only evaluate the 5,000 S1 entities that were benchmarked
benchmark_ids = set(candidates["source1_entity_id"])

gt = gt[gt["source1_entity_id"].isin(benchmark_ids)]

gt_map = dict(
    zip(
        gt["source1_entity_id"],
        gt["matched_entity_ids"].map(parse_ids)
    )
)

candidate_map = dict(
    zip(
        candidates["source1_entity_id"],
        candidates["candidate_entity_ids"].map(parse_ids)
    )
)

prediction_map = dict(
    zip(
        predictions["source1_entity_id"],
        predictions["matched_entity_ids"].map(parse_ids)
    )
)


# ---------------------------------------------------------
# 1. CANDIDATE RECALL
# ---------------------------------------------------------

total_true_matches = 0
true_matches_found_by_blocking = 0
entities_with_all_matches_found = 0

for entity_id in benchmark_ids:

    true_matches = gt_map.get(entity_id, set())
    candidate_set = candidate_map.get(entity_id, set())

    found = true_matches & candidate_set

    total_true_matches += len(true_matches)
    true_matches_found_by_blocking += len(found)

    if true_matches and true_matches.issubset(candidate_set):
        entities_with_all_matches_found += 1

    elif not true_matches:
        entities_with_all_matches_found += 1


candidate_recall = (
    true_matches_found_by_blocking / total_true_matches
    if total_true_matches
    else 0
)


# ---------------------------------------------------------
# 2. FINAL MODEL METRICS
# ---------------------------------------------------------

total_tp = 0
total_fp = 0
total_fn = 0

entity_f05_scores = []

for entity_id in benchmark_ids:

    true_matches = gt_map.get(entity_id, set())
    predicted = prediction_map.get(entity_id, set())

    tp = len(true_matches & predicted)
    fp = len(predicted - true_matches)
    fn = len(true_matches - predicted)

    total_tp += tp
    total_fp += fp
    total_fn += fn

    # Per-entity F0.5
    if tp == 0:
        if len(true_matches) == 0 and len(predicted) == 0:
            f05 = 1.0
        else:
            f05 = 0.0
    else:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

        beta2 = 0.25

        f05 = (
            (1 + beta2) * precision * recall
            / (beta2 * precision + recall)
        )

    entity_f05_scores.append(f05)


precision = (
    total_tp / (total_tp + total_fp)
    if total_tp + total_fp
    else 0
)

recall = (
    total_tp / (total_tp + total_fn)
    if total_tp + total_fn
    else 0
)

f05_micro = (
    (1.25 * precision * recall)
    / (0.25 * precision + recall)
    if precision + recall
    else 0
)

f05_macro = sum(entity_f05_scores) / len(entity_f05_scores)


# ---------------------------------------------------------
# 3. PRINT RESULTS
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("LAPTOP FINAL — TRAINING BENCHMARK")
print("=" * 60)

print(f"Benchmark entities:              {len(benchmark_ids):,}")

print("\nBLOCKING")
print("-" * 60)
print(f"Total true matches:              {total_true_matches:,}")
print(f"True matches found in candidates:{true_matches_found_by_blocking:,}")
print(f"Candidate Recall:                {candidate_recall:.4%}")
print(
    f"Entities with all true matches:  "
    f"{entities_with_all_matches_found:,}"
)

print("\nFINAL MODEL")
print("-" * 60)
print(f"True Positives:                  {total_tp:,}")
print(f"False Positives:                 {total_fp:,}")
print(f"False Negatives:                 {total_fn:,}")
print(f"Precision:                       {precision:.4%}")
print(f"Recall:                          {recall:.4%}")
print(f"F0.5 (micro):                    {f05_micro:.4%}")
print(f"F0.5 (macro):                    {f05_macro:.4%}")

print("=" * 60)