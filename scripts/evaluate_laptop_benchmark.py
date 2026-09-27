import argparse

import pandas as pd


def parse_ids(value):
    if pd.isna(value) or not str(value).strip():
        return set()
    return {x.strip() for x in str(value).split(",") if x.strip()}


def main():
    ap = argparse.ArgumentParser(description="Evaluate an entity-resolution benchmark.")
    ap.add_argument("--ground-truth", default="dataset/train/train_ground_truth.tsv")
    ap.add_argument("--candidate-pairs", default="output/final_fast_benchmark/candidate_pairs.tsv")
    ap.add_argument("--matching-results", default="output/final_fast_benchmark/matching_results.tsv")
    args = ap.parse_args()

    print("Loading files...")
    gt = pd.read_csv(args.ground_truth, sep="\t")
    candidates = pd.read_csv(args.candidate_pairs, sep="\t")
    predictions = pd.read_csv(args.matching_results, sep="\t")

    benchmark_ids = set(candidates["source1_entity_id"])
    gt = gt[gt["source1_entity_id"].isin(benchmark_ids)]

    gt_map = dict(zip(gt["source1_entity_id"], gt["matched_entity_ids"].map(parse_ids)))
    candidate_map = dict(zip(candidates["source1_entity_id"], candidates["candidate_entity_ids"].map(parse_ids)))
    prediction_map = dict(zip(predictions["source1_entity_id"], predictions["matched_entity_ids"].map(parse_ids)))

    total_true_matches = 0
    true_matches_found_by_blocking = 0
    entities_with_all_matches_found = 0

    for entity_id in benchmark_ids:
        true_matches = gt_map.get(entity_id, set())
        candidate_set = candidate_map.get(entity_id, set())
        found = true_matches & candidate_set
        total_true_matches += len(true_matches)
        true_matches_found_by_blocking += len(found)
        if true_matches.issubset(candidate_set):
            entities_with_all_matches_found += 1

    candidate_recall = (
        true_matches_found_by_blocking / total_true_matches
        if total_true_matches else 0
    )

    total_tp = total_fp = total_fn = 0
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

        if tp == 0:
            f05 = 1.0 if not true_matches and not predicted else 0.0
        else:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f05 = (1.25 * precision * recall) / (0.25 * precision + recall)

        entity_f05_scores.append(f05)

    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0
    recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0
    f05_micro = (
        (1.25 * precision * recall) / (0.25 * precision + recall)
        if precision + recall else 0
    )
    f05_macro = sum(entity_f05_scores) / len(entity_f05_scores)

    print("\n" + "=" * 60)
    print("LAPTOP FINAL — TRAINING BENCHMARK")
    print("=" * 60)
    print(f"Benchmark entities:               {len(benchmark_ids):,}")
    print("\nBLOCKING")
    print("-" * 60)
    print(f"Total true matches:               {total_true_matches:,}")
    print(f"True matches found in candidates: {true_matches_found_by_blocking:,}")
    print(f"Candidate Recall:                 {candidate_recall:.4%}")
    print(f"Entities with all true matches:   {entities_with_all_matches_found:,}")
    print("\nFINAL MODEL")
    print("-" * 60)
    print(f"True Positives:                   {total_tp:,}")
    print(f"False Positives:                  {total_fp:,}")
    print(f"False Negatives:                  {total_fn:,}")
    print(f"Precision:                        {precision:.4%}")
    print(f"Recall:                           {recall:.4%}")
    print(f"F0.5 (micro):                     {f05_micro:.4%}")
    print(f"F0.5 (macro):                     {f05_macro:.4%}")
    print("=" * 60)


if __name__ == "__main__":
    main()
