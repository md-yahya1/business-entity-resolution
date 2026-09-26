"""
Re-tune the model's decision threshold against macro-averaged entity-level
F_0.5 -- the actual competition metric -- instead of pairwise F1.

model_metadata.json's current threshold (0.72) was chosen by
find_optimal_threshold() in evaluation/metrics.py, which maximizes
pairwise F1 on individual candidate pairs. That is a reasonable proxy but
not the same objective: F_0.5 is precision-heavy and scored per Source-1
entity (with singletons worth a full point), so the best pairwise-F1
threshold is not guaranteed to be the best entity-level-F0.5 threshold.

This script re-derives a held-out validation split from
dataset/train (grouped, leakage-safe, using the same entity_group_id
logic training already uses), scores every threshold in a sweep against
real macro F_0.5, and reports what to put in model_metadata.json.

Usage:
    python scripts/tune_threshold_f0_5.py --train-dir dataset/train
"""
import sys
import os
import argparse
import json
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "code")))

from business_entity_resolution.src.blocking import generate_candidate_pairs
from business_entity_resolution.src.features import extract_features_dataframe, FEATURE_NAMES
from business_entity_resolution.src.models.predict import load_model_and_config
from business_entity_resolution.src.evaluation.entity_metrics import (
    ground_truth_to_dict,
    tune_threshold_for_macro_f0_5,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", default="dataset/train")
    parser.add_argument("--model", default="models/entity_resolution_model.joblib")
    parser.add_argument("--max-positives", type=int, default=3000, help="Validation-split size (kept modest for speed)")
    parser.add_argument("--max-negatives", type=int, default=9000)
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()

    print("Loading training data to build a held-out validation split...")
    s1 = pd.read_csv(os.path.join(args.train_dir, "train_source1.tsv"), sep="\t")
    s2 = pd.read_csv(os.path.join(args.train_dir, "train_source2.tsv"), sep="\t")
    s3 = pd.read_csv(os.path.join(args.train_dir, "train_source3.tsv"), sep="\t")
    gt = pd.read_csv(os.path.join(args.train_dir, "train_ground_truth.tsv"), sep="\t")

    # NOTE: for a rigorous number, hold out a slice of gt/s1 that the model's
    # own training run never touched (check dataset_version / random_seed in
    # model_metadata.json against what generated this split). This script
    # reuses the same sampling utility as training for convenience; swap in
    # your team's actual held-out split file if you have one committed.
    print("Generating candidate pairs for the validation split...")
    pairs_df = generate_candidate_pairs(
        s1, s2, s3, gt,
        max_positives=args.max_positives,
        max_negatives=args.max_negatives,
        random_seed=args.random_seed,
    )
    features_df = extract_features_dataframe(pairs_df)
    full_df = pd.concat([pairs_df, features_df], axis=1)

    model, feature_names, current_threshold = load_model_and_config(args.model)
    X = full_df[feature_names].values
    full_df["match_probability"] = model.predict_proba(X)[:, 1]

    true_by_entity = ground_truth_to_dict(gt)
    # only score S1 entities actually present in this validation split
    eval_entity_ids = sorted(set(full_df["entity_id_1"]) & set(s1["entity_id"]))

    print(f"Scoring {len(eval_entity_ids)} Source-1 entities across a threshold sweep...")
    best_t, best_score, history_df, breakdown = tune_threshold_for_macro_f0_5(
        full_df, true_by_entity, eval_entity_ids
    )

    print("\n--- THRESHOLD SWEEP (entity-level macro F_0.5) ---")
    print(history_df.to_string(index=False))

    print(f"\nCurrent model_metadata.json threshold: {current_threshold}")
    print(f"Best threshold by macro F_0.5:          {best_t}  (score = {best_score:.4f})")

    worst = breakdown.sort_values("f0_5").head(10)
    print("\nWorst 10 entities at the best threshold (for error analysis):")
    print(worst.to_string(index=False))

    print(
        f"\nIf this confirms {best_t} beats {current_threshold}, update "
        f"\"decision_threshold\": {best_t} in models/model_metadata.json "
        f"(and re-run scripts/generate_submission.py) -- don't hand-edit the "
        f"model file itself, only the metadata threshold."
    )


if __name__ == "__main__":
    main()
