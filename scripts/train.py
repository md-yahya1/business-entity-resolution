"""
Training script for Business Entity Resolution ML Model.

Usage:
    python scripts/train.py [--input artifacts/candidates/training_features.parquet] [--output models/entity_resolution_model.joblib]
"""
import sys
import os
import argparse
import json
import pandas as pd
import numpy as np

# Ensure code directory is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "code")))

from business_entity_resolution.src.blocking import generate_candidate_pairs
from business_entity_resolution.src.features import extract_features_dataframe, FEATURE_NAMES
from business_entity_resolution.src.models import (
    split_data_by_group,
    train_and_evaluate_models,
    save_model_artifacts,
)


def main():
    parser = argparse.ArgumentParser(description="Train Entity Resolution Machine Learning Model")
    parser.add_argument(
        "--input",
        type=str,
        default="artifacts/candidates/training_features.parquet",
        help="Path to training features parquet/tsv file (generated if missing)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/entity_resolution_model.joblib",
        help="Path to save trained model artifact"
    )
    parser.add_argument(
        "--max-positives",
        type=int,
        default=10000,
        help="Maximum positive ground truth pairs to sample"
    )
    parser.add_argument(
        "--max-negatives",
        type=int,
        default=30000,
        help="Maximum candidate negative pairs to sample"
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    print("==================================================")
    print("      BUSINESS ENTITY RESOLUTION MODEL TRAINING   ")
    print("==================================================")

    input_file = args.input
    os.makedirs(os.path.dirname(input_file), exist_ok=True)

    if os.path.exists(input_file):
        print(f"Loading pre-computed candidate feature dataset from: {input_file}")
        if input_file.endswith(".parquet"):
            df = pd.read_parquet(input_file)
        else:
            df = pd.read_csv(input_file, sep="\t")
    else:
        print("Candidate feature file not found. Generating candidate pairs from raw dataset...")
        train_dir = "dataset/train"
        s1_path = os.path.join(train_dir, "train_source1.tsv")
        s2_path = os.path.join(train_dir, "train_source2.tsv")
        s3_path = os.path.join(train_dir, "train_source3.tsv")
        gt_path = os.path.join(train_dir, "train_ground_truth.tsv")

        print("Reading raw source files and ground truth...")
        gt_all = pd.read_csv(gt_path, sep="\t")
        gt_sample = gt_all.dropna(subset=["matched_entity_ids"]).sample(
            n=min(args.max_positives, len(gt_all)), random_state=args.random_seed
        )

        needed_s1 = set(gt_sample["source1_entity_id"])
        needed_other = set()
        for m_str in gt_sample["matched_entity_ids"]:
            for m in m_str.split(","):
                if m.strip():
                    needed_other.add(m.strip())

        print(f"Loading targeted records for {len(needed_s1)} positive S1 anchors...")
        s1 = pd.read_csv(s1_path, sep="\t")
        s1 = s1[s1["entity_id"].isin(needed_s1)]

        s2 = pd.read_csv(s2_path, sep="\t")
        s2 = s2[s2["entity_id"].isin(needed_other) | (s2.index < 30000)]

        s3 = pd.read_csv(s3_path, sep="\t")
        s3 = s3[s3["entity_id"].isin(needed_other) | (s3.index < 30000)]

        print("Generating candidate pairs...")
        pairs_df = generate_candidate_pairs(
            s1, s2, s3, gt_sample,
            max_positives=args.max_positives,
            max_negatives=args.max_negatives,
            random_seed=args.random_seed
        )

        print("Extracting similarity comparison features...")
        features_df = extract_features_dataframe(pairs_df)

        df = pd.concat([pairs_df, features_df], axis=1)

        print(f"Saving extracted candidate features dataset to: {input_file}")
        if input_file.endswith(".parquet"):
            df.to_parquet(input_file, index=False)
        else:
            df.to_csv(input_file, sep="\t", index=False)

    print(f"\nTotal candidate dataset shape: {df.shape}")
    print(f"Class distribution:\n{df['label'].value_counts().to_dict()}")

    # Extract feature matrix
    features_df = df[FEATURE_NAMES]

    print("\nExecuting leakage-safe group split based on entity_group_id...")
    split_data = split_data_by_group(
        df, features_df, test_size=0.15, val_size=0.15, random_state=args.random_seed
    )

    print(f"Train size: {len(split_data['X_train'])}")
    print(f"Validation size: {len(split_data['X_val'])}")
    print(f"Test size: {len(split_data['X_test'])}")

    print("\nTraining and evaluating baseline models...")
    eval_results = train_and_evaluate_models(split_data, random_seed=args.random_seed)

    print("\n--- MODEL COMPARISON SUMMARY ---")
    for name, res in eval_results["all_model_results"].items():
        v_def = res["val_default_metrics"]
        v_opt = res["val_optimal_metrics"]
        print(f"Model: {name}")
        print(f"  Validation (thresh 0.5): F1={v_def['f1_score']:.4f}, Prec={v_def['precision']:.4f}, Rec={v_def['recall']:.4f}")
        print(f"  Validation (thresh {res['optimal_threshold']:.2f}): F1={v_opt['f1_score']:.4f}, Prec={v_opt['precision']:.4f}, Rec={v_opt['recall']:.4f}")

    best_name = eval_results["best_model_name"]
    best_model = eval_results["best_model"]
    best_thresh = eval_results["optimal_threshold"]
    test_metrics = eval_results["test_metrics"]

    print(f"\n==================================================")
    print(f"SELECTED FINAL MODEL: {best_name}")
    print(f"Optimal Decision Threshold: {best_thresh:.4f}")
    print(f"--- TEST SET PERFORMANCE ---")
    print(f"Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall:    {test_metrics['recall']:.4f}")
    print(f"F1-Score:  {test_metrics['f1_score']:.4f}")
    print(f"ROC-AUC:   {test_metrics['roc_auc']:.4f}" if test_metrics['roc_auc'] else "ROC-AUC: N/A")
    print(f"PR-AUC:    {test_metrics['pr_auc']:.4f}" if test_metrics['pr_auc'] else "PR-AUC: N/A")
    print(f"Confusion Matrix:\n{test_metrics['confusion_matrix']}")
    print(f"==================================================")

    output_dir = os.path.dirname(args.output) if os.path.dirname(args.output) else "models"
    model_path, config_path, meta_path = save_model_artifacts(
        model_obj=best_model,
        model_name=best_name,
        feature_names=FEATURE_NAMES,
        hyperparams=best_model.get_params(),
        metrics=test_metrics,
        decision_thresh=best_thresh,
        output_dir=output_dir,
        dataset_version="1.0",
        random_seed=args.random_seed
    )

    print(f"\nModel artifacts successfully saved:")
    print(f"  Model:   {model_path}")
    print(f"  Config:  {config_path}")
    print(f"  Metadata:{meta_path}")


if __name__ == "__main__":
    main()
