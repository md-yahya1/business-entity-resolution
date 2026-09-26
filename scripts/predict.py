"""
Inference script for Business Entity Resolution ML Model.

Usage:
    python scripts/predict.py --model models/entity_resolution_model.joblib --input artifacts/candidates/test_features.parquet --output output/predictions.tsv
"""
import sys
import os
import argparse
import pandas as pd

# Ensure code directory is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "code")))

from business_entity_resolution.src.models.predict import predict_candidate_pairs
from business_entity_resolution.src.blocking import generate_candidate_pairs
from business_entity_resolution.src.features import extract_features_dataframe, FEATURE_NAMES


def main():
    parser = argparse.ArgumentParser(description="Inference script for Entity Resolution Model")
    parser.add_argument(
        "--model",
        type=str,
        default="models/entity_resolution_model.joblib",
        help="Path to trained model joblib file"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="artifacts/candidates/test_features.parquet",
        help="Path to input candidate feature dataset (parquet or tsv)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/predictions.tsv",
        help="Path to output predictions TSV file"
    )

    args = parser.parse_args()

    print("==================================================")
    print("      BUSINESS ENTITY RESOLUTION INFERENCE       ")
    print("==================================================")

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Trained model not found at path: {args.model}")

    input_file = args.input
    if os.path.exists(input_file):
        print(f"Loading test feature dataset from: {input_file}")
        if input_file.endswith(".parquet"):
            df = pd.read_parquet(input_file)
        else:
            df = pd.read_csv(input_file, sep="\t")
    else:
        print(f"Input file {input_file} not found. Generating test features from dataset/test...")
        test_dir = "dataset/test"
        s1_path = os.path.join(test_dir, "test_source1.tsv")
        s2_path = os.path.join(test_dir, "test_source2.tsv")
        s3_path = os.path.join(test_dir, "test_source3.tsv")

        s1 = pd.read_csv(s1_path, sep="\t", nrows=1000)
        s2 = pd.read_csv(s2_path, sep="\t", nrows=5000)
        s3 = pd.read_csv(s3_path, sep="\t", nrows=5000)

        pairs_df = generate_candidate_pairs(s1, s2, s3, max_positives=0, max_negatives=2000)
        features_df = extract_features_dataframe(pairs_df)
        df = pd.concat([pairs_df, features_df], axis=1)

        os.makedirs(os.path.dirname(input_file), exist_ok=True)
        if input_file.endswith(".parquet"):
            df.to_parquet(input_file, index=False)
        else:
            df.to_csv(input_file, sep="\t", index=False)
        print(f"Saved generated test features to: {input_file}")

    print(f"Running model inference on {len(df)} candidate pairs...")
    predictions_df = predict_candidate_pairs(df, model_path=args.model, output_path=args.output)

    # Format matching results into source1_entity_id -> matched_entity_ids TSV format
    matches_df = predictions_df[predictions_df["is_match"] == 1]
    group_map = {}
    for _, row in matches_df.iterrows():
        id1, id2 = str(row["entity_id_1"]), str(row["entity_id_2"])
        if id1.startswith("S1-"):
            s1, other = id1, id2
        elif id2.startswith("S1-"):
            s1, other = id2, id1
        else:
            s1, other = id1, id2
        group_map.setdefault(s1, set()).add(other)

    matching_results_df = pd.DataFrame([
        {"source1_entity_id": s1, "matched_entity_ids": ",".join(sorted(list(m)))}
        for s1, m in group_map.items()
    ])
    matching_results_path = os.path.join(os.path.dirname(args.output), "matching_results.tsv")
    matching_results_df.to_csv(matching_results_path, sep="\t", index=False)

    print(f"\nInference completed.")
    print(f" - Detailed pair predictions saved to: {args.output}")
    print(f" - Formatted evaluation matching results saved to: {matching_results_path}")
    print(f"Summary of Predictions:")
    print(predictions_df["is_match"].value_counts().to_dict())
    print("\nSample predictions:")
    show_cols = [c for c in ["entity_id_1", "entity_id_2", "business_name_1", "business_name_2", "match_probability", "is_match"] if c in predictions_df.columns]
    print(predictions_df[show_cols].head())


if __name__ == "__main__":
    main()

