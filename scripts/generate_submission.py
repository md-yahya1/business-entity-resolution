"""
End-to-end submission generator for Business Entity Resolution.
Generates matching_results.tsv and candidate_pairs.tsv using the High-Speed
Streaming Inference Engine.

Usage:
    python scripts/generate_submission.py \
        --data-dir dataset/test \
        --model models/entity_resolution_model.joblib \
        --output-dir output \
        --top-k 25 \
        --chunk-size 50000
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import pandas as pd

if sys.version_info >= (3, 7):
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "code")))


from business_entity_resolution.src.blocking.inference_blocking import run_streaming_inference


def validate_outputs(matching_path: str, candidate_path: str, s1_df: pd.DataFrame, s2_df: pd.DataFrame, s3_df: pd.DataFrame) -> bool:
    """Lightweight local check mirroring the official validator's core rules."""
    print("\nRunning local output integrity validation...")
    issues = []

    matching_df = pd.read_csv(matching_path, sep="\t", dtype=str).fillna("")
    candidate_df = pd.read_csv(candidate_path, sep="\t", dtype=str).fillna("")

    valid_s1 = set(s1_df["entity_id"].astype(str))
    valid_other = set(s2_df["entity_id"].astype(str)) | set(s3_df["entity_id"].astype(str))

    if set(matching_df["source1_entity_id"]) != valid_s1:
        issues.append("matching_results.tsv does not cover exactly the S1 test entities.")
    if matching_df["source1_entity_id"].duplicated().any():
        issues.append("matching_results.tsv has duplicate source1_entity_id rows.")

    for df, col, name in [
        (matching_df, "matched_entity_ids", "matching_results.tsv"),
        (candidate_df, "candidate_entity_ids", "candidate_pairs.tsv"),
    ]:
        for row in df.itertuples(index=False):
            ids = [x for x in str(getattr(row, col)).split(",") if x]
            if len(ids) != len(set(ids)):
                issues.append(f"{name}: duplicate IDs in list for {row.source1_entity_id}")
            bad = [i for i in ids if i not in valid_other]
            if bad:
                issues.append(f"{name}: {row.source1_entity_id} references unknown/S1 ids: {bad[:3]}...")

    cand_map = dict(zip(candidate_df["source1_entity_id"], candidate_df["candidate_entity_ids"]))
    for row in matching_df.itertuples(index=False):
        matched = set(x for x in row.matched_entity_ids.split(",") if x)
        candidates = set(x for x in cand_map.get(row.source1_entity_id, "").split(",") if x)
        if not matched.issubset(candidates):
            issues.append(f"{row.source1_entity_id}: matched id(s) not present in its own candidate list.")

    if issues:
        print("\nVALIDATION ISSUES FOUND:")
        for i in issues[:10]:
            print(f"  - {i}")
        if len(issues) > 10:
            print(f"  ... and {len(issues) - 10} more issues.")
    else:
        print("Local validation PASSED: 100% compliant with challenge schema!")
    return len(issues) == 0


def main():
    parser = argparse.ArgumentParser(description="Generate matching_results.tsv and candidate_pairs.tsv")
    parser.add_argument("--data-dir", default="dataset/test", help="Folder with source1/2/3 tsv files")
    parser.add_argument("--model", default="models/entity_resolution_model.joblib")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--threshold", type=float, default=None, help="Override model_metadata.json's decision_threshold")
    parser.add_argument("--top-k", type=int, default=25, help="Max blocking candidates per S1 entity")
    parser.add_argument("--chunk-size", type=int, default=50000, help="Number of S1 entities processed per batch")
    parser.add_argument("--prefix", default="", help="File prefix, e.g. 'test_' or 'train_'")
    args = parser.parse_args()

    prefix = args.prefix
    if not prefix:
        prefix = "test_" if os.path.exists(os.path.join(args.data_dir, "test_source1.tsv")) else "train_"

    s1_path = os.path.join(args.data_dir, f"{prefix}source1.tsv")
    s2_path = os.path.join(args.data_dir, f"{prefix}source2.tsv")
    s3_path = os.path.join(args.data_dir, f"{prefix}source3.tsv")

    print(f"Loading datasets from {args.data_dir}...")
    s1 = pd.read_csv(s1_path, sep="\t")
    s2 = pd.read_csv(s2_path, sep="\t")
    s3 = pd.read_csv(s3_path, sep="\t")
    print(f"Loaded {len(s1):,} S1 / {len(s2):,} S2 / {len(s3):,} S3 records.")

    matching_path, candidate_path = run_streaming_inference(
        s1_df=s1,
        s2_df=s2,
        s3_df=s3,
        model_path=args.model,
        output_dir=args.output_dir,
        top_k=args.top_k,
        threshold_override=args.threshold,
        chunk_size=args.chunk_size,
    )

    validate_outputs(matching_path, candidate_path, s1, s2, s3)


if __name__ == "__main__":
    main()
