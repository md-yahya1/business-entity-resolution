"""
End-to-end submission generator for Business Entity Resolution.

This is the piece that was still missing from the pipeline: everything up
to here (preprocessing, feature engineering, model training) produces
either a trained model or pairwise predictions on an arbitrary feature
file. Nothing turned that into the two files the challenge actually
scores: matching_results.tsv and candidate_pairs.tsv, grouped by
source1_entity_id, in the exact column/format the validator expects.

Usage:
    python scripts/generate_submission.py \
        --data-dir dataset/test \
        --model models/entity_resolution_model.joblib \
        --output-dir output \
        --threshold 0.72          # optional override of model_metadata.json's threshold
        --top-k 25                # candidates per S1 entity from blocking

Also works against dataset/train (drop the ground truth file / it's
ignored) if you just want to sanity-check the format before test data
is available.
"""
import sys
import os
import argparse
import json
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "code")))

from business_entity_resolution.src.blocking.inference_blocking import (
    generate_inference_candidates,
    expand_candidates_to_pairs,
)
from business_entity_resolution.src.features import extract_features_dataframe
from business_entity_resolution.src.models.predict import load_model_and_config


def build_output_frames(s1_df, s2_df, s3_df, model_path, top_k, threshold_override):
    print(f"Blocking: generating up to {top_k} candidates per Source-1 entity...")
    candidates_df = generate_inference_candidates(s1_df, s2_df, s3_df, top_k=top_k)

    n_empty = (candidates_df["candidate_entity_ids"].apply(len) == 0).sum()
    print(f"  {len(candidates_df)} Source-1 entities, {n_empty} with zero candidates from blocking.")

    print("Expanding candidate lists into pairs and extracting features...")
    pairs_df = expand_candidates_to_pairs(candidates_df, s1_df, s2_df, s3_df)

    if len(pairs_df) == 0:
        print("WARNING: zero candidate pairs generated -- every S1 entity will be a singleton.")
        matching_rows = [{"source1_entity_id": eid, "matched_entity_ids": ""} for eid in s1_df["entity_id"]]
        candidate_rows = [{"source1_entity_id": eid, "candidate_entity_ids": ""} for eid in s1_df["entity_id"]]
        return pd.DataFrame(matching_rows), pd.DataFrame(candidate_rows)

    features_df = extract_features_dataframe(pairs_df)
    full_df = pd.concat([pairs_df, features_df], axis=1)

    model, feature_names, model_threshold = load_model_and_config(model_path)
    threshold = threshold_override if threshold_override is not None else model_threshold
    print(f"Running model inference (decision threshold = {threshold})...")

    X = full_df[feature_names].values
    full_df["match_probability"] = model.predict_proba(X)[:, 1]
    full_df["is_match"] = (full_df["match_probability"] >= threshold).astype(int)

    # candidate_pairs.tsv: everything fed to the model, grouped by S1 entity
    candidate_grouped = (
        full_df.groupby("entity_id_1")["entity_id_2"]
        .apply(lambda ids: ",".join(dict.fromkeys(ids)))  # de-dup, preserve order
        .reset_index()
        .rename(columns={"entity_id_1": "source1_entity_id", "entity_id_2": "candidate_entity_ids"})
    )

    # matching_results.tsv: only the pairs that cleared the threshold
    matched_only = full_df[full_df["is_match"] == 1]
    matching_grouped = (
        matched_only.groupby("entity_id_1")["entity_id_2"]
        .apply(lambda ids: ",".join(dict.fromkeys(ids)))
        .reset_index()
        .rename(columns={"entity_id_1": "source1_entity_id", "entity_id_2": "matched_entity_ids"})
    )

    # Reindex both against the FULL S1 id list so entities with no candidates
    # or no matches still get a row with an empty string, per the spec.
    all_s1_ids = pd.DataFrame({"source1_entity_id": s1_df["entity_id"]})
    candidate_out = all_s1_ids.merge(candidate_grouped, on="source1_entity_id", how="left")
    candidate_out["candidate_entity_ids"] = candidate_out["candidate_entity_ids"].fillna("")

    matching_out = all_s1_ids.merge(matching_grouped, on="source1_entity_id", how="left")
    matching_out["matched_entity_ids"] = matching_out["matched_entity_ids"].fillna("")

    return matching_out, candidate_out


def validate_outputs(matching_df, candidate_df, s1_df, s2_df, s3_df):
    """Lightweight local check mirroring the official validator's core rules
    (run the official utils/validate_submission.py too before you submit --
    this is just a fast first pass so you're not surprised by it)."""
    issues = []

    valid_s1 = set(s1_df["entity_id"])
    valid_other = set(s2_df["entity_id"]) | set(s3_df["entity_id"])

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

    # every matched id should also appear as a candidate for that entity
    cand_map = dict(zip(candidate_df["source1_entity_id"], candidate_df["candidate_entity_ids"]))
    for row in matching_df.itertuples(index=False):
        matched = set(x for x in row.matched_entity_ids.split(",") if x)
        candidates = set(x for x in cand_map.get(row.source1_entity_id, "").split(",") if x)
        if not matched.issubset(candidates):
            issues.append(f"{row.source1_entity_id}: matched id(s) not present in its own candidate list -- pipeline bug.")

    if issues:
        print("\nVALIDATION ISSUES FOUND:")
        for i in issues:
            print(f"  - {i}")
    else:
        print("\nLocal validation PASSED (still run the official validator before submitting).")
    return len(issues) == 0


def main():
    parser = argparse.ArgumentParser(description="Generate matching_results.tsv and candidate_pairs.tsv")
    parser.add_argument("--data-dir", default="dataset/test", help="Folder with source1/2/3 tsv files")
    parser.add_argument("--model", default="models/entity_resolution_model.joblib")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--threshold", type=float, default=None, help="Override model_metadata.json's decision_threshold")
    parser.add_argument("--top-k", type=int, default=25, help="Max blocking candidates per S1 entity")
    parser.add_argument("--prefix", default="", help="File prefix, e.g. 'test_' or 'train_'")
    args = parser.parse_args()

    prefix = args.prefix
    if not prefix:
        # infer from whichever files actually exist in data-dir
        prefix = "test_" if os.path.exists(os.path.join(args.data_dir, "test_source1.tsv")) else "train_"

    s1 = pd.read_csv(os.path.join(args.data_dir, f"{prefix}source1.tsv"), sep="\t")
    s2 = pd.read_csv(os.path.join(args.data_dir, f"{prefix}source2.tsv"), sep="\t")
    s3 = pd.read_csv(os.path.join(args.data_dir, f"{prefix}source3.tsv"), sep="\t")
    print(f"Loaded {len(s1)} S1 / {len(s2)} S2 / {len(s3)} S3 records from {args.data_dir}")

    matching_df, candidate_df = build_output_frames(s1, s2, s3, args.model, args.top_k, args.threshold)

    os.makedirs(args.output_dir, exist_ok=True)
    matching_path = os.path.join(args.output_dir, "matching_results.tsv")
    candidate_path = os.path.join(args.output_dir, "candidate_pairs.tsv")
    matching_df.to_csv(matching_path, sep="\t", index=False)
    candidate_df.to_csv(candidate_path, sep="\t", index=False)
    print(f"\nWrote {matching_path} and {candidate_path}")

    validate_outputs(matching_df, candidate_df, s1, s2, s3)


if __name__ == "__main__":
    main()
