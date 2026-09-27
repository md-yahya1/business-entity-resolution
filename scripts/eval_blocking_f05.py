"""
Measure candidate recall and macro F0.5 on a training slice.

Usage:
    python scripts/eval_blocking_f05.py --train-dir dataset/train --top-k 40
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "code")))

from business_entity_resolution.src.blocking.inference_blocking import FastCandidateIndex
from business_entity_resolution.src.evaluation.entity_decision import (
    select_entity_matches,
    tune_entity_decision_on_validation,
)
from business_entity_resolution.src.evaluation.entity_metrics import ground_truth_to_dict, macro_f0_5
from business_entity_resolution.src.features.pair_features import align_feature_matrix, compute_features_batch_fast
from business_entity_resolution.src.models.predict import load_model_and_config
from business_entity_resolution.src.preprocessing import preprocess_dataframe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", default="dataset/train")
    parser.add_argument("--model", default="models/entity_resolution_model.joblib")
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()

    gt_df = pd.read_csv(os.path.join(args.train_dir, "train_ground_truth.tsv"), sep="\t")
    sample_gt = gt_df.dropna(subset=["matched_entity_ids"]).sample(
        n=min(args.sample_size, len(gt_df)), random_state=args.random_seed
    )

    needed_s1 = set(sample_gt["source1_entity_id"])
    needed_cands = set()
    for m in sample_gt["matched_entity_ids"]:
        for x in str(m).split(","):
            if x.strip():
                needed_cands.add(x.strip())

    s1_df = pd.read_csv(os.path.join(args.train_dir, "train_source1.tsv"), sep="\t")
    sample_s1 = s1_df[s1_df["entity_id"].isin(needed_s1)]

    s2_chunks, s3_chunks = [], []
    for chunk in pd.read_csv(os.path.join(args.train_dir, "train_source2.tsv"), sep="\t", chunksize=100000):
        s2_chunks.append(chunk[chunk["entity_id"].isin(needed_cands) | (chunk.index < 50000)])
    for chunk in pd.read_csv(os.path.join(args.train_dir, "train_source3.tsv"), sep="\t", chunksize=100000):
        s3_chunks.append(chunk[chunk["entity_id"].isin(needed_cands) | (chunk.index < 50000)])
    s2_df = pd.concat(s2_chunks, ignore_index=True).drop_duplicates("entity_id")
    s3_df = pd.concat(s3_chunks, ignore_index=True).drop_duplicates("entity_id")

    index = FastCandidateIndex(s2_df, s3_df)
    s1p = preprocess_dataframe(sample_s1)
    model, feature_names, default_thresh = load_model_and_config(args.model)
    true_by_entity = ground_truth_to_dict(sample_gt)

    total_gt = sum(len(v) for v in true_by_entity.values())
    retrieved = 0
    pairs_by_entity = {}

    for row in s1p.itertuples(index=False):
        s1_id = row.entity_id
        cand_idxs = index.get_candidates_for_query(
            row.country_normalized,
            row.business_name_normalized,
            row.business_address_normalized,
            top_k=args.top_k,
        )
        cand_ids = [index.ids[ci] for ci in cand_idxs]
        true_set = true_by_entity.get(s1_id, set())
        retrieved += len(true_set.intersection(set(cand_ids)))

        if cand_idxs:
            n1 = [row.business_name_normalized] * len(cand_idxs)
            a1 = [row.business_address_normalized] * len(cand_idxs)
            c1 = [row.country_normalized] * len(cand_idxs)
            n2 = [index.names[ci] for ci in cand_idxs]
            a2 = [index.addresses[ci] for ci in cand_idxs]
            c2 = [index.countries[ci] for ci in cand_idxs]
            feats = compute_features_batch_fast(n1, a1, c1, n2, a2, c2, are_pre_normalized=True)
            feats = align_feature_matrix(feats, list(feature_names))
            probs = model.predict_proba(feats)[:, 1]
            pairs_by_entity[s1_id] = list(zip(cand_ids, probs.tolist()))
        else:
            pairs_by_entity[s1_id] = []

    recall = retrieved / max(total_gt, 1)
    print(f"Candidate recall @ top_k={args.top_k}: {retrieved}/{total_gt} ({recall:.2%})")

    baseline_pred = {
        eid: {cid for cid, p in pairs if p >= default_thresh}
        for eid, pairs in pairs_by_entity.items()
    }
    baseline_score, _ = macro_f0_5(true_by_entity, baseline_pred, needed_s1)
    print(f"Macro F0.5 (pairwise threshold={default_thresh:.2f} only): {baseline_score:.4f}")

    tuned_score, tuned_params, _ = tune_entity_decision_on_validation(
        pairs_by_entity, true_by_entity, needed_s1
    )
    print(f"Macro F0.5 (tuned entity decision): {tuned_score:.4f} params={tuned_params}")


if __name__ == "__main__":
    main()
