import sys, os
import pandas as pd
import numpy as np

sys.path.insert(0, 'code')
from business_entity_resolution.src.blocking.inference_blocking import FastCandidateIndex
from business_entity_resolution.src.features.pair_features import compute_features_batch_fast
from business_entity_resolution.src.models.predict import load_model_and_config
from business_entity_resolution.src.evaluation.entity_metrics import macro_f0_5, ground_truth_to_dict
from business_entity_resolution.src.preprocessing import preprocess_dataframe

print("Loading 1,000 ground truth samples and their actual S1, S2, S3 records...")
gt_df = pd.read_csv('dataset/train/train_ground_truth.tsv', sep='\t')
sample_gt = gt_df.dropna(subset=['matched_entity_ids']).sample(n=1000, random_state=42)

needed_s1 = set(sample_gt['source1_entity_id'])
needed_cands = set()
for m in sample_gt['matched_entity_ids']:
    for x in str(m).split(','):
        if x.strip():
            needed_cands.add(x.strip())

print(f"Sample contains {len(needed_s1)} S1 entities and {len(needed_cands)} true S2/S3 matches.")

# Load S1 records
s1_df = pd.read_csv('dataset/train/train_source1.tsv', sep='\t')
sample_s1 = s1_df[s1_df['entity_id'].isin(needed_s1)]

# Load S2 records: all needed + 50,000 background records
s2_chunks = []
for chunk in pd.read_csv('dataset/train/train_source2.tsv', sep='\t', chunksize=100000):
    hit = chunk[chunk['entity_id'].isin(needed_cands) | (chunk.index < 50000)]
    s2_chunks.append(hit)
s2_df = pd.concat(s2_chunks, ignore_index=True).drop_duplicates(subset=['entity_id'])

# Load S3 records: all needed + 50,000 background records
s3_chunks = []
for chunk in pd.read_csv('dataset/train/train_source3.tsv', sep='\t', chunksize=100000):
    hit = chunk[chunk['entity_id'].isin(needed_cands) | (chunk.index < 50000)]
    s3_chunks.append(hit)
s3_df = pd.concat(s3_chunks, ignore_index=True).drop_duplicates(subset=['entity_id'])

print(f"Index corpus: {len(s2_df)} S2 records, {len(s3_df)} S3 records (all true matches included).")

# Build Candidate Index
index = FastCandidateIndex(s2_df, s3_df)
s1p = preprocess_dataframe(sample_s1)
model, feature_names, current_thresh = load_model_and_config('models/entity_resolution_model.joblib')
print(f"Loaded model with threshold {current_thresh}")

true_by_entity = ground_truth_to_dict(sample_gt)

# Evaluate candidate recall
total_gt_matches = sum(len(v) for v in true_by_entity.values())
retrieved_gt_matches = 0

all_s1_pairs = {}
for row in s1p.itertuples(index=False):
    s1_id = row.entity_id
    cand_idxs = index.get_candidates_for_query(
        row.country_normalized,
        row.business_name_normalized,
        row.business_address_normalized,
        top_k=25
    )
    cand_ids = [index.ids[ci] for ci in cand_idxs]
    true_set = true_by_entity.get(s1_id, set())
    retrieved_gt_matches += len(true_set.intersection(set(cand_ids)))
    
    if cand_idxs:
        n1 = [row.business_name_normalized] * len(cand_idxs)
        a1 = [row.business_address_normalized] * len(cand_idxs)
        c1 = [row.country_normalized] * len(cand_idxs)
        n2 = [index.names[ci] for ci in cand_idxs]
        a2 = [index.addresses[ci] for ci in cand_idxs]
        c2 = [index.countries[ci] for ci in cand_idxs]
        
        feats = compute_features_batch_fast(n1, a1, c1, n2, a2, c2, are_pre_normalized=True)
        probs = model.predict_proba(feats)[:, 1]
        all_s1_pairs[s1_id] = list(zip(cand_ids, probs))
    else:
        all_s1_pairs[s1_id] = []

print(f"\n[BLOCKING RECALL] Retrieved {retrieved_gt_matches}/{total_gt_matches} true matches ({retrieved_gt_matches/total_gt_matches*100:.2f}%)")

print("\n[THRESHOLD SWEEP FOR MACRO F0.5]")
for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.62, 0.7, 0.8, 0.85, 0.9, 0.95]:
    pred_dict = {}
    for s1_id, pairs in all_s1_pairs.items():
        pred_dict[s1_id] = {cid for cid, p in pairs if p >= t}
    score, breakdown = macro_f0_5(true_by_entity, pred_dict, needed_s1)
    
    avg_preds = np.mean([len(v) for v in pred_dict.values()])
    # Calculate precision across all matched entities
    tp = sum(breakdown['n_correct'])
    pred_total = sum(breakdown['n_pred'])
    prec = tp / max(pred_total, 1)
    rec = tp / total_gt_matches
    print(f"Threshold {t:.2f} | Macro F0.5 = {score:.4f} | Avg Preds/Entity = {avg_preds:.2f} | Micro Prec = {prec:.3f} | Micro Rec = {rec:.3f}")
