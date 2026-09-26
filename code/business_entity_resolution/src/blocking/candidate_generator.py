"""
Candidate pair generation module for Entity Resolution pipeline.
Generates positive and negative candidate pairs using blocking strategies.
"""
from typing import Tuple, List, Dict, Set
import random
import pandas as pd
import numpy as np
from ..preprocessing import preprocess_dataframe, normalize_text


def generate_candidate_pairs(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    gt_df: pd.DataFrame = None,
    max_positives: int = 10000,
    max_negatives: int = 30000,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Generate candidate pairs with ground-truth labels and underlying entity group IDs for leakage-safe splitting.
    """
    random.seed(random_seed)
    np.random.seed(random_seed)

    # Combine all records into single lookups
    all_records = {}
    for df in [s1_df, s2_df, s3_df]:
        df_p = preprocess_dataframe(df)
        for row in df_p.itertuples(index=False):
            all_records[row.entity_id] = row

    positive_pairs = []
    positive_pair_set: Set[Tuple[str, str]] = set()

    entity_to_group: Dict[str, int] = {}
    group_counter = 0

    if gt_df is not None:
        gt_valid = gt_df.dropna(subset=['matched_entity_ids'])
        if len(gt_valid) > max_positives:
            gt_valid = gt_valid.sample(n=max_positives, random_state=random_seed)

        for row in gt_valid.itertuples(index=False):
            s1_id = row.source1_entity_id
            if s1_id not in all_records:
                continue

            matches = [m.strip() for m in str(row.matched_entity_ids).split(',') if m.strip()]
            valid_matches = [m for m in matches if m in all_records]
            if not valid_matches:
                continue

            group_id = group_counter
            group_counter += 1

            entity_to_group[s1_id] = group_id
            for m_id in valid_matches:
                entity_to_group[m_id] = group_id
                pair = tuple(sorted([s1_id, m_id]))
                if pair not in positive_pair_set:
                    positive_pair_set.add(pair)
                    positive_pairs.append((s1_id, m_id, 1, group_id))

    # Generate hard negative pairs via multi-pass blocking (country + first 3 chars + first token)
    negative_pairs = []
    negative_pair_set: Set[Tuple[str, str]] = set()

    blocks: Dict[str, List[str]] = {}
    all_eids = list(all_records.keys())
    if len(all_eids) > 100000:
        sample_eids = random.sample(all_eids, 100000)
    else:
        sample_eids = all_eids

    for eid in sample_eids:
        rec = all_records[eid]
        c = rec.country_normalized
        name_norm = rec.business_name_normalized
        if c and name_norm:
            # Key 1: country + first char
            key1 = f"{c}_{name_norm[0]}"
            blocks.setdefault(key1, []).append(eid)

            # Key 2: country + first 3 chars (hard negatives)
            if len(name_norm) >= 3:
                key2 = f"{c}_pre3_{name_norm[:3]}"
                blocks.setdefault(key2, []).append(eid)

            # Key 3: country + first token
            tokens = name_norm.split()
            if tokens and len(tokens[0]) >= 3:
                key3 = f"{c}_tok_{tokens[0]}"
                blocks.setdefault(key3, []).append(eid)

    block_keys = list(blocks.keys())
    random.shuffle(block_keys)

    for key in block_keys:
        eids_in_block = blocks[key]
        if len(eids_in_block) < 2:
            continue

        n_samples = min(40, len(eids_in_block))
        sub_eids = random.sample(eids_in_block, n_samples)
        for i in range(len(sub_eids)):
            for j in range(i + 1, min(i + 4, len(sub_eids))):
                e1, e2 = sub_eids[i], sub_eids[j]
                g1 = entity_to_group.get(e1, None)
                g2 = entity_to_group.get(e2, None)

                if g1 is not None and g2 is not None and g1 == g2:
                    continue

                pair = tuple(sorted([e1, e2]))
                if pair not in positive_pair_set and pair not in negative_pair_set:
                    negative_pair_set.add(pair)
                    group_id = g1 if g1 is not None else (g2 if g2 is not None else group_counter)
                    if g1 is None and g2 is None:
                        group_counter += 1
                    negative_pairs.append((e1, e2, 0, group_id))

                if len(negative_pairs) >= max_negatives:
                    break
            if len(negative_pairs) >= max_negatives:
                break
        if len(negative_pairs) >= max_negatives:
            break

    all_pairs = positive_pairs + negative_pairs
    random.shuffle(all_pairs)

    records = []
    for e1, e2, label, grp in all_pairs:
        r1 = all_records[e1]
        r2 = all_records[e2]
        records.append({
            "entity_id_1": e1,
            "business_name_1": r1.business_name,
            "business_address_1": r1.business_address,
            "country_1": r1.country,
            "entity_id_2": e2,
            "business_name_2": r2.business_name,
            "business_address_2": r2.business_address,
            "country_2": r2.country,
            "label": label,
            "entity_group_id": grp,
        })

    return pd.DataFrame(records)


def generate_test_candidates(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    max_candidates_per_s1: int = 15
) -> pd.DataFrame:
    """
    Generate candidate pairs for all S1 entities against S2/S3 using fast multi-pass blocking keys (country + first token).
    """
    blocks: Dict[Tuple[str, str], List[Dict[str, str]]] = {}

    for s_df in [s2_df, s3_df]:
        for row in s_df.itertuples(index=False):
            eid = getattr(row, "entity_id", "")
            bname = str(getattr(row, "business_name", ""))
            addr = str(getattr(row, "business_address", ""))
            c_raw = getattr(row, "country", "")
            country = normalize_country(str(c_raw))

            norm_name = normalize_business_name(bname)
            tokens = norm_name.split()
            first_token = tokens[0] if tokens else ""

            if country and first_token:
                key = (country, first_token)
                if key not in blocks:
                    blocks[key] = []
                blocks[key].append({
                    "entity_id": eid,
                    "business_name": bname,
                    "business_address": addr,
                    "country": c_raw,
                    "norm_name": norm_name
                })

    pairs = []
    for row in s1_df.itertuples(index=False):
        s1_id = getattr(row, "entity_id", "")
        s1_name = str(getattr(row, "business_name", ""))
        s1_addr = str(getattr(row, "business_address", ""))
        s1_country_raw = getattr(row, "country", "")
        s1_c = normalize_country(str(s1_country_raw))

        s1_norm = normalize_business_name(s1_name)
        tokens = s1_norm.split()
        first_token = tokens[0] if tokens else ""

        key = (s1_c, first_token)
        candidates = blocks.get(key, [])

        count = 0
        for cand in candidates:
            pairs.append({
                "entity_id_1": s1_id,
                "business_name_1": s1_name,
                "business_address_1": s1_addr,
                "country_1": s1_country_raw,
                "entity_id_2": cand["entity_id"],
                "business_name_2": cand["business_name"],
                "business_address_2": cand["business_address"],
                "country_2": cand["country"],
                "label": 0,
                "entity_group_id": 0
            })
            count += 1
            if count >= max_candidates_per_s1:
                break

    return pd.DataFrame(pairs)

