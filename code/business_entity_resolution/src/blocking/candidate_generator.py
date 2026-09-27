import random
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd

from ..preprocessing import (
    extract_address_hints,
    normalize_business_name,
    normalize_country,
    preprocess_dataframe,
)


def _block_keys(rec) -> List[Tuple[str, str]]:
    """High-recall multi-pass blocking keys for one preprocessed record."""
    country = rec.country_normalized
    name = rec.business_name_normalized
    address = rec.business_address_normalized
    hints = extract_address_hints(address)

    keys: List[Tuple[str, str]] = []

    # Exact/near-exact name passes. These are deliberately country-aware
    # where possible, but exact name is also useful when country is missing.
    if name:
        keys.append(("name_exact", name))
        if len(name) >= 3:
            keys.append(("name_pre3", name[:3]))
        if len(name) >= 4:
            keys.append(("name_pre4", name[:4]))
        for tok in name.split()[:4]:
            if len(tok) >= 3:
                keys.append(("name_tok", tok))

    # Geographic/address passes.
    if country:
        keys.append(("country", country))
    if hints["city"]:
        keys.append(("city", hints["city"]))
    if hints["postal_code"]:
        keys.append(("postal", hints["postal_code"]))
    if hints["state"]:
        keys.append(("state", hints["state"]))
    if hints["house_number"]:
        keys.append(("house", hints["house_number"]))

    for tok in address.split()[:6]:
        if len(tok) >= 4:
            keys.append(("addr_tok", tok))

    # Strong compound passes reduce collisions for common names/tokens.
    if country and name:
        if len(name) >= 3:
            keys.append(("country_name_pre3", country + "|" + name[:3]))
        for tok in name.split()[:3]:
            if len(tok) >= 3:
                keys.append(("country_name_tok", country + "|" + tok))
    if country and hints["city"]:
        keys.append(("country_city", country + "|" + hints["city"]))
    if country and hints["postal_code"]:
        keys.append(("country_postal", country + "|" + hints["postal_code"]))

    return list(set(keys))


def generate_candidate_pairs(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    gt_df: pd.DataFrame = None,
    max_positives: int = 800000,
    max_negatives: int = 900000,
    random_seed: int = 42,
    max_block_size: int = 500,
    max_candidates_per_s1: int = 120,
) -> pd.DataFrame:
    """
    Build a bounded, high-recall training set.

    Positives are sampled directly from ground truth up to max_positives
    *pairs*. Negatives come from the union of multiple blocking passes and
    are capped per S1 anchor. This avoids a Cartesian product while covering
    normalized names, tokens/prefixes, country, city, postal and address/state
    signals used by the inference pipeline.
    """
    random.seed(random_seed)
    np.random.seed(random_seed)

    s1p = preprocess_dataframe(s1_df)
    s2p = preprocess_dataframe(s2_df)
    s3p = preprocess_dataframe(s3_df)

    s1_records = {str(r.entity_id): r for r in s1p.itertuples(index=False)}
    other_records = {
        str(r.entity_id): r
        for r in pd.concat([s2p, s3p], ignore_index=True).itertuples(index=False)
    }

    entity_to_group = {s1_id: group_id for group_id, s1_id in enumerate(s1_records)}

    # Ground-truth positives. Sample rows first for speed, but enforce the
    # final pair cap because one S1 can have multiple matched IDs.
    positive_pairs: List[Tuple[str, str, int, int]] = []
    positive_pair_set: Set[Tuple[str, str]] = set()

    if gt_df is not None and max_positives > 0:
        gt_valid = gt_df.dropna(subset=["matched_entity_ids"])
        if len(gt_valid) > max_positives:
            # Sampling more rows than the final pair budget is unnecessary.
            gt_valid = gt_valid.sample(
                n=min(len(gt_valid), max_positives),
                random_state=random_seed,
            )

        for row in gt_valid.itertuples(index=False):
            s1_id = str(row.source1_entity_id)
            if s1_id not in s1_records:
                continue
            for m_id in str(row.matched_entity_ids).split(","):
                m_id = m_id.strip()
                pair = (s1_id, m_id)
                if m_id not in other_records or pair in positive_pair_set:
                    continue
                positive_pair_set.add(pair)
                positive_pairs.append((s1_id, m_id, 1, entity_to_group[s1_id]))
                if len(positive_pairs) >= max_positives:
                    break
            if len(positive_pairs) >= max_positives:
                break

    # Build bounded inverted indexes. Large generic blocks are clipped to
    # prevent common country/name tokens from creating millions of pairs.
    blocks: Dict[Tuple[str, str], List[str]] = {}
    for eid, rec in other_records.items():
        for key in _block_keys(rec):
            bucket = blocks.setdefault(key, [])
            if len(bucket) < max_block_size:
                bucket.append(eid)

    negative_pairs: List[Tuple[str, str, int, int]] = []
    negative_pair_set: Set[Tuple[str, str]] = set()

    s1_ids = list(s1_records)
    random.shuffle(s1_ids)

    for s1_id in s1_ids:
        if len(negative_pairs) >= max_negatives:
            break

        rec = s1_records[s1_id]
        candidate_ids: List[str] = []
        seen: Set[str] = set()

        # Union all blocking passes. Prefer candidates hit by multiple blocks
        # by counting hits; these are more useful hard negatives.
        hit_count: Dict[str, int] = {}
        for key in _block_keys(rec):
            for eid in blocks.get(key, []):
                if eid not in positive_pair_set and eid != s1_id:
                    hit_count[eid] = hit_count.get(eid, 0) + 1

        # Hard negatives first: candidates sharing multiple independent
        # blocking signals are more informative to the classifier.
        ranked = sorted(hit_count.items(), key=lambda x: (-x[1], x[0]))
        candidate_ids.extend(eid for eid, _ in ranked[:max_candidates_per_s1])

        # Add a randomized tail so training is not dominated by one blocking
        # strategy when several candidates have identical hit counts.
        tail = [eid for eid in hit_count if eid not in set(candidate_ids)]
        random.shuffle(tail)
        candidate_ids.extend(tail[:max_candidates_per_s1])

        for eid in candidate_ids:
            if len(negative_pairs) >= max_negatives:
                break
            pair = (s1_id, eid)
            if pair in positive_pair_set or pair in negative_pair_set:
                continue
            negative_pair_set.add(pair)
            negative_pairs.append((s1_id, eid, 0, entity_to_group[s1_id]))

    all_pairs = positive_pairs + negative_pairs
    random.shuffle(all_pairs)

    records = []
    for e1, e2, label, grp in all_pairs:
        r1 = s1_records[e1]
        r2 = other_records[e2]
        records.append(
            {
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
            }
        )

    result = pd.DataFrame(records)
    print(
        f"Generated {len(result):,} training pairs "
        f"({int((result['label'] == 1).sum()):,} positive / "
        f"{int((result['label'] == 0).sum()):,} negative)."
    )
    return result


def generate_test_candidates(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    max_candidates_per_s1: int = 15,
) -> pd.DataFrame:
    """Generate inference candidates using the existing country/name blocking."""
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
                blocks.setdefault(key, []).append(
                    {
                        "entity_id": eid,
                        "business_name": bname,
                        "business_address": addr,
                        "country": c_raw,
                        "norm_name": norm_name,
                    }
                )

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

        candidates = blocks.get((s1_c, first_token), [])
        for cand in candidates[:max_candidates_per_s1]:
            pairs.append(
                {
                    "entity_id_1": s1_id,
                    "business_name_1": s1_name,
                    "business_address_1": s1_addr,
                    "country_1": s1_country_raw,
                    "entity_id_2": cand["entity_id"],
                    "business_name_2": cand["business_name"],
                    "business_address_2": cand["business_address"],
                    "country_2": cand["country"],
                    "label": 0,
                    "entity_group_id": 0,
                }
            )

    return pd.DataFrame(pairs)
