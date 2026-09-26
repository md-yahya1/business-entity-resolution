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
    Generate training pairs strictly in the same direction as inference:
    Source-1 -> Source-2/Source-3.

    Every positive ground-truth pair is retained, and hard negatives are
    sampled only from S1-to-other-source blocks. This keeps the training
    distribution aligned with the submission pipeline.
    """
    random.seed(random_seed)
    np.random.seed(random_seed)

    s1p = preprocess_dataframe(s1_df)
    s2p = preprocess_dataframe(s2_df)
    s3p = preprocess_dataframe(s3_df)

    s1_records = {r.entity_id: r for r in s1p.itertuples(index=False)}
    other_records = {
        r.entity_id: r for r in pd.concat([s2p, s3p], ignore_index=True).itertuples(index=False)
    }

    positive_pairs = []
    positive_pair_set: Set[Tuple[str, str]] = set()
    entity_to_group: Dict[str, int] = {}

    # Give every S1 anchor its own group. All pairs for one S1 therefore
    # remain together during GroupShuffleSplit.
    for group_id, s1_id in enumerate(s1_records):
        entity_to_group[s1_id] = group_id

    if gt_df is not None:
        gt_valid = gt_df.dropna(subset=["matched_entity_ids"]).copy()
        if len(gt_valid) > max_positives:
            gt_valid = gt_valid.sample(n=max_positives, random_state=random_seed)

        for row in gt_valid.itertuples(index=False):
            s1_id = str(row.source1_entity_id)
            if s1_id not in s1_records:
                continue

            matches = [m.strip() for m in str(row.matched_entity_ids).split(",") if m.strip()]
            for m_id in matches:
                if m_id not in other_records:
                    continue
                pair = (s1_id, m_id)
                if pair not in positive_pair_set:
                    positive_pair_set.add(pair)
                    positive_pairs.append(
                        (s1_id, m_id, 1, entity_to_group[s1_id])
                    )

    # Multi-pass blocking over S2/S3 only.
    blocks: Dict[Tuple[str, str], List[str]] = {}
    for eid, rec in other_records.items():
        country = rec.country_normalized
        name = rec.business_name_normalized
        address = rec.business_address_normalized
        if not country:
            continue

        keys = []
        if len(name) >= 3:
            keys.append((country, "pre3:" + name[:3]))
        if name:
            for tok in name.split()[:3]:
                if len(tok) >= 3:
                    keys.append((country, "name:" + tok))
        if address:
            for tok in address.split()[:3]:
                if len(tok) >= 4:
                    keys.append((country, "addr:" + tok))

        for key in set(keys):
            blocks.setdefault(key, []).append(eid)

    negative_pairs: List[Tuple[str, str, int, int]] = []
    negative_pair_set: Set[Tuple[str, str]] = set()

    s1_ids = list(s1_records.keys())
    random.shuffle(s1_ids)

    for s1_id in s1_ids:
        if len(negative_pairs) >= max_negatives:
            break

        rec = s1_records[s1_id]
        country = rec.country_normalized
        name = rec.business_name_normalized
        address = rec.business_address_normalized

        keys = []
        if country and len(name) >= 3:
            keys.append((country, "pre3:" + name[:3]))
        if country and name:
            keys.extend((country, "name:" + tok) for tok in name.split()[:3] if len(tok) >= 3)
        if country and address:
            keys.extend((country, "addr:" + tok) for tok in address.split()[:3] if len(tok) >= 4)

        candidate_ids = []
        seen = set()
        for key in keys:
            for eid in blocks.get(key, []):
                if eid not in seen:
                    seen.add(eid)
                    candidate_ids.append(eid)

        random.shuffle(candidate_ids)
        for eid in candidate_ids:
            pair = (s1_id, eid)
            if pair in positive_pair_set or pair in negative_pair_set:
                continue
            negative_pair_set.add(pair)
            negative_pairs.append((s1_id, eid, 0, entity_to_group[s1_id]))
            if len(negative_pairs) >= max_negatives:
                break

    all_pairs = positive_pairs + negative_pairs
    random.shuffle(all_pairs)

    records = []
    for e1, e2, label, grp in all_pairs:
        r1 = s1_records[e1]
        r2 = other_records[e2]
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

