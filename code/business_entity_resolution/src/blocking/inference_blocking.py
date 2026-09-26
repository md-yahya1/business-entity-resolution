"""
Blocking for INFERENCE (test-time candidate generation).

candidate_generator.generate_candidate_pairs() is built for *training*: it
samples positive pairs from ground truth and negative pairs from anywhere
in the combined S1+S2+S3 pool (including S2-S2 / S2-S3 pairs), which is
fine for teaching the model what a mismatch looks like but is NOT what the
submission format needs.

For matching_results.tsv / candidate_pairs.tsv we need, for every single
Source-1 test entity (no exceptions, including a hypothetical unseen
country like France): a ranked shortlist of Source-2/Source-3 candidates
only, restricted to actual S1 <-> S2/S3 pairs.

Strategy: combine the (normalized country, first name character) block
with up to two selective same-country name/address token blocks. Rank this
bounded pool by its stronger name or address similarity. If all these
blocks are empty, fall back to same-country records, capped before ranking.
If an entity's country never appears on the S2/S3 side, it gets zero
candidates; an empty list is a valid, scoreable answer for a singleton.
"""
from typing import Dict, List
from collections import defaultdict
import pandas as pd
import rapidfuzz.fuzz as fuzz

from ..preprocessing import preprocess_dataframe


def _block_key(country: str, name: str) -> str:
    first_char = name[0] if name else ""
    return f"{country}|{first_char}"


def _rank_and_cap(
    query_name: str,
    query_address: str,
    candidate_ids: List[str],
    records_by_id: Dict[str, tuple],
    top_k: int,
) -> List[str]:
    scored = [
        (
            eid,
            max(
                fuzz.token_set_ratio(query_name, records_by_id[eid][0]),
                fuzz.token_set_ratio(query_address, records_by_id[eid][1]),
            ),
        )
        for eid in candidate_ids
    ]
    scored.sort(key=lambda x: -x[1])
    return [eid for eid, _ in scored[:top_k]]


def generate_inference_candidates(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    top_k: int = 25,
    country_fallback_cap: int = 500,
) -> pd.DataFrame:
    """
    Returns one row per Source-1 entity:
        source1_entity_id, candidate_entity_ids (list[str], S2/S3 only)

    Guarantees: every entity_id in s1_df appears exactly once, even when
    its candidate list is empty. Never includes an S1 id, an S2-S2 pair,
    an S3-S3 pair, or an id absent from s2_df/s3_df.
    """
    s1p = preprocess_dataframe(s1_df)
    s2p = preprocess_dataframe(s2_df)
    s3p = preprocess_dataframe(s3_df)

    # records_by_id: entity_id -> (normalized_name, normalized_address)
    records_by_id: Dict[str, tuple] = {}
    block_index: Dict[str, List[str]] = defaultdict(list)
    country_index: Dict[str, List[str]] = defaultdict(list)
    token_index: Dict[tuple, List[str]] = defaultdict(list)

    for other_df in (s2p, s3p):
        for row in other_df.itertuples(index=False):
            records_by_id[row.entity_id] = (
                row.business_name_normalized,
                row.business_address_normalized,
            )
            key = _block_key(row.country_normalized, row.business_name_normalized)
            block_index[key].append(row.entity_id)
            country_index[row.country_normalized].append(row.entity_id)
            for field_name, value in (
                ("name", row.business_name_normalized),
                ("address", row.business_address_normalized),
            ):
                for token in set(value.split()):
                    if len(token) >= 3:
                        token_index[(row.country_normalized, field_name, token)].append(row.entity_id)

    out_rows = []
    for row in s1p.itertuples(index=False):
        key = _block_key(row.country_normalized, row.business_name_normalized)
        candidates = list(block_index.get(key, []))

        # A shared non-leading token recovers spelling/order variants while
        # frequent tokens stay out of the candidate pool.
        token_blocks = []
        for field_name, value in (
            ("name", row.business_name_normalized),
            ("address", row.business_address_normalized),
        ):
            for token in set(value.split()):
                if len(token) < 3:
                    continue
                token_candidates = token_index.get((row.country_normalized, field_name, token), [])
                if 0 < len(token_candidates) <= country_fallback_cap:
                    token_blocks.append((len(token_candidates), field_name, token, token_candidates))
        token_blocks.sort(key=lambda block: (block[0], block[1], block[2]))
        for _, _, _, token_candidates in token_blocks[:2]:
            candidates.extend(token_candidates)
        candidates = list(dict.fromkeys(candidates))

        if not candidates:
            # fall back to same-country records only (cap before ranking --
            # a big country like US/India could otherwise be O(n) per entity)
            pool = country_index.get(row.country_normalized, [])
            candidates = pool[:country_fallback_cap]

        ranked = _rank_and_cap(
            row.business_name_normalized,
            row.business_address_normalized,
            candidates,
            records_by_id,
            top_k,
        )

        out_rows.append({
            "source1_entity_id": row.entity_id,
            "candidate_entity_ids": ranked,  # list, joined to a string later
        })

    return pd.DataFrame(out_rows)


def expand_candidates_to_pairs(
    candidates_df: pd.DataFrame,
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Explode the one-row-per-S1-entity candidate list into one row per
    (S1, candidate) pair with the raw attribute columns feature extraction
    expects: business_name_1/2, business_address_1/2, country_1/2.
    """
    s1_lookup = s1_df.set_index("entity_id")
    other_lookup = pd.concat([s2_df, s3_df], axis=0).set_index("entity_id")

    rows = []
    for row in candidates_df.itertuples(index=False):
        s1_id = row.source1_entity_id
        r1 = s1_lookup.loc[s1_id]
        for cand_id in row.candidate_entity_ids:
            r2 = other_lookup.loc[cand_id]
            rows.append({
                "entity_id_1": s1_id,
                "business_name_1": r1["business_name"],
                "business_address_1": r1["business_address"],
                "country_1": r1["country"],
                "entity_id_2": cand_id,
                "business_name_2": r2["business_name"],
                "business_address_2": r2["business_address"],
                "country_2": r2["country"],
            })

    return pd.DataFrame(rows, columns=[
        "entity_id_1", "business_name_1", "business_address_1", "country_1",
        "entity_id_2", "business_name_2", "business_address_2", "country_2",
    ])
