"""
High-Speed Candidate Blocking and Streaming Inference Engine.

Designed to scale to 1.7M+ Source-1 records and 10M+ candidate records (S2/S3)
with low memory usage, inverted index caching, and batched streaming inference.
"""
from __future__ import annotations

import os
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import rapidfuzz.fuzz as fuzz

from ..evaluation.entity_decision import select_entity_matches
from ..features.pair_features import compute_features_batch_fast, align_feature_matrix
from ..models.predict import load_model_and_config
from ..preprocessing import extract_address_hints, preprocess_dataframe

CORPORATE_STOPWORDS: Set[str] = {
    "the", "a", "an", "and", "or", "inc", "incorporated", "llc", "ltd",
    "limited", "corp", "corporation", "co", "company", "pvt", "private",
    "sa", "sas", "sarl", "gmbh", "ag", "spa", "srl", "bv", "nv", "oy", "ab",
    "in", "at", "of", "to", "for", "on", "by", "from",
}

COMMON_GENERIC_TOKENS: Set[str] = {
    "store", "shop", "market", "restaurant", "hotel", "cafe", "services",
    "service", "center", "centre", "group", "holdings", "enterprise",
    "enterprises", "solutions", "international", "global", "trading",
    "street", "road", "avenue", "drive", "lane", "building", "floor",
    "suite", "plaza", "block", "box", "post", "city", "north", "south",
    "east", "west", "new", "st", "rd", "ave", "dr",
}


def _extract_selective_tokens(text: str, max_tokens: int = 4) -> List[str]:
    """Extract selective non-generic alphanumeric tokens from normalized string."""
    tokens = text.split()
    selective = [
        t for t in tokens
        if len(t) >= 3 and t not in CORPORATE_STOPWORDS and t not in COMMON_GENERIC_TOKENS
    ]
    return selective[:max_tokens] if selective else tokens[:max_tokens]


class FastCandidateIndex:
    """
    Compact inverted index over S2 and S3 candidate records.
    Uses int32 indices and selective token/prefix blocking for ultra-fast lookup.
    """

    def __init__(self, s2_df: pd.DataFrame, s3_df: pd.DataFrame, max_token_postings: int = 2500):
        self.max_token_postings = max_token_postings

        print("Preprocessing and indexing Source-2 and Source-3 candidate records...")
        s2p = preprocess_dataframe(s2_df)
        s3p = preprocess_dataframe(s3_df)

        combined_df = pd.concat([s2p, s3p], ignore_index=True)

        self.ids: np.ndarray = combined_df["entity_id"].values.astype(str)
        self.names: np.ndarray = combined_df["business_name_normalized"].values.astype(str)
        self.addresses: np.ndarray = combined_df["business_address_normalized"].values.astype(str)
        self.countries: np.ndarray = combined_df["country_normalized"].values.astype(str)

        self.n_records: int = len(self.ids)
        print(f"Total candidate corpus size (S2+S3): {self.n_records:,} records.")

        # Inverted index tables:
        # 1. Prefix-3 index: (country, prefix_3) -> list[int]
        # 2. Token index: (country, token) -> list[int]
        # 3. Country fallback index: country -> list[int]
        self.prefix_index: Dict[Tuple[str, str], List[int]] = defaultdict(list)
        self.prefix2_index: Dict[Tuple[str, str], List[int]] = defaultdict(list)
        self.token_index: Dict[Tuple[str, str], List[int]] = defaultdict(list)
        self.postal_index: Dict[Tuple[str, str], List[int]] = defaultdict(list)
        self.house_index: Dict[Tuple[str, str], List[int]] = defaultdict(list)
        self.city_index: Dict[Tuple[str, str], List[int]] = defaultdict(list)
        self.country_index: Dict[str, List[int]] = defaultdict(list)

        t0 = time.time()
        for idx in range(self.n_records):
            c = self.countries[idx]
            name = self.names[idx]
            addr = self.addresses[idx]

            if not c:
                continue

            self.country_index[c].append(idx)

            if len(name) >= 3:
                self.prefix_index[(c, name[:3])].append(idx)
            if len(name) >= 2:
                self.prefix2_index[(c, name[:2])].append(idx)
            elif len(name) >= 1:
                self.prefix_index[(c, name[:1])].append(idx)

            for tok in _extract_selective_tokens(name, max_tokens=4):
                postings = self.token_index[(c, tok)]
                if len(postings) < self.max_token_postings:
                    postings.append(idx)

            for tok in _extract_selective_tokens(addr, max_tokens=4):
                postings = self.token_index[(c, tok)]
                if len(postings) < self.max_token_postings:
                    postings.append(idx)

            hints = extract_address_hints(addr)
            if hints["postal_code"]:
                self.postal_index[(c, hints["postal_code"])].append(idx)
            if hints["house_number"]:
                self.house_index[(c, hints["house_number"])].append(idx)
            if hints["city"]:
                self.city_index[(c, hints["city"])].append(idx)

        t_elapsed = time.time() - t0
        print(f"Inverted index built in {t_elapsed:.2f}s: {len(self.prefix_index):,} prefix blocks, {len(self.token_index):,} token blocks.")

    def _add_block(self, seen: Set[int], pool: Set[int], block_key, index_map: Dict) -> None:
        for idx in index_map.get(block_key, []):
            if idx not in seen:
                seen.add(idx)
                pool.add(idx)

    def get_candidates_for_query(
        self,
        country: str,
        name: str,
        address: str,
        top_k: int = 40,
    ) -> List[int]:
        """Union multiple blocking keys, then rank down to top_k."""
        if not country:
            return []

        seen: Set[int] = set()
        pool: Set[int] = set()
        hints = extract_address_hints(address)

        if len(name) >= 3:
            self._add_block(seen, pool, (country, name[:3]), self.prefix_index)
        if len(name) >= 2:
            self._add_block(seen, pool, (country, name[:2]), self.prefix2_index)
        elif len(name) >= 1:
            self._add_block(seen, pool, (country, name[:1]), self.prefix_index)

        for tok in _extract_selective_tokens(name, max_tokens=4):
            self._add_block(seen, pool, (country, tok), self.token_index)
        for tok in _extract_selective_tokens(address, max_tokens=4):
            self._add_block(seen, pool, (country, tok), self.token_index)

        if hints["postal_code"]:
            self._add_block(seen, pool, (country, hints["postal_code"]), self.postal_index)
        if hints["house_number"]:
            self._add_block(seen, pool, (country, hints["house_number"]), self.house_index)
        if hints["city"]:
            self._add_block(seen, pool, (country, hints["city"]), self.city_index)

        if not pool:
            for idx in self.country_index.get(country, [])[: max(top_k * 5, 100)]:
                if idx not in seen:
                    seen.add(idx)
                    pool.add(idx)

        if not pool:
            return []

        scored = [
            (
                idx,
                max(
                    fuzz.token_set_ratio(name, self.names[idx]),
                    fuzz.token_sort_ratio(name, self.names[idx]),
                    fuzz.partial_ratio(name, self.names[idx]),
                    fuzz.token_set_ratio(address, self.addresses[idx]),
                    fuzz.partial_ratio(address, self.addresses[idx]),
                ),
            )
            for idx in pool
        ]
        scored.sort(key=lambda x: (-x[1], self.ids[x[0]]))
        return [idx for idx, _ in scored[:top_k]]


def generate_inference_candidates(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    top_k: int = 25,
) -> pd.DataFrame:
    """
    Standard candidate retrieval function (compatible with existing tests).
    """
    index = FastCandidateIndex(s2_df, s3_df)
    s1p = preprocess_dataframe(s1_df)

    out_rows = []
    for row in s1p.itertuples(index=False):
        cand_indices = index.get_candidates_for_query(
            country=row.country_normalized,
            name=row.business_name_normalized,
            address=row.business_address_normalized,
            top_k=top_k,
        )
        cand_ids = [index.ids[idx] for idx in cand_indices]
        out_rows.append({
            "source1_entity_id": row.entity_id,
            "candidate_entity_ids": cand_ids,
        })

    return pd.DataFrame(out_rows)


def expand_candidates_to_pairs(
    candidates_df: pd.DataFrame,
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
) -> pd.DataFrame:
    """Expand candidate lists to DataFrame pairs."""
    s1_lookup = {r.entity_id: r for r in s1_df.itertuples(index=False)}
    combined_other = pd.concat([s2_df, s3_df], ignore_index=True)
    other_lookup = {r.entity_id: r for r in combined_other.itertuples(index=False)}

    rows = []
    for row in candidates_df.itertuples(index=False):
        s1_id = row.source1_entity_id
        r1 = s1_lookup.get(s1_id)
        if r1 is None:
            continue
        for cand_id in row.candidate_entity_ids:
            r2 = other_lookup.get(cand_id)
            if r2 is None:
                continue
            rows.append({
                "entity_id_1": s1_id,
                "business_name_1": getattr(r1, "business_name", ""),
                "business_address_1": getattr(r1, "business_address", ""),
                "country_1": getattr(r1, "country", ""),
                "entity_id_2": cand_id,
                "business_name_2": getattr(r2, "business_name", ""),
                "business_address_2": getattr(r2, "business_address", ""),
                "country_2": getattr(r2, "country", ""),
            })

    return pd.DataFrame(rows)


def run_streaming_inference(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    model_path: str,
    output_dir: str = "output",
    top_k: int = 40,
    threshold_override: Optional[float] = None,
    no_match_max_prob: float = 0.42,
    min_single_match_margin: float = 0.06,
    min_confident_single_match: float = 0.88,
    chunk_size: int = 50000,
) -> Tuple[str, str]:
    """
    High-Speed Chunked Streaming Inference Engine.
    Processes 1.7M+ S1 records in streaming chunks, extracting features and
    running model predictions with minimal memory footprint.
    """
    os.makedirs(output_dir, exist_ok=True)
    matching_path = os.path.join(output_dir, "matching_results.tsv")
    candidate_path = os.path.join(output_dir, "candidate_pairs.tsv")

    model, feature_names, default_thresh = load_model_and_config(model_path)
    threshold = threshold_override if threshold_override is not None else default_thresh
    print(f"Loaded model from {model_path}")
    print(f"Using match threshold: {threshold:.4f}")
    print(
        f"Entity gates: no_match_max_prob={no_match_max_prob:.2f}, "
        f"min_single_match_margin={min_single_match_margin:.2f}"
    )

    # Build inverted index once over S2 and S3
    index = FastCandidateIndex(s2_df, s3_df)

    # Preprocess S1 records
    print("\nPreprocessing Source-1 records...")
    s1p = preprocess_dataframe(s1_df)
    s1_ids: np.ndarray = s1p["entity_id"].values.astype(str)
    s1_names: np.ndarray = s1p["business_name_normalized"].values.astype(str)
    s1_addrs: np.ndarray = s1p["business_address_normalized"].values.astype(str)
    s1_countries: np.ndarray = s1p["country_normalized"].values.astype(str)

    total_s1 = len(s1_ids)
    num_chunks = int(np.ceil(total_s1 / chunk_size))
    print(f"Processing {total_s1:,} Source-1 entities across {num_chunks} chunks of size {chunk_size:,}...")

    total_candidates_count = 0
    total_matches_count = 0
    start_total_time = time.time()

    with open(matching_path, "w", encoding="utf-8") as f_match, open(candidate_path, "w", encoding="utf-8") as f_cand:
        # Write headers
        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")

        for chunk_idx in range(num_chunks):
            t_chunk_start = time.time()
            c_start = chunk_idx * chunk_size
            c_end = min(c_start + chunk_size, total_s1)
            chunk_len = c_end - c_start

            chunk_s1_ids = s1_ids[c_start:c_end]
            chunk_s1_names = s1_names[c_start:c_end]
            chunk_s1_addrs = s1_addrs[c_start:c_end]
            chunk_s1_countries = s1_countries[c_start:c_end]

            # Collect pairs for this chunk
            pair_s1_indices: List[int] = []
            pair_cand_indices: List[int] = []
            cand_lists_for_chunk: List[List[str]] = []

            for i in range(chunk_len):
                c = chunk_s1_countries[i]
                n = chunk_s1_names[i]
                a = chunk_s1_addrs[i]

                cand_idxs = index.get_candidates_for_query(c, n, a, top_k=top_k)
                cand_ids = [index.ids[ci] for ci in cand_idxs]
                cand_lists_for_chunk.append(cand_ids)

                for ci in cand_idxs:
                    pair_s1_indices.append(i)
                    pair_cand_indices.append(ci)

            total_pairs = len(pair_s1_indices)
            total_candidates_count += total_pairs

            matches_by_s1: Dict[int, List[str]] = {}
            probs_by_s1: Dict[int, List[float]] = defaultdict(list)
            ids_by_s1: Dict[int, List[str]] = defaultdict(list)

            if total_pairs > 0:
                n1_batch = [chunk_s1_names[pi] for pi in pair_s1_indices]
                a1_batch = [chunk_s1_addrs[pi] for pi in pair_s1_indices]
                c1_batch = [chunk_s1_countries[pi] for pi in pair_s1_indices]

                n2_batch = [index.names[ci] for ci in pair_cand_indices]
                a2_batch = [index.addresses[ci] for ci in pair_cand_indices]
                c2_batch = [index.countries[ci] for ci in pair_cand_indices]

                feat_matrix = compute_features_batch_fast(
                    n1_batch, a1_batch, c1_batch,
                    n2_batch, a2_batch, c2_batch,
                    are_pre_normalized=True,
                )
                feat_matrix = align_feature_matrix(feat_matrix, list(feature_names))

                probs = model.predict_proba(feat_matrix)[:, 1]

                for p_idx in range(total_pairs):
                    s1_local_idx = pair_s1_indices[p_idx]
                    cand_global_idx = pair_cand_indices[p_idx]
                    cand_eid = index.ids[cand_global_idx]
                    ids_by_s1[s1_local_idx].append(cand_eid)
                    probs_by_s1[s1_local_idx].append(float(probs[p_idx]))

            for s1_local_idx, cand_ids in ids_by_s1.items():
                entity_probs = probs_by_s1[s1_local_idx]
                matches_by_s1[s1_local_idx] = select_entity_matches(
                    cand_ids,
                    entity_probs,
                    match_threshold=threshold,
                    no_match_max_prob=no_match_max_prob,
                    min_single_match_margin=min_single_match_margin,
                    min_confident_single_match=min_confident_single_match,
                )

            # Write chunk results to disk
            chunk_matches = 0
            for i in range(chunk_len):
                s1_id = chunk_s1_ids[i]
                cands_str = ",".join(cand_lists_for_chunk[i])
                f_cand.write(f"{s1_id}\t{cands_str}\n")

                matched_list = matches_by_s1.get(i, [])
                if matched_list:
                    # Deduplicate while preserving order
                    deduped = list(dict.fromkeys(matched_list))
                    matched_str = ",".join(deduped)
                    chunk_matches += len(deduped)
                else:
                    matched_str = ""

                f_match.write(f"{s1_id}\t{matched_str}\n")

            total_matches_count += chunk_matches
            t_chunk = time.time() - t_chunk_start
            pct = ((chunk_idx + 1) / num_chunks) * 100.0
            print(
                f"[{pct:5.1f}%] Chunk {chunk_idx + 1:2d}/{num_chunks} processed "
                f"({chunk_len:,} S1 entities, {total_pairs:,} candidate pairs) | "
                f"Chunk matches: {chunk_matches:,} | Time: {t_chunk:.1f}s"
            )

    total_time = time.time() - start_total_time
    print(f"\nStreaming Inference Finished in {total_time / 60:.2f} minutes.")
    print(f"Total candidate pairs evaluated: {total_candidates_count:,}")
    print(f"Total matched pairs identified:  {total_matches_count:,}")
    print(f"Wrote submission files to:\n  - {matching_path}\n  - {candidate_path}")

    return matching_path, candidate_path
