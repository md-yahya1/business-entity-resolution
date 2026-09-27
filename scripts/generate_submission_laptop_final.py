"""Final compact inference: multi-pass blocking + batched ML."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "code")
    ),
)

from business_entity_resolution.src.preprocessing import (
    preprocess_dataframe,
    extract_address_hints,
)
from business_entity_resolution.src.blocking.laptop_final import (
    build_index,
    _hash_keys,
    selective_token,
)
from business_entity_resolution.src.features.laptop_final_features import (
    compute_final_features,
)
from business_entity_resolution.src.evaluation.entity_decision import (
    select_entity_matches,
)


def prepare_s1(df):
    s = preprocess_dataframe(df)
    hints = [
        extract_address_hints(x)
        for x in s.business_address_normalized.fillna("").astype(str)
    ]
    return {
        "id": s.entity_id.astype(str).to_numpy(),
        "name": s.business_name_normalized.fillna("").astype(str).to_numpy(),
        "address": s.business_address_normalized.fillna("").astype(str).to_numpy(),
        "country": s.country_normalized.fillna("").astype(str).to_numpy(),
        "postal": np.array([h["postal_code"] for h in hints], dtype=object),
        "house": np.array([h["house_number"] for h in hints], dtype=object),
        "city": np.array([h["city"] for h in hints], dtype=object),
        "state": np.array([h["state"] for h in hints], dtype=object),
        "prefix3": np.array(
            [
                x[:3]
                for x in s.business_name_normalized.fillna("").astype(str)
            ],
            dtype=object,
        ),
        "token": np.array(
            [
                selective_token(x)
                for x in s.business_name_normalized.fillna("").astype(str)
            ],
            dtype=object,
        ),
        "address_token": np.array(
            [
                selective_token(x)
                for x in s.business_address_normalized.fillna("").astype(str)
            ],
            dtype=object,
        ),
    }


def build_features(q, base, li, ri, index):
    # Keep the hot path in NumPy/RapidFuzz. The previous implementation
    # created Python lists for every feature column and every candidate pair.
    # With millions of test entities, that Python allocation overhead is
    # significant.
    return compute_final_features(
        q["name"][base + li],
        q["address"][base + li],
        q["country"][base + li],
        q["postal"][base + li],
        q["house"][base + li],
        q["city"][base + li],
        q["state"][base + li],
        index.names[ri],
        index.addresses[ri],
        index.countries[ri],
        index.postal[ri],
        index.house[ri],
        index.city[ri],
        index.state[ri],
    )
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="dataset/test")
    ap.add_argument("--source1-file", default="test_source1.tsv")
    ap.add_argument("--source2-file", default="test_source2.tsv")
    ap.add_argument("--source3-file", default="test_source3.tsv")
    ap.add_argument("--model-dir", default="models/laptop_final")
    ap.add_argument("--output-dir", default="output/laptop_final")
    ap.add_argument("--chunk-size", type=int, default=5000)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--posting-cap", type=int, default=64)
    ap.add_argument("--exact-cap", type=int, default=32)

    # The model is still the already-trained model. These knobs only change
    # inference-time retrieval/decision behavior; no retraining is required.
    ap.add_argument(
        "--retrieval-limit",
        type=int,
        default=10,
        help="internal candidates scored per entity; output remains <=10",
    )
    ap.add_argument(
        "--output-candidate-limit",
        type=int,
        default=10,
        help="maximum candidates written to candidate_pairs.tsv",
    )
    args = ap.parse_args()

    t0 = time.time()

    s1 = pd.read_csv(
        os.path.join(args.data_dir, args.source1_file),
        sep="\t",
    )
    s2 = pd.read_csv(
        os.path.join(args.data_dir, args.source2_file),
        sep="\t",
    )
    s3 = pd.read_csv(
        os.path.join(args.data_dir, args.source3_file),
        sep="\t",
    )
    print(
        f"Loaded {len(s1):,} S1 / {len(s2):,} S2 / {len(s3):,} S3"
    )

    model = joblib.load(
        os.path.join(
            args.model_dir,
            "entity_resolution_model.joblib",
        )
    )

    with open(
        os.path.join(args.model_dir, "model_metadata.json"),
        encoding="utf-8",
    ) as f:
        meta = json.load(f)

    threshold = float(meta["decision_threshold"])

    index = build_index(
        s2,
        s3,
        posting_cap=args.posting_cap,
        exact_cap=args.exact_cap,
    )

    print(
        f"Index: {index.n:,} candidates built in "
        f"{index.build_seconds:.1f}s"
    )

    q = prepare_s1(s1)
    total = (
        len(q["id"])
        if not args.limit
        else min(args.limit, len(q["id"]))
    )

    os.makedirs(args.output_dir, exist_ok=True)

    mp = os.path.join(
        args.output_dir,
        "matching_results.tsv",
    )
    cp = os.path.join(
        args.output_dir,
        "candidate_pairs.tsv",
    )

    total_pairs = 0
    total_matches = 0
    candidate_counts = []
    start = time.time()

    with open(
        mp,
        "w",
        encoding="utf-8",
        buffering=1,
    ) as fm, open(
        cp,
        "w",
        encoding="utf-8",
        buffering=1,
    ) as fc:

        fm.write(
            "source1_entity_id\tmatched_entity_ids\n"
        )
        fc.write(
            "source1_entity_id\tcandidate_entity_ids\n"
        )

        for base in range(
            0,
            total,
            args.chunk_size,
        ):
            end = min(
                base + args.chunk_size,
                total,
            )

            ct = time.time()

            cc = q["country"][base:end]

            h_name = _hash_keys(
                cc,
                q["name"][base:end],
            )
            h_prefix = _hash_keys(
                cc,
                q["prefix3"][base:end],
            )
            h_token = _hash_keys(
                cc,
                q["token"][base:end],
            )
            h_postal = _hash_keys(
                cc,
                q["postal"][base:end],
            )
            h_house = _hash_keys(
                cc,
                q["house"][base:end],
            )
            h_city = _hash_keys(
                cc,
                q["city"][base:end],
            )
            h_address_token = _hash_keys(
                cc,
                q["address_token"][base:end],
            )
            all_pairs = []
            candidate_lists = []
            direct = {}

            for local, i in enumerate(
                range(base, end)
            ):
                cands = index.candidates(
                    q["country"][i],
                    q["name"][i],
                    q["address"][i],
                    q["postal"][i],
                    q["house"][i],
                    q["city"][i],
                    q["state"][i],
                    limit=args.retrieval_limit,
                    hashes=(
                        h_name[i - base],
                        h_prefix[i - base],
                        h_token[i - base],
                        h_postal[i - base],
                        h_house[i - base],
                        h_city[i - base],
                        h_address_token[i - base],
                    ),
                )

                cands = cands.tolist()
                candidate_lists.append(cands)
                candidate_counts.append(len(cands))

                exact = [
                    j
                    for j in cands
                    if (
                        q["name"][i]
                        and index.names[j] == q["name"][i]
                    )
                ]

                # Exact-name unique remains the safest fast path.
                if len(exact) == 1:
                    direct[local] = index.ids[exact[0]]
                else:
                    for j in cands:
                        all_pairs.append((local, j))

            probs = {}

            if all_pairs:
                li = np.fromiter(
                    (x[0] for x in all_pairs),
                    dtype=np.int32,
                )
                ri = np.fromiter(
                    (x[1] for x in all_pairs),
                    dtype=np.int32,
                )

                X = build_features(
                    q,
                    base,
                    li,
                    ri,
                    index,
                )

                p = model.predict_proba(X)[:, 1]

                for k, (local, j) in enumerate(
                    all_pairs
                ):
                    probs.setdefault(
                        local,
                        [],
                    ).append(
                        (
                            index.ids[j],
                            float(p[k]),
                            j,
                        )
                    )

            chunk_matches = 0

            for local, i in enumerate(
                range(base, end)
            ):
                if local in direct:
                    selected = [
                        direct[local]
                    ]
                    selected_candidates = candidate_lists[local][:args.output_candidate_limit]
                else:
                    vals = probs.get(
                        local,
                        [],
                    )

                    vals_sorted = sorted(
                        vals,
                        key=lambda x: (
                            -x[1],
                            x[0],
                        ),
                    )

                    selected = select_entity_matches(
                        [x[0] for x in vals_sorted],
                        [x[1] for x in vals_sorted],
                        match_threshold=threshold,
                        no_match_max_prob=0.42,
                        min_single_match_margin=0.06,
                        min_confident_single_match=0.88,
                    )

                    # Keep the externally written candidate list compact.
                    selected_candidates = [
                        x[2]
                        for x in vals_sorted[
                            :args.output_candidate_limit
                        ]
                    ]

                    # If the entity has no scored candidates, retain the
                    # retrieval ordering so the output is still useful.
                    if not selected_candidates:
                        selected_candidates = candidate_lists[local][
                            :args.output_candidate_limit
                        ]

                cand_ids = [
                    index.ids[j]
                    for j in selected_candidates
                ]

                fm.write(
                    f"{q['id'][i]}\t"
                    f"{','.join(selected)}\n"
                )
                fc.write(
                    f"{q['id'][i]}\t"
                    f"{','.join(cand_ids)}\n"
                )

                chunk_matches += len(selected)

            total_pairs += sum(
                len(x)
                for x in candidate_lists
            )
            total_matches += chunk_matches

            done = end
            elapsed = time.time() - start
            rate = done / elapsed if elapsed else 0
            eta = (
                (total - done) / rate / 60
                if rate
                else 0
            )
            avg = (
                total_pairs / done
                if done
                else 0
            )

            print(
                f"[{done:,}/{total:,}] "
                f"{done / total * 100:5.2f}% | "
                f"candidates {total_pairs:,} | "
                f"avg/entity {avg:.2f} | "
                f"matches {total_matches:,} | "
                f"chunk {time.time() - ct:.2f}s | "
                f"throughput {done / elapsed:,.0f}/s | "
                f"ETA {eta:.1f}m",
                flush=True,
            )

    print(
        f"Finished in "
        f"{(time.time() - t0) / 60:.2f} min"
    )
    print(
        "Average retrieved candidates/entity: "
        f"{np.mean(candidate_counts):.3f}"
    )
    print(
        "Max retrieved candidates/entity: "
        f"{max(candidate_counts) if candidate_counts else 0}"
    )
    print(mp)
    print(cp)


if __name__ == "__main__":
    main()
