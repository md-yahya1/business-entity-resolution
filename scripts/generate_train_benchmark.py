"""Final compact inference: <=10 candidates/entity, batched ML, streaming TSV output."""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "code")))
from business_entity_resolution.src.preprocessing import preprocess_dataframe, extract_address_hints
from business_entity_resolution.src.blocking.laptop_final import build_index, _hash_keys, selective_token
from business_entity_resolution.src.features.laptop_final_features import compute_final_features
from business_entity_resolution.src.evaluation.entity_decision import select_entity_matches

def prepare_s1(df):
    s = preprocess_dataframe(df)
    hints = [extract_address_hints(x) for x in s.business_address_normalized.fillna("").astype(str)]
    return {
        "id": s.entity_id.astype(str).to_numpy(),
        "name": s.business_name_normalized.fillna("").astype(str).to_numpy(),
        "address": s.business_address_normalized.fillna("").astype(str).to_numpy(),
        "country": s.country_normalized.fillna("").astype(str).to_numpy(),
        "postal": np.array([h["postal_code"] for h in hints], dtype=object),
        "house": np.array([h["house_number"] for h in hints], dtype=object),
        "city": np.array([h["city"] for h in hints], dtype=object),
        "state": np.array([h["state"] for h in hints], dtype=object),
        "prefix3": np.array([x[:3] for x in s.business_name_normalized.fillna("").astype(str)], dtype=object),
        "token": np.array([selective_token(x) for x in s.business_name_normalized.fillna("").astype(str)], dtype=object),
        "address_token": np.array([selective_token(x) for x in s.business_address_normalized.fillna("").astype(str)], dtype=object),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="dataset/test")
    ap.add_argument("--model-dir", default="models/laptop_final")
    ap.add_argument("--output-dir", default="output/laptop_final")
    ap.add_argument("--chunk-size", type=int, default=5000)
    ap.add_argument("--limit", type=int, default=0, help="0=all; use 5000 for benchmark")
    ap.add_argument("--posting-cap", type=int, default=64)
    ap.add_argument("--exact-cap", type=int, default=32)
    args = ap.parse_args()

    t0 = time.time()
    s1 = pd.read_csv(os.path.join(args.data_dir, "train_source1.tsv"), sep="\t")
    s2 = pd.read_csv(os.path.join(args.data_dir, "train_source2.tsv"), sep="\t")
    s3 = pd.read_csv(os.path.join(args.data_dir, "train_source3.tsv"), sep="\t")
    print(f"Loaded {len(s1):,} S1 / {len(s2):,} S2 / {len(s3):,} S3")

    model = joblib.load(os.path.join(args.model_dir, "entity_resolution_model.joblib"))
    with open(os.path.join(args.model_dir, "model_metadata.json"), encoding="utf-8") as f:
        meta = json.load(f)
    threshold = float(meta["decision_threshold"])

    index = build_index(s2, s3, posting_cap=args.posting_cap, exact_cap=args.exact_cap)
    print(f"Index: {index.n:,} candidates built in {index.build_seconds:.1f}s")

    q = prepare_s1(s1)
    total = len(q["id"]) if not args.limit else min(args.limit, len(q["id"]))
    os.makedirs(args.output_dir, exist_ok=True)
    mp = os.path.join(args.output_dir, "matching_results.tsv")
    cp = os.path.join(args.output_dir, "candidate_pairs.tsv")

    total_pairs = 0
    total_matches = 0
    candidate_counts = []
    start = time.time()

    with open(mp, "w", encoding="utf-8", buffering=1) as fm, open(cp, "w", encoding="utf-8", buffering=1) as fc:
        fm.write("source1_entity_id\tmatched_entity_ids\n")
        fc.write("source1_entity_id\tcandidate_entity_ids\n")

        for base in range(0, total, args.chunk_size):
            end = min(base + args.chunk_size, total)
            ct = time.time()
            # Hash all query keys once per chunk; candidate lookups then use only binary search.
            cc = q["country"][base:end]
            h_name = _hash_keys(cc, q["name"][base:end])
            h_prefix = _hash_keys(cc, q["prefix3"][base:end])
            h_token = _hash_keys(cc, q["token"][base:end])
            h_postal = _hash_keys(cc, q["postal"][base:end])
            h_house = _hash_keys(cc, q["house"][base:end])
            h_city = _hash_keys(cc, q["city"][base:end])
            h_address_token = _hash_keys(cc, q["address_token"][base:end])
            h_address = _hash_keys(cc, q["address"][base:end])
            h_postal_house = _hash_keys(
                cc,
                np.array(
                    [p + "|" + h for p, h in zip(q["postal"][base:end], q["house"][base:end])],
                    dtype=object,
                ),
            )
            all_pairs = []
            candidate_lists = []
            direct = {}

            for local, i in enumerate(range(base, end)):
                cands = index.candidates(
                    q["country"][i], q["name"][i], q["address"][i],
                    q["postal"][i], q["house"][i], q["city"][i], q["state"][i],
                    limit=10,
                    hashes=(h_name[i-base], h_prefix[i-base], h_token[i-base],
                            h_postal[i-base], h_house[i-base], h_city[i-base], h_address_token[i-base]),
                )
                cands = cands.tolist()
                candidate_lists.append(cands)
                candidate_counts.append(len(cands))

                # Exact fast path: only bypass ML when the exact-name block is unique.
                exact = [j for j in cands if index.names[j] == q["name"][i] and q["name"][i]]
                if len(exact) == 1:
                    direct[local] = index.ids[exact[0]]
                else:
                    for j in cands:
                        all_pairs.append((local, j))

            probs = {}
            if all_pairs:
                li = np.fromiter((x[0] for x in all_pairs), dtype=np.int32)
                ri = np.fromiter((x[1] for x in all_pairs), dtype=np.int32)

                X = compute_final_features(
                    [q["name"][base+i] for i in li],
                    [q["address"][base+i] for i in li],
                    [q["country"][base+i] for i in li],
                    [q["postal"][base+i] for i in li],
                    [q["house"][base+i] for i in li],
                    [q["city"][base+i] for i in li],
                    [q["state"][base+i] for i in li],
                    index.names[ri].tolist(),
                    index.addresses[ri].tolist(),
                    index.countries[ri].tolist(),
                    index.postal[ri].tolist(),
                    index.house[ri].tolist(),
                    index.city[ri].tolist(),
                    index.state[ri].tolist(),
                )
                p = model.predict_proba(X)[:, 1]
                for k, (local, j) in enumerate(all_pairs):
                    probs.setdefault(local, []).append((index.ids[j], float(p[k])))

            chunk_matches = 0
            for local, i in enumerate(range(base, end)):
                if local in direct:
                    selected = [direct[local]]
                else:
                    vals = probs.get(local, [])
                    selected = select_entity_matches(
                        [x[0] for x in vals], [x[1] for x in vals],
                        match_threshold=threshold,
                        no_match_max_prob=0.42,
                        min_single_match_margin=0.06,
                        min_confident_single_match=0.88,
                    )

                cand_ids = [index.ids[j] for j in candidate_lists[local]]
                fm.write(f"{q['id'][i]}\t{','.join(selected)}\n")
                fc.write(f"{q['id'][i]}\t{','.join(cand_ids)}\n")
                chunk_matches += len(selected)

            total_pairs += sum(len(x) for x in candidate_lists)
            total_matches += chunk_matches
            done = end
            elapsed = time.time() - start
            rate = done / elapsed if elapsed else 0
            eta = (total - done) / rate / 60 if rate else 0
            avg = total_pairs / done if done else 0
            print(
                f"[{done:,}/{total:,}] {done/total*100:5.2f}% | "
                f"candidates {total_pairs:,} | avg/entity {avg:.2f} | "
                f"matches {total_matches:,} | chunk {time.time()-ct:.2f}s | "
                f"throughput {done/elapsed:,.0f}/s | ETA {eta:.1f}m",
                flush=True,
            )

    print(f"Finished in {(time.time()-t0)/60:.2f} min")
    print(f"Average candidates/entity: {np.mean(candidate_counts):.3f}")
    print(f"Max candidates/entity: {max(candidate_counts) if candidate_counts else 0}")
    print(mp)
    print(cp)

if __name__ == "__main__":
    main()
