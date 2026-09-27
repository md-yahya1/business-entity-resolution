"""Fast laptop-scale entity resolution inference.

Key design:
- exact normalized-name/address lookup first
- country + 3-character-name blocks
- RapidFuzz C++ top-N extraction inside each block (never Python-score the whole block)
- only 15 lightweight model features
- chunked streaming output
"""
from __future__ import annotations
import os, time, json
from collections import defaultdict
import numpy as np
import pandas as pd
import joblib
from rapidfuzz import fuzz, process
from ..preprocessing import preprocess_dataframe
from ..features.laptop_fast_features import compute_fast_features_batch
from ..evaluation.entity_decision import select_entity_matches

STOP={"the","a","an","and","or","inc","incorporated","llc","ltd","limited","corp","corporation","co","company","pvt","private","sa","sas","sarl","gmbh","ag","spa","srl","bv","nv","oy","ab","in","at","of","to","for","on","by","from"}
GENERIC={"store","shop","market","restaurant","hotel","cafe","services","service","center","centre","group","holdings","enterprise","enterprises","solutions","international","global","trading","street","road","avenue","drive","lane","building","floor","suite","plaza","block","box","post","city","north","south","east","west","new","st","rd","ave","dr"}

def _token(text):
    for t in text.split():
        if len(t)>=3 and t not in STOP and t not in GENERIC:
            return t
    return (text.split() or [""])[0]

class LaptopCandidateIndex:
    def __init__(self,s2_df,s3_df,max_token_postings=2500):
        t=time.time()
        print("Preprocessing candidate corpus...")
        s=pd.concat([preprocess_dataframe(s2_df),preprocess_dataframe(s3_df)],ignore_index=True)
        self.ids=s.entity_id.astype(str).to_numpy()
        self.names=s.business_name_normalized.fillna("").astype(str).to_numpy()
        self.addresses=s.business_address_normalized.fillna("").astype(str).to_numpy()
        self.countries=s.country_normalized.fillna("").astype(str).to_numpy()
        self.n=len(s)

        # Lists here are only block membership; fuzzy ranking is delegated to RapidFuzz.
        self.prefix=defaultdict(list)
        self.token=defaultdict(list)
        self.exact_name=defaultdict(list)
        self.exact_address=defaultdict(list)

        for i,(c,n,a) in enumerate(zip(self.countries,self.names,self.addresses)):
            if not c: continue
            if n:
                self.prefix[(c,n[:3])].append(i)
                self.exact_name[(c,n)].append(i)
                tok=_token(n)
                if tok and len(self.token[(c,tok)])<max_token_postings:
                    self.token[(c,tok)].append(i)
            if a:
                self.exact_address[(c,a)].append(i)

        self.prefix_names={k:[self.names[i] for i in v] for k,v in self.prefix.items()}
        self.token_names={k:[self.names[i] for i in v] for k,v in self.token.items()}
        print(f"Index ready: {self.n:,} candidates, {len(self.prefix):,} prefix blocks, "
              f"{len(self.token):,} token blocks in {time.time()-t:.1f}s",flush=True)

    def candidates(self,country,name,address,top_k=20):
        if not country: return []
        seen=set()

        # Exact matches are essentially free and should always survive.
        for i in self.exact_name.get((country,name),()):
            seen.add(i)
        for i in self.exact_address.get((country,address),()):
            seen.add(i)

        key=(country,name[:3]) if name else None
        choices=self.prefix_names.get(key,[]) if key else []
        indices=self.prefix.get(key,[]) if key else []
        if choices:
            # RapidFuzz performs the scan in optimized native code.
            hits=process.extract(name,choices,scorer=fuzz.ratio,limit=top_k,score_cutoff=35)
            for _,_,pos in hits:
                seen.add(indices[pos])
            # Address is a useful secondary signal; only scan the same small block.
            if address and len(seen)<top_k:
                ah=process.extract(address,[self._addr(i) for i in indices],
                                   scorer=fuzz.ratio,limit=top_k,score_cutoff=35)
                for _,_,pos in ah:
                    seen.add(indices[pos])
        else:
            tok=_token(name)
            inds=self.token.get((country,tok),[]) if tok else []
            choices=self.token_names.get((country,tok),[]) if tok else []
            if choices:
                hits=process.extract(name,choices,scorer=fuzz.ratio,limit=top_k,score_cutoff=35)
                for _,_,pos in hits: seen.add(inds[pos])

        # Final ordering uses only the small candidate set.
        scored=[]
        for i in seen:
            ns=fuzz.ratio(name,self.names[i]) if name else 0
            ad=fuzz.ratio(address,self.addresses[i]) if address else 0
            scored.append((max(ns,ad*0.85),i))
        scored.sort(key=lambda x:(-x[0],self.ids[x[1]]))
        return [i for _,i in scored[:top_k]]

    def _addr(self,i):
        return self.addresses[i]

def run_laptop_fast(s1_df,s2_df,s3_df,model_dir="models/laptop_fast",
                    output_dir="output_laptop_fast",top_k=20,chunk_size=5000,
                    no_match_max_prob=.42,min_single_match_margin=.06,
                    min_confident_single_match=.88):
    model=joblib.load(os.path.join(model_dir,"entity_resolution_model.joblib"))
    with open(os.path.join(model_dir,"model_metadata.json"),encoding="utf-8") as f: meta=json.load(f)
    threshold=float(meta.get("decision_threshold",.70))
    index=LaptopCandidateIndex(s2_df,s3_df)

    print("Preprocessing Source-1...")
    s1=preprocess_dataframe(s1_df)
    ids=s1.entity_id.astype(str).to_numpy()
    names=s1.business_name_normalized.fillna("").astype(str).to_numpy()
    addrs=s1.business_address_normalized.fillna("").astype(str).to_numpy()
    countries=s1.country_normalized.fillna("").astype(str).to_numpy()
    total=len(ids); start=time.time()
    os.makedirs(output_dir,exist_ok=True)
    mp=os.path.join(output_dir,"matching_results.tsv"); cp=os.path.join(output_dir,"candidate_pairs.tsv")

    with open(mp,"w",encoding="utf-8",buffering=1) as fm, open(cp,"w",encoding="utf-8",buffering=1) as fc:
        fm.write("source1_entity_id\tmatched_entity_ids\n")
        fc.write("source1_entity_id\tcandidate_entity_ids\n")
        for start_i in range(0,total,chunk_size):
            end_i=min(start_i+chunk_size,total)
            cand_lists=[]; pair_s=[]; pair_c=[]
            t0=time.time()
            for i in range(start_i,end_i):
                ci=index.candidates(countries[i],names[i],addrs[i],top_k)
                cand_lists.append(ci)
                pair_s.extend([i-start_i]*len(ci)); pair_c.extend(ci)

            probs_by={}
            if pair_s:
                X=compute_fast_features_batch(
                    [names[start_i+i] for i in pair_s],
                    [addrs[start_i+i] for i in pair_s],
                    [countries[start_i+i] for i in pair_s],
                    [index.names[j] for j in pair_c],
                    [index.addresses[j] for j in pair_c],
                    [index.countries[j] for j in pair_c])
                p=model.predict_proba(X)[:,1]
                for k,(i,j) in enumerate(zip(pair_s,pair_c)):
                    probs_by.setdefault(i,[]).append((index.ids[j],float(p[k])))

            chunk_matches=0
            for local,i in enumerate(range(start_i,end_i)):
                vals=probs_by.get(local,[])
                selected=select_entity_matches(
                    [x[0] for x in vals],[x[1] for x in vals],
                    match_threshold=threshold,no_match_max_prob=no_match_max_prob,
                    min_single_match_margin=min_single_match_margin,
                    min_confident_single_match=min_confident_single_match)
                fc.write(f"{ids[i]}\t{','.join(index.ids[j] for j in cand_lists[local])}\n")
                fm.write(f"{ids[i]}\t{','.join(selected)}\n")
                chunk_matches+=len(selected)

            done=end_i; elapsed=time.time()-start; rate=done/elapsed if elapsed else 0
            eta=(total-done)/rate/60 if rate else 0
            print(f"[{done:,}/{total:,}] {done/total*100:5.1f}% | "
                  f"pairs {sum(map(len,cand_lists)):,} | matches {chunk_matches:,} | "
                  f"chunk {time.time()-t0:.1f}s | elapsed {elapsed/60:.1f}m | ETA {eta:.1f}m",flush=True)

    print(f"Finished in {(time.time()-start)/60:.2f} min")
    print(mp); print(cp)
    return mp,cp
