"""Laptop-friendly streaming inference for business entity resolution.

Uses one prefix index and one selective-token index instead of the current
multi-index Python structure, then computes only 15 features for a small
candidate set.
"""
from __future__ import annotations
import os, time
from collections import defaultdict
from typing import Dict, List, Set, Tuple
import numpy as np
import pandas as pd
import joblib, json
from rapidfuzz import fuzz
from ..preprocessing import preprocess_dataframe
from ..features.laptop_fast_features import compute_fast_features_batch, FEATURE_NAMES_FAST
from ..evaluation.entity_decision import select_entity_matches

STOP={"the","a","an","and","or","inc","incorporated","llc","ltd","limited","corp","corporation","co","company","pvt","private","sa","sas","sarl","gmbh","ag","spa","srl","bv","nv","oy","ab","in","at","of","to","for","on","by","from"}
GENERIC={"store","shop","market","restaurant","hotel","cafe","services","service","center","centre","group","holdings","enterprise","enterprises","solutions","international","global","trading","street","road","avenue","drive","lane","building","floor","suite","plaza","block","box","post","city","north","south","east","west","new","st","rd","ave","dr"}

def _token(text):
    for t in text.split():
        if len(t)>=3 and t not in STOP and t not in GENERIC:
            return t
    return (text.split() or [""])[0]

class LaptopCandidateIndex:
    def __init__(self,s2_df,s3_df,max_postings=4000):
        print("Building laptop-fast candidate index...")
        t=time.time()
        s=pd.concat([preprocess_dataframe(s2_df),preprocess_dataframe(s3_df)],ignore_index=True)
        self.ids=s.entity_id.astype(str).to_numpy()
        self.names=s.business_name_normalized.fillna("").astype(str).to_numpy()
        self.addresses=s.business_address_normalized.fillna("").astype(str).to_numpy()
        self.countries=s.country_normalized.fillna("").astype(str).to_numpy()
        self.n=len(s)
        self.prefix=defaultdict(list)
        self.token=defaultdict(list)
        for i in range(self.n):
            c=self.countries[i]; n=self.names[i]
            if not c: continue
            if n:
                self.prefix[(c,n[:3])].append(i)
                tok=_token(n)
                if tok and len(self.token[(c,tok)])<max_postings:
                    self.token[(c,tok)].append(i)
        print(f"Index ready: {self.n:,} candidates, {len(self.prefix):,} prefix blocks, {len(self.token):,} token blocks in {time.time()-t:.1f}s")
    def candidates(self,country,name,address,top_k=20):
        if not country: return []
        pool=set()
        if name:
            pool.update(self.prefix.get((country,name[:3]),()))
            tok=_token(name)
            if tok: pool.update(self.token.get((country,tok),()))
        if not pool:
            return []
        # Cheap ranking before feature extraction.
        scored=[]
        for i in pool:
            ns=fuzz.ratio(name,self.names[i])
            if ns<35: continue
            ad=fuzz.ratio(address,self.addresses[i]) if address else 0
            scored.append((max(ns,ad*0.85),i))
        scored.sort(key=lambda x:(-x[0],self.ids[x[1]]))
        return [i for _,i in scored[:top_k]]

def run_laptop_fast(s1_df,s2_df,s3_df,model_dir="models/laptop_fast",
                    output_dir="output",top_k=20,chunk_size=5000,
                    no_match_max_prob=.42,min_single_match_margin=.06,
                    min_confident_single_match=.88):
    model=joblib.load(os.path.join(model_dir,"entity_resolution_model.joblib"))
    with open(os.path.join(model_dir,"model_metadata.json")) as f: meta=json.load(f)
    threshold=float(meta.get("decision_threshold",.70))
    index=LaptopCandidateIndex(s2_df,s3_df)
    s1=preprocess_dataframe(s1_df)
    ids=s1.entity_id.astype(str).to_numpy()
    names=s1.business_name_normalized.fillna("").astype(str).to_numpy()
    addrs=s1.business_address_normalized.fillna("").astype(str).to_numpy()
    countries=s1.country_normalized.fillna("").astype(str).to_numpy()
    os.makedirs(output_dir,exist_ok=True)
    mp=os.path.join(output_dir,"matching_results.tsv")
    cp=os.path.join(output_dir,"candidate_pairs.tsv")
    total=len(ids); start=time.time(); total_pairs=0; total_matches=0
    with open(mp,"w",encoding="utf-8",buffering=1) as fm, open(cp,"w",encoding="utf-8",buffering=1) as fc:
        fm.write("source1_entity_id\tmatched_entity_ids\n")
        fc.write("source1_entity_id\tcandidate_entity_ids\n")
        for start_i in range(0,total,chunk_size):
            end_i=min(start_i+chunk_size,total)
            cand_lists=[]; pair_s=[]; pair_c=[]
            for i in range(start_i,end_i):
                ci=index.candidates(countries[i],names[i],addrs[i],top_k)
                cand_lists.append(ci)
                for x in ci: pair_s.append(i); pair_c.append(x)
            probs_by={}
            if pair_s:
                X=compute_fast_features_batch(
                    [names[i] for i in pair_s],[addrs[i] for i in pair_s],[countries[i] for i in pair_s],
                    [index.names[j] for j in pair_c],[index.addresses[j] for j in pair_c],[index.countries[j] for j in pair_c])
                p=model.predict_proba(X)[:,1]
                for k,(i,j) in enumerate(zip(pair_s,pair_c)):
                    probs_by.setdefault(i,[]).append((index.ids[j],float(p[k])))
            chunk_matches=0
            for local,i in enumerate(range(start_i,end_i)):
                cand_ids=[index.ids[j] for j in cand_lists[local]]
                vals=probs_by.get(i,[])
                selected=select_entity_matches(
                    [x[0] for x in vals],[x[1] for x in vals],
                    match_threshold=threshold,no_match_max_prob=no_match_max_prob,
                    min_single_match_margin=min_single_match_margin,
                    min_confident_single_match=min_confident_single_match)
                fc.write(f"{ids[i]}\t{','.join(cand_ids)}\n")
                fm.write(f"{ids[i]}\t{','.join(selected)}\n")
                chunk_matches+=len(selected)
            total_pairs+=sum(len(x) for x in cand_lists); total_matches+=chunk_matches
            elapsed=time.time()-start; done=end_i; rate=done/elapsed if elapsed else 0
            eta=(total-done)/rate/60 if rate else 0
            print(f"[{done:,}/{total:,}] {done/total*100:5.1f}% | pairs {sum(len(x) for x in cand_lists):,} | matches {chunk_matches:,} | elapsed {elapsed/60:.1f}m | ETA {eta:.1f}m",flush=True)
    print(f"Finished in {(time.time()-start)/60:.2f} min; pairs={total_pairs:,}; matches={total_matches:,}")
    print(mp); print(cp)
    return mp,cp
