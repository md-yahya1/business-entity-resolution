"""Train the lightweight laptop-fast ER model from an existing feature parquet."""
from __future__ import annotations
import argparse, json, os
from datetime import datetime
import joblib, pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import GroupShuffleSplit
from business_entity_resolution.src.features.laptop_fast_features import FEATURE_NAMES_FAST
from business_entity_resolution.src.models.laptop_fast import build_laptop_fast_model

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",default="artifacts/candidates/training_features.parquet")
    p.add_argument("--output-dir",default="models/laptop_fast")
    p.add_argument("--seed",type=int,default=42)
    args=p.parse_args()
    if not os.path.exists(args.input):
        raise FileNotFoundError(args.input)
    df=pd.read_parquet(args.input)
    missing=[c for c in FEATURE_NAMES_FAST+["label","entity_group_id"] if c not in df.columns]
    if missing: raise ValueError(f"Missing columns: {missing}")
    X=df[FEATURE_NAMES_FAST].astype("float32").values
    y=df["label"].astype("int8").values
    groups=df["entity_group_id"].values
    g1=GroupShuffleSplit(n_splits=1,test_size=.20,random_state=args.seed)
    tr,va=next(g1.split(X,y,groups))
    model=build_laptop_fast_model()
    print(f"Training laptop-fast model on {len(tr):,} pairs / {len(FEATURE_NAMES_FAST)} features...")
    model.fit(X[tr],y[tr])
    prob=model.predict_proba(X[va])[:,1]
    # Lightweight threshold search; entity gates remain in submission inference.
    best=(0.0,0.5)
    for t in [x/100 for x in range(50,96,2)]:
        pred=(prob>=t).astype("int8")
        prec=precision_score(y[va],pred,zero_division=0)
        rec=recall_score(y[va],pred,zero_division=0)
        f05=(1.25*prec*rec)/(0.25*prec+rec) if prec+rec else 0.0
        if f05>best[0]: best=(f05,t)
    os.makedirs(args.output_dir,exist_ok=True)
    model_path=os.path.join(args.output_dir,"entity_resolution_model.joblib")
    joblib.dump(model,model_path,compress=3)
    meta={"model_type":"LaptopFastHistGradientBoosting","feature_names":FEATURE_NAMES_FAST,
          "decision_threshold":best[1],"validation_pair_f0_5":best[0],
          "training_date":datetime.now().isoformat(),"hyperparameters":model.get_params()}
    with open(os.path.join(args.output_dir,"model_metadata.json"),"w") as f: json.dump(meta,f,indent=2)
    with open(os.path.join(args.output_dir,"feature_config.json"),"w") as f: json.dump({"feature_names":FEATURE_NAMES_FAST,"num_features":len(FEATURE_NAMES_FAST)},f,indent=2)
    print(f"Validation pair F0.5: {best[0]:.4f}")
    print(f"Decision threshold: {best[1]:.2f}")
    print(f"Saved: {model_path}")

if __name__=="__main__": main()
