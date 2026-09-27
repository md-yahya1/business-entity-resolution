"""Train the compact final ER model from the existing training feature parquet."""
from __future__ import annotations
import argparse, json, os, sys
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "code")))
from business_entity_resolution.src.features.laptop_final_features import FEATURE_NAMES_FINAL
from business_entity_resolution.src.models.laptop_final import build_final_model

def f05(p, r):
    return (1.25*p*r)/(0.25*p+r) if (p+r) else 0.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="artifacts/candidates/training_features.parquet")
    ap.add_argument("--output-dir", default="models/laptop_final")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_parquet(args.input)
    missing = [c for c in FEATURE_NAMES_FINAL + ["label", "entity_group_id"] if c not in df.columns]
    if missing:
        raise ValueError("Missing columns: " + ", ".join(missing))

    X = df[FEATURE_NAMES_FINAL].astype("float32").to_numpy()
    y = df["label"].astype("int8").to_numpy()
    groups = df["entity_group_id"].to_numpy()

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=args.seed)
    train_idx, val_idx = next(splitter.split(X, y, groups))

    model = build_final_model()
    print(f"Training final model: {len(train_idx):,} pairs, {X.shape[1]} features")
    model.fit(X[train_idx], y[train_idx])

    prob = model.predict_proba(X[val_idx])[:, 1]
    best = (-1.0, 0.78, 0, 0, 0, 0)
    for t in np.arange(0.50, 0.951, 0.01):
        pred = (prob >= t).astype(np.int8)
        p = precision_score(y[val_idx], pred, zero_division=0)
        r = recall_score(y[val_idx], pred, zero_division=0)
        score = f05(p, r)
        if score > best[0]:
            best = (score, float(t), p, r, f1_score(y[val_idx], pred, zero_division=0), accuracy_score(y[val_idx], pred))

    os.makedirs(args.output_dir, exist_ok=True)
    model_path = os.path.join(args.output_dir, "entity_resolution_model.joblib")
    joblib.dump(model, model_path, compress=3)

    metadata = {
        "model_type": "HistGradientBoostingClassifier",
        "feature_names": FEATURE_NAMES_FINAL,
        "decision_threshold": best[1],
        "validation_metrics": {
            "precision": best[2], "recall": best[3], "f0_5": best[0],
            "f1": best[4], "accuracy": best[5]
        },
        "hyperparameters": model.get_params(),
        "training_timestamp": datetime.now().isoformat(timespec="seconds"),
        "random_seed": args.seed
    }
    with open(os.path.join(args.output_dir, "model_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    with open(os.path.join(args.output_dir, "feature_config.json"), "w", encoding="utf-8") as f:
        json.dump({"feature_names": FEATURE_NAMES_FINAL, "num_features": len(FEATURE_NAMES_FINAL)}, f, indent=2)

    print(f"Validation precision: {best[2]:.4f}")
    print(f"Validation recall:    {best[3]:.4f}")
    print(f"Validation F1:        {best[4]:.4f}")
    print(f"Validation F0.5:      {best[0]:.4f}")
    print(f"Decision threshold:   {best[1]:.2f}")
    print(f"Saved: {model_path}")

if __name__ == "__main__":
    main()
