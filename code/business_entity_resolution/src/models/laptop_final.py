"""Compact final HistGradientBoosting model."""
from sklearn.ensemble import HistGradientBoostingClassifier

FINAL_MODEL_PARAMS = dict(
    max_iter=100,
    learning_rate=0.09,
    max_leaf_nodes=15,
    max_depth=7,
    min_samples_leaf=10,
    l2_regularization=0.1,
    early_stopping=True,
    random_state=42,
)

def build_final_model():
    return HistGradientBoostingClassifier(**FINAL_MODEL_PARAMS)
