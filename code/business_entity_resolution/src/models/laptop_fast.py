"""Small HistGradientBoosting model used by the laptop-fast pipeline."""
from sklearn.ensemble import HistGradientBoostingClassifier

FAST_MODEL_PARAMS = dict(
    max_iter=120,
    learning_rate=0.08,
    max_leaf_nodes=15,
    max_depth=8,
    min_samples_leaf=10,
    l2_regularization=0.1,
    early_stopping=True,
    random_state=42,
)

def build_laptop_fast_model():
    return HistGradientBoostingClassifier(**FAST_MODEL_PARAMS)
