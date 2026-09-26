"""
Ensemble model registry for entity-resolution pair classification.

Centralizes base learner definitions and ensemble compositions so training
code stays thin and new models can be added in one place.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .tuning import optimize_soft_weights


EstimatorFactory = Callable[[int], BaseEstimator]


@dataclass(frozen=True)
class BaseLearnerSpec:
    """Metadata for a reusable base classifier."""

    name: str
    factory: EstimatorFactory


def _hist_gradient_boosting(seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=450,
        max_depth=12,
        learning_rate=0.045,
        min_samples_leaf=6,
        l2_regularization=0.08,
        max_bins=255,
        random_state=seed,
    )


def _hist_gradient_boosting_alt(seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=520,
        max_depth=9,
        learning_rate=0.035,
        min_samples_leaf=4,
        l2_regularization=0.04,
        max_bins=255,
        random_state=seed + 17,
    )


def _extra_trees(seed: int) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=250,
        max_depth=18,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )


def _random_forest(seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=250,
        max_depth=18,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )


def _gradient_boosting(seed: int) -> GradientBoostingClassifier:
    return GradientBoostingClassifier(
        n_estimators=200,
        max_depth=7,
        learning_rate=0.06,
        subsample=0.85,
        random_state=seed,
    )


def _logistic_regression(seed: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    C=2.0,
                    class_weight="balanced",
                    random_state=seed,
                ),
            ),
        ]
    )


def _ada_boost(seed: int) -> AdaBoostClassifier:
    return AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=4, random_state=seed),
        n_estimators=150,
        learning_rate=0.08,
        random_state=seed,
    )


def get_base_learner_specs() -> Tuple[BaseLearnerSpec, ...]:
    """Ordered catalog of diverse base learners for ensembles."""
    return (
        BaseLearnerSpec("hist", _hist_gradient_boosting),
        BaseLearnerSpec("extra", _extra_trees),
        BaseLearnerSpec("rf", _random_forest),
        BaseLearnerSpec("gb", _gradient_boosting),
        BaseLearnerSpec("lr", _logistic_regression),
        BaseLearnerSpec("ada", _ada_boost),
    )


def build_base_estimators(seed: int) -> List[Tuple[str, BaseEstimator]]:
    return [(spec.name, spec.factory(seed)) for spec in get_base_learner_specs()]


def _tree_voting_estimators(seed: int) -> List[Tuple[str, BaseEstimator]]:
    return [
        (spec.name, spec.factory(seed))
        for spec in get_base_learner_specs()
        if spec.name in {"hist", "extra", "rf", "gb"}
    ]


def _extended_voting_estimators(seed: int) -> List[Tuple[str, BaseEstimator]]:
    return _tree_voting_estimators(seed) + [
        ("hist_alt", _hist_gradient_boosting_alt(seed)),
    ]


def build_soft_voting_ensemble(
    seed: int,
    weights: Optional[Sequence[float]] = None,
) -> VotingClassifier:
    estimators = _tree_voting_estimators(seed)
    return VotingClassifier(
        estimators=estimators,
        voting="soft",
        weights=list(weights) if weights is not None else None,
        n_jobs=-1,
    )


def build_full_soft_voting_ensemble(
    seed: int,
    weights: Optional[Sequence[float]] = None,
) -> VotingClassifier:
    estimators = build_base_estimators(seed)
    return VotingClassifier(
        estimators=estimators,
        voting="soft",
        weights=list(weights) if weights is not None else None,
        n_jobs=-1,
    )


def build_stacking_ensemble(
    seed: int,
    *,
    final_estimator: Optional[BaseEstimator] = None,
    include_linear: bool = True,
) -> StackingClassifier:
    if include_linear:
        estimators = build_base_estimators(seed)
    else:
        estimators = _tree_voting_estimators(seed)

    meta = final_estimator or LogisticRegression(
        max_iter=2000,
        C=1.0,
        class_weight="balanced",
        random_state=seed,
    )

    return StackingClassifier(
        estimators=estimators,
        final_estimator=meta,
        cv=5,
        stack_method="predict_proba",
        passthrough=False,
        n_jobs=-1,
    )


class WeightedVotingEnsemble(BaseEstimator, ClassifierMixin):
    """
    Soft-voting ensemble with validation-tuned convex weights.

    Fits each base estimator on training data, then searches weight vectors
    on a held-out validation set to maximize F1 at the optimal threshold.
    """

    def __init__(
        self,
        estimators: Iterable[Tuple[str, BaseEstimator]],
        default_weights: Optional[Sequence[float]] = None,
    ):
        self.estimators = list(estimators)
        self.default_weights = default_weights

    def fit(self, X: np.ndarray, y: np.ndarray) -> "WeightedVotingEnsemble":
        self.estimators_ = [(name, clone(est)) for name, est in self.estimators]
        for _, est in self.estimators_:
            est.fit(X, y)
        n = len(self.estimators_)
        if self.default_weights is not None:
            w = np.asarray(self.default_weights, dtype=float)
            self.weights_ = w / w.sum()
        else:
            self.weights_ = np.full(n, 1.0 / n)
        return self

    def tune_weights(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
        target_metric: str = "f0_5",
        random_state: int = 42,
        n_random_restarts: int = 10,
    ) -> np.ndarray:
        """Continuous weight search on validation probabilities."""
        val_probs = self._collect_proba(X_val)
        self.weights_ = optimize_soft_weights(
            val_probs,
            y_val,
            target_metric=target_metric,
            random_state=random_state,
            n_random_restarts=n_random_restarts,
        )
        return self.weights_

    def _collect_proba(self, X: np.ndarray) -> np.ndarray:
        return np.column_stack(
            [est.predict_proba(X)[:, 1] for _, est in self.estimators_]
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        pos = self._collect_proba(X) @ self.weights_
        pos = np.clip(pos, 0.0, 1.0)
        return np.column_stack([1.0 - pos, pos])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return {
            "estimators": self.estimators,
            "default_weights": self.default_weights,
        }

    def set_params(self, **params: Any) -> "WeightedVotingEnsemble":
        for key, value in params.items():
            setattr(self, key, value)
        return self


def build_weighted_voting_ensemble(seed: int) -> WeightedVotingEnsemble:
    return WeightedVotingEnsemble(estimators=_tree_voting_estimators(seed))


def build_weighted_extended_voting_ensemble(seed: int) -> WeightedVotingEnsemble:
    return WeightedVotingEnsemble(estimators=_extended_voting_estimators(seed))


def build_weighted_full_voting_ensemble(seed: int) -> WeightedVotingEnsemble:
    return WeightedVotingEnsemble(estimators=build_base_estimators(seed))


def build_candidate_models(seed: int, profile: str = "full") -> Dict[str, BaseEstimator]:
    """
    Return standalone and ensemble models compared during training.

    profile:
      - \"minimal\": smoke-test subset (pytest only)
      - \"fast\": core bases + primary ensembles (quicker retrains)
      - \"full\": every ensemble, including extended weight tuning (best quality)
    """
    if profile == "minimal":
        return {
            "HistGradientBoosting": HistGradientBoostingClassifier(
                max_iter=80,
                max_depth=6,
                learning_rate=0.1,
                random_state=seed,
            ),
        }

    display_names = {
        "hist": "HistGradientBoosting",
        "extra": "ExtraTrees",
        "rf": "RandomForest",
        "gb": "GradientBoosting",
        "lr": "LogisticRegression",
        "ada": "AdaBoost",
    }
    base = {display_names[spec.name]: spec.factory(seed) for spec in get_base_learner_specs()}

    ensembles = {
        "SoftVotingEnsemble": build_soft_voting_ensemble(
            seed, weights=[0.40, 0.25, 0.20, 0.15]
        ),
        "StackingEnsemble_HGB": build_stacking_ensemble(
            seed,
            include_linear=False,
            final_estimator=_hist_gradient_boosting(seed),
        ),
        "WeightedVotingEnsemble": build_weighted_voting_ensemble(seed),
    }

    if profile == "full":
        ensembles.update(
            {
                "WeightedExtendedVotingEnsemble": build_weighted_extended_voting_ensemble(seed),
                "FullSoftVotingEnsemble": build_full_soft_voting_ensemble(seed),
                "StackingEnsemble_LR": build_stacking_ensemble(seed, include_linear=True),
                "WeightedFullVotingEnsemble": build_weighted_full_voting_ensemble(seed),
            }
        )

    return {**base, **ensembles}


def is_weight_tuned_ensemble(model: BaseEstimator) -> bool:
    return isinstance(model, WeightedVotingEnsemble)
