"""Hyperparameter search and decision cut-offs, used by ``train --tune``.

The search runs locally and its winners are saved to TUNED_PARAMS_PATH, so a
plain train (CI, Docker) reuses them without searching. Everything here sees
the training split only.
"""

import json
import logging

import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import (
    ParameterGrid,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
)

from creditlens.config import CV_FOLDS, RANDOM_STATE, TUNED_PARAMS_PATH, TUNING_ITERATIONS
from creditlens.ml.evaluate import make_pipeline

log = logging.getLogger(__name__)

# Search spaces, keyed like train.get_models().
PARAM_GRIDS = {
    "logistic_regression": {"C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]},
    "random_forest": {
        "max_depth": [6, 8, 10, 12, 16],
        "min_samples_leaf": [1, 2, 5, 10],
        "max_features": ["sqrt", 0.5, 0.8],
    },
    "hist_gradient_boosting": {
        "learning_rate": [0.02, 0.05, 0.1, 0.2],
        "max_depth": [3, 5, 8, None],
        "max_leaf_nodes": [15, 31, 63],
        "l2_regularization": [0.0, 0.1, 1.0],
    },
    "mlp": {
        "hidden_layer_sizes": [[32], [64], [64, 32], [128, 64]],
        "alpha": [0.0001, 0.001, 0.01, 0.1],
        "learning_rate_init": [0.001, 0.003, 0.01],
        "early_stopping": [True, False],
    },
}


def _folds() -> StratifiedKFold:
    return StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)


def search_params(name: str, estimator, X_train, y_train, categorical=()) -> dict:
    """Randomized search over ``PARAM_GRIDS[name]``, scored by ROC-AUC."""
    grid = {f"model__{key}": values for key, values in PARAM_GRIDS[name].items()}
    search = RandomizedSearchCV(
        make_pipeline(estimator, categorical),
        grid,
        n_iter=min(TUNING_ITERATIONS, len(ParameterGrid(grid))),
        scoring="roc_auc",
        cv=_folds(),
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    best = {key.removeprefix("model__"): value for key, value in search.best_params_.items()}
    log.info("  tuned %s: %s (cv roc_auc %.3f)", name, best, search.best_score_)
    return best


def best_threshold(estimator, X_train, y_train, categorical=()) -> float:
    """The probability cut-off with the best F1 on out-of-fold predictions.

    0.5 is only the library default; with 10-20% positives it is an arbitrary
    line. Out-of-fold predictions keep the test split out of the choice.
    """
    positive = cross_val_predict(
        make_pipeline(estimator, categorical),
        X_train,
        y_train,
        cv=_folds(),
        method="predict_proba",
    )[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_train, positive)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    # The last precision/recall pair has no matching threshold.
    return round(float(thresholds[np.argmax(f1[:-1])]), 4)


def load_tuned(dataset: str) -> dict | None:
    """Saved tuning for ``dataset`` ({model: {params, threshold}}), if any."""
    if not TUNED_PARAMS_PATH.exists():
        return None
    return json.loads(TUNED_PARAMS_PATH.read_text(encoding="utf-8")).get(dataset)
