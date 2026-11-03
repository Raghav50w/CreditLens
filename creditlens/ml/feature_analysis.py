"""Feature-importance analysis behind the per-dataset feature selection.

Diagnostic only - it ranks every cleaned column by permutation importance,
which is how the shortlist in ``config.DATASET_META`` was chosen. Nothing here
writes artifacts.

Permutation importance shuffles one column at a time and measures how much
ROC-AUC drops on rows the model was not fit on. Unlike a forest's built-in
``feature_importances_``, it does not favour columns with many distinct
values, so strong yes/no and category columns are ranked fairly.

Everything happens inside the same training split ``ml.train`` uses (same
seed): the model is fit on part of it and scored on the rest, so
the held-out test rows play no part in choosing the features.

    python -m creditlens.ml.feature_analysis
"""

import logging
import sys

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

from creditlens.config import DATASET_META, RANDOM_STATE, RAW_DIR, TEST_SIZE
from creditlens.ml.preprocess import CLEANERS

log = logging.getLogger(__name__)


def _report(dataset: str, X: pd.DataFrame, y: pd.Series) -> None:
    """Fit on part of the training split, rank features on the rest."""
    X_fit, X_val, y_fit, y_val = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )
    model = HistGradientBoostingClassifier(random_state=RANDOM_STATE).fit(X_fit, y_fit)
    result = permutation_importance(
        model, X_val, y_val, scoring="roc_auc", n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1
    )

    importances = pd.Series(result.importances_mean, index=X.columns).sort_values(ascending=False)
    spread = pd.Series(result.importances_std, index=X.columns)
    selected = set(DATASET_META[dataset]["features"])

    log.info("")
    log.info("=" * 64)
    log.info(" %s - permutation importance, ROC-AUC drop (%d features)", dataset, len(X.columns))
    log.info("=" * 64)
    for rank, (feature, importance) in enumerate(importances.items(), 1):
        marker = "  <-- SELECTED" if feature in selected else ""
        log.info("  %2d. %-32s %.4f +/- %.4f%s", rank, feature, importance, spread[feature], marker)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for dataset, clean in CLEANERS.items():
        X, y = clean(RAW_DIR)
        X_train, _, y_train, _ = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
        _report(dataset, X_train, y_train)
    return 0


if __name__ == "__main__":
    sys.exit(main())
