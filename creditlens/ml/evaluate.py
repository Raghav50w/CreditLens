"""Scores a model: 5-fold cross-validation on the training split, then a
single score against the held-out test split.

Everything here takes raw, possibly incomplete features and wraps the model
in a Pipeline with its own imputer, StandardScaler and one-hot encoder. That matters:
filling or scaling before cross-validation would let each fold see the rows
it is about to be scored on, which inflates the numbers. The Pipeline refits
both inside every fold.
"""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from creditlens.config import CV_FOLDS, RANDOM_STATE

SCORING = ("accuracy", "precision", "recall", "f1", "roc_auc")


def make_pipeline(estimator, categorical=()):
    """Glue preprocessing and model together so they fit and predict as one unit.

    Number columns are median-imputed and scaled; the imputer learns its
    medians from whatever rows the pipeline is fit on, so test rows never
    shape the fill values. Columns listed in ``categorical`` (by position) are
    one-hot encoded instead, so a code like loan_intent=3 is not read as "more"
    than loan_intent=1.
    """
    numeric = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    if categorical:
        prepare = ColumnTransformer(
            [
                (
                    "onehot",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    list(categorical),
                )
            ],
            remainder=numeric,
        )
    else:
        prepare = numeric
    return Pipeline([("prepare", prepare), ("model", estimator)])


def label_scores(y_true, y_pred):
    """Metrics computed from the 0/1 predictions."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def holdout_scores(fitted, X_test, y_test):
    """Score a fitted pipeline against data it has never seen."""
    scores = label_scores(y_test, fitted.predict(X_test))

    # ROC-AUC and the credit-scoring metrics need a confidence score, not just
    # a 0/1 answer. Every model here has predict_proba.
    positive = fitted.predict_proba(X_test)[:, 1]
    scores.update(probability_scores(y_test, positive))

    return scores


def probability_scores(y_true, positive):
    """Ranking and calibration metrics computed from P(class 1).

    Gini and KS are the standard credit-scoring views of ranking quality;
    PR-AUC matters more than ROC-AUC when positives are rare; Brier score
    measures how trustworthy the probabilities themselves are.
    """
    roc_auc = float(roc_auc_score(y_true, positive))
    fpr, tpr, _ = roc_curve(y_true, positive)
    return {
        "roc_auc": roc_auc,
        "gini": 2 * roc_auc - 1,
        "ks": float(np.max(tpr - fpr)),
        "pr_auc": float(average_precision_score(y_true, positive)),
        "brier": float(brier_score_loss(y_true, positive)),
    }


def baseline_scores(y_train, y_test):
    """What always predicting the training majority class would score.

    Any model worth shipping has to beat this. Its probability is the
    training positive rate, so the ranking metrics sit at chance.
    """
    majority = int(np.mean(y_train) >= 0.5)
    y_pred = np.full(len(y_test), majority)
    positive = np.full(len(y_test), float(np.mean(y_train)))

    scores = {"majority_class": majority, **label_scores(y_test, y_pred)}
    scores.update(probability_scores(y_test, positive))
    return scores


def cross_validation_scores(estimator, X_train, y_train, categorical=()):
    """5-fold stratified cross-validation. Returns {metric: {mean, std}}."""
    folds = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    results = cross_validate(
        make_pipeline(estimator, categorical),
        X_train,
        y_train,
        cv=folds,
        scoring=list(SCORING),
        error_score="raise",
    )

    summary = {}
    for metric in SCORING:
        fold_scores = results[f"test_{metric}"]
        summary[metric] = {
            "mean": float(np.mean(fold_scores)),
            "std": float(np.std(fold_scores)),
        }
    return summary


def evaluate(estimator, X_train, y_train, X_test, y_test, categorical=()):
    """Cross-validate on the training split, then score once on the test split."""
    fitted = make_pipeline(estimator, categorical).fit(X_train, y_train)
    return {
        "cross_validation": cross_validation_scores(estimator, X_train, y_train, categorical),
        "holdout": holdout_scores(fitted, X_test, y_test),
    }
