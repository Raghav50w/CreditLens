"""Train and evaluate every model for every dataset.

    python -m creditlens.ml.train          # reuse saved tuning (fast; CI, Docker)
    python -m creditlens.ml.train --tune   # search again, then save it

For each CSV in ``datasets/`` this script:

1. splits the data (stratified 80/20),
2. with ``--tune``: runs a randomized hyperparameter search and picks each
   model's decision cut-off, both on the training split only, and saves them
   to ``creditlens/tuned_params.json`` (see ``ml/tune.py``); otherwise loads
   that file,
3. cross-validates each estimator on the training split,
4. scores it once against the held-out test split,
5. refits the estimator on the full dataset and pickles that artifact,
6. writes every score to ``models/metrics.json``.

Step 5 is deliberate: the reported numbers come from data the model never
saw, while the shipped artifact is trained on everything available.
"""

import argparse
import json
import logging
import pickle
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import FixedThresholdClassifier, train_test_split
from sklearn.neural_network import MLPClassifier

from creditlens.config import (
    DATASET_META,
    DATASETS_DIR,
    METRICS_FILENAME,
    MODELS_DIR,
    RANDOM_STATE,
    TEST_SIZE,
    TUNED_PARAMS_PATH,
    feature_meta,
)
from creditlens.ml.evaluate import baseline_scores, evaluate, make_pipeline
from creditlens.ml.tune import best_threshold, load_tuned, search_params

log = logging.getLogger(__name__)


def get_models() -> dict:
    """Fresh, unfitted base estimators keyed by artifact name (no ensemble)."""
    return {
        "logistic_regression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            class_weight="balanced", random_state=RANDOM_STATE
        ),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=500,
            random_state=RANDOM_STATE,
        ),
    }


def build_ensemble(models: dict) -> VotingClassifier:
    """Soft-voting ensemble that averages the probabilities of ``models``."""
    return VotingClassifier(
        estimators=[(name, clone(model)) for name, model in models.items()],
        voting="soft",
    )


def tune_dataset(X_train, y_train, categorical=()) -> dict:
    """Search parameters and cut-offs for one dataset: {model: {params, threshold}}."""
    tuned = {}
    base = {}
    for name, estimator in get_models().items():
        params = search_params(name, estimator, X_train, y_train, categorical)
        base[name] = clone(estimator).set_params(**params)
        tuned[name] = {"params": params}

    base["ensemble"] = build_ensemble(base)
    tuned["ensemble"] = {"params": {}}

    for name, estimator in base.items():
        tuned[name]["threshold"] = best_threshold(estimator, X_train, y_train, categorical)
        log.info("  %s cut-off %.4f", name, tuned[name]["threshold"])
    return tuned


def saved_tuning(dataset: str) -> dict:
    """Saved tuning for ``dataset``; defaults with a 0.5 cut-off if none exists."""
    tuned = load_tuned(dataset)
    if tuned is None:
        log.warning("  no saved tuning for %s; using defaults (run with --tune)", dataset)
        tuned = {name: {"params": {}, "threshold": 0.5} for name in [*get_models(), "ensemble"]}
    return tuned


def build_models(tuned: dict) -> dict:
    """Every model with its tuned parameters and decision cut-off applied.

    FixedThresholdClassifier leaves predict_proba untouched and makes predict
    use the saved cut-off, so evaluation and the API both honour it.
    """
    base = {
        name: clone(estimator).set_params(**tuned[name]["params"])
        for name, estimator in get_models().items()
    }
    base["ensemble"] = build_ensemble(base)
    return {
        name: FixedThresholdClassifier(
            estimator, threshold=tuned[name]["threshold"], response_method="predict_proba"
        )
        for name, estimator in base.items()
    }


def resolve_feature_columns(df: pd.DataFrame, dataset: str) -> tuple[list[str], str]:
    """Feature columns, in DATASET_META order, and the target column.

    The order comes from config, never from the CSV, so the API, the UI and
    the trained model cannot disagree about it.
    """
    meta = DATASET_META[dataset]
    feature_cols = list(meta["features"])
    target_col = meta["target_column"]
    missing = [c for c in [*feature_cols, target_col] if c not in df.columns]
    if missing:
        raise ValueError(f"{dataset}: dataset is missing columns {missing}")
    return feature_cols, target_col


def train_dataset(
    csv_path: Path, dataset: str, models_dir: Path = MODELS_DIR, tune: bool = False
) -> dict:
    """Train, evaluate, and persist every model for one dataset."""
    log.info("=" * 62)
    log.info("  %s", dataset)
    log.info("=" * 62)

    df = pd.read_csv(csv_path)
    feature_cols, target_col = resolve_feature_columns(df, dataset)

    X = df[feature_cols].to_numpy(dtype=float)
    y = df[target_col].astype(int).to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    models_dir.mkdir(parents=True, exist_ok=True)
    # Positions of the unordered code columns, one-hot encoded by the pipeline.
    categorical = [
        i for i, col in enumerate(feature_cols) if feature_meta(dataset, col).get("categorical")
    ]

    tuned = tune_dataset(X_train, y_train, categorical) if tune else saved_tuning(dataset)
    models = build_models(tuned)

    results = {}
    for name, estimator in models.items():
        log.info("  training %s...", name)
        results[name] = evaluate(estimator, X_train, y_train, X_test, y_test, categorical)
        results[name]["params"] = tuned[name]["params"]
        results[name]["threshold"] = tuned[name]["threshold"]

        # The shipped artifact is the whole pipeline (imputer, scaler,
        # one-hot encoder, model) refit on the full dataset, so the API feeds
        # it raw form values exactly as training did.
        shipped = make_pipeline(estimator, categorical).fit(X, y)
        with open(models_dir / f"{dataset}_{name}.pkl", "wb") as fh:
            pickle.dump(shipped, fh)

        holdout = results[name]["holdout"]
        cv = results[name]["cross_validation"]["accuracy"]
        log.info(
            "    cv accuracy %.3f (+/- %.3f) | test accuracy %.3f | roc_auc %.3f | ks %.3f",
            cv["mean"],
            cv["std"],
            holdout["accuracy"],
            holdout["roc_auc"],
            holdout["ks"],
        )

    return {
        "n_samples": int(len(df)),
        "n_features": len(feature_cols),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "class_balance": {str(k): int(v) for k, v in pd.Series(y).value_counts().items()},
        "baseline": baseline_scores(y_train, y_test),
        "models": results,
    }


def _write_json(path: Path, obj) -> None:
    """Save the tuning as readable JSON."""
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _clear_artifacts(models_dir: Path) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)
    stale = list(models_dir.glob("*.pkl")) + list(models_dir.glob("*.json"))
    for path in stale:
        path.unlink()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Train and evaluate every model.")
    parser.add_argument(
        "--tune",
        action="store_true",
        help=f"search hyperparameters and cut-offs, and save them to {TUNED_PARAMS_PATH.name}",
    )
    args = parser.parse_args()

    if not DATASETS_DIR.is_dir():
        log.error("datasets directory not found: %s", DATASETS_DIR)
        log.error("run `python -m creditlens.ml.preprocess` first")
        return 1

    csv_files = sorted(DATASETS_DIR.glob("*.csv"))
    if not csv_files:
        log.error("no .csv files found in %s", DATASETS_DIR)
        return 1

    _clear_artifacts(MODELS_DIR)
    log.info("found %d dataset(s): %s", len(csv_files), [p.name for p in csv_files])

    metrics = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "datasets": {},
    }

    for csv_path in csv_files:
        dataset = csv_path.stem
        metrics["datasets"][dataset] = train_dataset(csv_path, dataset, tune=args.tune)

    metrics_path = MODELS_DIR / METRICS_FILENAME
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    if args.tune:
        tuned = {
            dataset: {
                name: {"params": model["params"], "threshold": model["threshold"]}
                for name, model in result["models"].items()
            }
            for dataset, result in metrics["datasets"].items()
        }
        _write_json(TUNED_PARAMS_PATH, tuned)
        log.info("  tuning saved to %s", TUNED_PARAMS_PATH)

    log.info("=" * 62)
    log.info("  artifacts written to %s", MODELS_DIR)
    log.info("  metrics written to %s", metrics_path)
    log.info("=" * 62)
    return 0


if __name__ == "__main__":
    sys.exit(main())
