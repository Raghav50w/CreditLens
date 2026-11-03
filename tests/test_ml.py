"""Training and preprocessing: are the published numbers honest?"""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from creditlens.config import DATASETS_DIR, MODEL_DISPLAY_NAMES, RANDOM_STATE
from creditlens.ml.evaluate import holdout_scores, make_pipeline
from creditlens.ml.train import train_dataset


@pytest.fixture(scope="session")
def trained(tmp_path_factory):
    """Train the Acquire dataset once for the whole session - it is the smallest."""
    models_dir = tmp_path_factory.mktemp("models")
    result = train_dataset(DATASETS_DIR / "loan_offer.csv", "loan_offer", models_dir=models_dir)
    return result, models_dir


def test_holdout_scores_are_not_training_scores():
    """The published metrics must describe data the model never saw.

    Trains a forest on pure random noise. It memorises what it was shown, so
    it scores near-perfectly there - and near chance on rows it has not seen.
    If both numbers came out high, the test split would be leaking.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    X = rng.normal(size=(200, 4))
    y = rng.integers(0, 2, size=200)

    X_train, X_test = X[:150], X[150:]
    y_train, y_test = y[:150], y[150:]

    forest = RandomForestClassifier(n_estimators=50, random_state=RANDOM_STATE)
    fitted = make_pipeline(forest).fit(X_train, y_train)

    assert holdout_scores(fitted, X_train, y_train)["accuracy"] > 0.95
    assert holdout_scores(fitted, X_test, y_test)["accuracy"] < 0.75


def test_pipeline_fills_missing_values():
    """The Credit Risk CSV keeps NaNs; the pipeline must fill them."""
    rng = np.random.default_rng(RANDOM_STATE)
    X = rng.normal(size=(100, 3))
    X[::7, 1] = np.nan
    y = rng.integers(0, 2, size=100)

    # LogisticRegression rejects NaN outright, so this only passes if the
    # pipeline imputes first.
    fitted = make_pipeline(LogisticRegression()).fit(X, y)

    assert fitted.predict_proba(X).shape == (100, 2)


def test_training_is_deterministic(trained, tmp_path):
    """random_state=42 means the published numbers are reproducible.

    Compared with approx, not ==. The forest sums its trees' votes in
    parallel (n_jobs=-1) and floating-point addition is not associative, so
    the very last decimal place can wobble between identical runs.
    """
    first, _ = trained
    second = train_dataset(
        DATASETS_DIR / "loan_offer.csv", "loan_offer", models_dir=tmp_path / "again"
    )

    for name in first["models"]:
        a = first["models"][name]
        b = second["models"][name]
        assert a["holdout"]["accuracy"] == pytest.approx(b["holdout"]["accuracy"])
        assert a["cross_validation"]["accuracy"]["mean"] == pytest.approx(
            b["cross_validation"]["accuracy"]["mean"]
        )


def test_scores_are_valid_probabilities(trained):
    result, _ = trained

    for model in result["models"].values():
        for name, value in model["holdout"].items():
            if value is not None:
                assert 0.0 <= value <= 1.0, name


def test_writes_every_expected_artifact(trained):
    _, models_dir = trained

    written = set()
    for path in models_dir.iterdir():
        written.add(path.name)

    expected = {f"loan_offer_{name}.pkl" for name in MODEL_DISPLAY_NAMES}
    assert written == expected


def test_models_beat_the_majority_class_baseline(trained):
    """A model that cannot out-rank "always predict the majority" is useless."""
    result, _ = trained

    assert result["baseline"]["roc_auc"] == pytest.approx(0.5)
    for model in result["models"].values():
        assert model["holdout"]["roc_auc"] > result["baseline"]["roc_auc"]


@pytest.mark.parametrize(
    ("dataset", "rows"),
    [("loan_offer", 5000), ("credit_risk", 32_409), ("attrition", 10_127)],
)
def test_expected_row_counts(dataset, rows):
    """Catches preprocessing silently dropping or duplicating records."""
    df = pd.read_csv(DATASETS_DIR / f"{dataset}.csv")

    assert len(df) == rows
    assert set(df.iloc[:, -1].unique()) <= {0, 1}
