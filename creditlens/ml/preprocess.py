"""Convert the raw source files into the clean CSVs used for training.

Reads from ``datasets/raw/`` and writes one CSV per dataset into
``datasets/``, keeping the configured features with the target as the final
column. Idempotent - safe to re-run at any time.

    python -m creditlens.ml.preprocess
"""

import logging
import sys
from pathlib import Path

import pandas as pd

from creditlens.config import (
    ATTRITION_CATEGORIES,
    CREDIT_CATEGORIES,
    DATASET_META,
    DATASETS_DIR,
    RAW_DIR,
)

log = logging.getLogger(__name__)


def _encode(series: pd.Series, categories: list[str]) -> pd.Series:
    """Map text categories to their position in ``categories``.

    The order is fixed in config so the form's dropdown codes always match
    the codes the model was trained on.
    """
    return series.map({name: code for code, name in enumerate(categories)}).astype(int)


# Each clean_* function returns every usable feature plus the 0/1 target,
# before any column selection. feature_analysis ranks these full frames.


def clean_loan_offer(raw_dir: Path = RAW_DIR) -> tuple[pd.DataFrame, pd.Series]:
    """Bank_Personal_Loan_Modelling.csv: Thera Bank, target ``Personal Loan``.

    Converted once from the Kaggle .xlsx (its "Data" sheet).
    """
    df = pd.read_csv(raw_dir / "Bank_Personal_Loan_Modelling.csv")

    # ID and ZIP code identify a customer; they carry no usable signal.
    df = df.drop(columns=["ID", "ZIP Code"])

    y = df.pop("Personal Loan").astype(int)
    return df, y


def clean_credit_risk(raw_dir: Path = RAW_DIR) -> tuple[pd.DataFrame, pd.Series]:
    """credit_risk_dataset.csv: Kaggle Credit Risk Dataset, target ``loan_status``."""
    df = pd.read_csv(raw_dir / "credit_risk_dataset.csv").drop_duplicates()

    # A handful of rows carry ages above 100 and 120+ years of employment.
    df = df[(df["person_age"] <= 100) & ~(df["person_emp_length"] > 60)]

    for col, categories in CREDIT_CATEGORIES.items():
        df[col] = _encode(df[col], categories)
    df["cb_person_default_on_file"] = (df["cb_person_default_on_file"] == "Y").astype(int)

    # Missing interest rates and employment lengths stay as NaN: the training
    # pipeline fills them with medians learned from the training split only.

    y = df.pop("loan_status").astype(int)
    return df, y


def clean_attrition(raw_dir: Path = RAW_DIR) -> tuple[pd.DataFrame, pd.Series]:
    """BankChurners.csv: Kaggle Credit Card Customers, target ``Attrition_Flag``."""
    df = pd.read_csv(raw_dir / "BankChurners.csv")

    # CLIENTNUM is an ID, and the two Naive_Bayes_* columns are another
    # model's predictions of the target - keeping them would leak the answer.
    df = df.drop(columns=["CLIENTNUM"]).drop(columns=df.filter(like="Naive_Bayes").columns)

    for col, categories in ATTRITION_CATEGORIES.items():
        df[col] = _encode(df[col], categories)

    y = (df.pop("Attrition_Flag") == "Attrited Customer").astype(int)
    return df, y


CLEANERS = {
    "loan_offer": clean_loan_offer,
    "credit_risk": clean_credit_risk,
    "attrition": clean_attrition,
}


def build_dataset(dataset: str, raw_dir: Path = RAW_DIR, out_dir: Path = DATASETS_DIR) -> Path:
    """Clean one dataset and keep its configured features."""
    X, y = CLEANERS[dataset](raw_dir)

    df = X[list(DATASET_META[dataset]["features"])].copy()
    df[DATASET_META[dataset]["target_column"]] = y

    out_path = out_dir / f"{dataset}.csv"
    df.to_csv(out_path, index=False)
    log.info("wrote %s (%d rows, %d cols)", out_path.name, len(df), len(df.columns))
    return out_path


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not RAW_DIR.is_dir():
        log.error("raw data directory not found: %s", RAW_DIR)
        return 1

    DATASETS_DIR.mkdir(parents=True, exist_ok=True)

    for name in CLEANERS:
        log.info("processing %s...", name)
        build_dataset(name)

    log.info("all datasets prepared in %s", DATASETS_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
