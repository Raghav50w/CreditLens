"""Paths, training constants, and the feature metadata for every dataset.

DATASET_META drives everything: which columns to keep, the model's feature
order, the API's validation bounds, and the form the frontend draws. Adding a
dataset means adding an entry here plus a cleaner in ml/preprocess.py.

Each feature's `default` is a typical value for that population, used to
pre-fill the form so it is submittable on load.
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATASETS_DIR = BASE_DIR / "datasets"
RAW_DIR = DATASETS_DIR / "raw"
MODELS_DIR = BASE_DIR / "models"
STATIC_DIR = BASE_DIR / "static"

METRICS_FILENAME = "metrics.json"

# Training constants.
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5  # cross-validation, tuning and cut-off search

# Hyperparameter search budget for `python -m creditlens.ml.train --tune`.
# The search runs locally and its winners are saved to TUNED_PARAMS_PATH, so
# a plain train (CI, Docker) reuses them without searching.
TUNING_ITERATIONS = 20
TUNED_PARAMS_PATH = BASE_DIR / "creditlens" / "tuned_params.json"

# Model selected when a request omits ``model_type``, and pre-selected by the
# frontend. Kept here so the API default and the UI default cannot disagree.
DEFAULT_MODEL = "random_forest"

# Human-readable names for the model keys produced by ``ml.train``.
MODEL_DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "hist_gradient_boosting": "Gradient Boosting (HistGB)",
    "mlp": "Neural Network (MLP)",
    "ensemble": "Ensemble (soft voting)",
}

# Fixed category orders for text columns, shared by preprocessing and the form
# dropdowns so the integer codes cannot drift apart.
CREDIT_CATEGORIES = {
    "person_home_ownership": ["RENT", "OWN", "MORTGAGE", "OTHER"],
    "loan_intent": [
        "PERSONAL",
        "EDUCATION",
        "MEDICAL",
        "VENTURE",
        "HOMEIMPROVEMENT",
        "DEBTCONSOLIDATION",
    ],
    # Grades are ordered, A (best) to G (worst), so they stay a single number.
    "loan_grade": ["A", "B", "C", "D", "E", "F", "G"],
}

ATTRITION_CATEGORIES = {
    "Gender": ["F", "M"],
    "Education_Level": [
        "Unknown",
        "Uneducated",
        "High School",
        "College",
        "Graduate",
        "Post-Graduate",
        "Doctorate",
    ],
    "Marital_Status": ["Unknown", "Single", "Married", "Divorced"],
    "Income_Category": [
        "Unknown",
        "Less than $40K",
        "$40K - $60K",
        "$60K - $80K",
        "$80K - $120K",
        "$120K +",
    ],
    "Card_Category": ["Blue", "Silver", "Gold", "Platinum"],
}


def _options(categories: list[str]) -> list[dict]:
    """Dropdown options whose values are the codes preprocessing assigns."""
    return [{"value": code, "label": name} for code, name in enumerate(categories)]


# Every dataset also sets what a positive (1) prediction means for the bank:
# `positive_is_good` decides whether the result box is shown green or red.
# Features flagged `categorical` are codes with no order (home ownership, loan purpose) and
# are one-hot encoded by the training pipeline.
DATASET_META = {
    "loan_offer": {
        "display_name": "Acquire - Personal Loan Offer",
        "description": "Kaggle Bank Personal Loan Modelling, Thera Bank (5,000 customers).",
        "target_column": "Personal Loan",
        "positive_is_good": True,
        "positive_label": "Likely to accept the loan offer",
        "negative_label": "Unlikely to accept the loan offer",
        "features": {
            "Income": {
                "label": "Annual Income",
                "unit": "USD thousands",
                "min": 8,
                "max": 250,
                "step": 1,
                "default": 64,
                "help": "Customer's yearly income.",
            },
            "Education": {
                "label": "Education",
                "unit": "",
                "min": 1,
                "max": 3,
                "step": 1,
                "default": 2,
                "help": "Highest education level.",
                "options": [
                    {"value": 1, "label": "Undergraduate"},
                    {"value": 2, "label": "Graduate"},
                    {"value": 3, "label": "Advanced / Professional"},
                ],
            },
            "Family": {
                "label": "Family Size",
                "unit": "people",
                "min": 1,
                "max": 4,
                "step": 1,
                "default": 2,
                "help": "Number of people in the customer's family.",
            },
            "CCAvg": {
                "label": "Card Spending",
                "unit": "USD thousands / month",
                "min": 0,
                "max": 10,
                "step": 0.1,
                "default": 1.5,
                "help": "Average monthly credit card spending.",
            },
            "CD Account": {
                "label": "Certificate of Deposit",
                "unit": "",
                "min": 0,
                "max": 1,
                "step": 1,
                "default": 0,
                "help": "Whether the customer holds a certificate of deposit account.",
                "options": [{"value": 0, "label": "No"}, {"value": 1, "label": "Yes"}],
            },
        },
    },
    "credit_risk": {
        "display_name": "Lend - Loan Default",
        "description": "Kaggle Credit Risk Dataset (32,409 loans after cleaning).",
        "target_column": "loan_status",
        "positive_is_good": False,
        "positive_label": "Likely to default",
        "negative_label": "Unlikely to default",
        "features": {
            "loan_percent_income": {
                "label": "Loan to Income",
                "unit": "ratio",
                "min": 0,
                "max": 1,
                "step": 0.01,
                "default": 0.15,
                "help": "Loan amount divided by yearly income.",
            },
            "loan_grade": {
                "label": "Loan Grade",
                "unit": "",
                "min": 0,
                "max": 6,
                "step": 1,
                "default": 1,
                "help": "Lender's risk grade, A (best) to G (worst).",
                "options": _options(CREDIT_CATEGORIES["loan_grade"]),
            },
            "person_income": {
                "label": "Annual Income",
                "unit": "USD",
                "min": 4000,
                "max": 2000000,
                "step": 1000,
                "default": 55000,
                "help": "Borrower's yearly income.",
            },
            "person_home_ownership": {
                "label": "Home Ownership",
                "unit": "",
                "min": 0,
                "max": 3,
                "step": 1,
                "default": 0,
                "help": "Whether the borrower rents, owns or has a mortgage.",
                "options": _options(CREDIT_CATEGORIES["person_home_ownership"]),
                "categorical": True,
            },
            "loan_intent": {
                "label": "Loan Purpose",
                "unit": "",
                "min": 0,
                "max": 5,
                "step": 1,
                "default": 2,
                "help": "What the loan is for.",
                "options": _options(CREDIT_CATEGORIES["loan_intent"]),
                "categorical": True,
            },
            "person_emp_length": {
                "label": "Employment Length",
                "unit": "years",
                "min": 0,
                "max": 60,
                "step": 1,
                "default": 4,
                "help": "Years in current employment.",
            },
        },
    },
    "attrition": {
        "display_name": "Retain - Card Customer Attrition",
        "description": "Kaggle Credit Card Customers, BankChurners (10,127 customers).",
        "target_column": "Attrition_Flag",
        "positive_is_good": False,
        "positive_label": "Likely to leave the bank",
        "negative_label": "Likely to stay",
        "features": {
            "Total_Trans_Ct": {
                "label": "Transactions (12 months)",
                "unit": "count",
                "min": 10,
                "max": 150,
                "step": 1,
                "default": 67,
                "help": "Number of card transactions in the last 12 months.",
            },
            "Total_Trans_Amt": {
                "label": "Transaction Amount (12 months)",
                "unit": "USD",
                "min": 500,
                "max": 20000,
                "step": 1,
                "default": 3899,
                "help": "Total card spending in the last 12 months.",
            },
            "Total_Relationship_Count": {
                "label": "Products Held",
                "unit": "count",
                "min": 1,
                "max": 6,
                "step": 1,
                "default": 4,
                "help": "Number of bank products the customer uses.",
                "options": [{"value": n, "label": str(n)} for n in range(1, 7)],
            },
            "Total_Revolving_Bal": {
                "label": "Revolving Balance",
                "unit": "USD",
                "min": 0,
                "max": 2600,
                "step": 1,
                "default": 1276,
                "help": "Card balance carried over month to month.",
            },
            "Total_Amt_Chng_Q4_Q1": {
                "label": "Spending Change Q4/Q1",
                "unit": "ratio",
                "min": 0,
                "max": 4,
                "step": 0.01,
                "default": 0.74,
                "help": "Q4 spending divided by Q1 spending.",
            },
            "Total_Ct_Chng_Q4_Q1": {
                "label": "Transaction Change Q4/Q1",
                "unit": "ratio",
                "min": 0,
                "max": 4,
                "step": 0.01,
                "default": 0.7,
                "help": "Q4 transaction count divided by Q1 count.",
            },
        },
    },
}


def feature_meta(dataset: str, feature: str) -> dict:
    """Return metadata for a single feature."""
    return DATASET_META[dataset]["features"][feature]
