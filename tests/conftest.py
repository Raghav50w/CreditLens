"""Shared test fixtures.

Tests run against the real datasets and the real trained models, so run
`python -m creditlens.ml.preprocess` and `python -m creditlens.ml.train`
before `pytest`.
"""

import pytest
from fastapi.testclient import TestClient

from creditlens.config import DATASET_META
from creditlens.main import app


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def valid_payload():
    """A request body that satisfies every attrition validation rule."""
    return {
        "dataset": "attrition",
        "model_type": "random_forest",
        "Total_Trans_Ct": 40,
        "Total_Trans_Amt": 2000.0,
        "Total_Relationship_Count": 2,
        "Total_Revolving_Bal": 0,
        "Total_Amt_Chng_Q4_Q1": 0.5,
        "Total_Ct_Chng_Q4_Q1": 0.4,
    }


@pytest.fixture
def attrition_features():
    return list(DATASET_META["attrition"]["features"])
