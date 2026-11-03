"""The HTTP endpoints, and the input rules they enforce."""

import pytest

from creditlens.config import DATASET_META, MODEL_DISPLAY_NAMES
from creditlens.schemas import validate_features


def test_predict_happy_path(client, valid_payload):
    response = client.post("/api/predict", json=valid_payload)

    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in (0, 1)
    assert body["model_used"] == "random_forest"
    assert body["summary"] in ("Likely to leave the bank", "Likely to stay")
    assert 0 <= body["probability"] <= 100


@pytest.mark.parametrize("model_type", list(MODEL_DISPLAY_NAMES))
def test_every_model_serves_predictions(client, valid_payload, model_type):
    valid_payload["model_type"] = model_type
    response = client.post("/api/predict", json=valid_payload)

    assert response.status_code == 200
    assert response.json()["model_used"] == model_type
    assert response.json()["probability"] is not None


@pytest.mark.parametrize(
    ("dataset", "positive_outcome"),
    [("loan_offer", "good"), ("credit_risk", "bad"), ("attrition", "bad")],
)
def test_outcome_colour_follows_dataset_meaning(client, dataset, positive_outcome):
    """An accepted offer is good news; a default or attrition is not."""
    payload = {"dataset": dataset, "model_type": "random_forest"}
    for name, meta in DATASET_META[dataset]["features"].items():
        payload[name] = meta["default"]

    body = client.post("/api/predict", json=payload).json()

    negative_outcome = "bad" if positive_outcome == "good" else "good"
    expected = positive_outcome if body["prediction"] == 1 else negative_outcome
    assert body["outcome"] == expected


def test_datasets_catalog_shape(client):
    """The payload the whole frontend is built from."""
    response = client.get("/api/datasets")

    assert response.status_code == 200
    datasets = response.json()["datasets"]

    # Listed in lifecycle order, so the dropdown opens on Acquire.
    assert list(datasets) == ["loan_offer", "credit_risk", "attrition"]

    attrition = datasets["attrition"]
    assert [f["name"] for f in attrition["features"]] == list(DATASET_META["attrition"]["features"])
    assert {m["name"] for m in attrition["models"]} == set(MODEL_DISPLAY_NAMES)


def test_missing_field_returns_400(client, valid_payload):
    del valid_payload["Total_Trans_Ct"]
    response = client.post("/api/predict", json=valid_payload)

    assert response.status_code == 400
    assert "Total_Trans_Ct" in response.json()["fields"]


def test_out_of_range_returns_400(client, valid_payload):
    valid_payload["Total_Trans_Ct"] = 99999
    response = client.post("/api/predict", json=valid_payload)

    assert response.status_code == 400
    assert "at most" in response.json()["fields"]["Total_Trans_Ct"]


def test_unknown_dataset_returns_404(client, valid_payload):
    valid_payload["dataset"] = "mortgages"
    response = client.post("/api/predict", json=valid_payload)

    assert response.status_code == 404
    assert "Unknown dataset" in response.json()["error"]


def test_index_serves_the_frontend(client):
    response = client.get("/")

    assert response.status_code == 200
    assert b"CreditLens" in response.content


def test_returns_values_in_model_feature_order(attrition_features):
    """Feature order is the one bug that fails silently.

    Sent in the wrong order the model still answers, it is just answering a
    different question, so nothing errors and the result is quietly wrong.
    """
    payload = {
        "Total_Ct_Chng_Q4_Q1": 0.4,
        "Total_Revolving_Bal": 0,
        "Total_Trans_Amt": 2000.0,
        "Total_Amt_Chng_Q4_Q1": 0.5,
        "Total_Relationship_Count": 2,
        "Total_Trans_Ct": 40,
    }

    values, errors = validate_features("attrition", attrition_features, payload)

    assert errors == {}
    assert values == [40.0, 2000.0, 2.0, 0.0, 0.5, 0.4]


@pytest.mark.parametrize("dataset", sorted(DATASET_META))
def test_configured_defaults_pass_their_own_validation(dataset):
    """The form pre-fills every field with its `default`.

    A default outside its own min/max would break the form the moment someone
    clicked submit without touching anything.
    """
    features = DATASET_META[dataset]["features"]
    payload = {}
    for name, meta in features.items():
        payload[name] = meta["default"]

    values, errors = validate_features(dataset, list(features), payload)

    assert errors == {}
    assert len(values) == len(features)
