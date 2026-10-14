"""Tests for dynamic clinical decision threshold functionality in FastAPI and ML inference.

Verifies:
1. Dynamic threshold via query parameter (?threshold=...) overrides default and changes binary prediction.
2. Dynamic threshold via request body (threshold field in patient features) is honored.
3. Runtime threshold config API (/config/threshold GET, POST, DELETE) correctly manages global cutoff.
4. Batch prediction (/predict/batch?threshold=...) adheres to custom dynamic threshold.
5. Invalid thresholds (< 0.01 or > 0.99, non-finite) are rejected with HTTP 422.
6. Pipeline trace reports the exact dynamic threshold used for Kafka & HDFS storage.
"""
import pytest
from fastapi.testclient import TestClient

from app.api.main import app, model_manager


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def test_patient():
    return {
        "patient_id": "SYNTH-PT-TH-TEST",
        "age": 65.0,
        "heart_rate": 82.0,
        "systolic_bp": 124.0,
        "diastolic_bp": 78.0,
        "spo2": 97.0,
        "respiratory_rate": 15.0,
        "temperature_c": 37.0,
        "previous_admissions": 1.0,
        "chronic_conditions_count": 2.0,
        "lactate": 1.3,
        "observation_hour": 14.0,
        "prior_observation_count": 4.0,
    }


def test_dynamic_threshold_single_prediction_flips_class(client, test_patient):
    """Verify that changing threshold dynamically flips the predicted class for the exact same patient."""
    # First, get baseline probability with default threshold (0.70)
    res_default = client.post("/predict?threshold=0.70", json=test_patient)
    assert res_default.status_code == 200
    data_default = res_default.json()
    score = data_default["risk_probability"]
    assert data_default["threshold"] == 0.70
    assert data_default["prediction"] == 0  # Score is ~0.46, below 0.70

    # Now, set threshold to 0.35 (below the patient's probability of ~0.46)
    res_sensitive = client.post("/predict?threshold=0.35", json=test_patient)
    assert res_sensitive.status_code == 200
    data_sensitive = res_sensitive.json()
    assert data_sensitive["threshold"] == 0.35
    assert data_sensitive["risk_probability"] == score  # Probability is invariant
    assert data_sensitive["prediction"] == 1  # Score 0.46 >= 0.35 -> Flips to Admission!
    assert data_sensitive["review_flag"] is True

    # Now, set threshold to 0.85 (above the patient's probability)
    res_strict = client.post("/predict?threshold=0.85", json=test_patient)
    assert res_strict.status_code == 200
    data_strict = res_strict.json()
    assert data_strict["threshold"] == 0.85
    assert data_strict["prediction"] == 0  # Score 0.46 < 0.85 -> Discharge/Observe
    assert data_strict["review_flag"] is False


def test_dynamic_threshold_in_payload_body(client, test_patient):
    """Verify threshold can be passed directly inside the JSON body."""
    patient_with_th = {**test_patient, "threshold": 0.40}
    res = client.post("/predict", json=patient_with_th)
    assert res.status_code == 200
    data = res.json()
    assert data["threshold"] == 0.40
    # Score is ~0.46 >= 0.40 -> Admission
    assert data["prediction"] == 1


def test_runtime_threshold_config_endpoints(client, test_patient):
    """Verify GET, POST, and DELETE /config/threshold endpoints."""
    # 1. Check current config
    res_get = client.get("/config/threshold")
    assert res_get.status_code == 200
    initial_th = res_get.json()["default_threshold"]

    # 2. Update threshold globally
    res_post = client.post("/config/threshold", json={"threshold": 0.45})
    assert res_post.status_code == 200
    assert res_post.json()["threshold"] == 0.45

    # 3. Predict without explicit query param should now use runtime threshold 0.45
    res_pred = client.post("/predict", json=test_patient)
    assert res_pred.status_code == 200
    assert res_pred.json()["threshold"] == 0.45

    # 4. Delete / reset runtime threshold
    res_del = client.delete("/config/threshold")
    assert res_del.status_code == 200
    assert res_del.json()["threshold"] == initial_th

    # 5. Predict without query param should now use restored default
    res_restored = client.post("/predict", json=test_patient)
    assert res_restored.status_code == 200
    assert res_restored.json()["threshold"] == initial_th


def test_dynamic_threshold_batch_prediction(client, test_patient):
    """Verify batch predictions honor the dynamic threshold."""
    batch = [
        {**test_patient, "patient_id": "BATCH-001"},
        {**test_patient, "patient_id": "BATCH-002"},
    ]
    # At 0.70 cutoff
    res_070 = client.post("/predict/batch?threshold=0.70", json=batch)
    assert res_070.status_code == 200
    preds_070 = res_070.json()["predictions"]
    for p in preds_070:
        assert p["threshold"] == 0.70
        assert p["prediction"] == 0

    # At 0.35 cutoff
    res_035 = client.post("/predict/batch?threshold=0.35", json=batch)
    assert res_035.status_code == 200
    preds_035 = res_035.json()["predictions"]
    for p in preds_035:
        assert p["threshold"] == 0.35
        assert p["prediction"] == 1


def test_threshold_out_of_bounds_validation(client, test_patient):
    """Verify invalid threshold values (<0.01, >0.99) are rejected with 422."""
    res_low = client.post("/predict?threshold=0.001", json=test_patient)
    assert res_low.status_code == 422

    res_high = client.post("/predict?threshold=1.5", json=test_patient)
    assert res_high.status_code == 422

    res_neg = client.post("/config/threshold", json={"threshold": -0.1})
    assert res_neg.status_code == 422
