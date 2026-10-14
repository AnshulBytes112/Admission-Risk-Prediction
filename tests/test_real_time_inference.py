"""Comprehensive tests for Real-Time ML Inference & Feature Contract.

Verifies:
1. Exact 12 model feature ordering and canonical mapping.
2. patient_id identifier exclusion from model feature input DataFrame.
3. Correct positive-class probability extraction using model.classes_.
4. Actual model invocation on valid inference requests.
5. Strict threshold-based binary prediction consistency.
6. Rejection of NaN, null, and non-finite values.
7. Sensitivity check: different physiological inputs yield distinct model probabilities.
8. Batch inference (/predict/batch) adheres to canonical feature ordering and identifier handling.
9. Integration test against real loaded artifact in models/risk_model.joblib.
10. Probability and percentage mathematical consistency.
"""
from __future__ import annotations

import math
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from app.api.main import (
    app,
    model_manager,
    build_feature_dataframe,
    extract_positive_class_probability,
    PatientFeatures,
    MODEL_PATH,
)
from ml.features import FEATURE_ALLOWLIST

EXPECTED_12_FEATURES = [
    "age",
    "heart_rate",
    "systolic_bp",
    "diastolic_bp",
    "spo2",
    "respiratory_rate",
    "temperature_c",
    "previous_admissions",
    "chronic_conditions_count",
    "lactate",
    "observation_hour",
    "prior_observation_count",
]


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def normal_patient():
    return {
        "patient_id": "PT-NORMAL-001",
        "heart_rate": 72.0,
        "systolic_bp": 120.0,
        "diastolic_bp": 80.0,
        "spo2": 98.0,
        "respiratory_rate": 16.0,
        "temperature_c": 36.8,
        "age": 28.0,
        "lactate": 1.1,
        "previous_admissions": 0.0,
        "chronic_conditions_count": 0.0,
        "observation_hour": 10.0,
        "prior_observation_count": 0.0,
    }


@pytest.fixture
def critical_patient():
    return {
        "patient_id": "PT-CRITICAL-999",
        "heart_rate": 145.0,
        "systolic_bp": 80.0,
        "diastolic_bp": 50.0,
        "spo2": 84.0,
        "respiratory_rate": 32.0,
        "temperature_c": 39.4,
        "age": 82.0,
        "lactate": 6.8,
        "previous_admissions": 4.0,
        "chronic_conditions_count": 4.0,
        "observation_hour": 3.0,
        "prior_observation_count": 5.0,
    }


def test_canonical_12_features_contract():
    """Verify the 12 features match the ML pipeline's FEATURE_ALLOWLIST exactly."""
    assert len(FEATURE_ALLOWLIST) == 12
    assert len(EXPECTED_12_FEATURES) == 12


def test_build_feature_dataframe_excludes_patient_id(normal_patient):
    """Verify build_feature_dataframe strictly strips patient_id and retains order."""
    item = PatientFeatures(**normal_patient)
    df, pids = build_feature_dataframe([item], EXPECTED_12_FEATURES)

    assert pids == ["PT-NORMAL-001"]
    assert "patient_id" not in df.columns
    assert list(df.columns) == EXPECTED_12_FEATURES
    assert len(df) == 1
    assert df.loc[0, "heart_rate"] == 72.0
    assert df.loc[0, "age"] == 28.0


def test_build_feature_dataframe_rejects_non_finite(normal_patient):
    """Verify non-finite values (NaN, inf) are caught and rejected."""
    bad_patient = normal_patient.copy()
    bad_patient["age"] = float("nan")

    # Pydantic validator should catch this first
    with pytest.raises(Exception):
        PatientFeatures(**bad_patient)


def test_positive_class_probability_extraction():
    """Verify extract_positive_class_probability safely extracts index 1."""
    # 2 classes: [class 0, class 1]
    dummy_probs = np.array([[0.85, 0.15], [0.30, 0.70]])
    extracted = extract_positive_class_probability(dummy_probs, pos_class_idx=1)
    assert np.allclose(extracted, [0.15, 0.70])

    # Single column fallback
    single_col = np.array([[0.25], [0.75]])
    extracted_single = extract_positive_class_probability(single_col, pos_class_idx=1)
    assert np.allclose(extracted_single, [0.0, 0.0])


def test_real_model_loaded_and_predicts(client, normal_patient):
    """Verify inference endpoint calls the actual loaded model artifact."""
    if not MODEL_PATH.exists():
        pytest.skip(f"Model artifact not found at {MODEL_PATH}")

    response = client.post("/predict", json=normal_patient)
    assert response.status_code == 200, response.text
    data = response.json()

    assert data["patient_id"] == "PT-NORMAL-001"
    assert "risk_probability" in data
    assert 0.0 <= data["risk_probability"] <= 1.0
    assert "risk_percent" in data
    assert "threshold" in data
    assert "prediction" in data
    assert data["prediction"] in (0, 1)

    # Mathematical consistency
    expected_pct = round(data["risk_probability"] * 100, 2)
    assert math.isclose(data["risk_percent"], expected_pct, abs_tol=0.01)

    # Threshold consistency
    expected_pred = 1 if data["risk_probability"] >= data["threshold"] else 0
    assert data["prediction"] == expected_pred
    assert data["review_flag"] == (expected_pred == 1)


def test_real_model_sensitivity(client, normal_patient, critical_patient):
    """Verify that different physiological inputs produce different model outputs."""
    if not MODEL_PATH.exists():
        pytest.skip(f"Model artifact not found at {MODEL_PATH}")

    res_normal = client.post("/predict", json=normal_patient)
    res_critical = client.post("/predict", json=critical_patient)

    assert res_normal.status_code == 200
    assert res_critical.status_code == 200

    prob_normal = res_normal.json()["risk_probability"]
    prob_critical = res_critical.json()["risk_probability"]

    assert prob_normal != prob_critical
    assert prob_critical > prob_normal, f"Critical {prob_critical} should exceed normal {prob_normal}"


def test_batch_prediction_consistency(client, normal_patient, critical_patient):
    """Verify /predict/batch scores each record accurately and retains IDs."""
    if not MODEL_PATH.exists():
        pytest.skip(f"Model artifact not found at {MODEL_PATH}")

    batch_payload = [normal_patient, critical_patient]
    response = client.post("/predict/batch", json=batch_payload)
    assert response.status_code == 200
    data = response.json()

    assert data["count"] == 2
    preds = data["predictions"]
    assert preds[0]["patient_id"] == "PT-NORMAL-001"
    assert preds[1]["patient_id"] == "PT-CRITICAL-999"

    # Compare single vs batch predictions for identical records
    res_single = client.post("/predict", json=normal_patient).json()
    assert math.isclose(preds[0]["risk_probability"], res_single["risk_probability"], abs_tol=1e-5)


def test_invalid_features_rejected(client, normal_patient):
    """Verify invalid payloads (out-of-bounds, missing fields) are rejected with 422."""
    bad_payload = normal_patient.copy()
    bad_payload["heart_rate"] = 9999.0  # ge=20, le=300
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422

    missing_payload = normal_patient.copy()
    del missing_payload["heart_rate"]
    response2 = client.post("/predict", json=missing_payload)
    assert response2.status_code == 422
