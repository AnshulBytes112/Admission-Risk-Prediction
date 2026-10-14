"""Automated tests for Phase 6 FastAPI inference API, model lifecycle, validation, and serving.

Verifies:
- Liveness (/health) operates independently from model readiness.
- Readiness (/ready) reports 503 when model is unloaded/missing and 200 when loaded.
- Model info (/model-info) returns safe governance metadata and 503 when unavailable.
- Inference (/predict) performs validated scoring and positive-class mapping.
- Vectorized batch inference (/predict/batch) handles multiple records and enforces limits.
- Input validation rejects missing features, nulls, NaN, Infinity, out-of-bounds, and extra fields.
- Positive-class probability maps correctly through model.classes_.
- Missing, corrupt, and schema-invalid artifacts produce clean, safe HTTP errors.
- Experimental flag and synthetic data disclaimers propagate across endpoints.
"""
from __future__ import annotations

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.api.main import app, model_manager, FEATURE_RANGES
from ml.features import FEATURE_ALLOWLIST
from ml.training import build_pipeline, save_model_bundle_atomically


@pytest.fixture
def test_client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def valid_patient_payload() -> dict[str, Any]:
    """Valid patient feature payload containing all 12 required features."""
    return {
        "patient_id": "SYN-TEST-001",
        "age": 64.0,
        "heart_rate": 86.0,
        "systolic_bp": 128.0,
        "diastolic_bp": 82.0,
        "spo2": 96.0,
        "respiratory_rate": 18.0,
        "temperature_c": 37.0,
        "previous_admissions": 1.0,
        "chronic_conditions_count": 2.0,
        "lactate": 1.4,
        "observation_hour": 14.0,
        "prior_observation_count": 2.0,
    }


@pytest.fixture
def mock_trained_bundle(tmp_path) -> Path:
    """Create a temporary valid model bundle on disk."""
    np.random.seed(42)
    features = list(FEATURE_ALLOWLIST)
    n = 20
    X = pd.DataFrame(np.random.uniform(20.0, 70.0, size=(n, len(features))), columns=features)
    y = np.array([1 if i < 5 else 0 for i in range(n)])

    pipe = build_pipeline("logistic_regression", numeric_features=features, seed=42)
    pipe.fit(X, y)

    bundle_path = tmp_path / "mock_risk_model.joblib"
    bundle = {
        "model": pipe,
        "model_name": "logistic_regression",
        "features": features,
        "threshold": 0.70,
        "target": "admission_within_24h",
        "dataset_type": "synthetic",
        "is_experimental": True,
        "not_for_clinical_use": True,
    }
    save_model_bundle_atomically(bundle, bundle_path)

    # Also save companion metadata and evaluation report
    metadata = {
        "artifact_version": "1.0.0",
        "model_name": "logistic_regression",
        "target": "admission_within_24h",
        "features": features,
        "threshold": 0.70,
        "is_experimental": True,
        "dataset_type": "synthetic",
        "safety_disclaimer": "Academic synthetic demonstration; not for clinical use.",
    }
    (tmp_path / "model_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    report = {
        "project": "Emergency Admission Risk Prediction",
        "dataset_metadata": {"total_rows": 50, "total_columns": 22},
        "held_out_test_metrics": {"accuracy": 1.0, "f1_score": 0.0},
        "model_selection": {"selection_criterion": "validation PR-AUC"},
    }
    (tmp_path / "evaluation_report.json").write_text(json.dumps(report), encoding="utf-8")

    return bundle_path


# -----------------------------------------------------------------------------
# 1. Health and Readiness Probe Tests
# -----------------------------------------------------------------------------
def test_health_liveness_probe_without_model(test_client, monkeypatch, tmp_path):
    """Verify GET /health reports liveness even when no model is available."""
    missing_path = tmp_path / "nonexistent.joblib"
    monkeypatch.setattr("app.api.main.MODEL_PATH", missing_path)

    response = test_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["application_health"] == "healthy"
    assert data["model_readiness"] == "not_loaded"
    assert data["model_file_exists"] is False
    assert "disclaimer" in data


def test_readiness_probe_when_model_missing(test_client, monkeypatch, tmp_path):
    """Verify GET /ready returns HTTP 503 when the model artifact is missing."""
    missing_path = tmp_path / "nonexistent.joblib"
    monkeypatch.setattr("app.api.main.MODEL_PATH", missing_path)

    response = test_client.get("/ready")
    assert response.status_code == 503
    data = response.json()
    assert data["status"] == "not_ready"
    assert data["model_loaded"] is False
    assert "not found" in data["error_reason"].lower()


def test_readiness_probe_when_model_loaded(test_client, monkeypatch, mock_trained_bundle):
    """Verify GET /ready returns HTTP 200 with operational metrics when model is loaded."""
    monkeypatch.setattr("app.api.main.MODEL_PATH", mock_trained_bundle)

    response = test_client.get("/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert data["model_loaded"] is True
    assert data["model_name"] == "logistic_regression"
    assert data["artifact_status"] == "loaded"
    assert data["feature_count"] == 12
    assert data["model_threshold"] == 0.70
    assert data["is_experimental"] is True
    assert data["data_origin"] == "synthetic"
    assert data["error_reason"] is None


# -----------------------------------------------------------------------------
# 2. Model Info Endpoint Tests
# -----------------------------------------------------------------------------
def test_model_info_when_model_missing(test_client, monkeypatch, tmp_path):
    """Verify GET /model-info returns HTTP 503 when model is not available."""
    monkeypatch.setattr("app.api.main.MODEL_PATH", tmp_path / "nonexistent.joblib")

    response = test_client.get("/model-info")
    assert response.status_code == 503
    data = response.json()
    assert data["status"] == "unavailable"
    assert "disclaimer" in data


def test_model_info_when_model_loaded(test_client, monkeypatch, mock_trained_bundle):
    """Verify GET /model-info returns governance metadata without exposing secrets."""
    monkeypatch.setattr("app.api.main.MODEL_PATH", mock_trained_bundle)

    response = test_client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "logistic_regression"
    assert data["target_name"] == "admission_within_24h"
    assert data["feature_count"] == 12
    assert data["is_experimental"] is True
    assert data["data_origin"] == "synthetic"
    assert "disclaimer" in data
    # Ensure sensitive paths are not leaked
    assert "C:\\" not in json.dumps(data)
    assert "/home" not in json.dumps(data)


# -----------------------------------------------------------------------------
# 3. Single Prediction & Probability Tests
# -----------------------------------------------------------------------------
def test_predict_single_patient_success(test_client, monkeypatch, mock_trained_bundle, valid_patient_payload):
    """Verify POST /predict calculates risk, returns binary prediction, and disclaimers."""
    monkeypatch.setattr("app.api.main.MODEL_PATH", mock_trained_bundle)

    response = test_client.post("/predict", json=valid_patient_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["patient_id"] == "SYN-TEST-001"
    assert "risk_probability" in data
    assert 0.0 <= data["risk_probability"] <= 1.0
    assert "risk_percent" in data
    assert data["threshold"] == 0.70
    assert isinstance(data["review_flag"], bool)
    assert data["prediction"] in (0, 1)
    assert data["prediction"] == int(data["review_flag"])
    assert data["is_experimental"] is True
    assert data["dataset_type"] == "synthetic"
    assert "not a clinically validated prediction" in data["notice"].lower()


def test_predict_batch_success(test_client, monkeypatch, mock_trained_bundle, valid_patient_payload):
    """Verify POST /predict/batch scores multiple patients in a single call."""
    monkeypatch.setattr("app.api.main.MODEL_PATH", mock_trained_bundle)

    batch_payload = [
        {**valid_patient_payload, "patient_id": "SYN-BATCH-001", "age": 45.0},
        {**valid_patient_payload, "patient_id": "SYN-BATCH-002", "age": 78.0},
    ]
    response = test_client.post("/predict/batch", json=batch_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 2
    assert len(data["predictions"]) == 2
    assert data["predictions"][0]["patient_id"] == "SYN-BATCH-001"
    assert data["predictions"][1]["patient_id"] == "SYN-BATCH-002"


def test_predict_batch_empty_rejected(test_client, monkeypatch, mock_trained_bundle):
    """Verify POST /predict/batch rejects empty list."""
    monkeypatch.setattr("app.api.main.MODEL_PATH", mock_trained_bundle)
    response = test_client.post("/predict/batch", json=[])
    assert response.status_code == 422


# -----------------------------------------------------------------------------
# 4. Input Validation and Security Tests
# -----------------------------------------------------------------------------
def test_predict_missing_feature_rejected(test_client, monkeypatch, mock_trained_bundle, valid_patient_payload):
    """Verify missing required numeric feature returns 422."""
    monkeypatch.setattr("app.api.main.MODEL_PATH", mock_trained_bundle)
    invalid = valid_patient_payload.copy()
    del invalid["heart_rate"]

    response = test_client.post("/predict", json=invalid)
    assert response.status_code == 422


def test_predict_null_value_rejected(test_client, monkeypatch, mock_trained_bundle, valid_patient_payload):
    """Verify null values in numeric fields return 422."""
    monkeypatch.setattr("app.api.main.MODEL_PATH", mock_trained_bundle)
    invalid = valid_patient_payload.copy()
    invalid["spo2"] = None

    response = test_client.post("/predict", json=invalid)
    assert response.status_code == 422


def test_predict_nan_string_rejected(test_client, monkeypatch, mock_trained_bundle, valid_patient_payload):
    """Verify NaN strings return 422."""
    monkeypatch.setattr("app.api.main.MODEL_PATH", mock_trained_bundle)
    invalid = valid_patient_payload.copy()
    invalid["lactate"] = "NaN"

    response = test_client.post("/predict", json=invalid)
    assert response.status_code == 422


def test_predict_infinity_rejected(test_client, monkeypatch, mock_trained_bundle, valid_patient_payload):
    """Verify Infinity strings return 422."""
    monkeypatch.setattr("app.api.main.MODEL_PATH", mock_trained_bundle)
    invalid = valid_patient_payload.copy()
    invalid["temperature_c"] = "Infinity"

    response = test_client.post("/predict", json=invalid)
    assert response.status_code == 422


def test_predict_out_of_bounds_rejected(test_client, monkeypatch, mock_trained_bundle, valid_patient_payload):
    """Verify out-of-bounds measurements return 422."""
    monkeypatch.setattr("app.api.main.MODEL_PATH", mock_trained_bundle)

    # Age > 100
    invalid_age = {**valid_patient_payload, "age": 105.0}
    assert test_client.post("/predict", json=invalid_age).status_code == 422

    # SpO2 < 50
    invalid_spo2 = {**valid_patient_payload, "spo2": 45.0}
    assert test_client.post("/predict", json=invalid_spo2).status_code == 422


def test_predict_extra_field_forbidden(test_client, monkeypatch, mock_trained_bundle, valid_patient_payload):
    """Verify unexpected fields are forbidden and return 422."""
    monkeypatch.setattr("app.api.main.MODEL_PATH", mock_trained_bundle)
    invalid = {**valid_patient_payload, "unapproved_admin_field": "hack"}
    assert test_client.post("/predict", json=invalid).status_code == 422


# -----------------------------------------------------------------------------
# 5. Error Handling and Corrupt Model Tests
# -----------------------------------------------------------------------------
def test_corrupt_model_artifact_returns_500(test_client, monkeypatch, tmp_path, valid_patient_payload):
    """Verify corrupt model file triggers HTTP 500 with clear message."""
    corrupt_file = tmp_path / "corrupt_model.joblib"
    corrupt_file.write_bytes(b"CORRUPT_NOT_A_VALID_JOBLIB_FILE")
    monkeypatch.setattr("app.api.main.MODEL_PATH", corrupt_file)

    response = test_client.post("/predict", json=valid_patient_payload)
    assert response.status_code == 500
    assert "Could not load model bundle" in response.json()["detail"]


def test_invalid_bundle_schema_missing_features(test_client, monkeypatch, tmp_path, valid_patient_payload):
    """Verify bundle missing features key triggers HTTP 500."""
    invalid_file = tmp_path / "no_features_bundle.joblib"
    joblib.dump({"model": "dummy_model"}, invalid_file)
    monkeypatch.setattr("app.api.main.MODEL_PATH", invalid_file)

    response = test_client.post("/predict", json=valid_patient_payload)
    assert response.status_code == 500
    assert "missing required key" in response.json()["detail"]


class InvertedClassifier:
    """Mock classifier with inverted classes [1, 0]."""
    classes_ = np.array([1, 0])

    def predict_proba(self, X):
        # Column 0 is class 1 (prob 0.85), Column 1 is class 0 (prob 0.15)
        n = len(X)
        return np.tile([0.85, 0.15], (n, 1))


def test_positive_class_index_mapping(test_client, monkeypatch, tmp_path, valid_patient_payload):
    """Verify that positive-class probability correctly maps to label 1 even if classes are ordered [1, 0]."""
    bundle_path = tmp_path / "inverted_classes.joblib"
    bundle = {
        "model": InvertedClassifier(),
        "model_name": "mock_inverted",
        "features": list(FEATURE_ALLOWLIST),
        "threshold": 0.70,
        "is_experimental": True,
    }
    joblib.dump(bundle, bundle_path)
    monkeypatch.setattr("app.api.main.MODEL_PATH", bundle_path)

    response = test_client.post("/predict", json=valid_patient_payload)
    assert response.status_code == 200
    data = response.json()
    # It must select 0.85 (class 1), NOT 0.15 (index 1 which is class 0)
    assert data["risk_probability"] == 0.85
    assert data["review_flag"] is True
    assert data["prediction"] == 1
