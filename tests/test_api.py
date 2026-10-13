"""Automated tests for FastAPI starter endpoints in Phase 1.

Verifies:
- Root endpoint provides project info and synthetic data declaration.
- Health endpoint returns JSON application status.
- Metadata endpoint exposes configuration and disclaimers without requiring a model.
- Model-dependent endpoints return 503 with helpful messages when no model is trained.
- Request validation rejects invalid or malformed payloads.
"""
from pathlib import Path
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient
from app.api.main import app

client = TestClient(app)


def test_api_root():
    """Verify that root endpoint returns HTTP 200 and project information."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["project"] == "Emergency Admission Risk Prediction"
    assert data["phase"] == 1
    assert data["status"] == "ready"
    assert data["data_type"] == "synthetic"
    assert "notice" in data


def test_api_health():
    """Verify that health endpoint distinguishes application health from model readiness."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["application_health"] == "healthy"
    assert "model_readiness" in data
    assert isinstance(data["model_file_exists"], bool)


def test_api_metadata():
    """Verify that metadata endpoint returns configuration and safety notice."""
    response = client.get("/metadata")
    assert response.status_code == 200
    data = response.json()
    assert data["dataset_type"] == "synthetic"
    assert data["clinical_use"] is False
    assert data["not_for_clinical_use"] is True
    assert "risk_threshold" in data


def test_predict_without_trained_model():
    """Verify that prediction endpoint returns HTTP 503 when no model bundle exists."""
    valid_payload = {
        "patient_id": "SYN000001",
        "age": 65,
        "heart_rate": 88,
        "systolic_bp": 125,
        "diastolic_bp": 80,
        "spo2": 96,
        "respiratory_rate": 18,
        "temperature_c": 37.1,
        "previous_admissions": 1,
        "chronic_conditions_count": 2,
        "lactate": 1.4,
        "observation_hour": 14,
        "prior_observation_count": 1,
    }
    response = client.post("/predict", json=valid_payload)
    # In Phase 1 without a trained model, it should return 503 Service Unavailable
    assert response.status_code == 503
    assert "Trained model not found" in response.json()["detail"]


def test_predict_invalid_schema():
    """Verify that Pydantic rejects invalid patient feature payloads with 422."""
    invalid_payload = {
        "patient_id": "SYN000001",
        "age": 150,  # exceeds allowed age range (le 100)
    }
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422
