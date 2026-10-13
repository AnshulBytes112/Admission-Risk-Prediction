"""API contract tests. These tests run without requiring a trained model."""
import os
import sys
from pathlib import Path

# Import the app module from the repository when run from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient
from app.api.main import app

client = TestClient(app)

def test_root_describes_synthetic_project():
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["data_type"] == "synthetic"

def test_health_endpoint_is_json():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_metadata_has_safety_context():
    response = client.get("/metadata")
    assert response.status_code == 200
    body = response.json()
    assert body["dataset_type"] == "synthetic"
    assert body.get("not_for_clinical_use") is True or body.get("clinical_use") is False

def test_predict_rejects_missing_required_fields():
    response = client.post("/predict", json={"patient_id": "SYN-TEST"})
    assert response.status_code == 422

def test_predict_rejects_extra_fields():
    payload = {
        "patient_id": "SYN-TEST",
        "age": 55,
        "heart_rate": 80,
        "systolic_bp": 120,
        "diastolic_bp": 75,
        "spo2": 97,
        "respiratory_rate": 16,
        "temperature_c": 36.8,
        "previous_admissions": 0,
        "chronic_conditions_count": 1,
        "lactate": 1.2,
        "observation_hour": 12,
        "prior_observation_count": 0,
        "unexpected": "not-allowed",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
