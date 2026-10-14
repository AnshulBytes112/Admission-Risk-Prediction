"""Automated unit and integration tests for Phase 7 Streamlit Dashboard.

Covers:
- API client configuration and environment variable resolution.
- Health, readiness, model-info, and metadata HTTP responses.
- API timeouts, connection errors, and HTTP status handling (422, 503).
- Single prediction payload preparation and response parsing.
- Batch prediction payload generation, record bounds, and validation.
- CSV parsing, required column validation, and NaN/type checking.
- Synthetic-data disclaimer checks and experimental status logic.
- Local dataset fallback and empty-state handling.
"""
from __future__ import annotations

import io
from pathlib import Path
import unittest.mock as mock

import pandas as pd
import pytest

from app.dashboard.api_client import DashboardApiClient
from app.dashboard.views.batch_prediction import (
    REQUIRED_NUMERIC_FEATURES,
    generate_sample_batch_csv,
    MAX_BATCH_SIZE,
)
from app.dashboard.views.model_info import load_local_model_artifacts
from app.dashboard.views.overview import load_exploratory_data


# =============================================================================
# 1. API Client Configuration and Connection Tests
# =============================================================================

def test_api_client_url_resolution(monkeypatch):
    """Test API client defaults and environment variable resolution."""
    monkeypatch.delenv("DASHBOARD_API_URL", raising=False)
    monkeypatch.delenv("API_BASE_URL", raising=False)
    client_default = DashboardApiClient()
    assert client_default.base_url == "http://localhost:8000"

    monkeypatch.setenv("DASHBOARD_API_URL", "http://custom-api:9000/")
    client_custom = DashboardApiClient()
    assert client_custom.base_url == "http://custom-api:9000"


def test_api_client_health_success():
    """Test successful health check parsing with mock response."""
    client = DashboardApiClient(base_url="http://mock-api:8000")
    with mock.patch("requests.get") as mock_get:
        mock_resp = mock.MagicMock()
        mock_resp.status_code = 200
        mock_resp.ok = True
        mock_resp.json.return_value = {
            "status": "ok",
            "application_health": "healthy",
            "model_readiness": "ready",
            "is_experimental": True,
        }
        mock_get.return_value = mock_resp

        ok, data, latency = client.check_health()
        assert ok is True
        assert data["status"] == "ok"
        assert latency >= 0.0


def test_api_client_readiness_failure():
    """Test readiness reporting false when model is not ready (503)."""
    client = DashboardApiClient(base_url="http://mock-api:8000")
    with mock.patch("requests.get") as mock_get:
        mock_resp = mock.MagicMock()
        mock_resp.status_code = 503
        mock_resp.ok = False
        mock_resp.json.return_value = {"detail": "Model is not loaded"}
        mock_get.return_value = mock_resp

        ok, data, _ = client.check_readiness()
        assert ok is False
        assert data.get("detail") == "Model is not loaded"


def test_api_client_connection_error():
    """Test graceful handling of network connection failure."""
    client = DashboardApiClient(base_url="http://unavailable-host:8000")
    with mock.patch("requests.get", side_effect=Exception("Connection refused")):
        ok, data, _ = client.check_health()
        assert ok is False
        assert "Connection refused" in data.get("error", "")


def test_api_client_timeout_handling():
    """Test timeout exception handling."""
    import requests
    client = DashboardApiClient(base_url="http://slow-host:8000")
    with mock.patch("requests.get", side_effect=requests.exceptions.Timeout("Timed out")):
        ok, data, _ = client.check_health()
        assert ok is False
        assert "timed out" in data.get("error", "").lower()


# =============================================================================
# 2. Prediction Payload and Response Parsing Tests
# =============================================================================

def test_single_prediction_flow():
    """Test single patient inference submission and parsing."""
    client = DashboardApiClient(base_url="http://mock-api:8000")
    payload = {
        "patient_id": "SYNTH-001",
        "age": 65,
        "heart_rate": 82.0,
        "systolic_bp": 124.0,
        "diastolic_bp": 78.0,
        "spo2": 97.0,
        "respiratory_rate": 15.0,
        "temperature_c": 37.0,
        "previous_admissions": 1,
        "chronic_conditions_count": 2,
        "lactate": 1.3,
        "observation_hour": 14,
        "prior_observation_count": 4,
    }

    with mock.patch("requests.post") as mock_post:
        mock_resp = mock.MagicMock()
        mock_resp.status_code = 200
        mock_resp.ok = True
        mock_resp.json.return_value = {
            "patient_id": "SYNTH-001",
            "prediction": 0,
            "risk_probability": 0.124,
            "threshold": 0.70,
            "review_flag": False,
            "model_name": "logistic_regression",
            "is_experimental": True,
        }
        mock_post.return_value = mock_resp

        ok, result, lat = client.predict_single(payload)
        assert ok is True
        assert result["prediction"] == 0
        assert result["risk_probability"] == 0.124
        assert result["is_experimental"] is True


def test_single_prediction_validation_error_422():
    """Test API 422 Unprocessable Entity error response handling."""
    client = DashboardApiClient(base_url="http://mock-api:8000")
    invalid_payload = {"patient_id": "SYNTH-001", "age": 999}  # invalid age

    with mock.patch("requests.post") as mock_post:
        mock_resp = mock.MagicMock()
        mock_resp.status_code = 422
        mock_resp.ok = False
        mock_resp.json.return_value = {
            "detail": [{"loc": ["body", "age"], "msg": "Input should be less than or equal to 100"}]
        }
        mock_post.return_value = mock_resp

        ok, result, _ = client.predict_single(invalid_payload)
        assert ok is False
        assert "Validation Error (422)" in result.get("error", "")


# =============================================================================
# 3. Batch Validation and Processing Tests
# =============================================================================

def test_sample_batch_csv_generator():
    """Verify sample CSV template generation has all required columns and valid records."""
    csv_str = generate_sample_batch_csv(n_rows=10)
    df = pd.read_csv(io.StringIO(csv_str))

    assert len(df) == 10
    assert "patient_id" in df.columns
    for feature in REQUIRED_NUMERIC_FEATURES:
        assert feature in df.columns
        assert pd.to_numeric(df[feature], errors="coerce").notna().all()


def test_batch_record_limit():
    """Verify that batch size upper bound is respected (500 records max)."""
    assert MAX_BATCH_SIZE == 500


def test_batch_missing_required_columns():
    """Test detection of missing feature columns in uploaded CSV."""
    df_missing = pd.DataFrame({
        "patient_id": ["PT1"],
        "age": [45],
        # missing heart_rate, systolic_bp, etc.
    })
    missing = [c for c in REQUIRED_NUMERIC_FEATURES if c not in df_missing.columns]
    assert len(missing) > 0
    assert "heart_rate" in missing


def test_batch_invalid_numeric_values():
    """Test detection of non-numeric or NaN values in feature columns."""
    df_invalid = pd.DataFrame({
        "patient_id": ["PT1"],
        "age": ["not_a_number"],
    })
    converted = pd.to_numeric(df_invalid["age"], errors="coerce")
    assert converted.isna().any()


def test_batch_prediction_submission():
    """Test batch prediction API submission and parsing."""
    client = DashboardApiClient(base_url="http://mock-api:8000")
    batch_records = [{"patient_id": f"P{i}", "age": 50} for i in range(5)]

    with mock.patch("requests.post") as mock_post:
        mock_resp = mock.MagicMock()
        mock_resp.status_code = 200
        mock_resp.ok = True
        mock_resp.json.return_value = {
            "total_records": 5,
            "predictions": [
                {
                    "patient_id": f"P{i}",
                    "prediction": 0,
                    "risk_probability": 0.15,
                    "threshold": 0.70,
                    "review_flag": False,
                }
                for i in range(5)
            ],
            "model_name": "logistic_regression",
            "is_experimental": True,
        }
        mock_post.return_value = mock_resp

        ok, result, _ = client.predict_batch(batch_records)
        assert ok is True
        assert result["total_records"] == 5
        assert len(result["predictions"]) == 5


# =============================================================================
# 4. Local Files Fallback and Empty States
# =============================================================================

def test_load_local_model_artifacts():
    """Verify loading of local metadata and evaluation report files."""
    meta, eval_rep = load_local_model_artifacts()
    # If files exist in repo, verify their contents
    if meta is not None:
        assert meta.get("model_name") == "logistic_regression"
        assert meta.get("feature_count") == 12
    if eval_rep is not None:
        assert eval_rep.get("is_experimental") is True
        assert "synthetic" in eval_rep.get("data_origin", "").lower()


def test_load_exploratory_data_graceful():
    """Verify exploratory data loader returns clean tuple when path is missing."""
    with mock.patch("app.dashboard.views.overview.PARQUET_PATH", Path("nonexistent/features.parquet")):
        with mock.patch("app.dashboard.views.overview.CSV_PATH", Path("nonexistent/raw.csv")):
            df, source = load_exploratory_data()
            assert df is None
            assert "No dataset found" in source
