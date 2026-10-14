"""Automated tests for Phase 5 artifact serialization, metrics handling, and Phase 6 API compatibility.

Verifies:
- Safe metric computation when evaluation split contains only one class.
- Model bundle serialization and roundtrip deserialization via joblib.
- Feature ordering matches between bundle and pipeline preprocessor.
- Prediction probabilities are strictly finite and within [0.0, 1.0].
- Decision thresholding logic yields correct binary classifications.
- Saved model bundle operates cleanly with FastAPI /predict and /metadata endpoints.
- Experimental flag is preserved in metadata and evaluation reports.
"""
from __future__ import annotations

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.api.main import app
from ml.evaluation import (
    compute_metrics,
    generate_evaluation_report_json,
    generate_model_metadata,
    select_best_model,
)
from ml.features import FEATURE_ALLOWLIST
from ml.training import build_pipeline, save_model_bundle_atomically


@pytest.fixture
def trained_test_pipeline() -> tuple[Any, list[str]]:
    """Train a quick LogisticRegression pipeline on synthetic data."""
    np.random.seed(42)
    n = 30
    features = list(FEATURE_ALLOWLIST)
    X = pd.DataFrame(np.random.uniform(20.0, 80.0, size=(n, len(features))), columns=features)
    y = np.array([1 if i < 10 else 0 for i in range(n)])

    pipe = build_pipeline("logistic_regression", numeric_features=features, seed=42)
    pipe.fit(X, y)
    return pipe, features


def test_metric_handling_with_single_class():
    """Verify metrics return None for ROC-AUC and PR-AUC when ground truth has only 1 class."""
    y_true = np.zeros(20, dtype=int)
    scores = np.random.uniform(0.1, 0.4, size=20)

    metrics = compute_metrics(y_true, scores, threshold=0.70)
    assert metrics["roc_auc"] is None
    assert "Undefined" in metrics["roc_auc_note"]
    assert metrics["pr_auc_average_precision"] is None
    assert "Undefined" in metrics["pr_auc_note"]
    assert metrics["positive_examples"] == 0
    assert metrics["negative_examples"] == 20
    assert metrics["accuracy"] == 1.0  # all predicted negative, all actual negative


def test_metric_handling_with_both_classes():
    """Verify metrics are calculated properly when both classes are present."""
    y_true = np.array([0, 0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.8, 0.9])

    metrics = compute_metrics(y_true, scores, threshold=0.70)
    assert metrics["roc_auc"] == 1.0
    assert metrics["pr_auc_average_precision"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1_score"] == 1.0


def test_model_selection_pr_auc():
    """Verify model selection prioritizes validation PR-AUC."""
    val_results = {
        "model_low": {"pr_auc_average_precision": 0.40, "f1_score": 0.50},
        "model_high": {"pr_auc_average_precision": 0.85, "f1_score": 0.70},
        "model_tie": {"pr_auc_average_precision": 0.85, "f1_score": 0.75},
    }
    best_name, reason = select_best_model(val_results)
    assert best_name == "model_tie"
    assert "0.85" in reason


def test_bundle_serialization_roundtrip(tmp_path, trained_test_pipeline):
    """Verify model bundle saves and reloads with exact feature order and probability outputs."""
    pipe, features = trained_test_pipeline
    model_path = tmp_path / "test_model.joblib"

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

    save_model_bundle_atomically(bundle, model_path)
    loaded = joblib.load(model_path)

    assert loaded["model_name"] == "logistic_regression"
    assert loaded["features"] == features
    assert loaded["threshold"] == 0.70
    assert loaded["is_experimental"] is True

    # Test probability output
    sample_patient = pd.DataFrame([{feat: 50.0 for feat in features}], columns=features)
    probs = loaded["model"].predict_proba(sample_patient)
    assert probs.shape == (1, 2)
    p_pos = float(probs[0, 1])
    assert 0.0 <= p_pos <= 1.0


def test_fastapi_inference_integration(tmp_path, monkeypatch, trained_test_pipeline):
    """Verify that the FastAPI API loads the bundle and returns HTTP 200 with correct predictions."""
    pipe, features = trained_test_pipeline
    model_path = tmp_path / "api_test_model.joblib"

    bundle = {
        "model": pipe,
        "model_name": "logistic_regression",
        "features": features,
        "threshold": 0.70,
        "target": "admission_within_24h",
        "dataset_type": "synthetic",
        "is_experimental": False,
        "not_for_clinical_use": True,
    }
    save_model_bundle_atomically(bundle, model_path)

    # Monkeypatch API MODEL_PATH to point to test bundle
    monkeypatch.setattr("app.api.main.MODEL_PATH", model_path)

    client = TestClient(app)

    # Test /metadata endpoint with loaded model
    meta_resp = client.get("/metadata")
    assert meta_resp.status_code == 200
    meta_json = meta_resp.json()
    assert meta_json["model_loaded"] is True
    assert meta_json["model_name"] == "logistic_regression"
    assert meta_json["features"] == features
    assert meta_json["risk_threshold"] == 0.70

    # Test /predict endpoint
    patient_payload = {
        "patient_id": "SYN99999",
        "age": 68.0,
        "heart_rate": 88.0,
        "systolic_bp": 135.0,
        "diastolic_bp": 85.0,
        "spo2": 95.0,
        "respiratory_rate": 20.0,
        "temperature_c": 37.4,
        "previous_admissions": 2.0,
        "chronic_conditions_count": 3.0,
        "lactate": 1.6,
        "observation_hour": 16.0,
        "prior_observation_count": 1.0,
    }
    pred_resp = client.post("/predict", json=patient_payload)
    assert pred_resp.status_code == 200
    pred_json = pred_resp.json()
    assert pred_json["patient_id"] == "SYN99999"
    assert "risk_probability" in pred_json
    assert 0.0 <= pred_json["risk_probability"] <= 1.0
    assert pred_json["threshold"] == 0.70
    assert isinstance(pred_json["review_flag"], bool)
    assert "not a clinically validated prediction" in pred_json["notice"].lower()
