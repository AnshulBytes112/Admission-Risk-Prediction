"""Phase 5.1: Model Selection Audit, Reproducibility, Feature Ordering, and Leakage Prevention Tests.

Academic demonstration disclaimer:
All records and labels are SYNTHETIC. Features, model architectures, and metrics
must never be interpreted as clinically validated or suitable for medical decisions.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest

from ml.evaluation import compute_metrics, select_best_model, tune_threshold_on_validation
from ml.features import (
    EXCLUDED_COLUMNS,
    FEATURE_ALLOWLIST,
    FEATURE_ALLOWLIST_V2,
    TARGET_COLUMN,
    select_and_validate_features,
)
from ml.training import split_dataset

MODEL_PATH = Path("models/risk_model.joblib")
METADATA_PATH = Path("models/model_metadata.json")
REPORT_PATH = Path("models/evaluation_report.json")


def test_model_selection_not_hardcoded():
    """Verify that select_best_model dynamically selects the model with highest PR-AUC and never hardcodes a winner."""
    # Scenario A: Candidate 'model_a' has higher PR-AUC
    metrics_a_wins = {
        "dummy": {"pr_auc_average_precision": 0.15, "f1_score": 0.0},
        "model_a": {"pr_auc_average_precision": 0.35, "f1_score": 0.40},
        "model_b": {"pr_auc_average_precision": 0.25, "f1_score": 0.30},
    }
    winner_a, reason_a = select_best_model(metrics_a_wins)
    assert winner_a == "model_a"
    assert "0.35" in reason_a

    # Scenario B: Candidate 'model_b' has higher PR-AUC
    metrics_b_wins = {
        "dummy": {"pr_auc_average_precision": 0.15, "f1_score": 0.0},
        "model_a": {"pr_auc_average_precision": 0.22, "f1_score": 0.25},
        "model_b": {"pr_auc_average_precision": 0.42, "f1_score": 0.45},
    }
    winner_b, reason_b = select_best_model(metrics_b_wins)
    assert winner_b == "model_b"
    assert "0.42" in reason_b

    # Scenario C: Tie on PR-AUC, broken by F1-score
    metrics_tie = {
        "model_x": {"pr_auc_average_precision": 0.30, "f1_score": 0.28},
        "model_y": {"pr_auc_average_precision": 0.30, "f1_score": 0.36},
    }
    winner_tie, _ = select_best_model(metrics_tie)
    assert winner_tie == "model_y"


def test_dummy_baseline_excluded_from_candidate_selection():
    """Verify dummy prevalence baseline is never chosen when non-dummy models are available."""
    metrics_dummy_higher_acc = {
        "dummy": {"pr_auc_average_precision": 0.85, "f1_score": 0.0, "accuracy": 0.85},
        "real_model": {"pr_auc_average_precision": 0.30, "f1_score": 0.35, "accuracy": 0.65},
    }
    winner, _ = select_best_model(metrics_dummy_higher_acc)
    assert winner == "real_model"


def test_reproducibility_deterministic_splits():
    """Verify that dataset splitting produces identical partitions across runs with the same seed."""
    raw_csv = Path("data/raw/train.csv")
    if not raw_csv.exists():
        pytest.skip("data/raw/train.csv not found")

    df = pd.read_csv(raw_csv)
    split1 = split_dataset(df, test_size=0.20, val_size=0.20, seed=42)
    split2 = split_dataset(df, test_size=0.20, val_size=0.20, seed=42)

    pd.testing.assert_frame_equal(split1.train_df, split2.train_df)
    pd.testing.assert_frame_equal(split1.val_df, split2.val_df)
    pd.testing.assert_frame_equal(split1.test_df, split2.test_df)
    assert split1.train_positives == split2.train_positives
    assert split1.val_positives == split2.val_positives
    assert split1.test_positives == split2.test_positives


def test_no_target_or_provenance_leakage():
    """Verify that prohibited identifier, metadata, target, and disposition columns are excluded from features."""
    all_raw_cols = [
        "record_id", "patient_id", "age", "heart_rate", "systolic_bp", "diastolic_bp",
        "oxygen_saturation", "spo2", "respiratory_rate", "temperature_c", "previous_admissions",
        "comorbidity_count", "chronic_conditions_count", "esi", "arrival_by_ambulance",
        "altered_mental_status", "admission_within_24h", "is_synthetic", "data_origin", "disposition"
    ]
    features = select_and_validate_features(all_raw_cols)

    prohibited = {"record_id", "is_synthetic", "data_origin", "admission_within_24h", "disposition"}
    intersection = prohibited.intersection(set(features))
    assert not intersection, f"Target or provenance leakage found in features: {intersection}"


def test_threshold_tuning_uses_only_validation_data():
    """Verify threshold tuning evaluates candidates purely on validation ground truth and scores."""
    # Synthetic validation set with known probabilities
    y_val = np.array([0, 0, 0, 0, 1, 1, 0, 1, 0, 0])
    # Scores where true positives score high (0.6 - 0.8) and negatives score low (0.1 - 0.3)
    val_scores = np.array([0.15, 0.22, 0.28, 0.12, 0.75, 0.68, 0.35, 0.82, 0.18, 0.25])

    tuned_th, summary = tune_threshold_on_validation(
        y_val, val_scores, candidate_thresholds=[0.30, 0.40, 0.50, 0.60]
    )
    assert summary["tuning_applied"] is True
    # Cutoff between 0.40 and 0.60 perfectly separates positives
    assert tuned_th in (0.40, 0.50, 0.60)
    assert summary["validation_best_f1"] == 1.0


def test_single_and_batch_prediction_parity_on_artifact():
    """Verify single-record and batch prediction parity on the active model bundle."""
    if not MODEL_PATH.exists():
        pytest.skip(f"Model artifact not found at {MODEL_PATH}")

    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    features = bundle["features"]

    # Create 3 distinct test records
    pt1 = {f: 50.0 for f in features}
    pt1["age"] = 35.0
    pt1["heart_rate"] = 72.0

    pt2 = {f: 50.0 for f in features}
    pt2["age"] = 78.0
    pt2["heart_rate"] = 115.0

    pt3 = {f: 50.0 for f in features}
    pt3["age"] = 22.0
    pt3["heart_rate"] = 65.0

    batch_df = pd.DataFrame([pt1, pt2, pt3], columns=features)
    batch_probs = model.predict_proba(batch_df)[:, 1]

    single_probs = []
    for pt in [pt1, pt2, pt3]:
        single_df = pd.DataFrame([pt], columns=features)
        prob = float(model.predict_proba(single_df)[0, 1])
        single_probs.append(prob)

    max_diff = np.max(np.abs(batch_probs - np.array(single_probs)))
    assert max_diff < 1e-6, f"Single vs batch prediction parity error: max_diff={max_diff}"


def test_synthetic_warnings_present_in_metadata_and_report():
    """Verify all saved evaluation reports and model metadata explicitly declare synthetic academic nature."""
    if METADATA_PATH.exists():
        meta = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        assert meta.get("clinical_use") is False
        assert meta.get("not_for_clinical_use") is True
        assert meta.get("dataset_type") == "synthetic"
        assert "synthetic" in meta.get("safety_disclaimer", "").lower()

    if REPORT_PATH.exists():
        report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
        assert report.get("data_origin") == "synthetic"
        assert "synthetic" in report.get("disclaimer", "").lower()
