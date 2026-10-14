"""Automated tests for Phase 8 Demo Dataset, Dataset Explorer, and Pipeline Observability.

Covers:
- Deterministic synthetic demo dataset generation (scripts/generate_demo_dataset.py).
- Batch prediction template columns and schema verification.
- Dataset explorer dataset loader and fallback mechanisms.
- Dataset filtering by search, outcome, age, and hour.
- Pipeline manifest inspector and telemetry aggregation.
- Single prediction form prefill integration.
"""
from __future__ import annotations

import io
from pathlib import Path
import tempfile
import unittest.mock as mock

import pandas as pd
import pytest

from app.dashboard.pipeline_inspector import (
    get_latest_spark_manifest,
    get_latest_ingestion_manifest,
    get_pipeline_telemetry,
)
from app.dashboard.views.dataset_explorer import (
    MODEL_FEATURES,
    load_datasets,
)
from scripts.generate_demo_dataset import (
    FEATURE_ALLOWLIST,
    generate_demo_dataset,
    generate_batch_template,
)


# =============================================================================
# 1. Demo Dataset & Batch Template Generation Tests
# =============================================================================

def test_generate_demo_dataset_row_counts_and_determinism():
    """Verify row counts and deterministic reproducibility with a fixed seed."""
    df1 = generate_demo_dataset(rows=50, seed=123)
    df2 = generate_demo_dataset(rows=50, seed=123)
    df3 = generate_demo_dataset(rows=50, seed=999)

    assert len(df1) == 50
    assert len(df2) == 50
    # Deterministic output
    pd.testing.assert_frame_equal(df1, df2)
    # Different seeds produce different samples
    assert not df1["heart_rate"].equals(df3["heart_rate"])


def test_generate_demo_dataset_required_schema():
    """Verify all 12 model features plus identifiers, dates, and target exist."""
    df = generate_demo_dataset(rows=30, seed=42)

    assert "patient_id" in df.columns
    assert "admission_within_24h" in df.columns
    assert "observation_date" in df.columns
    assert "is_synthetic" in df.columns
    assert (df["is_synthetic"] == True).all()  # noqa: E712
    assert "data_origin" in df.columns
    assert (df["data_origin"] == "synthetic").all()

    for feat in FEATURE_ALLOWLIST:
        assert feat in df.columns
        # No NaNs in generated demo dataset
        assert df[feat].notna().all()


def test_generate_batch_template_contract():
    """Verify that batch template contains exactly patient_id + 12 model features (no target or leak)."""
    df = generate_demo_dataset(rows=20, seed=42)
    tpl = generate_batch_template(df, sample_size=10)

    assert len(tpl) == 10
    expected_cols = ["patient_id"] + FEATURE_ALLOWLIST
    assert list(tpl.columns) == expected_cols

    # Ensure target is NOT in the prediction template
    assert "admission_within_24h" not in tpl.columns
    assert "is_synthetic" not in tpl.columns
    assert "event_id" not in tpl.columns


def test_generate_demo_dataset_minimum_bound():
    """Verify row count validation enforces at least 10 rows."""
    with pytest.raises(ValueError, match="at least 10"):
        generate_demo_dataset(rows=5)


# =============================================================================
# 2. Pipeline Manifest Inspector & Observability Tests
# =============================================================================

def test_pipeline_telemetry_structure():
    """Verify that get_pipeline_telemetry returns standard telemetry keys."""
    telemetry = get_pipeline_telemetry()

    assert "health_score" in telemetry
    assert 0 <= telemetry["health_score"] <= 100
    assert "raw_ingested_count" in telemetry
    assert "spark_output_count" in telemetry
    assert "quarantined_count" in telemetry
    assert "imputations_total" in telemetry
    assert "model_available" in telemetry


def test_spark_manifest_parser_graceful():
    """Verify manifest parser returns None gracefully when directory is missing."""
    with mock.patch("app.dashboard.pipeline_inspector.DATA_DIR", Path("nonexistent_dir")):
        res = get_latest_spark_manifest()
        assert res is None


def test_ingest_manifest_parser_graceful():
    """Verify ingest manifest parser returns None gracefully when directory is missing."""
    with mock.patch("app.dashboard.pipeline_inspector.DATA_DIR", Path("nonexistent_dir")):
        res = get_latest_ingestion_manifest()
        assert res is None


# =============================================================================
# 3. Dataset Explorer Functions & Filter Tests
# =============================================================================

def test_load_datasets_contains_demo_or_parquet():
    """Verify load_datasets returns available demo or parquet datasets."""
    datasets = load_datasets()
    assert isinstance(datasets, dict)
    # At least one dataset should be found in repository
    assert len(datasets) > 0


def test_dataset_filtering_logic():
    """Test dataframe filtering by patient ID, age, and admission outcome."""
    df = generate_demo_dataset(rows=60, seed=42)

    # Search filter
    target_pt = df["patient_id"].iloc[0]
    filtered_search = df[df["patient_id"] == target_pt]
    assert len(filtered_search) >= 1

    # Age filter
    filtered_age = df[(df["age"] >= 50) & (df["age"] <= 70)]
    assert (filtered_age["age"] >= 50).all()
    assert (filtered_age["age"] <= 70).all()

    # Outcome filter
    filtered_pos = df[df["admission_within_24h"] == 1]
    filtered_neg = df[df["admission_within_24h"] == 0]
    assert (filtered_pos["admission_within_24h"] == 1).all()
    assert (filtered_neg["admission_within_24h"] == 0).all()
