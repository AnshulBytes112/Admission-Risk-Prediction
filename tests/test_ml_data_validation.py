"""Unit tests for Phase 5 data validation, schema integrity, and adequacy safeguards.

Verifies:
- Missing target column raises ValueError.
- Null target values raise ValueError.
- Non-binary target values raise ValueError.
- Empty dataset raises ValueError.
- Duplicate event ID detection.
- Synthetic-data provenance detection.
- Insufficient positive-class examples trigger DataAdequacyError when not overridden.
- Existing artifacts are preserved when training fails.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from ml.features import (
    FEATURE_ALLOWLIST,
    TARGET_COLUMN,
    DataAdequacyError,
    inspect_and_diagnose_dataset,
    load_feature_dataset,
    validate_dataset_adequacy,
)
from ml.training import save_model_bundle_atomically


@pytest.fixture
def sample_valid_df() -> pd.DataFrame:
    """Create a minimal valid synthetic dataframe with 20 rows and 2 classes."""
    np.random.seed(42)
    n = 20
    data = {
        "event_id": [f"EVT{i:05d}" for i in range(n)],
        "patient_id": [f"SYN{i % 5:05d}" for i in range(n)],
        "observation_ts": pd.date_range("2026-01-01", periods=n, freq="h"),
        "observation_date": ["2026-01-01"] * n,
        "observation_hour": [12] * n,
        "admission_within_24h": [1 if i < 5 else 0 for i in range(n)],
        "is_synthetic": [True] * n,
        "data_origin": ["synthetic"] * n,
        "prior_observation_count": [0] * n,
        "kafka_topic": ["patient-observations"] * n,
        "kafka_partition": [0] * n,
        "kafka_offset": list(range(n)),
    }
    for feat in FEATURE_ALLOWLIST:
        if feat not in data:
            data[feat] = np.random.uniform(20.0, 100.0, size=n)
    return pd.DataFrame(data)


def test_missing_target_column(sample_valid_df):
    """Verify that a missing target column raises ValueError."""
    df_missing_target = sample_valid_df.drop(columns=[TARGET_COLUMN])
    with pytest.raises(ValueError, match="Required target column .* is missing"):
        inspect_and_diagnose_dataset(df_missing_target)


def test_null_target_values(sample_valid_df):
    """Verify that null/NaN in target raises ValueError."""
    df_null_target = sample_valid_df.copy()
    df_null_target.loc[0, TARGET_COLUMN] = np.nan
    with pytest.raises(ValueError, match="missing/null values"):
        inspect_and_diagnose_dataset(df_null_target)


def test_non_binary_target_values(sample_valid_df):
    """Verify that target values other than 0 and 1 raise ValueError."""
    df_invalid_target = sample_valid_df.copy()
    df_invalid_target.loc[0, TARGET_COLUMN] = 99
    with pytest.raises(ValueError, match="non-binary values"):
        inspect_and_diagnose_dataset(df_invalid_target)


def test_empty_dataset(tmp_path):
    """Verify that an empty DataFrame or empty Parquet raises ValueError."""
    empty_df = pd.DataFrame()
    with pytest.raises(ValueError, match="empty dataset"):
        inspect_and_diagnose_dataset(empty_df)

    empty_csv = tmp_path / "empty.csv"
    empty_csv.write_text("event_id,admission_within_24h\n", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        load_feature_dataset(empty_csv)


def test_duplicate_event_ids(sample_valid_df):
    """Verify duplicate event IDs are correctly detected in diagnostics."""
    df_dups = sample_valid_df.copy()
    df_dups.loc[1, "event_id"] = df_dups.loc[0, "event_id"]
    diagnostics = inspect_and_diagnose_dataset(df_dups)
    assert diagnostics.duplicate_event_ids == 1


def test_synthetic_provenance_detection(sample_valid_df):
    """Verify synthetic provenance markers are recorded."""
    diagnostics = inspect_and_diagnose_dataset(sample_valid_df)
    prov = diagnostics.synthetic_provenance
    assert prov["is_synthetic_column_present"] is True
    assert prov["is_synthetic_all_true"] is True
    assert "synthetic" in prov["data_origin_unique"]


def test_insufficient_positive_class_fails_safe(sample_valid_df):
    """Verify that having fewer positive examples than minimum required raises DataAdequacyError."""
    # Set only 1 positive row
    df_skewed = sample_valid_df.copy()
    df_skewed[TARGET_COLUMN] = 0
    df_skewed.loc[0, TARGET_COLUMN] = 1

    diagnostics = inspect_and_diagnose_dataset(df_skewed)
    assert diagnostics.target_class_counts[1] == 1

    # Should raise error when allow_small_dataset is False
    with pytest.raises(DataAdequacyError, match="Dataset adequacy check failed"):
        validate_dataset_adequacy(diagnostics, minimum_positive_examples=5, allow_small_dataset=False)

    # Should NOT raise error when allow_small_dataset is True
    validate_dataset_adequacy(diagnostics, minimum_positive_examples=5, allow_small_dataset=True)


def test_missing_binary_class_entirely(sample_valid_df):
    """Verify error when a class is completely absent even with allow_small_dataset."""
    df_all_zero = sample_valid_df.copy()
    df_all_zero[TARGET_COLUMN] = 0
    diagnostics = inspect_and_diagnose_dataset(df_all_zero)
    with pytest.raises(DataAdequacyError, match="does not contain both binary classes"):
        validate_dataset_adequacy(diagnostics, minimum_positive_examples=5, allow_small_dataset=True)


def test_preservation_of_existing_artifacts_on_failure(tmp_path):
    """Verify existing model artifacts are not overwritten if saving fails or bundle is invalid."""
    model_file = tmp_path / "risk_model.joblib"
    initial_content = b"ORIGINAL_VALID_MODEL_DATA"
    model_file.write_bytes(initial_content)

    # Attempt saving an invalid bundle (missing required keys)
    invalid_bundle = {"invalid_key": 123}
    with pytest.raises(ValueError, match="missing required keys"):
        save_model_bundle_atomically(invalid_bundle, model_file)

    # Verify original file content was preserved
    assert model_file.read_bytes() == initial_content
