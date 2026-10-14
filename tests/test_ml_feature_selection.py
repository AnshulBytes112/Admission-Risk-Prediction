"""Unit tests for Phase 5 feature selection, leakage prevention, and patient grouping.

Verifies:
- All identifiers, metadata, and target columns are strictly excluded from predictors.
- Feature allowlist filters out unapproved columns.
- Group-aware train/val/test splitting guarantees zero patient overlap across splits.
- Splitting is deterministic given the same random seed.
- Fallback strategy handles datasets without patient grouping gracefully.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.features import (
    EXCLUDED_COLUMNS,
    FEATURE_ALLOWLIST,
    TARGET_COLUMN,
    select_and_validate_features,
)
from ml.training import split_dataset


@pytest.fixture
def multi_patient_df() -> pd.DataFrame:
    """Create a DataFrame with 10 distinct patients and 4 observations each (40 rows)."""
    np.random.seed(42)
    n_patients = 10
    obs_per_patient = 4
    total = n_patients * obs_per_patient

    records = []
    for p_idx in range(n_patients):
        patient_id = f"SYN{p_idx:04d}"
        for obs_idx in range(obs_per_patient):
            row = {
                "event_id": f"EVT{len(records):05d}",
                "patient_id": patient_id,
                "observation_ts": f"2026-01-0{obs_idx+1} 12:00:00",
                "observation_date": f"2026-01-0{obs_idx+1}",
                "observation_hour": 12,
                "admission_within_24h": 1 if p_idx < 3 else 0,
                "is_synthetic": True,
                "data_origin": "synthetic",
                "prior_observation_count": obs_idx,
                "kafka_topic": "patient-observations",
                "kafka_partition": 0,
                "kafka_offset": len(records),
                "unapproved_future_leak_column": 999.0,
            }
            for feat in FEATURE_ALLOWLIST:
                if feat not in row:
                    row[feat] = float(np.random.uniform(20.0, 100.0))
            records.append(row)
    return pd.DataFrame(records)


def test_feature_allowlist_and_exclusion(multi_patient_df):
    """Verify that only approved features are selected and excluded columns are stripped."""
    cols = multi_patient_df.columns.tolist()
    selected = select_and_validate_features(cols)

    # Check that no excluded column is in selected features
    for excluded in EXCLUDED_COLUMNS:
        assert excluded not in selected, f"Leaked column '{excluded}' found in selected features!"

    # Check unapproved columns are not selected
    assert "unapproved_future_leak_column" not in selected

    # Check all selected features are in allowlist
    for feat in selected:
        assert feat in FEATURE_ALLOWLIST


def test_patient_level_leakage_prevention(multi_patient_df):
    """Verify that patient IDs never overlap between train, val, and test splits."""
    split = split_dataset(
        multi_patient_df,
        test_size=0.20,
        val_size=0.20,
        seed=42,
    )

    assert split.patient_grouping_enforced is True

    train_patients = set(split.train_df["patient_id"].unique())
    val_patients = set(split.val_df["patient_id"].unique())
    test_patients = set(split.test_df["patient_id"].unique())

    # Strict disjunction checks
    assert len(train_patients & val_patients) == 0, "Patient overlap between train and val!"
    assert len(train_patients & test_patients) == 0, "Patient overlap between train and test!"
    assert len(val_patients & test_patients) == 0, "Patient overlap between val and test!"

    # Total rows must match original
    total_split_rows = len(split.train_df) + len(split.val_df) + len(split.test_df)
    assert total_split_rows == len(multi_patient_df)


def test_split_reproducibility(multi_patient_df):
    """Verify that identical random seeds yield identical splits."""
    split1 = split_dataset(multi_patient_df, test_size=0.20, val_size=0.20, seed=123)
    split2 = split_dataset(multi_patient_df, test_size=0.20, val_size=0.20, seed=123)

    pd.testing.assert_frame_equal(split1.train_df, split2.train_df)
    pd.testing.assert_frame_equal(split1.val_df, split2.val_df)
    pd.testing.assert_frame_equal(split1.test_df, split2.test_df)


def test_fallback_when_patient_grouping_unavailable(multi_patient_df):
    """Verify that split_dataset falls back to random split when patient_id is absent."""
    df_no_patient = multi_patient_df.drop(columns=["patient_id"])
    split = split_dataset(df_no_patient, test_size=0.20, val_size=0.20, seed=42)

    assert split.patient_grouping_enforced is False
    assert "Stratified/Simple Random Split" in split.strategy
    total_split_rows = len(split.train_df) + len(split.val_df) + len(split.test_df)
    assert total_split_rows == len(df_no_patient)
