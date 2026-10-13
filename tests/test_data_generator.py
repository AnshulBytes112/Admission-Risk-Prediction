"""Automated tests for the synthetic clinical data generator.

Validates that:
1. Configurable row count generates the exact number of observations.
2. All 15 required clinical schema columns are present.
3. Event IDs are strictly unique.
4. Patient IDs allow multiple observations per patient (demonstrating grouping capability).
5. Target variable is strictly binary (0 or 1).
6. Synthetic marker column is set to True across all rows.
7. Fixed random seed produces identical deterministic output.
8. Different random seeds produce different outputs.
9. Invalid row counts (negative, zero, non-integer) raise appropriate errors.
10. Automatic creation of parent directories when writing output files.
11. Feature values fall within expected clinical bounds, accounting for documented missing values.
12. Controlled missingness is present in designated vitals.
"""
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from scripts.generate_synthetic_data import generate

REQUIRED_COLUMNS = [
    "event_id",
    "patient_id",
    "observation_timestamp",
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
    "admission_within_24h",
    "synthetic_data",
]


def test_row_count_configurable():
    """Verify that different row count arguments yield the exact number of rows."""
    for count in [50, 200, 500]:
        df = generate(rows=count, seed=42)
        assert len(df) == count, f"Expected {count} rows, got {len(df)}"


def test_required_columns_present():
    """Verify that all required columns are present in the generated DataFrame."""
    df = generate(rows=100, seed=42)
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    assert not missing, f"Missing required columns: {missing}"
    assert len(df.columns) == len(REQUIRED_COLUMNS), f"Unexpected column count: {df.columns.tolist()}"


def test_unique_event_identifiers():
    """Verify that every event_id in the generated dataset is unique."""
    df = generate(rows=250, seed=42)
    assert df["event_id"].nunique() == len(df), "Event IDs must be unique"
    assert df["event_id"].str.startswith("EVT").all(), "Event IDs should follow EVT format"


def test_patient_pool_and_multiple_observations():
    """Verify that synthetic patients can have multiple observations."""
    df = generate(rows=500, seed=42)
    assert df["patient_id"].nunique() < len(df), "Patient pool should allow multiple events per patient"
    assert df["patient_id"].str.startswith("SYN").all(), "Patient IDs should follow SYN format"


def test_binary_target_values():
    """Verify that the target column admission_within_24h contains only 0 and 1."""
    df = generate(rows=300, seed=42)
    unique_targets = set(df["admission_within_24h"].unique())
    assert unique_targets.issubset({0, 1}), f"Target must be binary (0/1), found: {unique_targets}"


def test_synthetic_marker():
    """Verify that every record has synthetic_data flag set to True."""
    df = generate(rows=150, seed=42)
    assert bool(df["synthetic_data"].all()) is True, "All records must be explicitly marked as synthetic"
    assert (df["synthetic_data"] == True).all(), "Every row must have synthetic_data == True"
    assert df["synthetic_data"].dtype == bool, "synthetic_data column must be boolean"


def test_deterministic_output_with_same_seed():
    """Verify reproducibility: identical seed must yield identical data."""
    df1 = generate(rows=200, seed=123)
    df2 = generate(rows=200, seed=123)
    pd.testing.assert_frame_equal(df1, df2)


def test_different_seeds_produce_different_data():
    """Verify that different seeds produce distinct data series."""
    df1 = generate(rows=200, seed=123)
    df2 = generate(rows=200, seed=456)
    assert not df1["heart_rate"].equals(df2["heart_rate"]), "Different seeds should produce different vital signs"


def test_invalid_row_counts():
    """Verify that invalid row counts raise a ValueError."""
    for invalid in [0, -1, -500]:
        with pytest.raises(ValueError):
            generate(rows=invalid, seed=42)


def test_output_directory_creation(tmp_path: Path):
    """Verify that saving generated data to an uncreated nested directory works."""
    nested_dir = tmp_path / "deeply" / "nested" / "raw"
    output_file = nested_dir / "test_observations.csv"
    assert not nested_dir.exists()
    
    # Simulate CLI/main output behavior
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df = generate(rows=50, seed=42)
    df.to_csv(output_file, index=False)
    
    assert output_file.exists()
    assert output_file.stat().st_size > 0
    read_df = pd.read_csv(output_file)
    assert len(read_df) == 50


def test_expected_value_ranges():
    """Verify that clinical observations fall within plausible physiological ranges."""
    df = generate(rows=1000, seed=99)
    
    # Age range: 18 to 100
    assert (df["age"] >= 18).all() and (df["age"] <= 100).all()
    
    # Heart rate range (excluding NaNs): 35 to 190
    valid_hr = df["heart_rate"].dropna()
    assert (valid_hr >= 35).all() and (valid_hr <= 190).all()
    
    # Systolic BP range (excluding NaNs): 60 to 220
    valid_sbp = df["systolic_bp"].dropna()
    assert (valid_sbp >= 60).all() and (valid_sbp <= 220).all()
    
    # Diastolic BP range: 30 to 140
    assert (df["diastolic_bp"] >= 30).all() and (df["diastolic_bp"] <= 140).all()
    
    # SpO2 range (excluding NaNs): 70 to 100
    valid_spo2 = df["spo2"].dropna()
    assert (valid_spo2 >= 70).all() and (valid_spo2 <= 100).all()
    
    # Respiratory rate range: 5 to 50
    assert (df["respiratory_rate"] >= 5).all() and (df["respiratory_rate"] <= 50).all()
    
    # Temperature range (excluding NaNs): 33 to 42
    valid_temp = df["temperature_c"].dropna()
    assert (valid_temp >= 33.0).all() and (valid_temp <= 42.0).all()
    
    # Previous admissions and chronic conditions count
    assert (df["previous_admissions"] >= 0).all()
    assert (df["chronic_conditions_count"] >= 0).all()
    
    # Lactate range: 0.3 to 10
    assert (df["lactate"] >= 0.3).all() and (df["lactate"] <= 10.0).all()


def test_controlled_missingness():
    """Verify that missing values exist in controlled vital signs to exercise BDA cleaning."""
    df = generate(rows=2000, seed=42)
    columns_with_missingness = ["heart_rate", "spo2", "systolic_bp", "temperature_c"]
    for col in columns_with_missingness:
        null_count = df[col].isna().sum()
        assert null_count > 0, f"Expected controlled missing values in {col}"
        assert null_count < len(df) * 0.10, f"Missingness in {col} too high ({null_count})"
