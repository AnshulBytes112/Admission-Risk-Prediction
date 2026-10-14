"""Feature selection, data loading, leakage prevention, and data quality diagnostics.

Academic demonstration disclaimer:
All records and labels are SYNTHETIC. Features, model architectures, and metrics
must never be interpreted as clinically validated or suitable for medical decisions.
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

TARGET_COLUMN = "admission_within_24h"
GROUP_COLUMN = "patient_id"

# Explicit allowlist of candidate numeric features for model training and inference
FEATURE_ALLOWLIST = [
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
    "observation_hour",
    "prior_observation_count",
]

# Explicit list of columns to be excluded from feature inputs to prevent target leakage,
# patient identity memorization, or streaming/lake metadata contamination.
EXCLUDED_COLUMNS = [
    "event_id",
    "patient_id",
    "observation_ts",
    "observation_date",
    "admission_within_24h",
    "is_synthetic",
    "data_origin",
    "synthetic_data",
    "kafka_topic",
    "kafka_partition",
    "kafka_offset",
]


class DataAdequacyError(ValueError):
    """Raised when the input dataset fails minimum statistical or structural adequacy checks."""
    pass


@dataclass
class DatasetDiagnostics:
    """Summary metrics of dataset structure, class balance, and data hygiene."""
    total_rows: int
    total_columns: int
    unique_patients: int
    observations_per_patient_min: int
    observations_per_patient_max: int
    observations_per_patient_mean: float
    duplicate_event_ids: int
    missing_target_count: int
    target_class_counts: dict[int, int]
    target_class_proportions: dict[int, float]
    missing_features_counts: dict[str, int]
    usable_numeric_features: list[str]
    synthetic_provenance: dict[str, Any]
    dataset_fingerprint_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_dataset_fingerprint(df: pd.DataFrame) -> str:
    """Compute deterministic SHA-256 fingerprint from dataframe columns and row sample."""
    hasher = hashlib.sha256()
    sorted_cols = sorted(df.columns.tolist())
    hasher.update(",".join(sorted_cols).encode("utf-8"))
    hasher.update(str(len(df)).encode("utf-8"))
    sample_size = min(len(df), 50)
    if sample_size > 0:
        sample_vals = df[sorted_cols].head(sample_size).to_string().encode("utf-8")
        hasher.update(sample_vals)
    return hasher.hexdigest()


def load_feature_dataset(path: str | Path) -> pd.DataFrame:
    """Load feature dataset from a partitioned Parquet directory, Parquet file, or CSV.

    Args:
        path: Path to Parquet directory, Parquet file, or CSV file.

    Returns:
        pd.DataFrame containing the loaded dataset with non-finite values mapped to NaN.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Feature dataset path does not exist: {p}")

    if p.is_dir():
        try:
            # Reads directory of partitioned Parquet files using pyarrow
            df = pd.read_parquet(p, engine="pyarrow")
        except Exception as exc:
            raise RuntimeError(
                f"Failed to read Parquet directory at {p} with PyArrow: {exc}"
            ) from exc
    elif p.suffix.lower() in (".parquet", ".pq"):
        df = pd.read_parquet(p, engine="pyarrow")
    elif p.suffix.lower() == ".csv":
        df = pd.read_csv(p)
    else:
        raise ValueError(
            f"Unsupported file format for feature dataset: {p}. "
            "Expected a directory containing Parquet partitions, a .parquet file, or a .csv file."
        )

    if df.empty:
        raise ValueError(f"Feature dataset at {p} is empty (0 rows).")

    # Replace infinite values with NaN for controlled imputation
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def select_and_validate_features(columns: list[str]) -> list[str]:
    """Select allowed model features and verify that no identifier/metadata columns leak.

    Args:
        columns: Available column names in dataframe.

    Returns:
        Sorted/ordered list of validated features.
    """
    # Enforce leak prevention: none of the excluded columns can be selected
    selected = [c for c in FEATURE_ALLOWLIST if c in columns and c not in EXCLUDED_COLUMNS]

    leaked = [c for c in selected if c in EXCLUDED_COLUMNS]
    if leaked:
        raise ValueError(f"Data leakage detected! Excluded columns selected as features: {leaked}")

    if not selected:
        raise ValueError(
            f"None of the required features from the allowlist were found. "
            f"Expected features within: {FEATURE_ALLOWLIST}"
        )

    return selected


def inspect_and_diagnose_dataset(df: pd.DataFrame) -> DatasetDiagnostics:
    """Inspect input dataset and compute comprehensive diagnostic metrics.

    Args:
        df: Input DataFrame.

    Returns:
        DatasetDiagnostics summary.
    """
    if df.empty:
        raise ValueError("Cannot inspect an empty dataset.")

    # Target column validation
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Required target column '{TARGET_COLUMN}' is missing from dataset.")

    target_series = df[TARGET_COLUMN]
    missing_target = int(target_series.isna().sum())

    # Check for non-binary target values
    non_null_target = target_series.dropna()
    invalid_targets = (~non_null_target.isin([0, 1, 0.0, 1.0])).sum()
    if invalid_targets > 0:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' contains {invalid_targets} non-binary values. "
            "Only 0 and 1 are permitted."
        )

    if missing_target > 0:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' contains {missing_target} missing/null values."
        )

    # Convert target to integer counts
    int_target = non_null_target.astype(int)
    counts = int_target.value_counts().to_dict()
    total_valid = len(int_target)
    proportions = {cls: (cnt / total_valid) if total_valid > 0 else 0.0 for cls, cnt in counts.items()}

    # Patient statistics
    unique_patients = int(df[GROUP_COLUMN].nunique()) if GROUP_COLUMN in df.columns else 0
    if GROUP_COLUMN in df.columns and unique_patients > 0:
        counts_per_patient = df[GROUP_COLUMN].value_counts()
        obs_min = int(counts_per_patient.min())
        obs_max = int(counts_per_patient.max())
        obs_mean = round(float(counts_per_patient.mean()), 2)
    else:
        obs_min = 0
        obs_max = 0
        obs_mean = 0.0

    # Duplicate events
    dup_events = int(df["event_id"].duplicated().sum()) if "event_id" in df.columns else 0

    # Usable features and missing counts
    usable_features = select_and_validate_features(df.columns.tolist())
    missing_features = {feat: int(df[feat].isna().sum()) for feat in usable_features}

    # Synthetic provenance
    synthetic_provenance: dict[str, Any] = {
        "is_synthetic_column_present": "is_synthetic" in df.columns,
        "data_origin_column_present": "data_origin" in df.columns,
        "is_synthetic_all_true": bool((df["is_synthetic"] == True).all()) if "is_synthetic" in df.columns else False,  # noqa: E712
        "data_origin_unique": df["data_origin"].unique().tolist() if "data_origin" in df.columns else [],
        "synthetic_notice": "Academic demonstration: all records and labels are synthetically generated.",
    }

    fingerprint = compute_dataset_fingerprint(df)

    return DatasetDiagnostics(
        total_rows=len(df),
        total_columns=len(df.columns),
        unique_patients=unique_patients,
        observations_per_patient_min=obs_min,
        observations_per_patient_max=obs_max,
        observations_per_patient_mean=obs_mean,
        duplicate_event_ids=dup_events,
        missing_target_count=missing_target,
        target_class_counts=counts,
        target_class_proportions=proportions,
        missing_features_counts=missing_features,
        usable_numeric_features=usable_features,
        synthetic_provenance=synthetic_provenance,
        dataset_fingerprint_sha256=fingerprint,
    )


def validate_dataset_adequacy(
    diagnostics: DatasetDiagnostics,
    minimum_positive_examples: int = 5,
    allow_small_dataset: bool = False,
) -> None:
    """Enforce data adequacy thresholds before model training and evaluation.

    Args:
        diagnostics: Computed DatasetDiagnostics.
        minimum_positive_examples: Minimum number of positive cases required for standard training.
        allow_small_dataset: Flag to permit experimental/smoke-test runs on undersized datasets.

    Raises:
        DataAdequacyError: If dataset fails criteria and allow_small_dataset is False.
    """
    total_rows = diagnostics.total_rows
    pos_count = diagnostics.target_class_counts.get(1, 0)
    neg_count = diagnostics.target_class_counts.get(0, 0)

    # Both classes must exist in the dataset
    if pos_count == 0 or neg_count == 0:
        raise DataAdequacyError(
            f"Dataset does not contain both binary classes! "
            f"Class 0 count: {neg_count}, Class 1 count: {pos_count}. "
            "Binary classification models cannot be trained or evaluated without examples of both classes."
        )

    # Enforce minimum positive examples for reliable validation
    if pos_count < minimum_positive_examples and not allow_small_dataset:
        raise DataAdequacyError(
            f"Dataset adequacy check failed: Only {pos_count} positive example(s) found "
            f"among {total_rows} total rows (minimum required: {minimum_positive_examples}).\n"
            f"A single positive sample or tiny minority class cannot be divided across "
            f"Train / Validation / Test sets without leaving evaluation splits with 0 positive cases.\n\n"
            f"Remediation:\n"
            f"  1. Generate a larger synthetic observation dataset with sufficient admission events:\n"
            f"     py -3.13 scripts\\generate_synthetic_data.py --rows 1000 --seed 42 --output data\\raw\\patient_observations.csv\n"
            f"  2. Ingest through Kafka/HDFS/Spark pipeline to produce balanced Parquet features.\n"
            f"  3. For technical pipeline smoke-testing only, re-run with '--allow-small-dataset':\n"
            f"     py -3.13 scripts\\train_risk_model.py --allow-small-dataset\n"
            f"     (Note: artifacts will be explicitly tagged as experimental and unvalidated)."
        )
