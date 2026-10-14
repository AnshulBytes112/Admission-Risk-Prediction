"""Model definitions, training pipelines, patient-grouped splitting, and atomic persistence.

Academic demonstration disclaimer:
All records and labels are SYNTHETIC. Features, model architectures, and metrics
must never be interpreted as clinically validated or suitable for medical decisions.
"""
from __future__ import annotations

import logging
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.features import GROUP_COLUMN, TARGET_COLUMN

logger = logging.getLogger(__name__)


@dataclass
class DatasetSplit:
    """Holds train, validation, and test splits along with verification metadata."""
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    strategy: str
    patient_grouping_enforced: bool
    train_size: int
    val_size: int
    test_size: int
    train_positives: int
    val_positives: int
    test_positives: int
    train_negatives: int
    val_negatives: int
    test_negatives: int

    def summary(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "patient_grouping_enforced": self.patient_grouping_enforced,
            "train": {
                "total": self.train_size,
                "positives": self.train_positives,
                "negatives": self.train_negatives,
                "positive_prevalence": round(self.train_positives / self.train_size, 4) if self.train_size else 0.0,
            },
            "validation": {
                "total": self.val_size,
                "positives": self.val_positives,
                "negatives": self.val_negatives,
                "positive_prevalence": round(self.val_positives / self.val_size, 4) if self.val_size else 0.0,
            },
            "test": {
                "total": self.test_size,
                "positives": self.test_positives,
                "negatives": self.test_negatives,
                "positive_prevalence": round(self.test_positives / self.test_size, 4) if self.test_size else 0.0,
            },
        }


def split_dataset(
    df: pd.DataFrame,
    test_size: float = 0.20,
    val_size: float = 0.20,
    seed: int = 42,
    group_col: str = GROUP_COLUMN,
    allow_small_dataset: bool = False,
) -> DatasetSplit:
    """Split dataset into train, validation, and test sets.

    Enforces patient-level grouping to prevent identical patients from leaking across
    train and evaluation sets.

    Args:
        df: Input DataFrame.
        test_size: Fraction of data reserved for final held-out testing.
        val_size: Fraction of data reserved for model selection and validation.
        seed: Random seed for reproducible splitting.
        group_col: Column name identifying patient groups.
        allow_small_dataset: Whether experimental small-dataset overrides are permitted.

    Returns:
        DatasetSplit containing train, validation, and test sets.
    """
    total_len = len(df)
    if total_len < 3:
        raise ValueError(f"Dataset has only {total_len} rows; cannot split into 3 subsets.")

    has_groups = group_col in df.columns and df[group_col].nunique() >= 3

    # If small dataset with only 1 positive, group shuffle split could put that 1 positive in val/test,
    # leaving train with 0 positives which breaks training of some classifiers.
    total_positives = int(df[TARGET_COLUMN].sum())

    if has_groups:
        strategy = "Patient-level GroupShuffleSplit"
        # First split into train_val and test
        gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        trainval_idx, test_idx = next(gss_test.split(df, df[TARGET_COLUMN], groups=df[group_col]))

        trainval_df = df.iloc[trainval_idx]
        test_df = df.iloc[test_idx]

        # Calculate relative validation size for the remaining trainval set
        relative_val_size = val_size / (1.0 - test_size)
        relative_val_size = min(max(relative_val_size, 0.05), 0.5)

        gss_val = GroupShuffleSplit(n_splits=1, test_size=relative_val_size, random_state=seed + 1)
        tr_idx, val_idx = next(gss_val.split(trainval_df, trainval_df[TARGET_COLUMN], groups=trainval_df[group_col]))

        train_df = trainval_df.iloc[tr_idx]
        val_df = trainval_df.iloc[val_idx]

        # If small dataset has only 1 positive and it landed in val or test, move it to train
        # so models can at least fit during experimental smoke testing
        if allow_small_dataset and total_positives == 1 and int(train_df[TARGET_COLUMN].sum()) == 0:
            logger.warning(
                "Experimental mode: Single positive example placed into training split to allow model fitting."
            )
            # Find the patient with the positive label
            pos_row = df[df[TARGET_COLUMN] == 1].iloc[0]
            pos_patient = pos_row[group_col]
            # Ensure all records for this patient are in train
            patient_records = df[df[group_col] == pos_patient]
            train_df = pd.concat([train_df, patient_records]).drop_duplicates(subset=["event_id"] if "event_id" in df.columns else None)
            val_df = val_df[val_df[group_col] != pos_patient]
            test_df = test_df[test_df[group_col] != pos_patient]

        # Verify strict patient isolation
        train_patients = set(train_df[group_col].dropna().unique())
        val_patients = set(val_df[group_col].dropna().unique())
        test_patients = set(test_df[group_col].dropna().unique())

        leak_train_val = train_patients & val_patients
        leak_train_test = train_patients & test_patients
        leak_val_test = val_patients & test_patients

        if leak_train_val or leak_train_test or leak_val_test:
            raise RuntimeError(
                f"Patient leakage detected across splits! "
                f"Train/Val overlap: {len(leak_train_val)}, "
                f"Train/Test overlap: {len(leak_train_test)}, "
                f"Val/Test overlap: {len(leak_val_test)}"
            )
        patient_grouping_enforced = True
    else:
        strategy = "Stratified/Simple Random Split (no patient grouping)"
        patient_grouping_enforced = False
        stratify_col = df[TARGET_COLUMN] if df[TARGET_COLUMN].nunique() > 1 else None
        try:
            trainval_df, test_df = train_test_split(
                df, test_size=test_size, random_state=seed, stratify=stratify_col
            )
            relative_val_size = val_size / (1.0 - test_size)
            stratify_val = trainval_df[TARGET_COLUMN] if trainval_df[TARGET_COLUMN].nunique() > 1 else None
            train_df, val_df = train_test_split(
                trainval_df, test_size=relative_val_size, random_state=seed + 1, stratify=stratify_val
            )
        except ValueError:
            # Fallback when class distribution is too skewed for stratification
            trainval_df, test_df = train_test_split(df, test_size=test_size, random_state=seed)
            relative_val_size = val_size / (1.0 - test_size)
            train_df, val_df = train_test_split(trainval_df, test_size=relative_val_size, random_state=seed + 1)

    return DatasetSplit(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        strategy=strategy,
        patient_grouping_enforced=patient_grouping_enforced,
        train_size=len(train_df),
        val_size=len(val_df),
        test_size=len(test_df),
        train_positives=int(train_df[TARGET_COLUMN].sum()),
        val_positives=int(val_df[TARGET_COLUMN].sum()),
        test_positives=int(test_df[TARGET_COLUMN].sum()),
        train_negatives=int((train_df[TARGET_COLUMN] == 0).sum()),
        val_negatives=int((val_df[TARGET_COLUMN] == 0).sum()),
        test_negatives=int((test_df[TARGET_COLUMN] == 0).sum()),
    )


def build_pipeline(model_name: str, numeric_features: list[str], seed: int = 42) -> Pipeline:
    """Build a complete scikit-learn training pipeline with leakage-safe preprocessors.

    Args:
        model_name: One of 'logistic_regression', 'random_forest', 'hist_gradient_boosting'.
        numeric_features: Ordered list of numeric feature column names.
        seed: Random seed for model training.

    Returns:
        Configured Pipeline.
    """
    if model_name == "logistic_regression":
        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "numeric",
                    Pipeline([
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]),
                    numeric_features,
                )
            ],
            remainder="drop",
        )
        classifier = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=seed,
            solver="lbfgs",
        )
    elif model_name == "random_forest":
        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "numeric",
                    Pipeline([
                        ("imputer", SimpleImputer(strategy="median")),
                    ]),
                    numeric_features,
                )
            ],
            remainder="drop",
        )
        classifier = RandomForestClassifier(
            n_estimators=100,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        )
    elif model_name == "hist_gradient_boosting":
        # Consistent median imputation across pipelines ensures fair comparison
        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "numeric",
                    Pipeline([
                        ("imputer", SimpleImputer(strategy="median")),
                    ]),
                    numeric_features,
                )
            ],
            remainder="drop",
        )
        classifier = HistGradientBoostingClassifier(
            max_iter=100,
            learning_rate=0.08,
            random_state=seed,
            class_weight="balanced",
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])


def save_model_bundle_atomically(
    bundle: dict[str, Any],
    output_path: str | Path,
) -> None:
    """Save model bundle to a temporary file, verify its readability, and replace destination atomically.

    Args:
        bundle: Model bundle dictionary conforming to Phase 6 API contract.
        output_path: Target .joblib path.
    """
    dest = Path(output_path).resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Validate essential contract keys before saving
    required_keys = {"model", "features", "model_name", "threshold"}
    missing = required_keys - set(bundle.keys())
    if missing:
        raise ValueError(f"Cannot save invalid bundle; missing required keys: {missing}")

    # Write to temporary file in the same directory (or system temp)
    temp_dir = dest.parent / ".tmp_models"
    temp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = temp_dir / f"tmp_{os.getpid()}_{dest.name}"

    try:
        joblib.dump(bundle, tmp_path)
        # Verify readability
        loaded = joblib.load(tmp_path)
        if not hasattr(loaded["model"], "predict_proba"):
            raise ValueError("Serialized model lacks predict_proba method.")

        # Atomic replacement
        if dest.exists():
            backup = dest.with_suffix(".joblib.bak")
            try:
                shutil.copy2(dest, backup)
            except Exception:
                pass
        shutil.move(str(tmp_path), str(dest))
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass
        if temp_dir.exists():
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception:
                pass
