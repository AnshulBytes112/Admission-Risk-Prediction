#!/usr/bin/env python3
"""Train/evaluate models on the synthetic processed feature dataset.

This is an academic demonstration pipeline. Labels are synthetic, so metrics
must not be interpreted as real-world clinical performance.
"""
import argparse
import json
import os
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, average_precision_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

FEATURES = [
    "age", "heart_rate", "systolic_bp", "diastolic_bp", "spo2",
    "respiratory_rate", "temperature_c", "previous_admissions",
    "chronic_conditions_count", "lactate", "observation_hour",
    "prior_observation_count",
]
TARGET = "admission_within_24h"
GROUP = "patient_id"
RANDOM_SEED = int(os.getenv("TRAIN_RANDOM_SEED", "42"))
RF_ESTIMATORS = int(os.getenv("RF_N_ESTIMATORS", "250"))
RF_MIN_SAMPLES_LEAF = int(os.getenv("RF_MIN_SAMPLES_LEAF", "3"))
LOGISTIC_MAX_ITER = int(os.getenv("LOGISTIC_MAX_ITER", "2000"))
HGB_MAX_ITER = int(os.getenv("HGB_MAX_ITER", "150"))
HGB_LEARNING_RATE = float(os.getenv("HGB_LEARNING_RATE", "0.08"))

def load_frame(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.is_dir():
        try:
            import pyarrow.parquet as pq  # noqa: F401
            return pd.read_parquet(p)
        except Exception as exc:
            raise RuntimeError(
                f"Could not read Parquet path {p}. Install pyarrow and ensure Spark output was exported locally, "
                f"or run this script with a local Parquet dataset. Details: {exc}"
            ) from exc
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    if p.suffix.lower() in (".parquet", ".pq"):
        return pd.read_parquet(p)
    raise ValueError("Input must be a local CSV, Parquet file, or directory containing Parquet files.")

def make_split(df):
    # Keep patient IDs separated between splits when possible to reduce identity leakage.
    if GROUP in df.columns and df[GROUP].nunique() >= 3:
        gss1 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_SEED)
        trainval_idx, test_idx = next(gss1.split(df, df[TARGET], groups=df[GROUP]))
        trainval = df.iloc[trainval_idx]
        test = df.iloc[test_idx]
        gss2 = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=RANDOM_SEED + 1)
        tr_idx, val_idx = next(gss2.split(trainval, trainval[TARGET], groups=trainval[GROUP]))
        train = trainval.iloc[tr_idx]
        val = trainval.iloc[val_idx]
    else:
        trainval, test = train_test_split(df, test_size=0.2, random_state=RANDOM_SEED, stratify=df[TARGET])
        train, val = train_test_split(trainval, test_size=0.25, random_state=RANDOM_SEED + 1, stratify=trainval[TARGET])
    return train, val, test

def build_pipeline(model_name, numeric_features):
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    prep = ColumnTransformer([("numeric", numeric, numeric_features)], remainder="drop")
    if model_name == "logistic_regression":
        model = LogisticRegression(max_iter=LOGISTIC_MAX_ITER, class_weight="balanced", random_state=RANDOM_SEED)
    elif model_name == "random_forest":
        model = RandomForestClassifier(
            n_estimators=RF_ESTIMATORS, min_samples_leaf=RF_MIN_SAMPLES_LEAF, class_weight="balanced_subsample",
            n_jobs=-1, random_state=RANDOM_SEED
        )
    elif model_name == "hist_gradient_boosting":
        # HGB accepts NaNs natively, but imputation is still used consistently for comparability.
        model = HistGradientBoostingClassifier(max_iter=HGB_MAX_ITER, learning_rate=HGB_LEARNING_RATE, random_state=RANDOM_SEED)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return Pipeline([("preprocess", prep), ("model", model)])

def safe_auc(fn, y, scores):
    try:
        return float(fn(y, scores)) if len(np.unique(y)) == 2 else None
    except Exception:
        return None

def metrics_for(y, scores, threshold):
    pred = (scores >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": safe_auc(roc_auc_score, y, scores),
        "pr_auc_average_precision": safe_auc(average_precision_score, y, scores),
        "confusion_matrix_labels_0_1": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
        "classification_report": classification_report(y, pred, labels=[0, 1], output_dict=True, zero_division=0),
    }

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="data/processed/features.parquet",
                    help="Local Parquet/CSV input. Spark HDFS output must be copied/exported locally first.")
    ap.add_argument("--output-dir", default="models")
    ap.add_argument("--threshold", type=float, default=float(os.getenv("RISK_THRESHOLD", "0.7")))
    args = ap.parse_args()

    if not 0 < args.threshold < 1:
        ap.error("--threshold must be between 0 and 1")
    df = load_frame(args.input)
    required = set(FEATURES + [TARGET])
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")
    df = df.copy()
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df[df[TARGET].isin([0, 1])].copy()
    df[TARGET] = df[TARGET].astype(int)
    if len(df) < 100 or df[TARGET].nunique() != 2:
        raise ValueError("Need at least 100 labelled rows and both target classes for a useful demo.")
    df = df.replace([np.inf, -np.inf], np.nan)
    feature_cols = [c for c in FEATURES if c in df.columns]
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=[TARGET])
    train, val, test = make_split(df)
    X_train, y_train = train[feature_cols], train[TARGET]
    X_val, y_val = val[feature_cols], val[TARGET]
    X_test, y_test = test[feature_cols], test[TARGET]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates = ["logistic_regression", "random_forest", "hist_gradient_boosting"]
    results = {}
    fitted = {}

    for name in candidates:
        pipe = build_pipeline(name, feature_cols)
        pipe.fit(X_train, y_train)
        val_scores = pipe.predict_proba(X_val)[:, 1]
        # Compare models using validation PR-AUC; fall back to F1 if only one class is present.
        ap_score = safe_auc(average_precision_score, y_val, val_scores)
        val_report = metrics_for(y_val, val_scores, args.threshold)
        results[name] = {"validation": val_report, "validation_selection_score_pr_auc": ap_score}
        fitted[name] = pipe
        print(f"{name}: validation PR-AUC={ap_score}")

    def selection_key(item):
        name, result = item
        ap_score = result["validation_selection_score_pr_auc"]
        return (-1 if ap_score is None else ap_score, result["validation"]["f1"])
    best_name = max(results.items(), key=selection_key)[0]
    best_model = fitted[best_name]
    test_scores = best_model.predict_proba(X_test)[:, 1]
    test_metrics = metrics_for(y_test, test_scores, args.threshold)

    bundle = {
        "model": best_model,
        "model_name": best_name,
        "features": feature_cols,
        "threshold": args.threshold,
        "target": TARGET,
        "dataset_type": "synthetic",
        "not_for_clinical_use": True,
    }
    joblib.dump(bundle, out_dir / "risk_model.joblib")
    report = {
        "project": "Emergency Admission Risk Prediction",
        "dataset_type": "synthetic",
        "row_count": int(len(df)),
        "positive_rate": float(df[TARGET].mean()),
        "split_sizes": {"train": len(train), "validation": len(val), "test": len(test)},
        "patient_group_split": GROUP in df.columns,
        "selected_model": best_name,
        "risk_threshold": args.threshold,
        "validation_results": results,
        "test_metrics": test_metrics,
        "limitations": [
            "All labels and records are synthetic; these metrics are not clinical evidence.",
            "The 0.7 threshold is a project demo setting, not a clinically validated cutoff.",
            "This model must not be used to make medical decisions.",
        ],
    }
    (out_dir / "evaluation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\n=== Selected model ===")
    print(best_name)
    print("\n=== Held-out test metrics ===")
    print(json.dumps({k: v for k, v in test_metrics.items() if k != "classification_report"}, indent=2))
    print(f"\nSaved model: {out_dir / 'risk_model.joblib'}")
    print(f"Saved report: {out_dir / 'evaluation_report.json'}")
    print("NOTICE: synthetic demonstration only; not for clinical use.")

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
