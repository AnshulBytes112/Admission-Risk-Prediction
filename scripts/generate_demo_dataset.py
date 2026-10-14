#!/usr/bin/env python3
"""Generate a rich, reproducible synthetic demo dataset for frontend exploration and batch prediction.

Creates:
1. Full demo dataset with all 12 model features + identifiers, dates, and simulated target labels.
2. Clean batch prediction template containing ONLY the 12 features + patient_id (no target/leakage).

All data is strictly synthetic for academic demonstration and software verification.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import numpy as np
import pandas as pd

# The exact 12 model input features accepted by Phase 5/6 API
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


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Logistic sigmoid function."""
    return 1.0 / (1.0 + np.exp(-x))


def generate_demo_dataset(rows: int = 500, seed: int = 42) -> pd.DataFrame:
    """Generate fictional clinical observation records for UI exploration and model demonstration.

    Args:
        rows: Number of observation records to generate.
        seed: Random seed for deterministic reproducibility.

    Returns:
        DataFrame containing complete feature schema and synthetic target labels.
    """
    if rows < 10:
        raise ValueError("Row count must be at least 10.")

    rng = np.random.default_rng(seed)

    # Simulate realistic emergency department patient IDs
    patient_pool_size = max(5, int(rows * 0.4))
    patient_pool = [f"SYN-DEMO-{i:04d}" for i in range(1, patient_pool_size + 1)]
    patient_ids = rng.choice(patient_pool, size=rows, replace=True)

    # Base patient age distribution
    age_by_pt = {pid: int(np.clip(rng.normal(62, 16), 18, 98)) for pid in patient_pool}
    ages = np.array([age_by_pt[pid] for pid in patient_ids])

    # Vital signs with realistic physiological correlations
    heart_rate = np.clip(rng.normal(84 + (ages > 70) * 4, 18, rows), 45, 195).round(1)
    systolic_bp = np.clip(rng.normal(126 - (ages > 75) * 6, 22, rows), 65, 230).round(1)
    diastolic_bp = np.clip(rng.normal(76, 13, rows), 35, 135).round(1)
    spo2 = np.clip(rng.normal(96.2, 2.8, rows), 72, 100).round(1)
    respiratory_rate = np.clip(rng.normal(17.5, 4.8, rows), 8, 48).round(1)
    temperature = np.clip(rng.normal(37.0, 0.8, rows), 34.5, 41.5).round(1)
    previous_admissions = np.clip(rng.poisson(1.1, rows), 0, 15).astype(int)
    chronic_conditions = np.clip(rng.poisson(1.5 + (ages > 65) * 1.0, rows), 0, 9).astype(int)
    lactate = np.clip(rng.normal(1.6, 0.9, rows), 0.4, 12.0).round(2)

    # Observation timing and temporal features
    observation_hour = rng.integers(0, 24, size=rows)
    prior_obs_count = np.clip(rng.poisson(3.5, rows), 0, 30).astype(int)

    # Simulated admission probability (synthetic propensity score for demonstration only)
    linear_score = (
        -3.2
        + 0.022 * (ages - 50)
        + 0.024 * (heart_rate - 80)
        + 0.026 * (100 - systolic_bp)
        + 0.16 * (95 - spo2)
        + 0.08 * (respiratory_rate - 18)
        + 0.42 * (temperature - 37.0)
        + 0.22 * previous_admissions
        + 0.18 * chronic_conditions
        + 0.32 * (lactate - 1.5)
        + 0.015 * prior_obs_count
    )
    admission_prob = sigmoid(linear_score)
    admission_outcome = rng.binomial(1, admission_prob)

    # Observation timestamps across 2025/2026
    start_date = np.datetime64("2026-01-01T00:00:00")
    minute_offsets = rng.integers(0, 180 * 24 * 60, size=rows).astype("timedelta64[m]")
    timestamps = start_date + minute_offsets
    dates = pd.to_datetime(timestamps).strftime("%Y-%m-%d")

    df = pd.DataFrame({
        "event_id": [f"EVT-DEMO-{i:06d}" for i in range(1, rows + 1)],
        "patient_id": patient_ids,
        "observation_date": dates,
        "observation_hour": observation_hour,
        "age": ages,
        "heart_rate": heart_rate,
        "systolic_bp": systolic_bp,
        "diastolic_bp": diastolic_bp,
        "spo2": spo2,
        "respiratory_rate": respiratory_rate,
        "temperature_c": temperature,
        "previous_admissions": previous_admissions,
        "chronic_conditions_count": chronic_conditions,
        "lactate": lactate,
        "prior_observation_count": prior_obs_count,
        "admission_within_24h": admission_outcome,
        "is_synthetic": True,
        "data_origin": "synthetic",
    })

    return df


def generate_batch_template(demo_df: pd.DataFrame, sample_size: int = 25) -> pd.DataFrame:
    """Generate a clean batch prediction CSV containing only API-accepted predictor features."""
    sample_rows = demo_df.head(sample_size).copy()
    template_cols = ["patient_id"] + FEATURE_ALLOWLIST
    return sample_rows[template_cols]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic demo dataset and batch template.")
    parser.add_argument("--rows", type=int, default=500, help="Number of records to generate (default: 500)")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed (default: 42)")
    parser.add_argument(
        "--output",
        type=str,
        default="data/demo/synthetic_demo_patients.csv",
        help="Path for full exploratory dataset CSV",
    )
    parser.add_argument(
        "--template-output",
        type=str,
        default="data/demo/batch_prediction_template.csv",
        help="Path for batch prediction upload template CSV",
    )
    args = parser.parse_args()

    out_path = Path(args.output)
    tpl_path = Path(args.template_output)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tpl_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.rows} synthetic demonstration patient records (seed={args.seed})...")
    df = generate_demo_dataset(rows=args.rows, seed=args.seed)
    df.to_csv(out_path, index=False)
    print(f" Saved full demo dataset: {out_path} ({len(df)} rows, {len(df.columns)} columns)")

    template_df = generate_batch_template(df, sample_size=min(50, len(df)))
    template_df.to_csv(tpl_path, index=False)
    print(f" Saved batch prediction template: {tpl_path} ({len(template_df)} rows, {len(template_df.columns)} columns)")

    # Print summary metrics
    pos_count = int(df["admission_within_24h"].sum())
    neg_count = len(df) - pos_count
    pos_rate = (pos_count / len(df)) * 100.0
    unique_pts = df["patient_id"].nunique()

    print("\nDataset Summary:")
    print(f"  • Total Rows: {len(df)}")
    print(f"  • Unique Patients: {unique_pts}")
    print(f"  • Positive Admissions: {pos_count} ({pos_rate:.1f}%)")
    print(f"  • Negative (Non-admitted): {neg_count} ({100 - pos_rate:.1f}%)")
    print(f"  • Features: {', '.join(FEATURE_ALLOWLIST)}")
    print(f"  • Provenance: 100% Synthetic (Academic demonstration only)")


if __name__ == "__main__":
    main()
