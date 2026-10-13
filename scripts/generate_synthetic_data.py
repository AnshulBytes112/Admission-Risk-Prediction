#!/usr/bin/env python3
"""Generate fictional clinical observations for a BDA software demonstration.

The labels are simulated using transparent rules and random variation. This is not
real clinical data and must not be used to make healthcare decisions.
"""
import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def generate(rows: int, seed: int) -> pd.DataFrame:
    if not isinstance(rows, int) or rows <= 0:
        raise ValueError("Row count must be a positive integer.")
    if rows < 10:
        raise ValueError("Row count must be at least 10.")
    rng = np.random.default_rng(seed)
    patient_count = max(5, min(500, int(rows * 0.35))) if rows < 1500 else max(500, int(rows * 0.35))
    patient_ids = np.array([f"SYN{n:06d}" for n in range(1, patient_count + 1)])
    ids = rng.choice(patient_ids, size=rows, replace=True)

    age_by_id = {pid: int(np.clip(rng.normal(59, 19), 18, 100)) for pid in patient_ids}
    ages = np.array([age_by_id[p] for p in ids])
    heart_rate = np.clip(rng.normal(82 + (ages > 70) * 3, 16, rows), 35, 190).round(0)
    systolic_bp = np.clip(rng.normal(124 - (ages > 75) * 5, 19, rows), 60, 220).round(0)
    diastolic_bp = np.clip(rng.normal(77, 12, rows), 30, 140).round(0)
    spo2 = np.clip(rng.normal(97, 2.2, rows), 70, 100).round(0)
    respiratory_rate = np.clip(rng.normal(17, 4.5, rows), 5, 50).round(0)
    temperature = np.clip(rng.normal(36.9, 0.7, rows), 33, 42).round(1)
    previous_admissions = np.clip(rng.poisson(0.8, rows), 0, 12)
    chronic_conditions = np.clip(rng.poisson(1.3 + (ages > 65) * 0.8, rows), 0, 8)
    lactate = np.clip(rng.normal(1.5, 0.7, rows), 0.3, 10).round(1)

    # A synthetic propensity only. It is intentionally not a medical model.
    linear = (
        -3.1
        + 0.018 * (ages - 50)
        + 0.025 * (heart_rate - 80)
        + 0.028 * (95 - systolic_bp)
        + 0.18 * (95 - spo2)
        + 0.07 * (respiratory_rate - 17)
        + 0.45 * (temperature - 37)
        + 0.20 * previous_admissions
        + 0.16 * chronic_conditions
        + 0.28 * (lactate - 1.5)
    )
    probability = sigmoid(linear)
    outcome = rng.binomial(1, probability)

    start = np.datetime64("2025-01-01T00:00:00")
    offsets = rng.integers(0, 365 * 24 * 60, size=rows).astype("timedelta64[m]")
    timestamps = start + offsets

    df = pd.DataFrame({
        "event_id": [f"EVT{n:09d}" for n in range(1, rows + 1)],
        "patient_id": ids,
        "observation_timestamp": pd.to_datetime(timestamps).astype(str),
        "age": ages,
        "heart_rate": heart_rate.astype(int),
        "systolic_bp": systolic_bp.astype(int),
        "diastolic_bp": diastolic_bp.astype(int),
        "spo2": spo2.astype(int),
        "respiratory_rate": respiratory_rate.astype(int),
        "temperature_c": temperature,
        "previous_admissions": previous_admissions.astype(int),
        "chronic_conditions_count": chronic_conditions.astype(int),
        "lactate": lactate,
        "admission_within_24h": outcome.astype(int),
        "synthetic_data": True,
    })

    # Introduce a small amount of missingness to exercise cleaning in the BDA pipeline.
    for col in ["heart_rate", "spo2", "systolic_bp", "temperature_c"]:
        mask = rng.random(rows) < 0.025
        df.loc[mask, col] = np.nan

    return df


def main():
    default_output = os.getenv(
        "SYNTHETIC_OUTPUT_PATH",
        os.getenv("DATASET_OUTPUT_PATH", "data/raw/patient_observations.csv")
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=int(os.getenv("SYNTHETIC_ROWS", "10000")))
    parser.add_argument("--seed", type=int, default=int(os.getenv("SYNTHETIC_SEED", "42")))
    parser.add_argument("--output", default=default_output)
    args = parser.parse_args()
    if args.rows < 100:
        parser.error("--rows must be at least 100")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df = generate(args.rows, args.seed)
    df.to_csv(output, index=False)
    print(f"Generated rows: {len(df):,}")
    print(f"Output location: {output.resolve()}")
    print(f"Random seed: {args.seed}")
    print(f"Synthetic target prevalence: {df['admission_within_24h'].mean():.3f}")
    print("WARNING: All generated records are fictional. Target is simulated and not a clinical label.")
    print("NOTICE: Academic software demonstration data; not MIMIC-III and not for clinical decisions.")


if __name__ == "__main__":
    main()
