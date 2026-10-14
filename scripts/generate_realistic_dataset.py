#!/usr/bin/env python3
"""Generate a physiologically calibrated, realistic synthetic training dataset.

This generator creates 5,000 patient observations with proper clinical feature
correlations and a realistic ~20% admission rate. Healthy patients (normal vitals,
young age, no prior history) reliably score LOW risk; critical patients (abnormal
vitals, elderly, multiple comorbidities, elevated lactate) score HIGH risk.

Features match the V1 schema used by the FastAPI form:
  age, heart_rate, systolic_bp, diastolic_bp, spo2, respiratory_rate,
  temperature_c, previous_admissions, chronic_conditions_count, lactate,
  observation_hour, prior_observation_count

Output: data/raw/train.csv
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

NORMAL = {
    "heart_rate":       (60, 100),
    "systolic_bp":      (90, 140),
    "diastolic_bp":     (60, 90),
    "spo2":             (95, 100),
    "respiratory_rate": (12, 20),
    "temperature_c":    (36.5, 37.5),
    "lactate":          (0.5, 2.0),
}

ABNORMAL = {
    "heart_rate":       [(40, 59), (101, 150)],
    "systolic_bp":      [(60, 89), (160, 220)],
    "diastolic_bp":     [(30, 59), (91, 130)],
    "spo2":             [(70, 94)],
    "respiratory_rate": [(5, 11), (21, 40)],
    "temperature_c":    [(34.0, 36.4), (37.6, 41.5)],
    "lactate":          [(2.1, 12.0)],
}


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def sample_normal_vital(rng, name, n):
    lo, hi = NORMAL[name]
    center = (lo + hi) / 2
    std = (hi - lo) / 6
    return np.clip(rng.normal(center, std, n), lo, hi)


def sample_abnormal_vital(rng, name, n):
    ranges = ABNORMAL[name]
    idx = rng.integers(0, len(ranges), n)
    result = np.zeros(n)
    for i, (lo, hi) in enumerate(ranges):
        mask = idx == i
        center = (lo + hi) / 2
        std = max((hi - lo) / 6, 0.1)
        result[mask] = np.clip(rng.normal(center, std, mask.sum()), lo, hi)
    return result


def generate_vital(rng, admitted, healthy, name, abnormal_frac_admitted=0.75, abnormal_frac_healthy=0.04):
    n_rows = len(admitted) + len(healthy)
    arr = np.zeros(admitted.sum() + healthy.sum())
    # Admitted
    adm_indices = np.where(admitted)[0]
    rng.shuffle(adm_indices)
    n_abn = int(len(adm_indices) * abnormal_frac_admitted)
    arr[adm_indices[:n_abn]]  = sample_abnormal_vital(rng, name, n_abn)
    arr[adm_indices[n_abn:]]  = sample_normal_vital(rng, name, len(adm_indices) - n_abn)
    # Healthy
    hlt_indices = np.where(healthy)[0]
    rng.shuffle(hlt_indices)
    n_abn_h = int(len(hlt_indices) * abnormal_frac_healthy)
    arr[hlt_indices[:n_abn_h]] = sample_abnormal_vital(rng, name, n_abn_h)
    arr[hlt_indices[n_abn_h:]] = sample_normal_vital(rng, name, len(hlt_indices) - n_abn_h)
    return arr


def generate_dataset(n_rows=5000, seed=42):
    rng = np.random.default_rng(seed)
    target_rate = 0.20
    n_admitted = int(n_rows * target_rate)
    n_healthy = n_rows - n_admitted
    labels = np.concatenate([np.ones(n_admitted, dtype=int), np.zeros(n_healthy, dtype=int)])
    shuffle_idx = rng.permutation(n_rows)
    labels = labels[shuffle_idx]
    admitted = labels == 1
    healthy  = labels == 0

    age = np.zeros(n_rows)
    age[admitted] = np.clip(rng.normal(72, 14, admitted.sum()), 18, 100).round()
    age[healthy]  = np.clip(rng.normal(44, 16, healthy.sum()),  18, 100).round()

    heart_rate       = generate_vital(rng, admitted, healthy, "heart_rate",       0.70, 0.05).round(1)
    systolic_bp      = generate_vital(rng, admitted, healthy, "systolic_bp",      0.65, 0.05).round(1)
    diastolic_bp     = generate_vital(rng, admitted, healthy, "diastolic_bp",     0.55, 0.04).round(1)
    spo2             = generate_vital(rng, admitted, healthy, "spo2",             0.75, 0.03).round(1)
    respiratory_rate = generate_vital(rng, admitted, healthy, "respiratory_rate", 0.65, 0.04).round(1)
    temperature_c    = generate_vital(rng, admitted, healthy, "temperature_c",    0.60, 0.05).round(1)
    lactate          = generate_vital(rng, admitted, healthy, "lactate",          0.80, 0.03).round(2)

    previous_admissions = np.zeros(n_rows, dtype=int)
    previous_admissions[admitted] = np.clip(rng.poisson(2.8, admitted.sum()), 0, 15)
    previous_admissions[healthy]  = np.clip(rng.poisson(0.3, healthy.sum()),  0, 15)

    chronic_conditions_count = np.zeros(n_rows, dtype=int)
    chronic_conditions_count[admitted] = np.clip(rng.poisson(3.2, admitted.sum()), 0, 10)
    chronic_conditions_count[healthy]  = np.clip(rng.poisson(0.5, healthy.sum()),  0, 10)

    observation_hour = rng.integers(0, 24, n_rows)

    prior_observation_count = np.zeros(n_rows, dtype=int)
    prior_observation_count[admitted] = np.clip(rng.poisson(6.0, admitted.sum()), 0, 30)
    prior_observation_count[healthy]  = np.clip(rng.poisson(1.5, healthy.sum()),  0, 30)

    # Verify separation
    linear = (
        -4.0
        + 0.040 * (age - 50)
        + 0.030 * np.abs(heart_rate - 80) / 10
        + 0.025 * np.maximum(0, 90 - systolic_bp) / 5
        + 0.025 * np.maximum(0, systolic_bp - 150) / 5
        + 0.200 * np.maximum(0, 95 - spo2)
        + 0.060 * np.maximum(0, respiratory_rate - 20)
        + 0.060 * np.maximum(0, 12 - respiratory_rate)
        + 0.300 * np.maximum(0, temperature_c - 37.5)
        + 0.500 * np.maximum(0, lactate - 2.0)
        + 0.250 * previous_admissions
        + 0.200 * chronic_conditions_count
        + 0.050 * prior_observation_count
    )
    prob = sigmoid(linear)
    print(f"  Calibration check: admitted mean_p={prob[admitted].mean():.3f}  healthy mean_p={prob[healthy].mean():.3f}")

    n_patients = max(100, int(n_rows * 0.55))
    patient_pool = [f"PT-{i:05d}" for i in range(1, n_patients + 1)]
    patient_ids  = rng.choice(patient_pool, n_rows, replace=True)
    event_ids = [f"EVT-{i:07d}" for i in range(1, n_rows + 1)]
    start = np.datetime64("2025-01-01T00:00:00")
    offsets = rng.integers(0, 365 * 24 * 60, n_rows).astype("timedelta64[m]")
    dates = pd.to_datetime(start + offsets).strftime("%Y-%m-%d")

    df = pd.DataFrame({
        "event_id":                event_ids,
        "patient_id":              patient_ids,
        "observation_date":        dates,
        "observation_hour":        observation_hour.astype(int),
        "age":                     age.astype(int),
        "heart_rate":              heart_rate,
        "systolic_bp":             systolic_bp,
        "diastolic_bp":            diastolic_bp,
        "spo2":                    spo2,
        "respiratory_rate":        respiratory_rate,
        "temperature_c":           temperature_c,
        "previous_admissions":     previous_admissions,
        "chronic_conditions_count": chronic_conditions_count,
        "lactate":                 lactate,
        "prior_observation_count": prior_observation_count.astype(int),
        "admission_within_24h":    labels,
        "is_synthetic":            True,
        "data_origin":             "realistic_synthetic_v2",
    })
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows",   type=int, default=5000)
    parser.add_argument("--seed",   type=int, default=42)
    parser.add_argument("--output", type=str, default="data/raw/train.csv")
    args = parser.parse_args()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"Generating {args.rows} physiologically calibrated patient records (seed={args.seed})...")
    df = generate_dataset(args.rows, args.seed)
    pos  = int(df["admission_within_24h"].sum())
    neg  = len(df) - pos
    print(f"  Admitted (label=1)    : {pos} ({pos/len(df)*100:.1f}%)")
    print(f"  Not admitted (label=0): {neg} ({neg/len(df)*100:.1f}%)")
    print(f"  Unique patients       : {df['patient_id'].nunique()}")
    df.to_csv(out, index=False)
    print(f"\nSaved => {out}")
    print("\n  Mean vitals by class:")
    for col in ["heart_rate", "systolic_bp", "spo2", "lactate", "age", "chronic_conditions_count"]:
        m0 = df.loc[df["admission_within_24h"] == 0, col].mean()
        m1 = df.loc[df["admission_within_24h"] == 1, col].mean()
        print(f"    {col:30s}  healthy={m0:7.2f}  admitted={m1:7.2f}")


if __name__ == "__main__":
    main()
