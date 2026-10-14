#!/usr/bin/env python3
"""Phase 7: Systematic Model Sensitivity and Plausibility Analysis.

Performs controlled single-parameter perturbations across all 12 model features,
holding all other features fixed at a clinically typical baseline.
Outputs diagnostic tables and saves reports to models/sensitivity_report.json and .md.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import joblib
import numpy as np
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.features import FEATURE_ALLOWLIST

BASELINE_RECORD = {
    "age": 55.0,
    "heart_rate": 78.0,
    "systolic_bp": 122.0,
    "diastolic_bp": 78.0,
    "spo2": 98.0,
    "respiratory_rate": 16.0,
    "temperature_c": 37.0,
    "previous_admissions": 1.0,
    "chronic_conditions_count": 1.0,
    "lactate": 1.4,
    "observation_hour": 14.0,
    "prior_observation_count": 2.0,
}

FEATURE_RANGES = {
    "age": [18.0, 35.0, 55.0, 70.0, 85.0, 100.0],
    "heart_rate": [45.0, 60.0, 78.0, 100.0, 130.0, 160.0, 190.0],
    "systolic_bp": [70.0, 95.0, 122.0, 150.0, 180.0, 220.0],
    "diastolic_bp": [45.0, 60.0, 78.0, 95.0, 115.0, 130.0],
    "spo2": [75.0, 82.0, 88.0, 93.0, 98.0, 100.0],
    "respiratory_rate": [8.0, 12.0, 16.0, 24.0, 32.0, 42.0],
    "temperature_c": [35.0, 36.2, 37.0, 38.2, 39.5, 41.0],
    "previous_admissions": [0.0, 1.0, 2.0, 4.0, 8.0, 12.0],
    "chronic_conditions_count": [0.0, 1.0, 2.0, 4.0, 6.0, 8.0],
    "lactate": [0.6, 1.0, 1.4, 2.5, 5.0, 8.5, 12.0],
    "observation_hour": [0.0, 4.0, 8.0, 14.0, 18.0, 23.0],
    "prior_observation_count": [0.0, 2.0, 5.0, 10.0, 20.0, 35.0],
}


def run_sensitivity_analysis(model_path: Path = Path("models/risk_model.joblib")) -> dict[str, Any]:
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")

    bundle = joblib.load(model_path)
    model = bundle["model"]
    features = bundle.get("features", FEATURE_ALLOWLIST)
    threshold = float(bundle.get("threshold", 0.70))
    model_name = bundle.get("model_name", "unknown")

    # Baseline prediction
    df_base = pd.DataFrame([BASELINE_RECORD], columns=features)
    base_prob = float(model.predict_proba(df_base)[0, 1])
    base_pred = int(base_prob >= threshold)

    print("================================================================================")
    print("PHASE 7: MODEL SENSITIVITY AND PLAUSIBILITY AUDIT")
    print(f"Model: {model_name} | Decision Threshold: {threshold}")
    print(f"Baseline Admission Risk Probability: {base_prob:.4f} ({base_prob * 100:.2f}%) | Pred: {base_pred}")
    print("================================================================================\n")

    results_by_feature = {}

    for feat_name, sweep_vals in FEATURE_RANGES.items():
        sweep_rows = []
        for val in sweep_vals:
            rec = dict(BASELINE_RECORD)
            rec[feat_name] = float(val)
            df_sweep = pd.DataFrame([rec], columns=features)
            prob = float(model.predict_proba(df_sweep)[0, 1])
            pred = int(prob >= threshold)
            delta = prob - base_prob

            sweep_rows.append({
                "value": float(val),
                "risk_probability": round(prob, 6),
                "risk_percent": round(prob * 100, 2),
                "prediction": pred,
                "delta_from_baseline": round(delta, 6),
            })

        min_p = min(r["risk_probability"] for r in sweep_rows)
        max_p = max(r["risk_probability"] for r in sweep_rows)
        p_range = max_p - min_p
        is_flat = p_range < 0.0001
        is_saturated = any(r["risk_probability"] >= 0.999 or r["risk_probability"] <= 0.001 for r in sweep_rows)

        results_by_feature[feat_name] = {
            "tested_values": sweep_vals,
            "min_probability": round(min_p, 6),
            "max_probability": round(max_p, 6),
            "probability_range": round(p_range, 6),
            "is_flat": is_flat,
            "is_saturated": is_saturated,
            "sweep_results": sweep_rows,
        }

        print(f"Feature: {feat_name:24s} | Min Prob: {min_p:.4f} | Max Prob: {max_p:.4f} | Range: {p_range:.4f}")
        for r in sweep_rows:
            sign = "+" if r["delta_from_baseline"] >= 0 else ""
            print(f"   val={r['value']:7.1f} -> p={r['risk_probability']:.4f} ({r['risk_percent']:5.1f}%) | pred={r['prediction']} | d={sign}{r['delta_from_baseline']:.4f}")
        print()

    report = {
        "model_name": model_name,
        "threshold": threshold,
        "baseline_record": BASELINE_RECORD,
        "baseline_probability": round(base_prob, 6),
        "baseline_prediction": base_pred,
        "features_audited": len(results_by_feature),
        "sensitivity_by_feature": results_by_feature,
        "notes": [
            "All values generated via controlled ceteris paribus perturbations.",
            "Probabilities reflect the actual loaded model pipeline predict_proba output.",
            "Synthetic academic demo model only; not for clinical triage or patient management.",
        ],
    }

    # Save to models/
    out_dir = model_path.parent
    json_path = out_dir / "sensitivity_report.json"
    md_path = out_dir / "sensitivity_report.md"

    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved JSON report: {json_path}")

    # Build Markdown report
    lines = [
        "# Model Sensitivity and Plausibility Audit Report",
        "",
        "> [!NOTE]",
        "> Academic synthetic demonstration model. Probabilities demonstrate software responsiveness and learned weights.",
        "> Not clinically validated for healthcare decision making.",
        "",
        f"- **Model**: `{model_name}`",
        f"- **Decision Threshold**: `{threshold}`",
        f"- **Baseline Risk**: `{base_prob * 100:.2f}%` (Probability: `{base_prob:.4f}`)",
        "",
        "## Feature Perturbation Summary",
        "",
        "| Feature | Tested Range | Min Risk | Max Risk | Risk Spread | Status |",
        "|---|---|---|---|---|---|",
    ]
    for feat, data in results_by_feature.items():
        v_min, v_max = min(data["tested_values"]), max(data["tested_values"])
        status = "Active & Sensitive" if not data["is_flat"] else "Constant/Flat"
        lines.append(
            f"| `{feat}` | {v_min} – {v_max} | {data['min_probability'] * 100:.1f}% | "
            f"{data['max_probability'] * 100:.1f}% | {data['probability_range'] * 100:.1f}% | {status} |"
        )

    lines.append("\n## Detailed Value Sweeps\n")
    for feat, data in results_by_feature.items():
        lines.append(f"### `{feat}`")
        lines.append("| Value | Risk Probability | Percentage | Predicted Class | Delta from Baseline |")
        lines.append("|---|---|---|---|---|")
        for r in data["sweep_results"]:
            lines.append(
                f"| {r['value']} | `{r['risk_probability']:.4f}` | {r['risk_percent']}% | `{r['prediction']}` | {r['delta_from_baseline']:+.4f} |"
            )
        lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved Markdown report: {md_path}")
    return report


if __name__ == "__main__":
    run_sensitivity_analysis()
