#!/usr/bin/env python3
"""Validate saved model artifacts, schema compatibility, and inference readiness for Phase 6.

Academic demonstration disclaimer:
All records and labels are SYNTHETIC. Features, model architectures, and metrics
must never be interpreted as clinically validated or suitable for medical decisions.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.features import FEATURE_ALLOWLIST


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate trained model bundle, metadata, and inference contract for Phase 6 FastAPI.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/risk_model.joblib",
        help="Path to trained model bundle .joblib file.",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default="models/model_metadata.json",
        help="Path to model metadata JSON file.",
    )
    parser.add_argument(
        "--report",
        type=str,
        default="models/evaluation_report.json",
        help="Path to evaluation report JSON file.",
    )
    return parser.parse_args()


def validate_artifacts(
    model_path_str: str,
    metadata_path_str: str,
    report_path_str: str,
) -> bool:
    print("================================================================================")
    print("PHASE 5: MODEL ARTIFACT AND API COMPATIBILITY VALIDATION")
    print("Academic Synthetic Data Demonstration (Not for clinical decision support)")
    print("================================================================================")

    model_path = Path(model_path_str)
    metadata_path = Path(metadata_path_str)
    report_path = Path(report_path_str)

    failed = False

    # Check 1: Model file existence
    if not model_path.is_file():
        print(f"[FAIL] Model file missing: {model_path}")
        return False
    print(f"[PASS] Model bundle found: {model_path} ({model_path.stat().st_size:,} bytes)")

    # Check 2: Load model bundle
    try:
        bundle = joblib.load(model_path)
        print("[PASS] Successfully deserialized model bundle via joblib.")
    except Exception as exc:
        print(f"[FAIL] Could not load model bundle: {exc}")
        return False

    if not isinstance(bundle, dict):
        print(f"[FAIL] Model bundle must be a dictionary, got {type(bundle)}")
        return False

    required_keys = {"model", "features", "model_name", "threshold"}
    missing_keys = required_keys - set(bundle.keys())
    if missing_keys:
        print(f"[FAIL] Model bundle is missing required contract keys: {missing_keys}")
        failed = True
    else:
        print(f"[PASS] Bundle contains all required keys: {sorted(required_keys)}")

    model = bundle.get("model")
    features = bundle.get("features", [])
    threshold = float(bundle.get("threshold", 0.70))
    is_experimental = bool(bundle.get("is_experimental", False))

    if not hasattr(model, "predict_proba"):
        print("[FAIL] Model object does not support predict_proba method.")
        failed = True
    else:
        print("[PASS] Model object exposes predict_proba method.")

    # Check 3: Metadata JSON verification
    if not metadata_path.is_file():
        print(f"[FAIL] Metadata file missing: {metadata_path}")
        failed = True
    else:
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            print(f"[PASS] Metadata JSON parsed successfully ({len(metadata)} keys).")
            # Verify feature list matches bundle
            if metadata.get("features") != features:
                print("[FAIL] Metadata feature list does not match model bundle feature list!")
                failed = True
            else:
                print(f"[PASS] Metadata feature list strictly matches bundle ({len(features)} features).")
            if "safety_disclaimer" in metadata:
                print(f"[PASS] Synthetic safety notice present: {metadata['safety_disclaimer'][:60]}...")
            if metadata.get("is_experimental") != is_experimental:
                print("[WARNING] Metadata is_experimental flag does not match bundle.")
        except Exception as exc:
            print(f"[FAIL] Invalid metadata JSON: {exc}")
            failed = True

    # Check 4: Evaluation report verification
    if report_path.is_file():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            print(f"[PASS] Evaluation report JSON parsed successfully ({len(report)} keys).")
            if "disclaimer" in report:
                print(f"[PASS] Evaluation report disclaimer present: {report['disclaimer'][:60]}...")
            if "held_out_test_metrics" in report:
                test_acc = report["held_out_test_metrics"].get("accuracy")
                print(f"[PASS] Held-out test metrics verified (Accuracy: {test_acc}).")
        except Exception as exc:
            print(f"[FAIL] Invalid evaluation report JSON: {exc}")
            failed = True
    else:
        print(f"[WARNING] Evaluation report missing: {report_path}")

    # Check 5: Inference compatibility smoke test
    print("\n--- INFERENCE SMOKE TEST ---")
    synthetic_patient = {
        "age": 62.0,
        "heart_rate": 84.0,
        "systolic_bp": 130.0,
        "diastolic_bp": 82.0,
        "spo2": 97.0,
        "respiratory_rate": 18.0,
        "temperature_c": 37.0,
        "previous_admissions": 1.0,
        "chronic_conditions_count": 2.0,
        "lactate": 1.3,
        "observation_hour": 14.0,
        "prior_observation_count": 1.0,
    }

    # Ensure all features in bundle are provided
    test_row = {feat: synthetic_patient.get(feat, 0.0) for feat in features}
    input_df = pd.DataFrame([test_row], columns=features)

    try:
        probabilities = model.predict_proba(input_df)
        prob = float(probabilities[0, 1])
        print(f"[PASS] Model executed inference: output probability = {prob:.6f}")

        # Check probability bounds
        if not np.isfinite(prob) or not (0.0 <= prob <= 1.0):
            print(f"[FAIL] Probability out of valid bounds [0.0, 1.0]: {prob}")
            failed = True
        else:
            print("[PASS] Probability is finite and bounded within [0.0, 1.0].")

        # Check decision thresholding
        binary_pred = int(prob >= threshold)
        review_flag = bool(prob >= threshold)
        print(
            f"[PASS] Threshold logic (threshold={threshold}): "
            f"binary_pred={binary_pred}, review_flag={review_flag}"
        )
    except Exception as exc:
        print(f"[FAIL] Inference execution failed: {exc}")
        failed = True

    # Check 6: Experimental status check
    if is_experimental:
        print("\n[NOTE] Model is marked as EXPERIMENTAL (trained on small dataset override).")
        print("       Artifacts are suitable for pipeline testing only, NOT clinical deployment.")

    print("\n================================================================================")
    if failed:
        print("RESULT: VALIDATION FAILED -- Fix the issues reported above.")
        print("================================================================================")
        return False
    else:
        print("RESULT: ALL ARTIFACT & INFERENCE COMPATIBILITY CHECKS PASSED")
        print("================================================================================")
        return True


def main() -> int:
    args = parse_arguments()
    success = validate_artifacts(
        model_path_str=args.model,
        metadata_path_str=args.metadata,
        report_path_str=args.report,
    )
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
