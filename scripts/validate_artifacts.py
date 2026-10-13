#!/usr/bin/env python3
"""Check expected local artifacts before the BDA demo; does not train or score a model."""
import argparse
import json
from pathlib import Path
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    root = Path(args.root)
    checks = [
        ("synthetic CSV", root / "data/raw/patient_observations.csv"),
        ("trained model bundle", root / "models/risk_model.joblib"),
        ("model evaluation report", root / "models/evaluation_report.json"),
    ]
    failed = False
    for label, path in checks:
        if path.is_file() and path.stat().st_size > 0:
            print(f"[PASS] {label}: {path} ({path.stat().st_size:,} bytes)")
        else:
            print(f"[MISSING] {label}: {path}")
            failed = True
    report = root / "models/evaluation_report.json"
    if report.is_file():
        try:
            obj = json.loads(report.read_text(encoding="utf-8"))
            print(f"[INFO] Evaluation report keys: {', '.join(sorted(obj.keys()))}")
            print("[INFO] Review split strategy, target prevalence and metrics manually.")
        except Exception as exc:
            print(f"[INVALID] Evaluation report is not valid JSON: {exc}")
            failed = True
    print("Reminder: data and labels in this project are synthetic; this check is not a clinical validation.")
    return 1 if failed else 0

if __name__ == "__main__":
    sys.exit(main())
