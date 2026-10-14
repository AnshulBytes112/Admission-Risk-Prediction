#!/usr/bin/env python3
"""Train, evaluate, and register baseline models for emergency admission risk prediction.

Academic demonstration disclaimer:
All records and labels are SYNTHETIC. Features, model architectures, and metrics
must never be interpreted as clinically validated or suitable for medical decisions.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.evaluation import (
    compute_metrics,
    generate_evaluation_report_json,
    generate_evaluation_report_md,
    generate_model_metadata,
    select_best_model,
)
from ml.features import (
    TARGET_COLUMN,
    DataAdequacyError,
    inspect_and_diagnose_dataset,
    load_feature_dataset,
    select_and_validate_features,
    validate_dataset_adequacy,
)
from ml.training import (
    build_pipeline,
    save_model_bundle_atomically,
    split_dataset,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_risk_model")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 5 ML Training and Evaluation Pipeline for Emergency Admission Risk Prediction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/processed/features.parquet",
        help="Path to processed feature Parquet directory, Parquet file, or CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Output directory where model bundle and evaluation reports are written.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=int(os.getenv("TRAIN_RANDOM_SEED", "42")),
        help="Random seed for splitting and estimators.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.20,
        help="Fraction of data reserved for final held-out testing.",
    )
    parser.add_argument(
        "--validation-size",
        type=float,
        default=0.20,
        help="Fraction of dataset reserved for validation and model selection.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=float(os.getenv("RISK_THRESHOLD", "0.70")),
        help="Decision threshold for positive risk classification.",
    )
    parser.add_argument(
        "--minimum-positive-examples",
        type=int,
        default=5,
        help="Minimum positive examples required before training can proceed.",
    )
    parser.add_argument(
        "--allow-small-dataset",
        action="store_true",
        help="Permit experimental smoke-testing when dataset is below statistical adequacy limits.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()

    if not 0.0 < args.threshold < 1.0:
        logger.error(f"--threshold must be strictly between 0 and 1; got {args.threshold}")
        return 1
    if not 0.0 < args.test_size < 1.0:
        logger.error(f"--test-size must be strictly between 0 and 1; got {args.test_size}")
        return 1
    if not 0.0 < args.validation_size < 1.0:
        logger.error(f"--validation-size must be strictly between 0 and 1; got {args.validation_size}")
        return 1

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    print("================================================================================")
    print("PHASE 5: EMERGENCY ADMISSION RISK PREDICTION -- ML PIPELINE")
    print("Academic Synthetic Data Demonstration (Not for clinical decision support)")
    print("================================================================================")
    print(f"Input source      : {input_path}")
    print(f"Output directory  : {output_dir}")
    print(f"Random seed       : {args.seed}")
    print(f"Decision threshold: {args.threshold}")
    print(f"Allow small data  : {args.allow_small_dataset}")
    print("--------------------------------------------------------------------------------")

    # Step 1: Load and inspect dataset
    try:
        df = load_feature_dataset(input_path)
    except Exception as exc:
        logger.error(f"Failed to load dataset: {exc}")
        return 1

    try:
        diagnostics = inspect_and_diagnose_dataset(df)
    except Exception as exc:
        logger.error(f"Dataset inspection error: {exc}")
        return 1

    print("\n--- DATASET DIAGNOSTICS ---")
    print(f"Total records               : {diagnostics.total_rows}")
    print(f"Total columns               : {diagnostics.total_columns}")
    print(f"Unique patients             : {diagnostics.unique_patients}")
    print(f"Observations/patient (mean) : {diagnostics.observations_per_patient_mean}")
    print(f"Duplicate event IDs         : {diagnostics.duplicate_event_ids}")
    print(f"Target distribution         : {diagnostics.target_class_counts}")
    print(f"Target proportions          : {diagnostics.target_class_proportions}")
    print(f"Usable numeric features     : {len(diagnostics.usable_numeric_features)} features")
    print(f"Dataset fingerprint (SHA256): {diagnostics.dataset_fingerprint_sha256[:16]}...")

    # Step 2: Validate dataset adequacy
    try:
        validate_dataset_adequacy(
            diagnostics,
            minimum_positive_examples=args.minimum_positive_examples,
            allow_small_dataset=args.allow_small_dataset,
        )
    except DataAdequacyError as exc:
        print("\n================================================================================")
        print("ERROR: DATASET STATISTICAL ADEQUACY CHECK FAILED")
        print("================================================================================")
        print(str(exc))
        print("================================================================================")
        print("Existing model artifacts in output directory (if any) have been preserved.")
        return 2

    is_experimental = bool(
        args.allow_small_dataset
        and diagnostics.target_class_counts.get(1, 0) < args.minimum_positive_examples
    )
    if is_experimental:
        print("\n================================================================================")
        print("ATTENTION: EXPERIMENTAL SMOKE TEST MODE ACTIVATED (--allow-small-dataset)")
        print("Dataset does not meet minimum statistical criteria for clinical generalization.")
        print("Artifacts will be labeled as EXPERIMENTAL and MUST NOT be treated as reliable.")
        print("================================================================================")

    # Step 3: Feature selection and leak prevention
    features = select_and_validate_features(df.columns.tolist())
    print(f"\nModel feature allowlist ({len(features)} features):")
    for idx, feat in enumerate(features, 1):
        print(f"  {idx:2d}. {feat}")

    # Step 4: Split dataset
    try:
        split = split_dataset(
            df,
            test_size=args.test_size,
            val_size=args.validation_size,
            seed=args.seed,
            allow_small_dataset=args.allow_small_dataset,
        )
    except Exception as exc:
        logger.error(f"Failed to split dataset: {exc}")
        return 1

    print("\n--- SPLIT COMPOSITION ---")
    print(f"Strategy: {split.strategy} (Patient Grouping Enforced: {split.patient_grouping_enforced})")
    print(f"  Train set : {split.train_size} rows ({split.train_positives} pos, {split.train_negatives} neg)")
    print(f"  Val set   : {split.val_size} rows ({split.val_positives} pos, {split.val_negatives} neg)")
    print(f"  Test set  : {split.test_size} rows ({split.test_positives} pos, {split.test_negatives} neg)")

    # Step 5: Train and compare baseline models
    candidate_names = ["logistic_regression", "random_forest", "hist_gradient_boosting"]
    validation_results: dict[str, dict[str, Any]] = {}
    fitted_models = {}

    X_train = split.train_df[features]
    y_train = split.train_df[TARGET_COLUMN].astype(int)
    X_val = split.val_df[features]
    y_val = split.val_df[TARGET_COLUMN].astype(int)
    X_test = split.test_df[features]
    y_test = split.test_df[TARGET_COLUMN].astype(int)

    print("\n--- TRAINING CANDIDATE MODELS ---")
    for name in candidate_names:
        print(f"Fitting pipeline: {name} ...")
        pipe = build_pipeline(name, numeric_features=features, seed=args.seed)
        pipe.fit(X_train, y_train)
        fitted_models[name] = pipe

        val_scores = pipe.predict_proba(X_val)[:, 1]
        val_metrics = compute_metrics(y_val, val_scores, threshold=args.threshold)
        validation_results[name] = val_metrics

        pr_auc = val_metrics.get("pr_auc_average_precision")
        pr_str = f"{pr_auc:.4f}" if pr_auc is not None else "N/A"
        f1_str = f"{val_metrics.get('f1_score', 0):.4f}"
        print(f"  -> Validation PR-AUC: {pr_str} | F1: {f1_str} | Accuracy: {val_metrics.get('accuracy', 0):.4f}")

    # Step 6: Select best model on validation PR-AUC (with F1 tie-breaker)
    best_name, selection_reason = select_best_model(validation_results)
    best_pipeline = fitted_models[best_name]

    print("\n--- MODEL SELECTION ---")
    print(f"Selected model   : {best_name}")
    print(f"Selection reason : {selection_reason}")

    # Step 7: Freeze model and evaluate once on held-out test set
    test_scores = best_pipeline.predict_proba(X_test)[:, 1]
    test_metrics = compute_metrics(y_test, test_scores, threshold=args.threshold)

    print("\n--- HELD-OUT TEST EVALUATION ---")
    print(f"Accuracy         : {test_metrics.get('accuracy')}")
    print(f"Precision (th={args.threshold}): {test_metrics.get('precision')}")
    print(f"Recall (th={args.threshold})   : {test_metrics.get('recall')}")
    print(f"F1-Score         : {test_metrics.get('f1_score')}")
    print(f"ROC-AUC          : {test_metrics.get('roc_auc')} ({test_metrics.get('roc_auc_note', 'evaluated')})")
    print(f"PR-AUC           : {test_metrics.get('pr_auc_average_precision')} ({test_metrics.get('pr_auc_note', 'evaluated')})")
    print(f"Confusion Matrix : {test_metrics.get('confusion_matrix_labels_0_1')}")

    # Step 8: Assemble artifacts
    warnings_list = [
        "All patient observations and target outcomes are SYNTHETIC. Not for clinical use.",
        f"Configured risk threshold of {args.threshold} is an academic demo setting, not an optimized clinical cutoff.",
        "Model features represent snapshot observations and should be strictly temporal in a real clinical deployment.",
    ]
    if is_experimental:
        warnings_list.append(
            "EXPERIMENTAL SMOKE TEST: Dataset size was below minimum threshold for reliable generalization. "
            "Metrics reflect an undersized smoke-test and must not be used to claim model accuracy."
        )

    bundle = {
        "model": best_pipeline,
        "model_name": best_name,
        "features": features,
        "threshold": args.threshold,
        "target": TARGET_COLUMN,
        "dataset_type": "synthetic",
        "is_experimental": is_experimental,
        "not_for_clinical_use": True,
    }

    report_dict = generate_evaluation_report_json(
        project_name="Emergency Admission Risk Prediction",
        phase="Phase 5 - Machine Learning Training & Evaluation",
        data_origin=str(df["data_origin"].iloc[0]) if "data_origin" in df.columns else "synthetic",
        input_path=str(input_path),
        fingerprint=diagnostics.dataset_fingerprint_sha256,
        total_rows=diagnostics.total_rows,
        total_columns=diagnostics.total_columns,
        target_distribution=diagnostics.target_class_counts,
        features=features,
        split_summary=split.summary(),
        seed=args.seed,
        validation_metrics=validation_results,
        selected_model=best_name,
        selection_reason=selection_reason,
        threshold=args.threshold,
        test_metrics=test_metrics,
        is_experimental=is_experimental,
        warnings_list=warnings_list,
    )

    report_md = generate_evaluation_report_md(report_dict)

    metadata_dict = generate_model_metadata(
        artifact_version="1.0.0",
        selected_model_name=best_name,
        model_object=best_pipeline,
        features=features,
        threshold=args.threshold,
        fingerprint=diagnostics.dataset_fingerprint_sha256,
        is_experimental=is_experimental,
    )

    # Step 9: Save artifacts atomically
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "risk_model.joblib"
    report_json_path = output_dir / "evaluation_report.json"
    report_md_path = output_dir / "evaluation_report.md"
    metadata_json_path = output_dir / "model_metadata.json"

    print("\n--- SAVING ARTIFACTS ---")
    save_model_bundle_atomically(bundle, model_path)
    print(f"Saved model bundle       : {model_path}")

    report_json_path.write_text(json.dumps(report_dict, indent=2), encoding="utf-8")
    print(f"Saved evaluation report   : {report_json_path}")

    report_md_path.write_text(report_md, encoding="utf-8")
    print(f"Saved markdown report    : {report_md_path}")

    metadata_json_path.write_text(json.dumps(metadata_dict, indent=2), encoding="utf-8")
    print(f"Saved model metadata     : {metadata_json_path}")

    print("\n================================================================================")
    print("PHASE 5 EXECUTION SUCCESSFUL")
    print(f"Model: {best_name} | Experimental: {is_experimental}")
    print("Disclaimer: Synthetic data demo only; not for clinical triage or patient decisions.")
    print("================================================================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
