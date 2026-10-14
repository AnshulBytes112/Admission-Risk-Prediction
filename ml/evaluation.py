"""Model evaluation, metrics computation, model selection, and reporting.

Academic demonstration disclaimer:
All records and labels are SYNTHETIC. Features, model architectures, and metrics
must never be interpreted as clinically validated or suitable for medical decisions.
"""
from __future__ import annotations

import datetime
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ml.features import EXCLUDED_COLUMNS, FEATURE_ALLOWLIST, TARGET_COLUMN


def compute_metrics(
    y_true: pd.Series | np.ndarray,
    y_scores: np.ndarray,
    threshold: float = 0.70,
) -> dict[str, Any]:
    """Compute comprehensive classification metrics at a given decision threshold.

    Handles single-class evaluation splits gracefully without fabricating metrics.

    Args:
        y_true: Ground truth binary targets.
        y_scores: Predicted probabilities for class 1.
        threshold: Decision cutoff threshold for positive classification.

    Returns:
        Dictionary of computed metrics and explanations.
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred = (y_scores >= threshold).astype(int)

    unique_classes = np.unique(y_true_arr)
    has_both_classes = len(unique_classes) == 2
    total_evaluated = int(len(y_true_arr))
    positives_evaluated = int((y_true_arr == 1).sum())
    negatives_evaluated = int((y_true_arr == 0).sum())
    positive_prevalence = float(positives_evaluated / total_evaluated) if total_evaluated > 0 else 0.0

    # ROC-AUC and PR-AUC require both classes to be present in y_true
    if has_both_classes:
        roc_auc = float(roc_auc_score(y_true_arr, y_scores))
        roc_auc_note = None
        pr_auc = float(average_precision_score(y_true_arr, y_scores))
        pr_auc_note = None
        brier = float(brier_score_loss(y_true_arr, y_scores))
    else:
        roc_auc = None
        roc_auc_note = f"Undefined: ground truth contains only class(es) {unique_classes.tolist()}."
        pr_auc = None
        pr_auc_note = f"Undefined: ground truth contains only class(es) {unique_classes.tolist()}."
        brier = float(brier_score_loss(y_true_arr, y_scores)) if total_evaluated > 0 else None

    # Threshold-dependent metrics with zero_division safety
    acc = float(accuracy_score(y_true_arr, y_pred))
    prec = float(precision_score(y_true_arr, y_pred, zero_division=0))
    rec = float(recall_score(y_true_arr, y_pred, zero_division=0))
    f1 = float(f1_score(y_true_arr, y_pred, zero_division=0))
    cm = confusion_matrix(y_true_arr, y_pred, labels=[0, 1]).tolist()
    clf_report = classification_report(y_true_arr, y_pred, labels=[0, 1], output_dict=True, zero_division=0)

    return {
        "decision_threshold": float(threshold),
        "total_evaluated": total_evaluated,
        "positive_examples": positives_evaluated,
        "negative_examples": negatives_evaluated,
        "positive_prevalence": round(positive_prevalence, 4),
        "no_skill_pr_auc_baseline": round(positive_prevalence, 4),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
        "roc_auc_note": roc_auc_note,
        "pr_auc_average_precision": round(pr_auc, 4) if pr_auc is not None else None,
        "pr_auc_note": pr_auc_note,
        "brier_score": round(brier, 4) if brier is not None else None,
        "confusion_matrix_labels_0_1": cm,
        "classification_report": clf_report,
    }


def select_best_model(
    validation_results: dict[str, dict[str, Any]],
) -> tuple[str, str]:
    """Select the best candidate model using validation PR-AUC (with F1 as tie-breaker).

    Args:
        validation_results: Mapping of model_name -> validation metrics dict.

    Returns:
        Tuple of (selected_model_name, selection_reason).
    """
    def score_key(item: tuple[str, dict[str, Any]]):
        _name, res = item
        pr_auc = res.get("pr_auc_average_precision")
        f1 = res.get("f1_score", 0.0)
        # If PR-AUC is None, score as -1.0 so valid PR-AUC wins
        pr_score = -1.0 if pr_auc is None else pr_auc
        return (pr_score, f1)

    sorted_models = sorted(validation_results.items(), key=score_key, reverse=True)
    best_name, best_metrics = sorted_models[0]

    if best_metrics.get("pr_auc_average_precision") is not None:
        reason = f"Highest validation PR-AUC ({best_metrics['pr_auc_average_precision']}) with F1 ({best_metrics['f1_score']})"
    else:
        reason = f"Validation PR-AUC unavailable due to single-class validation set; selected by highest F1 ({best_metrics['f1_score']})"

    return best_name, reason


def generate_evaluation_report_json(
    project_name: str,
    phase: str,
    data_origin: str,
    input_path: str,
    fingerprint: str,
    total_rows: int,
    total_columns: int,
    target_distribution: dict[str, Any],
    features: list[str],
    split_summary: dict[str, Any],
    seed: int,
    validation_metrics: dict[str, dict[str, Any]],
    selected_model: str,
    selection_reason: str,
    threshold: float,
    test_metrics: dict[str, Any],
    is_experimental: bool,
    warnings_list: list[str],
) -> dict[str, Any]:
    """Construct structured JSON evaluation report adhering to repository contract."""
    return {
        "project": project_name,
        "phase": phase,
        "evaluation_timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "data_origin": data_origin,
        "disclaimer": (
            "All records and labels are SYNTHETIC. Academic software demonstration only; "
            "this model is NOT clinically validated and MUST NOT be used for triage or medical decisions."
        ),
        "is_experimental": is_experimental,
        "dataset_metadata": {
            "input_dataset_path": str(input_path),
            "fingerprint_sha256": fingerprint,
            "total_rows": total_rows,
            "total_columns": total_columns,
            "target_distribution": target_distribution,
            "feature_allowlist": features,
            "excluded_columns": EXCLUDED_COLUMNS,
        },
        "split_configuration": {
            "random_seed": seed,
            "split_summary": split_summary,
        },
        "validation_model_comparison": validation_metrics,
        "model_selection": {
            "selected_model": selected_model,
            "selection_criterion": "validation PR-AUC (Average Precision) with F1 tie-breaker",
            "selection_reason": selection_reason,
            "decision_threshold": threshold,
        },
        "held_out_test_metrics": test_metrics,
        "warnings_and_limitations": warnings_list,
        "training_status": "EXPERIMENTAL_SMOKE_TEST" if is_experimental else "COMPLETED_VALIDATED",
    }


def generate_evaluation_report_md(report_dict: dict[str, Any]) -> str:
    """Format human-readable markdown evaluation report with tables and alerts."""
    val_comp = report_dict.get("validation_model_comparison", {})
    test_m = report_dict.get("held_out_test_metrics", {})
    splits = report_dict.get("split_configuration", {}).get("split_summary", {})
    ds_meta = report_dict.get("dataset_metadata", {})

    rows = []
    for model_name, m in val_comp.items():
        pr_auc = m.get("pr_auc_average_precision")
        pr_str = f"{pr_auc:.4f}" if pr_auc is not None else "N/A"
        roc = m.get("roc_auc")
        roc_str = f"{roc:.4f}" if roc is not None else "N/A"
        rows.append(
            f"| `{model_name}` | {m.get('accuracy', 0):.4f} | {m.get('precision', 0):.4f} | "
            f"{m.get('recall', 0):.4f} | {m.get('f1_score', 0):.4f} | {roc_str} | {pr_str} |"
        )
    val_table = "\n".join(rows)

    warnings_md = "\n".join(f"- {w}" for w in report_dict.get("warnings_and_limitations", []))

    md = f"""# Phase 5 — Machine Learning Evaluation Report

> [!CAUTION]
> **SYNTHETIC DATA DISCLAIMER**:
> {report_dict.get('disclaimer')}

## Executive Summary
- **Project**: {report_dict.get('project')}
- **Phase**: {report_dict.get('phase')}
- **Evaluation Timestamp**: `{report_dict.get('evaluation_timestamp_utc')}`
- **Training Status**: `{report_dict.get('training_status')}` (Experimental: `{report_dict.get('is_experimental')}`)
- **Selected Model**: `{report_dict.get('model_selection', {}).get('selected_model')}`
- **Selection Reason**: {report_dict.get('model_selection', {}).get('selection_reason')}
- **Decision Threshold**: `{report_dict.get('model_selection', {}).get('decision_threshold')}`

## Dataset & Splitting
- **Input Path**: `{ds_meta.get('input_dataset_path')}`
- **Fingerprint (SHA-256)**: `{ds_meta.get('fingerprint_sha256')}`
- **Total Records**: `{ds_meta.get('total_rows')}`
- **Splitting Strategy**: `{splits.get('strategy')}` (Patient Grouping Enforced: `{splits.get('patient_grouping_enforced')}`)
- **Train Set**: {splits.get('train', {}).get('total')} records ({splits.get('train', {}).get('positives')} positive)
- **Validation Set**: {splits.get('validation', {}).get('total')} records ({splits.get('validation', {}).get('positives')} positive)
- **Held-Out Test Set**: {splits.get('test', {}).get('total')} records ({splits.get('test', {}).get('positives')} positive)

## Model Comparison (Validation Set)
| Candidate Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
{val_table}

> [!NOTE]
> PR-AUC is the primary selection metric for imbalanced emergency risk prediction.
> No-skill baseline PR-AUC equals positive prevalence on the evaluation split.

## Final Held-Out Test Evaluation
*Evaluated once on the unseen test partition using the frozen selected model:*
- **Accuracy**: `{test_m.get('accuracy')}`
- **Precision (Threshold {test_m.get('decision_threshold')})**: `{test_m.get('precision')}`
- **Recall (Threshold {test_m.get('decision_threshold')})**: `{test_m.get('recall')}`
- **F1-Score**: `{test_m.get('f1_score')}`
- **ROC-AUC**: `{test_m.get('roc_auc')}` ({test_m.get('roc_auc_note', 'evaluated')})
- **PR-AUC**: `{test_m.get('pr_auc_average_precision')}` ({test_m.get('pr_auc_note', 'evaluated')})
- **Confusion Matrix [[TN, FP], [FN, TP]]**: `{test_m.get('confusion_matrix_labels_0_1')}`

## Warnings and Limitations
{warnings_md}
"""
    return md


def generate_model_metadata(
    artifact_version: str,
    selected_model_name: str,
    model_object: Any,
    features: list[str],
    threshold: float,
    fingerprint: str,
    is_experimental: bool,
) -> dict[str, Any]:
    """Generate registry metadata matching Phase 6 API expectations."""
    clf = model_object.named_steps.get("classifier") if hasattr(model_object, "named_steps") else model_object
    model_class = clf.__class__.__name__ if clf else "Unknown"
    params = clf.get_params() if hasattr(clf, "get_params") else {}
    # Convert numpy types in params to standard Python types for JSON serialization
    clean_params = {}
    for k, v in params.items():
        if isinstance(v, (np.integer, np.floating, np.bool_)):
            clean_params[k] = v.item()
        elif isinstance(v, (str, int, float, bool, list, dict, type(None))):
            clean_params[k] = v
        else:
            clean_params[k] = str(v)

    return {
        "artifact_version": artifact_version,
        "model_name": selected_model_name,
        "model_class": model_class,
        "hyperparameters": clean_params,
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "target": TARGET_COLUMN,
        "features": features,
        "feature_count": len(features),
        "threshold": float(threshold),
        "dataset_type": "synthetic",
        "training_dataset_fingerprint_sha256": fingerprint,
        "is_experimental": is_experimental,
        "clinical_use": False,
        "not_for_clinical_use": True,
        "environment": {
            "python_version": sys.version,
            "platform": platform.platform(),
            "scikit_learn_version": sklearn.__version__,
        },
        "evaluation_report_location": "models/evaluation_report.json",
        "safety_disclaimer": (
            "Academic demonstration: all records and labels are SYNTHETIC. "
            "This model is not clinically validated and must not be used for actual patient care."
        ),
    }
