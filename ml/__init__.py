"""Phase 5 Machine Learning Training and Evaluation Package.

Academic demonstration disclaimer:
All records and labels are SYNTHETIC. Features, model architectures, and metrics
must never be interpreted as clinically validated or suitable for medical decisions.
"""
from ml.evaluation import (
    compute_metrics,
    generate_evaluation_report_json,
    generate_evaluation_report_md,
    generate_model_metadata,
    select_best_model,
)
from ml.features import (
    EXCLUDED_COLUMNS,
    FEATURE_ALLOWLIST,
    GROUP_COLUMN,
    TARGET_COLUMN,
    DataAdequacyError,
    DatasetDiagnostics,
    compute_dataset_fingerprint,
    inspect_and_diagnose_dataset,
    load_feature_dataset,
    select_and_validate_features,
    validate_dataset_adequacy,
)
from ml.training import (
    DatasetSplit,
    build_pipeline,
    save_model_bundle_atomically,
    split_dataset,
)

__all__ = [
    "EXCLUDED_COLUMNS",
    "FEATURE_ALLOWLIST",
    "GROUP_COLUMN",
    "TARGET_COLUMN",
    "DataAdequacyError",
    "DatasetDiagnostics",
    "DatasetSplit",
    "build_pipeline",
    "compute_dataset_fingerprint",
    "compute_metrics",
    "generate_evaluation_report_json",
    "generate_evaluation_report_md",
    "generate_model_metadata",
    "inspect_and_diagnose_dataset",
    "load_feature_dataset",
    "save_model_bundle_atomically",
    "select_and_validate_features",
    "select_best_model",
    "split_dataset",
    "validate_dataset_adequacy",
]
