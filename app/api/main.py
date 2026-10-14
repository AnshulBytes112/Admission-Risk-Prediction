"""FastAPI inference API for the Emergency Admission Risk Prediction project.

Phase 6: Model loading, lifecycle management, input validation, and prediction serving.
Academic software demonstration using synthetic data. Not for clinical decision-support.
"""
from __future__ import annotations

import json
import logging
import math
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

# -----------------------------------------------------------------------------
# Configuration and Logging
# -----------------------------------------------------------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO), format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("admission_api")

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/risk_model.joblib"))
THRESHOLD = float(os.getenv("RISK_THRESHOLD", "0.7"))
CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "API_CORS_ORIGINS",
        "http://localhost:8501,http://127.0.0.1:8501,http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")
    if origin.strip()
]

SYNTHETIC_DISCLAIMER = (
    "Synthetic academic demonstration only. Not a clinically validated prediction; "
    "do not use for medical decisions or triage."
)

FEATURE_RANGES: dict[str, tuple[float, float]] = {
    "age": (18.0, 100.0),
    "heart_rate": (25.0, 240.0),
    "systolic_bp": (40.0, 300.0),
    "diastolic_bp": (20.0, 200.0),
    "spo2": (50.0, 100.0),
    "respiratory_rate": (3.0, 80.0),
    "temperature_c": (30.0, 44.0),
    "previous_admissions": (0.0, 100.0),
    "chronic_conditions_count": (0.0, 50.0),
    "lactate": (0.0, 30.0),
    "observation_hour": (0.0, 23.0),
    "prior_observation_count": (0.0, 100000.0),
}


# -----------------------------------------------------------------------------
# Request and Response Schemas
# -----------------------------------------------------------------------------
class PatientFeatures(BaseModel):
    """Validated patient telemetry and clinical history features."""
    model_config = ConfigDict(extra="forbid")

    patient_id: str = Field(min_length=1, max_length=80, description="Patient identifier")
    age: float = Field(ge=18.0, le=100.0, description="Patient age in years")
    heart_rate: float = Field(ge=25.0, le=240.0, description="Heart rate in beats per minute")
    systolic_bp: float = Field(ge=40.0, le=300.0, description="Systolic blood pressure in mmHg")
    diastolic_bp: float = Field(ge=20.0, le=200.0, description="Diastolic blood pressure in mmHg")
    spo2: float = Field(ge=50.0, le=100.0, description="Blood oxygen saturation percentage")
    respiratory_rate: float = Field(ge=3.0, le=80.0, description="Breaths per minute")
    temperature_c: float = Field(ge=30.0, le=44.0, description="Body temperature in degrees Celsius")
    previous_admissions: float = Field(ge=0.0, le=100.0, description="Count of prior emergency admissions")
    chronic_conditions_count: float = Field(ge=0.0, le=50.0, description="Count of documented chronic conditions")
    lactate: float = Field(ge=0.0, le=30.0, description="Blood serum lactate level in mmol/L")
    observation_hour: float = Field(ge=0.0, le=23.0, description="Hour of arrival observation (0-23)")
    prior_observation_count: float = Field(ge=0.0, le=100000.0, description="Count of strictly preceding observations")

    @field_validator(
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
        mode="after",
    )
    @classmethod
    def validate_finite_number(cls, v: float, info) -> float:
        """Reject non-finite values (NaN, Infinity)."""
        if v is None:
            raise ValueError(f"{info.field_name} must not be null")
        if math.isnan(v):
            raise ValueError(f"{info.field_name} must be a finite number, not NaN")
        if math.isinf(v):
            raise ValueError(f"{info.field_name} must be a finite number, not Infinity")
        return v


# -----------------------------------------------------------------------------
# Model Lifecycle and Manager
# -----------------------------------------------------------------------------
class ModelManager:
    """Manages model loading, verification, caching, and inference readiness."""

    def __init__(self) -> None:
        self.model_bundle: dict[str, Any] | None = None
        self.metadata: dict[str, Any] | None = None
        self.evaluation_report: dict[str, Any] | None = None
        self.loaded_path: Path | None = None
        self.artifact_status: str = "unloaded"
        self.error_reason: str | None = None
        self.pos_class_index: int = 1

    def resolve_model_path(self, target_path: Path | str | None = None) -> Path:
        """Resolve current active model path, honoring environment or monkeypatching."""
        if target_path is not None:
            return Path(target_path)
        # Check current module-level MODEL_PATH to preserve test monkeypatching
        return Path(MODEL_PATH)

    def load_bundle(self, force: bool = False, target_path: Path | str | None = None) -> dict[str, Any]:
        """Load and validate model bundle from disk once, or return cached bundle.

        Security Note: joblib uses pickle serialization internally. Artifacts must
        only be loaded from trusted local workspace directories.
        """
        active_path = self.resolve_model_path(target_path)

        # Return cached bundle if path hasn't changed and not forced
        if not force and self.model_bundle is not None and self.loaded_path == active_path:
            return self.model_bundle

        logger.info(f"Loading model bundle from: {active_path}")

        if not active_path.exists():
            self.model_bundle = None
            self.loaded_path = active_path
            self.artifact_status = "missing"
            self.error_reason = f"Trained model not found at {active_path}. Complete Phase 5 and mount the models directory."
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=self.error_reason)

        try:
            bundle = joblib.load(active_path)
        except Exception as exc:
            self.model_bundle = None
            self.loaded_path = active_path
            self.artifact_status = "corrupt"
            self.error_reason = f"Could not load model bundle: {exc}"
            logger.error(self.error_reason)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=self.error_reason) from exc

        # Bundle contract validation
        if not isinstance(bundle, dict):
            self.model_bundle = None
            self.loaded_path = active_path
            self.artifact_status = "invalid_bundle"
            self.error_reason = f"Model bundle must be a dictionary, got {type(bundle)}"
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=self.error_reason)

        for req_key in ("model", "features"):
            if req_key not in bundle:
                self.model_bundle = None
                self.loaded_path = active_path
                self.artifact_status = "invalid_bundle"
                self.error_reason = f"Model bundle missing required key: {req_key}"
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=self.error_reason)

        features = bundle.get("features", [])
        if not isinstance(features, list) or not features:
            self.model_bundle = None
            self.loaded_path = active_path
            self.artifact_status = "invalid_bundle"
            self.error_reason = "Model bundle features list is empty or invalid."
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=self.error_reason)

        if len(features) != len(set(features)):
            self.model_bundle = None
            self.loaded_path = active_path
            self.artifact_status = "invalid_bundle"
            self.error_reason = "Model bundle contains duplicate feature names."
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=self.error_reason)

        model = bundle.get("model")
        if not hasattr(model, "predict_proba"):
            self.model_bundle = None
            self.loaded_path = active_path
            self.artifact_status = "invalid_bundle"
            self.error_reason = "Loaded model does not support probability predictions (predict_proba)."
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=self.error_reason)

        # Positive-class mapping via model.classes_
        classes = getattr(model, "classes_", None)
        if classes is None and hasattr(model, "named_steps"):
            classifier = model.named_steps.get("classifier") or model.named_steps.get("model")
            if classifier is not None:
                classes = getattr(classifier, "classes_", None)

        if classes is not None:
            matched_indices = [i for i, c in enumerate(classes) if c in (1, 1.0, "1")]
            if matched_indices:
                self.pos_class_index = matched_indices[0]
            else:
                self.model_bundle = None
                self.loaded_path = active_path
                self.artifact_status = "invalid_bundle"
                self.error_reason = "Model classes do not include positive class label 1."
        else:
            self.pos_class_index = 1

        # Defensive scikit-learn cross-version compatibility: ensure SimpleImputer has _fill_dtype
        try:
            if hasattr(model, "named_steps"):
                prep = model.named_steps.get("preprocessor") or model.named_steps.get("preprocess")
                if prep and hasattr(prep, "transformers_"):
                    for _trans_name, trans_pipe, _cols in prep.transformers_:
                        if hasattr(trans_pipe, "named_steps"):
                            imputer = trans_pipe.named_steps.get("imputer")
                            if imputer is not None and not hasattr(imputer, "_fill_dtype") and hasattr(imputer, "statistics_"):
                                imputer._fill_dtype = imputer.statistics_.dtype
        except Exception:
            pass

        # Check threshold validity
        bundle_thresh = bundle.get("threshold", THRESHOLD)
        try:
            bundle_thresh_val = float(bundle_thresh)
            if not 0.0 <= bundle_thresh_val <= 1.0:
                raise ValueError("Threshold must be between 0 and 1.")
        except Exception as exc:
            self.model_bundle = None
            self.loaded_path = active_path
            self.artifact_status = "invalid_bundle"
            self.error_reason = f"Invalid decision threshold in bundle: {exc}"
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=self.error_reason) from exc

        # Read sibling metadata and evaluation reports if present
        meta_file = active_path.parent / "model_metadata.json"
        if meta_file.is_file():
            try:
                self.metadata = json.loads(meta_file.read_text(encoding="utf-8"))
            except Exception:
                self.metadata = None
        else:
            self.metadata = None

        report_file = active_path.parent / "evaluation_report.json"
        if report_file.is_file():
            try:
                self.evaluation_report = json.loads(report_file.read_text(encoding="utf-8"))
            except Exception:
                self.evaluation_report = None
        else:
            self.evaluation_report = None

        self.model_bundle = bundle
        self.loaded_path = active_path
        self.artifact_status = "loaded"
        self.error_reason = None
        logger.info(
            f"Model bundle verified successfully: name={bundle.get('model_name')}, "
            f"features={len(features)}, threshold={bundle_thresh_val}, experimental={bundle.get('is_experimental', False)}"
        )
        return self.model_bundle

    def is_ready(self) -> bool:
        """Check if model bundle is currently loaded and ready for inference."""
        active_path = self.resolve_model_path()
        if self.model_bundle is None or self.loaded_path != active_path:
            try:
                self.load_bundle()
            except Exception:
                return False
        return self.artifact_status == "loaded" and self.model_bundle is not None


model_manager = ModelManager()


def load_bundle() -> dict[str, Any]:
    """Compatibility wrapper delegating to ModelManager."""
    return model_manager.load_bundle()


# -----------------------------------------------------------------------------
# FastAPI Lifespan and Application Setup
# -----------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI application lifecycle: loads model on startup."""
    logger.info("Initializing Emergency Admission Risk Prediction API...")
    try:
        model_manager.load_bundle()
    except Exception as exc:
        logger.warning(f"Model startup load notification: {exc}")
    yield
    logger.info("Shutting down Emergency Admission Risk Prediction API.")


app = FastAPI(
    title="Emergency Admission Risk Prediction API",
    description="Academic demo using a model trained on synthetic data. Not for clinical use.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


# -----------------------------------------------------------------------------
# API Endpoints
# -----------------------------------------------------------------------------
@app.get("/")
def root():
    """Project foundation and starter status."""
    is_ready = model_manager.is_ready()
    return {
        "project": "Emergency Admission Risk Prediction",
        "phase": 1,
        "phase_name": "Project Foundation and Infrastructure",
        "status": "ready",
        "data_type": "synthetic",
        "model_readiness": "ready" if is_ready else "unloaded (Phase 1 foundation)",
        "notice": "Academic software demonstration using synthetic data. Not for clinical decision-support.",
    }


@app.get("/health")
def health():
    """Process liveness probe: verifies the service is alive regardless of model readiness."""
    active_path = model_manager.resolve_model_path()
    is_ready = model_manager.is_ready()
    return {
        "status": "ok",
        "service": "emergency-admission-api",
        "application_health": "healthy",
        "model_readiness": "loaded" if is_ready else "not_loaded",
        "model_file_exists": active_path.exists(),
        "disclaimer": "Academic demo; not for clinical use.",
    }


@app.get("/ready")
def ready():
    """Inference readiness probe: verifies model artifact is loaded and operational."""
    active_path = model_manager.resolve_model_path()
    is_ready = model_manager.is_ready()

    if is_ready and model_manager.model_bundle is not None:
        bundle = model_manager.model_bundle
        features = bundle.get("features", [])
        return {
            "status": "ready",
            "model_loaded": True,
            "model_name": bundle.get("model_name", "unknown"),
            "artifact_status": model_manager.artifact_status,
            "is_experimental": bool(bundle.get("is_experimental", False)),
            "data_origin": bundle.get("dataset_type", "synthetic"),
            "model_threshold": float(bundle.get("threshold", THRESHOLD)),
            "feature_count": len(features),
            "error_reason": None,
        }

    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "status": "not_ready",
            "model_loaded": False,
            "model_name": None,
            "artifact_status": model_manager.artifact_status,
            "is_experimental": None,
            "data_origin": "synthetic",
            "model_threshold": THRESHOLD,
            "feature_count": 0,
            "error_reason": model_manager.error_reason or f"Model not loaded from {active_path}.",
        },
    )


@app.get("/metadata")
def metadata():
    """Model registry metadata and configuration notice."""
    active_path = model_manager.resolve_model_path()
    if not model_manager.is_ready():
        return {
            "project_name": os.getenv("PROJECT_NAME", "emergency-admission-bda"),
            "model_loaded": False,
            "model_path": str(active_path),
            "risk_threshold": THRESHOLD,
            "dataset_type": "synthetic",
            "clinical_use": False,
            "not_for_clinical_use": True,
            "phase": "Phase 1 - Foundation and Infrastructure",
            "disclaimer": "Academic synthetic demonstration; not for clinical use or triage decisions.",
        }

    bundle = model_manager.load_bundle()
    return {
        "project_name": os.getenv("PROJECT_NAME", "emergency-admission-bda"),
        "model_loaded": True,
        "model_name": bundle.get("model_name", "unknown"),
        "features": bundle.get("features", []),
        "risk_threshold": float(bundle.get("threshold", THRESHOLD)),
        "dataset_type": bundle.get("dataset_type", "synthetic"),
        "clinical_use": False,
        "not_for_clinical_use": True,
        "is_experimental": bool(bundle.get("is_experimental", False)),
        "disclaimer": "Academic synthetic demonstration; not for clinical use or triage decisions.",
    }


@app.get("/model-info")
def model_info():
    """Safe, non-sensitive model governance and evaluation summary."""
    if not model_manager.is_ready():
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unavailable",
                "detail": model_manager.error_reason or "Trained model is not loaded.",
                "disclaimer": SYNTHETIC_DISCLAIMER,
            },
        )

    bundle = model_manager.load_bundle()
    meta = model_manager.metadata or {}
    report = model_manager.evaluation_report or {}

    model_obj = bundle.get("model")
    clf = getattr(model_obj, "named_steps", {}).get("classifier", model_obj)
    model_class = clf.__class__.__name__ if clf else "Unknown"

    ds_meta = report.get("dataset_metadata", {})
    test_metrics = report.get("held_out_test_metrics", {})

    return {
        "model_name": bundle.get("model_name", "unknown"),
        "model_version": meta.get("artifact_version", "1.0.0"),
        "model_class": model_class,
        "target_name": bundle.get("target", "admission_within_24h"),
        "features": bundle.get("features", []),
        "feature_count": len(bundle.get("features", [])),
        "threshold": float(bundle.get("threshold", THRESHOLD)),
        "is_experimental": bool(bundle.get("is_experimental", False)),
        "data_origin": bundle.get("dataset_type", "synthetic"),
        "training_dataset_dimensions": {
            "rows": ds_meta.get("total_rows"),
            "columns": ds_meta.get("total_columns"),
        } if ds_meta else None,
        "evaluation_summary": {
            "selection_criterion": report.get("model_selection", {}).get("selection_criterion"),
            "held_out_test_accuracy": test_metrics.get("accuracy"),
            "held_out_test_f1": test_metrics.get("f1_score"),
            "notes": "Single-class evaluation set; metrics reflect experimental smoke-test." if bundle.get("is_experimental") else "Validated evaluation split.",
        } if test_metrics else None,
        "disclaimer": SYNTHETIC_DISCLAIMER,
    }


def build_feature_dataframe(payloads: list[PatientFeatures], expected_features: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Authoritative feature-mapping function for Phase 6 FastAPI inference.

    1. Validates presence of all expected model features in strict canonical order.
    2. Strips patient_id and prevents any identifier or metadata contamination.
    3. Guarantees DataFrame columns are in the exact expected feature order.
    4. Rejects NaN, null, or non-finite values before inference.
    """
    rows = []
    patient_ids = []
    for item in payloads:
        data = item.model_dump()
        pid = str(data.pop("patient_id"))
        patient_ids.append(pid)

        row = {}
        for feat in expected_features:
            val = data.get(feat)
            if val is None or not np.isfinite(val):
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"Feature '{feat}' must be a finite number; got {val} for patient {pid}",
                )
            row[feat] = float(val)
        rows.append(row)

    return pd.DataFrame(rows, columns=expected_features), patient_ids


def extract_positive_class_probability(probabilities: np.ndarray, pos_class_idx: int) -> np.ndarray:
    """Safely extract positive-class probabilities from model.predict_proba output.

    Maps using model.classes_ positive class index, with defensive column bounds checking.
    """
    n_cols = probabilities.shape[1]
    if n_cols > pos_class_idx:
        return probabilities[:, pos_class_idx]
    elif n_cols == 1:
        return np.zeros(probabilities.shape[0])
    return probabilities[:, -1]


@app.post("/predict")
def predict(payload: PatientFeatures):
    """Predict emergency admission risk for a single patient."""
    bundle = model_manager.load_bundle()
    features = bundle["features"]
    model = bundle["model"]

    frame, patient_ids = build_feature_dataframe([payload], features)
    patient_id = patient_ids[0]

    try:
        probabilities = model.predict_proba(frame)
        pos_idx = model_manager.pos_class_index
        scores = extract_positive_class_probability(probabilities, pos_idx)
        score = float(scores[0])
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Prediction failed; check feature compatibility: {exc}",
        ) from exc

    if not np.isfinite(score) or not (0.0 <= score <= 1.0):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Computed score out of probability bounds: {score}",
        )

    threshold = float(bundle.get("threshold", THRESHOLD))
    review_flag = bool(score >= threshold)
    binary_pred = int(review_flag)

    return {
        "patient_id": patient_id,
        "prediction": binary_pred,
        "risk_probability": round(score, 6),
        "risk_percent": round(score * 100, 2),
        "threshold": threshold,
        "review_flag": review_flag,
        "model_name": bundle.get("model_name", "unknown"),
        "dataset_type": bundle.get("dataset_type", "synthetic"),
        "is_experimental": bool(bundle.get("is_experimental", False)),
        "notice": SYNTHETIC_DISCLAIMER,
    }


@app.post("/predict/batch")
def predict_batch(payloads: list[PatientFeatures]):
    """Predict emergency admission risk for a batch of up to 500 patients."""
    if not payloads:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Provide at least one patient record.",
        )
    if len(payloads) > 500:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Maximum 500 records per batch.",
        )

    bundle = model_manager.load_bundle()
    features = bundle["features"]
    model = bundle["model"]
    threshold = float(bundle.get("threshold", THRESHOLD))
    pos_idx = model_manager.pos_class_index
    is_exp = bool(bundle.get("is_experimental", False))
    dataset_type = bundle.get("dataset_type", "synthetic")
    model_name = bundle.get("model_name", "unknown")

    frame, patient_ids = build_feature_dataframe(payloads, features)

    try:
        probabilities = model.predict_proba(frame)
        scores = extract_positive_class_probability(probabilities, pos_idx)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Batch prediction failed: {exc}",
        ) from exc

    results = []
    for idx, pid in enumerate(patient_ids):
        score = float(scores[idx])
        review_flag = bool(score >= threshold)
        results.append({
            "patient_id": pid,
            "prediction": int(review_flag),
            "risk_probability": round(score, 6),
            "risk_percent": round(score * 100, 2),
            "threshold": threshold,
            "review_flag": review_flag,
            "model_name": model_name,
            "dataset_type": dataset_type,
            "is_experimental": is_exp,
            "notice": SYNTHETIC_DISCLAIMER,
        })

    return {"predictions": results, "count": len(results)}


# -----------------------------------------------------------------------------
# Big Data Pipeline Observability & Diagnostics Endpoints
# -----------------------------------------------------------------------------
DATA_DIR = Path(os.getenv("DATA_DIR", "/app/data" if Path("/app/data").exists() else "data"))


def _probe_socket(targets: list[tuple[str, int]], timeout: float = 0.8) -> dict[str, Any]:
    """Safely probe network connectivity to a list of host:port targets with strict timeout."""
    import socket
    last_error = None
    for host, port in targets:
        try:
            with socket.create_connection((host, port), timeout=timeout):
                return {"reachable": True, "target": f"{host}:{port}", "error": None}
        except Exception as exc:
            last_error = str(exc)
    return {"reachable": False, "target": f"{targets[0][0]}:{targets[0][1]}", "error": last_error}


def _get_latest_manifest(subdir: str, filename: str) -> dict[str, Any] | None:
    """Safely load the most recently modified JSON manifest from a data/metadata directory."""
    dir_path = DATA_DIR / "metadata" / subdir
    if not dir_path.exists():
        return None
    files = list(dir_path.glob(f"**/{filename}"))
    if not files:
        return None
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    try:
        data = json.loads(files[0].read_text(encoding="utf-8"))
        data["_filepath"] = str(files[0])
        return data
    except Exception:
        return None


@app.get("/pipeline/status")
def get_pipeline_status():
    """End-to-end BDA pipeline observability status and evidence summary."""
    from datetime import datetime, timezone

    # 1. Probes
    kafka_probe = _probe_socket([("kafka", 29092), ("localhost", 9092), ("127.0.0.1", 9092)])
    hdfs_probe = _probe_socket([("namenode", 9870), ("localhost", 9870), ("127.0.0.1", 9870), ("namenode", 9000), ("localhost", 9000)])

    # 2. Local Streaming Artifacts
    stream_file = DATA_DIR / "streamed" / "patient_events.jsonl"
    stream_lines = 0
    stream_bytes = 0
    if stream_file.is_file():
        try:
            stream_bytes = stream_file.stat().st_size
            with open(stream_file, "r", encoding="utf-8") as f:
                stream_lines = sum(1 for _ in f if _.strip())
        except Exception:
            pass

    quarantine_file = DATA_DIR / "streamed" / "quarantine_events.jsonl"
    quarantine_lines = 0
    if quarantine_file.is_file():
        try:
            with open(quarantine_file, "r", encoding="utf-8") as f:
                quarantine_lines = sum(1 for _ in f if _.strip())
        except Exception:
            pass

    # 3. Parquet Feature Dataset
    parquet_dir = DATA_DIR / "processed" / "features.parquet"
    parquet_exists = parquet_dir.is_dir()
    partitions = []
    if parquet_exists:
        try:
            partitions = [p.name for p in parquet_dir.iterdir() if p.is_dir() and p.name.startswith("observation_date=")]
        except Exception:
            pass

    # 4. Manifests
    spark_manifest = _get_latest_manifest("spark_manifests", "processing_manifest.json")
    ingest_manifest = _get_latest_manifest("ingestion_manifests", "manifest.json")

    # 5. Model Readiness
    is_ready = model_manager.is_ready()

    now_utc = datetime.now(timezone.utc).isoformat()

    return {
        "timestamp_utc": now_utc,
        "disclaimer": SYNTHETIC_DISCLAIMER,
        "components": {
            "kafka": {
                "status": "verified" if kafka_probe["reachable"] else "unavailable",
                "probe_result": kafka_probe,
                "topic": "patient-observations",
                "consumer_group": "emergency-admission-bda-consumer",
                "evidence_source": "Live TCP probe to Kafka broker (9092/29092)",
            },
            "hdfs": {
                "status": "verified" if hdfs_probe["reachable"] else "unavailable",
                "probe_result": hdfs_probe,
                "raw_path": "/emergency-admission-bda/raw/kafka-events",
                "processed_path": "/emergency-admission-bda/processed/patient-features",
                "evidence_source": "Live TCP probe to Hadoop NameNode (9870/9000)",
            },
            "spark": {
                "status": "historical_success" if spark_manifest and spark_manifest.get("status") == "SUCCESS" else "unknown",
                "last_run_id": spark_manifest.get("run_id") if spark_manifest else None,
                "last_run_timestamp": spark_manifest.get("processing_timestamp_utc") if spark_manifest else None,
                "raw_records": spark_manifest.get("raw_record_count") if spark_manifest else 0,
                "valid_records": spark_manifest.get("valid_record_count") if spark_manifest else 0,
                "quarantined_records": spark_manifest.get("quarantined_record_count") if spark_manifest else 0,
                "duplicate_count": spark_manifest.get("duplicate_count") if spark_manifest else 0,
                "output_records": spark_manifest.get("output_record_count") if spark_manifest else 0,
                "evidence_source": f"Historical Spark Manifest ({spark_manifest.get('_filepath', 'none')})" if spark_manifest else "No manifest found",
            },
            "parquet": {
                "status": "verified" if parquet_exists else "missing",
                "dataset_path": str(parquet_dir),
                "partition_count": len(partitions),
                "partition_column": "observation_date",
                "success_marker": (parquet_dir / "_SUCCESS").exists() if parquet_exists else False,
                "evidence_source": "Local filesystem verification of partitioned Parquet files",
            },
            "model": {
                "status": "loaded" if is_ready else "not_loaded",
                "model_name": model_manager.model_bundle.get("model_name") if is_ready else None,
                "threshold": float(model_manager.model_bundle.get("threshold", THRESHOLD)) if is_ready else THRESHOLD,
                "is_experimental": bool(model_manager.model_bundle.get("is_experimental", True)) if is_ready else True,
                "evidence_source": "Phase 6 FastAPI ModelManager memory registry",
            },
            "fastapi": {
                "status": "online",
                "service": "emergency-admission-api",
                "checked_at": now_utc,
                "evidence_source": "FastAPI internal process health probe",
            },
        },
        "streaming_storage": {
            "patient_events_lines": stream_lines,
            "patient_events_bytes": stream_bytes,
            "quarantine_events_lines": quarantine_lines,
            "patient_events_path": str(stream_file) if stream_file.exists() else None,
        },
        "latest_ingest_manifest": ingest_manifest,
        "latest_spark_manifest": spark_manifest,
    }


@app.get("/pipeline/kafka")
def get_pipeline_kafka():
    """Detailed Kafka topic, broker, and event archive diagnostic telemetry."""
    probe = _probe_socket([("kafka", 29092), ("localhost", 9092), ("127.0.0.1", 9092)])
    stream_file = DATA_DIR / "streamed" / "patient_events.jsonl"
    quarantine_file = DATA_DIR / "streamed" / "quarantine_events.jsonl"

    stream_count = 0
    stream_bytes = 0
    if stream_file.is_file():
        try:
            stream_bytes = stream_file.stat().st_size
            with open(stream_file, "r", encoding="utf-8") as f:
                stream_count = sum(1 for _ in f if _.strip())
        except Exception:
            pass

    return {
        "service": "kafka-streaming",
        "topic": "patient-observations",
        "consumer_group": "emergency-admission-bda-consumer",
        "broker_probe": probe,
        "status": "verified" if probe["reachable"] else "unavailable",
        "partitions_configured": 3,
        "replication_factor": 1,
        "streamed_events_archive": {
            "path": str(stream_file),
            "exists": stream_file.exists(),
            "record_count": stream_count,
            "size_bytes": stream_bytes,
            "format": "jsonl (newline-delimited JSON)",
        },
        "quarantine_archive": {
            "path": str(quarantine_file),
            "exists": quarantine_file.exists(),
            "record_count": 0,
        },
        "scripts": {
            "producer": "scripts/produce_kafka_stream.py",
            "consumer": "scripts/consume_kafka_stream.py",
        },
        "evidence_source": "Live socket check + streamed JSONL event archive on disk",
    }


@app.get("/pipeline/hdfs")
def get_pipeline_hdfs():
    """Detailed HDFS cluster connectivity and raw ingestion manifest status."""
    probe = _probe_socket([("namenode", 9870), ("localhost", 9870), ("127.0.0.1", 9870), ("namenode", 9000), ("localhost", 9000)])
    ingest_manifest = _get_latest_manifest("ingestion_manifests", "manifest.json")

    return {
        "service": "hadoop-hdfs",
        "namenode_probe": probe,
        "status": "verified" if probe["reachable"] else "unavailable",
        "configured_paths": {
            "raw": "/emergency-admission-bda/raw/kafka-events",
            "processed": "/emergency-admission-bda/processed/patient-features",
            "quarantine": "/emergency-admission-bda/quarantine/spark-processing",
            "manifests": "/emergency-admission-bda/metadata/ingestion-manifests",
        },
        "latest_ingestion_manifest": ingest_manifest,
        "manifest_type": "historical_record",
        "evidence_source": "Live NameNode socket check + historical ingestion manifest",
    }


@app.get("/pipeline/spark/latest")
def get_pipeline_spark_latest():
    """Retrieve the most recent PySpark batch feature engineering run manifest."""
    spark_manifest = _get_latest_manifest("spark_manifests", "processing_manifest.json")
    if not spark_manifest:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No PySpark processing manifests found in data/metadata/spark_manifests.")

    return {
        "service": "pyspark-batch-processing",
        "manifest": spark_manifest,
        "status": spark_manifest.get("status", "UNKNOWN"),
        "manifest_type": "historical_record",
        "evidence_source": f"Historical file: {spark_manifest.get('_filepath')}",
    }


@app.get("/pipeline/parquet")
def get_pipeline_parquet():
    """Inspect the local processed Parquet dataset, partitions, and schema."""
    parquet_dir = DATA_DIR / "processed" / "features.parquet"
    if not parquet_dir.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Parquet directory not found at {parquet_dir}")

    partitions = []
    try:
        partitions = sorted([p.name for p in parquet_dir.iterdir() if p.is_dir() and p.name.startswith("observation_date=")])
    except Exception:
        pass

    success_marker = (parquet_dir / "_SUCCESS").exists()

    # Read schema & dimensions if pyarrow/fastparquet/pandas available
    schema_info = []
    total_rows = 0
    try:
        df_sample = pd.read_parquet(parquet_dir)
        total_rows = len(df_sample)
        schema_info = [{"column": col, "dtype": str(dtype)} for col, dtype in df_sample.dtypes.items()]
    except Exception:
        pass

    return {
        "service": "parquet-feature-dataset",
        "path": str(parquet_dir),
        "exists": True,
        "partition_column": "observation_date",
        "partition_count": len(partitions),
        "partitions_sample": partitions[:10],
        "has_success_marker": success_marker,
        "total_rows": total_rows,
        "schema": schema_info,
        "data_origin": "synthetic",
        "dataset_nature": "Local copy of processed HDFS Parquet dataset",
        "evidence_source": "Local filesystem scan of partitioned Parquet directory",
    }


@app.get("/pipeline/manifests")
def get_pipeline_manifests():
    """List all available historical ingestion and Spark execution manifests."""
    ingest_dir = DATA_DIR / "metadata" / "ingestion_manifests"
    spark_dir = DATA_DIR / "metadata" / "spark_manifests"

    ingest_manifests = []
    if ingest_dir.exists():
        for path in sorted(ingest_dir.glob("**/manifest.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                ingest_manifests.append({
                    "ingestion_date": data.get("ingestion_date"),
                    "record_count": data.get("record_count"),
                    "valid_record_count": data.get("valid_record_count"),
                    "verification_status": data.get("verification_status"),
                    "filepath": str(path),
                })
            except Exception:
                pass

    spark_manifests = []
    if spark_dir.exists():
        for path in sorted(spark_dir.glob("**/processing_manifest.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                spark_manifests.append({
                    "run_id": data.get("run_id"),
                    "timestamp": data.get("processing_timestamp_utc"),
                    "raw_record_count": data.get("raw_record_count"),
                    "output_record_count": data.get("output_record_count"),
                    "status": data.get("status"),
                    "filepath": str(path),
                })
            except Exception:
                pass

    return {
        "ingestion_manifests": ingest_manifests,
        "spark_manifests": spark_manifests,
        "disclaimer": "All manifests represent historical execution records, not active live operations.",
    }

