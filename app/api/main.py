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
        "API_CORS_ORIGINS", "http://localhost:8501,http://127.0.0.1:8501"
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


@app.post("/predict")
def predict(payload: PatientFeatures):
    """Predict emergency admission risk for a single patient."""
    bundle = model_manager.load_bundle()
    features = bundle["features"]
    values = payload.model_dump()
    patient_id = values.pop("patient_id")

    # Build ordered feature row
    row = {name: values.get(name) for name in features}
    unknown = [name for name in features if name not in row or row[name] is None]
    if unknown:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Model requires unsupported features: {unknown}")

    frame = pd.DataFrame([row], columns=features)
    model = bundle["model"]

    try:
        probabilities = model.predict_proba(frame)
        pos_idx = model_manager.pos_class_index
        score = float(probabilities[0, pos_idx])
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=f"Prediction failed; check feature compatibility: {exc}") from exc

    if not np.isfinite(score) or not (0.0 <= score <= 1.0):
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Computed score out of probability bounds: {score}")

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
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Provide at least one patient record.")
    if len(payloads) > 500:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="Maximum 500 records per batch.")

    bundle = model_manager.load_bundle()
    features = bundle["features"]
    model = bundle["model"]
    threshold = float(bundle.get("threshold", THRESHOLD))
    pos_idx = model_manager.pos_class_index
    is_exp = bool(bundle.get("is_experimental", False))
    dataset_type = bundle.get("dataset_type", "synthetic")
    model_name = bundle.get("model_name", "unknown")

    # Vectorized DataFrame construction for batch performance
    rows = []
    patient_ids = []
    for item in payloads:
        vals = item.model_dump()
        patient_ids.append(vals.pop("patient_id"))
        rows.append({name: vals.get(name) for name in features})

    frame = pd.DataFrame(rows, columns=features)
    try:
        probabilities = model.predict_proba(frame)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=f"Batch prediction failed: {exc}") from exc

    results = []
    for idx, pid in enumerate(patient_ids):
        score = float(probabilities[idx, pos_idx])
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
