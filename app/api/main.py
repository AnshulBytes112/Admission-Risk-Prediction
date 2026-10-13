"""FastAPI inference API for the Emergency Admission Risk Prediction demo.

Loads a model bundle created by scripts/train_model.py. The API refuses to
silently fall back to heuristic scores: if no trained model is present, the
prediction endpoint returns HTTP 503 with setup instructions.
"""
import os
from pathlib import Path
from typing import Any
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/risk_model.joblib"))
THRESHOLD = float(os.getenv("RISK_THRESHOLD", "0.7"))
CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "API_CORS_ORIGINS", "http://localhost:8501,http://127.0.0.1:8501"
    ).split(",")
    if origin.strip()
]

app = FastAPI(
    title="Emergency Admission Risk Prediction API",
    description="Academic demo using a model trained on synthetic data. Not for clinical use.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

FEATURE_RANGES = {
    "age": (18, 100),
    "heart_rate": (25, 240),
    "systolic_bp": (40, 300),
    "diastolic_bp": (20, 200),
    "spo2": (50, 100),
    "respiratory_rate": (3, 80),
    "temperature_c": (30, 44),
    "previous_admissions": (0, 100),
    "chronic_conditions_count": (0, 50),
    "lactate": (0, 30),
    "observation_hour": (0, 23),
    "prior_observation_count": (0, 100000),
}

class PatientFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")
    patient_id: str = Field(min_length=1, max_length=80)
    age: float = Field(ge=18, le=100)
    heart_rate: float | None = Field(default=None, ge=25, le=240)
    systolic_bp: float | None = Field(default=None, ge=40, le=300)
    diastolic_bp: float | None = Field(default=None, ge=20, le=200)
    spo2: float | None = Field(default=None, ge=50, le=100)
    respiratory_rate: float | None = Field(default=None, ge=3, le=80)
    temperature_c: float | None = Field(default=None, ge=30, le=44)
    previous_admissions: float | None = Field(default=None, ge=0, le=100)
    chronic_conditions_count: float | None = Field(default=None, ge=0, le=50)
    lactate: float | None = Field(default=None, ge=0, le=30)
    observation_hour: float | None = Field(default=12, ge=0, le=23)
    prior_observation_count: float | None = Field(default=0, ge=0, le=100000)

def load_bundle() -> dict[str, Any]:
    if not MODEL_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail=f"Trained model not found at {MODEL_PATH}. Complete Phase 5 and mount the models directory.",
        )
    try:
        bundle = joblib.load(MODEL_PATH)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not load model bundle: {exc}") from exc
    if not isinstance(bundle, dict) or "model" not in bundle or "features" not in bundle:
        raise HTTPException(status_code=500, detail="Model bundle format is invalid.")
    return bundle

@app.get("/")
def root():
    return {
        "project": "Emergency Admission Risk Prediction",
        "phase": 1,
        "phase_name": "Project Foundation and Infrastructure",
        "status": "ready",
        "data_type": "synthetic",
        "model_readiness": "ready" if MODEL_PATH.exists() else "unloaded (Phase 1 foundation)",
        "notice": "Academic software demonstration using synthetic data. Not for clinical decision-support.",
    }

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "emergency-admission-api",
        "application_health": "healthy",
        "model_readiness": "loaded" if MODEL_PATH.exists() else "not_loaded",
        "model_file_exists": MODEL_PATH.exists(),
        "disclaimer": "Academic demo; not for clinical use.",
    }

@app.get("/metadata")
def metadata():
    if not MODEL_PATH.exists():
        return {
            "project_name": os.getenv("PROJECT_NAME", "emergency-admission-bda"),
            "model_loaded": False,
            "model_path": str(MODEL_PATH),
            "risk_threshold": THRESHOLD,
            "dataset_type": "synthetic",
            "clinical_use": False,
            "not_for_clinical_use": True,
            "phase": "Phase 1 - Foundation and Infrastructure",
            "disclaimer": "Academic synthetic demonstration; not for clinical use or triage decisions.",
        }
    bundle = load_bundle()
    return {
        "project_name": os.getenv("PROJECT_NAME", "emergency-admission-bda"),
        "model_loaded": True,
        "model_name": bundle.get("model_name", "unknown"),
        "features": bundle.get("features", []),
        "risk_threshold": float(bundle.get("threshold", THRESHOLD)),
        "dataset_type": bundle.get("dataset_type", "synthetic"),
        "clinical_use": False,
        "not_for_clinical_use": True,
    }

@app.post("/predict")
def predict(payload: PatientFeatures):
    bundle = load_bundle()
    features = bundle["features"]
    values = payload.model_dump()
    patient_id = values.pop("patient_id")
    row = {name: values.get(name) for name in features}
    unknown = [name for name in features if name not in row]
    if unknown:
        raise HTTPException(status_code=500, detail=f"Model requires unsupported features: {unknown}")
    frame = pd.DataFrame([row], columns=features)
    try:
        model = bundle["model"]
        if hasattr(model, "predict_proba"):
            score = float(model.predict_proba(frame)[0][1])
        else:
            raise HTTPException(status_code=500, detail="Loaded model does not support probability predictions.")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Prediction failed; check feature compatibility: {exc}") from exc
    threshold = float(bundle.get("threshold", THRESHOLD))
    return {
        "patient_id": patient_id,
        "risk_probability": round(score, 6),
        "risk_percent": round(score * 100, 2),
        "threshold": threshold,
        "review_flag": score >= threshold,
        "model_name": bundle.get("model_name", "unknown"),
        "dataset_type": bundle.get("dataset_type", "synthetic"),
        "notice": "Synthetic academic demonstration only. Not a clinically validated prediction; do not use for medical decisions.",
    }

@app.post("/predict/batch")
def predict_batch(payloads: list[PatientFeatures]):
    if not payloads:
        raise HTTPException(status_code=422, detail="Provide at least one patient record.")
    if len(payloads) > 500:
        raise HTTPException(status_code=413, detail="Maximum 500 records per batch.")
    return {"predictions": [predict(item) for item in payloads], "count": len(payloads)}
