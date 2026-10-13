# Phase 6 — Model Prediction API

## What this phase adds
- FastAPI endpoints for health, model metadata, single prediction and batch prediction.
- Pydantic request validation and broad input bounds.
- Loads `models/risk_model.joblib` produced in Phase 5.
- Returns model probability, configured threshold, review flag, model name and synthetic-data disclaimer.
- Does not fall back to a heuristic if the model is missing.

## Before starting
Complete Phase 5 and make sure this file exists:
`models/risk_model.joblib`

The Docker API service mounts the local `models` folder into `/app/models`, so the model can be updated without baking it into the image.

## Run API locally

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements-api.txt
$env:MODEL_PATH="models/risk_model.joblib"
$env:RISK_THRESHOLD="0.7"
uvicorn app.api.main:app --reload --port 8000
```

Open http://localhost:8000/docs.

## Example request

```json
{
  "patient_id": "SYN-DEMO-001",
  "age": 67,
  "heart_rate": 112,
  "systolic_bp": 95,
  "diastolic_bp": 62,
  "spo2": 91,
  "respiratory_rate": 26,
  "temperature_c": 38.6,
  "previous_admissions": 2,
  "chronic_conditions_count": 3,
  "lactate": 2.4,
  "observation_hour": 14,
  "prior_observation_count": 4
}
```

This returns a model probability for the synthetic model. The example values are fictitious and do not indicate actual patient risk.

## Endpoints
- `GET /`: project and API status.
- `GET /health`: service health and model-file existence.
- `GET /metadata`: model name, features and configured threshold.
- `POST /predict`: one record.
- `POST /predict/batch`: up to 500 records.

## Security and limitations
- Local academic development only; no authentication is configured.
- Do not expose this API to the internet or submit real patient data.
- The model was trained on synthetic data and must not be used for medical decisions.
- The API's feature schema must match the exact feature list in the trained model bundle.
