# Phase 7 — Streamlit Dashboard Integration

## Features
- Summary metrics for observations, patients and records above a display threshold.
- Patient ID search and age filtering.
- Observation table and heart-rate/score distributions.
- Prediction form that calls `POST /predict` on the FastAPI service.
- Explicit distinction between Phase 1 illustrative heuristic scores and Phase 6 model-generated predictions.
- Visible disclaimer that synthetic model scores are not clinically valid.

## Run
Ensure the Phase 5 model file exists at `models/risk_model.joblib`.

```powershell
docker compose up --build -d api dashboard
```

Open http://localhost:8501. API docs are at http://localhost:8000/docs.

## Configuration
- `API_BASE_URL`: internal Docker API URL is set to `http://api:8000` in Compose.
- `RISK_THRESHOLD`: default display threshold is `0.7`; the dashboard slider can adjust only the display filter and does not retrain the model.

## Troubleshooting
- **Dashboard cannot contact API:** `docker compose logs -f api dashboard`; confirm both containers are running.
- **Model missing:** run Phase 5 and confirm `models/risk_model.joblib` exists in the project directory.
- **Table shows heuristic scores:** the list endpoint may still be serving a demo-only score. The single-record form calls the trained-model `/predict` endpoint. Do not label heuristic values as model predictions.
- **No data:** generate synthetic CSV and complete Phase 2/3/4 as required.
