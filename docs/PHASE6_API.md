# Phase 6 — FastAPI Model Inference API, Input Validation & Prediction Serving

> [!CAUTION]
> **ACADEMIC DEMONSTRATION & SYNTHETIC DATA NOTICE**:
> All patient records, physiological vital signs, and binary admission labels in this project are **SYNTHETIC**.
> The API inference service and predictive outputs described herein are an **academic Big Data Analytics software engineering demonstration**.
> This service is **NOT clinically validated** and **MUST NEVER be used for actual medical decisions, patient diagnosis, or emergency department triage**.

---

## 1. Phase 6 Architecture & Objectives

Phase 6 implements a robust, secure, asynchronous REST API using **FastAPI** to serve real-time predictions from the versioned machine learning model bundle trained in Phase 5 (`models/risk_model.joblib`).

```
+-------------------------------------------------------------+
|             Client Applications / Web Dashboards            |
|       (Streamlit Dashboard, cURL, PowerShell, Swagger)      |
+-------------------------------------------------------------+
                              | HTTP REST
                              v
+-------------------------------------------------------------+
|                     FastAPI Inference API                   |
|                   (app/api/main.py : 8000)                  |
|  - CORS Middleware (ports 8501, 127.0.0.1:8501)             |
|  - Lifespan Model Manager (load once at startup)            |
|  - Strict Pydantic Schema Validation (NaN, Inf, Range)      |
+-------------------------------------------------------------+
                              |
       +----------------------+-----------------------+
       |                      |                       |
       v                      v                       v
[GET /health]           [GET /ready]            [POST /predict]
Process Liveness      Inference Readiness      Vectorized Scoring
 (Always 200 OK)     (200 Ready / 503 Down)   (Strict Feature Order)
                              |
                              v
+-------------------------------------------------------------+
|               Loaded Model Bundle & Governance              |
|  - models/risk_model.joblib (Scikit-Learn Pipeline)         |
|  - Positive-class mapping via model.classes_ (class 1)      |
|  - models/model_metadata.json (Registry Governance)         |
|  - models/evaluation_report.json (Audit Trail)              |
+-------------------------------------------------------------+
```

### Key Design Principles
1. **Decoupled Liveness vs. Readiness**:
   - `GET /health` acts as a Kubernetes/container liveness probe (returns 200 OK whenever the FastAPI process is running).
   - `GET /ready` acts as an inference readiness probe (returns 200 OK only when the model bundle is verified and ready; returns 503 if missing or corrupt).
2. **Cached Model Lifecycle Management**:
   The model is loaded once at application startup using FastAPI's `lifespan` context manager, eliminating disk I/O latency during prediction.
3. **Strict Software Input Validation**:
   Inputs are validated against physiological/operational bounds, non-nullity, and finiteness (rejecting `NaN` and `Infinity`). Extra fields are forbidden (`extra="forbid"`).
4. **Positive-Class Index Mapping**:
   The service queries `model.classes_` to identify the column index corresponding to class label `1`, avoiding silent bugs caused by inverted class orders.
5. **Experimental Smoke-Test Propagation**:
   The API dynamically checks whether the loaded model was trained with the small-dataset override (`is_experimental: true`) and surfaces this disclaimer in all response payloads.

---

## 2. API Endpoints Specification

### Summary Table

| Method | Endpoint | Purpose | Success Status | Failure Status |
|---|---|---|---|---|
| `GET` | `/` | Root foundation & service identity | `200 OK` | — |
| `GET` | `/health` | Process liveness probe | `200 OK` | — |
| `GET` | `/ready` | Model inference readiness probe | `200 OK` | `503 Service Unavailable` |
| `GET` | `/metadata` | Registry metadata & configuration | `200 OK` | — |
| `GET` | `/model-info` | Safe model governance & metrics | `200 OK` | `503 Service Unavailable` |
| `POST` | `/predict` | Single patient risk prediction | `200 OK` | `422 Unprocessable` / `503 Down` |
| `POST` | `/predict/batch` | Batch risk prediction (up to 500) | `200 OK` | `413 Too Large` / `422 Unprocessable` |

---

### Detailed Endpoint Contracts

#### 1. `GET /health`
- **Purpose**: Verify that the web server is running and responsive.
- **Sample Response**:
  ```json
  {
    "status": "ok",
    "service": "emergency-admission-api",
    "application_health": "healthy",
    "model_readiness": "loaded",
    "model_file_exists": true,
    "disclaimer": "Academic demo; not for clinical use."
  }
  ```

#### 2. `GET /ready`
- **Purpose**: Verify that the ML model bundle is loaded, validated, and operational.
- **Success Response (`200 OK`)**:
  ```json
  {
    "status": "ready",
    "model_loaded": true,
    "model_name": "logistic_regression",
    "artifact_status": "loaded",
    "is_experimental": true,
    "data_origin": "synthetic",
    "model_threshold": 0.7,
    "feature_count": 12,
    "error_reason": null
  }
  ```
- **Not Ready Response (`503 Service Unavailable`)**:
  ```json
  {
    "status": "not_ready",
    "model_loaded": false,
    "model_name": null,
    "artifact_status": "missing",
    "is_experimental": null,
    "data_origin": "synthetic",
    "model_threshold": 0.7,
    "feature_count": 0,
    "error_reason": "Trained model not found at models/risk_model.joblib."
  }
  ```

#### 3. `GET /model-info`
- **Purpose**: Expose safe model registry metadata without leaking filesystem paths or credentials.
- **Sample Response (`200 OK`)**:
  ```json
  {
    "model_name": "logistic_regression",
    "model_version": "1.0.0",
    "model_class": "LogisticRegression",
    "target_name": "admission_within_24h",
    "features": ["age", "heart_rate", "systolic_bp", "diastolic_bp", "spo2", "respiratory_rate", "temperature_c", "previous_admissions", "chronic_conditions_count", "lactate", "observation_hour", "prior_observation_count"],
    "feature_count": 12,
    "threshold": 0.7,
    "is_experimental": true,
    "data_origin": "synthetic",
    "training_dataset_dimensions": {
      "rows": 50,
      "columns": 22
    },
    "evaluation_summary": {
      "selection_criterion": "validation PR-AUC (Average Precision) with F1 tie-breaker",
      "held_out_test_accuracy": 1.0,
      "held_out_test_f1": 0.0,
      "notes": "Single-class evaluation set; metrics reflect experimental smoke-test."
    },
    "disclaimer": "Synthetic academic demonstration only. Not a clinically validated prediction; do not use for medical decisions or triage."
  }
  ```

#### 4. `POST /predict`
- **Purpose**: Compute admission probability and threshold decision for a single patient.
- **Request Body**:
  ```json
  {
    "patient_id": "SYN-DEMO-001",
    "age": 67.0,
    "heart_rate": 112.0,
    "systolic_bp": 95.0,
    "diastolic_bp": 62.0,
    "spo2": 91.0,
    "respiratory_rate": 26.0,
    "temperature_c": 38.6,
    "previous_admissions": 2.0,
    "chronic_conditions_count": 3.0,
    "lactate": 2.4,
    "observation_hour": 14.0,
    "prior_observation_count": 4.0
  }
  ```
- **Response Body (`200 OK`)**:
  ```json
  {
    "patient_id": "SYN-DEMO-001",
    "prediction": 0,
    "risk_probability": 0.068412,
    "risk_percent": 6.84,
    "threshold": 0.7,
    "review_flag": false,
    "model_name": "logistic_regression",
    "dataset_type": "synthetic",
    "is_experimental": true,
    "notice": "Synthetic academic demonstration only. Not a clinically validated prediction; do not use for medical decisions or triage."
  }
  ```

#### 5. `POST /predict/batch`
- **Purpose**: High-throughput vectorized prediction for up to 500 patient records in a single request.
- **Request Body**: JSON array of patient objects `[ { ... }, { ... } ]`.
- **Response Body (`200 OK`)**:
  ```json
  {
    "predictions": [ ... ],
    "count": 2
  }
  ```

---

## 3. Input Validation Bounds & Rules

Pydantic enforces strict software typing, bounds, and finiteness:

| Feature | Type | Range | Meaning / Rationale |
|---|---|---|---|
| `patient_id` | `str` | $1 \le \text{len} \le 80$ | Audit identifier (not used as a predictor) |
| `age` | `float` | $[18.0, 100.0]$ | Adult emergency admissions cohort |
| `heart_rate` | `float` | $[25.0, 240.0]$ | Severe bradycardia to extreme tachycardia |
| `systolic_bp` | `float` | $[40.0, 300.0]$ | Profound hypotension to severe hypertensive crisis |
| `diastolic_bp` | `float` | $[20.0, 200.0]$ | Vascular pressure bounds |
| `spo2` | `float` | $[50.0, 100.0]$ | Critical hypoxia to normal arterial oxygenation |
| `respiratory_rate` | `float` | $[3.0, 80.0]$ | Extreme bradypnea to severe tachypnea |
| `temperature_c` | `float` | $[30.0, 44.0]$ | Severe hypothermia to extreme hyperpyrexia |
| `previous_admissions` | `float` | $[0.0, 100.0]$ | Historic emergency visits |
| `chronic_conditions_count` | `float` | $[0.0, 50.0]$ | Documented comorbidities |
| `lactate` | `float` | $[0.0, 30.0]$ | Cellular hypoperfusion / sepsis biomarker |
| `observation_hour` | `float` | $[0.0, 23.0]$ | Arrival circadian cycle hour |
| `prior_observation_count`| `float` | $[0.0, 100000.0]$| Prior longitudinal events recorded |

### Quality Rules
- **Non-nullity**: Nulls are strictly rejected with HTTP 422.
- **Finiteness**: `NaN`, `Infinity`, and `-Infinity` are rejected with HTTP 422.
- **Extra Fields**: Unapproved keys (e.g., `is_synthetic`, `admission_within_24h`) are forbidden and rejected with HTTP 422.

---

## 4. Environment Variables Configuration

| Variable | Default Value | Description |
|---|---|---|
| `MODEL_PATH` | `/app/models/risk_model.joblib` (Docker)<br>`models/risk_model.joblib` (Local) | Location of serialized Phase 5 model bundle |
| `RISK_THRESHOLD` | `0.70` | Default decision threshold for binary review flag |
| `API_PORT` | `8000` | Port on which FastAPI listener binds |
| `API_CORS_ORIGINS`| `http://localhost:8501,http://127.0.0.1:8501` | Allowed browser origins for Streamlit dashboard |
| `LOG_LEVEL` | `INFO` | Application log verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |

---

## 5. Windows PowerShell Execution Guide

Run all commands from the repository root: `C:\Users\ANSHUL\BDA-PROJECT\emergency_admission_bda`

### Option A: Run Locally with Python

1. **Install API Requirements**:
   ```powershell
   py -3.13 -m pip install -r requirements-api.txt
   ```

2. **Start FastAPI with Uvicorn**:
   ```powershell
   $env:MODEL_PATH="models/risk_model.joblib"
   $env:RISK_THRESHOLD="0.70"
   py -3.13 -m uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload
   ```

3. **Interactive Documentation (Swagger)**:
   Open [http://localhost:8000/docs](http://localhost:8000/docs) in your browser.

---

### Option B: Run in Docker Container

1. **Build Container Image**:
   ```powershell
   docker compose build api
   ```

2. **Start API Container**:
   ```powershell
   docker compose up -d api
   docker compose ps api
   ```

3. **Inspect Logs**:
   ```powershell
   docker compose logs api --tail 50
   ```

4. **Stop Container**:
   ```powershell
   docker compose down
   ```

---

## 6. Testing Endpoints via Windows PowerShell

Use PowerShell's `Invoke-RestMethod` to verify each endpoint:

### 1. Test Liveness Probe (`GET /health`)
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/health" -Method Get | ConvertTo-Json
```

### 2. Test Inference Readiness Probe (`GET /ready`)
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/ready" -Method Get | ConvertTo-Json
```

### 3. Test Model Governance Info (`GET /model-info`)
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/model-info" -Method Get | ConvertTo-Json -Depth 4
```

### 4. Test Single Patient Prediction (`POST /predict`)
```powershell
$body = @{
    patient_id = "SYN-POWERSHELL-01"
    age = 71.0
    heart_rate = 92.0
    systolic_bp = 138.0
    diastolic_bp = 88.0
    spo2 = 94.0
    respiratory_rate = 20.0
    temperature_c = 37.2
    previous_admissions = 1.0
    chronic_conditions_count = 2.0
    lactate = 1.8
    observation_hour = 16.0
    prior_observation_count = 1.0
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8000/predict" -Method Post -ContentType "application/json" -Body $body | ConvertTo-Json
```

### 5. Test Batch Prediction (`POST /predict/batch`)
```powershell
$batch = @(
    @{
        patient_id = "SYN-PATIENT-A"
        age = 55.0; heart_rate = 78.0; systolic_bp = 120.0; diastolic_bp = 80.0
        spo2 = 98.0; respiratory_rate = 16.0; temperature_c = 36.8
        previous_admissions = 0.0; chronic_conditions_count = 1.0; lactate = 1.1
        observation_hour = 10.0; prior_observation_count = 0.0
    },
    @{
        patient_id = "SYN-PATIENT-B"
        age = 82.0; heart_rate = 110.0; systolic_bp = 90.0; diastolic_bp = 60.0
        spo2 = 90.0; respiratory_rate = 24.0; temperature_c = 38.5
        previous_admissions = 4.0; chronic_conditions_count = 4.0; lactate = 3.2
        observation_hour = 22.0; prior_observation_count = 5.0
    }
) | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8000/predict/batch" -Method Post -ContentType "application/json" -Body $batch | ConvertTo-Json -Depth 4
```

---

## 7. Security Best Practices & Governance

1. **Joblib Serialization Warning**:
   `joblib` uses Python's pickle protocol to serialize models. Deserializing pickled files can execute arbitrary code. Only load models from trusted workspace directories. Never expose an endpoint that accepts raw `.joblib` or `.pkl` uploads from users.
2. **CORS Restrictions**:
   Browser cross-origin resource sharing is restricted to localhost Streamlit origins (`http://localhost:8501`). Wildcard origins (`*`) are disabled.
3. **No Patient Identifiers Stored**:
   The API does not persist incoming patient feature vectors or medical measurements to disk or database.
