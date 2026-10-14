# Phase 7 — Modern Streamlit Analytics Dashboard & FastAPI Serving Integration

## Overview & Architecture

Phase 7 implements a modern, responsive, healthcare analytics SaaS frontend for the **Emergency Admission Risk Prediction** Big Data Analytics (BDA) platform.

The dashboard integrates directly with the **Phase 6 FastAPI** inference microservice (`http://api:8000` inside Docker Compose, `http://localhost:8000` locally in Windows PowerShell) and surfaces Phase 5 model governance artifacts.

```
┌─────────────────────────────────────────────────────────────┐
│                      Phase 7 Frontend                       │
│             Streamlit Analytics Dashboard (:8501)           │
│   (Persistent Dark Navy Sidebar + Light SaaS Workspace)     │
└──────────────┬───────────────────────────────┬──────────────┘
               │ HTTP GET /health, /ready      │ Local Parquet /
               │ HTTP POST /predict, /batch    │ Metadata Read
               ▼                               ▼
┌───────────────────────────────┐  ┌───────────────────────────┐
│        Phase 6 Backend        │  │      Phase 4 & 5 Data     │
│      FastAPI Serving (:8000)  │  │   data/processed/features │
│  (Pydantic Schema Validation) │  │   models/model_metadata   │
└──────────────┬────────────────┘  │   models/evaluation_report│
               │                   └───────────────────────────┘
               ▼
┌───────────────────────────────┐
│       Phase 5 Pipeline        │
│   models/risk_model.joblib    │
│  (12-Feature Classification)  │
└───────────────────────────────┘
```

---

## Non-Negotiable Academic Disclaimer & Governance

> [!WARNING]
> **ACADEMIC DEMONSTRATION ONLY:**
> All patient records, physiological measurements, laboratory markers, and admission outcomes are **100% synthetic**.
> The model is experimental, trained on limited synthetic data, and is **not clinically validated**. It must **never** be used for real emergency medical decisions, clinical diagnosis, or hospital triaging.

### Operational Dimension Distinctions
The UI clearly differentiates three separate concepts:
1. **API Service Status:** Operational (FastAPI HTTP daemon responding).
2. **Model Artifact Status:** Loaded in memory (`models/risk_model.joblib` loaded via `joblib`).
3. **Model Evaluation Maturity:** Undersized Experimental Smoke-Test (statistical note prominently warns that test-set accuracy of 1.0 is an artifact of 0 positive examples in the test split; ROC-AUC and PR-AUC are mathematically undefined).

---

## Dashboard Pages & Functional Capabilities

### 1. Dashboard Overview (`views/overview.py`)
- **Header & Badges:** Displays `EXPERIMENTAL MODEL (SYNTHETIC DATA)` badge, last successful API health probe, and timestamp.
- **Live Status Cards:**
  - **API Status:** Live health status and round-trip latency (`/health`).
  - **Model Readiness:** Operational readiness status (`/ready`).
  - **Model Version:** Active model class, 12 features, decision threshold (`/model-info`).
  - **Data Origin:** Synthetic data origin and academic limitation badge.
- **Exploratory Analytics:** Reads `data/processed/features.parquet` to chart target class distribution, observation arrival distribution across hours, and vital sign correlations.
- **Recent Inferences:** Displays session-isolated recent predictions table without persisting patient state.

### 2. Single Patient Risk Inference (`views/single_prediction.py`)
- **12 Required Model Features:**
  - *Demographics:* `age` [18–100]
  - *Vitals:* `heart_rate` [25–240], `systolic_bp` [40–300], `diastolic_bp` [20–200], `spo2` [50–100], `respiratory_rate` [3–80], `temperature_c` [30.0–44.0], `lactate` [0.0–30.0]
  - *Clinical History:* `previous_admissions` [0–100], `chronic_conditions_count` [0–50]
  - *Observation Context:* `observation_hour` [0–23], `prior_observation_count` [0–100,000]
- **Demonstration Presets:**
  - *Typical / Low Risk Case*
  - *Elevated / Critical Case*
  - *Borderline / Moderate Case*
  - *Random Synthetic Case Generator*
- **Prediction Result Panel:**
  - Large risk status card (Low Risk / High Risk: Review Flag)
  - Percentage admission probability
  - Progress bar calibrated against the `0.70` decision threshold
  - Latency in milliseconds and experimental indicator

### 3. Batch Risk Inference (`views/batch_prediction.py`)
- **CSV Upload & Bounds Checking:** Accepts CSV files up to **500 records** (API-enforced upper bound).
- **Automated Validation:** Validates headers, flags missing columns, checks for NaNs or non-numeric entries.
- **Sample CSV Generator:** Built-in downloadable 15-patient valid template (`synthetic_patients_batch_sample.csv`).
- **Interactive Results Table:** Displays patient IDs, probabilities, prediction labels, and review flags.
- **CSV Export:** Allows downloading the scored results as a clean CSV file.

### 4. Model Information & Governance (`views/model_info.py`)
- Distinguishes API liveness, artifact availability, and evaluation maturity.
- Displays full architecture specifications (`LogisticRegression`, `L2 penalty`, `lbfgs solver`).
- Validated feature allowlist with leak-prevention explanations.
- Evaluation report analysis detailing single-class limitations.

### 5. System Status & Diagnostics (`views/system_status.py`)
- Probes `/health`, `/ready`, `/model-info`, and `/metadata`.
- Measures real-time latencies (ms) for each route.
- Inspects JSON payloads in interactive tabs.
- Includes a diagnostic remediation checklist.

### 6. About & Architecture (`views/about.py`)
- End-to-end BDA lifecycle narrative covering Phases 1 through 7.
- Technology stack overview and governance details.

---

## Windows PowerShell Verification Commands

All commands below are strictly formatted for **Windows PowerShell**:

### 1. Docker Compose Verification (Recommended)

```powershell
# Navigate to repository root
Set-Location -Path "C:\Users\ANSHUL\BDA-PROJECT\emergency_admission_bda"

# Build API and Dashboard images
docker compose build api dashboard

# Start services in the background
docker compose up -d api dashboard

# Verify container statuses
docker compose ps

# Inspect logs
docker compose logs api --tail 30
docker compose logs dashboard --tail 30
```

- **Dashboard UI:** [http://localhost:8501](http://localhost:8501)
- **FastAPI Documentation:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **Health Endpoint:** [http://localhost:8000/health](http://localhost:8000/health)

### 2. Local Python 3.13 Execution

```powershell
# Navigate to repository root
Set-Location -Path "C:\Users\ANSHUL\BDA-PROJECT\emergency_admission_bda"

# Install dashboard dependencies
py -3.13 -m pip install -r requirements-dashboard.txt

# Set environment variable pointing to the API service
$env:API_BASE_URL="http://localhost:8000"

# Launch the Streamlit dashboard
py -3.13 -m streamlit run app/dashboard/Home.py
```

### 3. Automated Test Suite Execution

```powershell
# Run Phase 7 specific dashboard unit tests
py -3.13 -m pytest tests/test_dashboard_phase7.py -v

# Run the complete regression test suite (105 tests across all phases)
py -3.13 -m pytest tests/ -v
```
