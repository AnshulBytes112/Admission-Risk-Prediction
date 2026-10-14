# Emergency Admission Risk Prediction — BDA Project

A local-first academic Big Data Analytics project scaffold for the topic:

**Emergency Admission Risk Prediction**

The first phase provides the repository foundation, synthetic-data generator, Docker Compose infrastructure, and starter API/dashboard. The dataset is synthetic and is **not MIMIC-III**. No clinical use is intended.

## Architecture

Synthetic CSV → Kafka (Phase 2) → HDFS (Phase 3) → PySpark processing (Phase 4) → ML training/scoring (Phase 5) → FastAPI → Streamlit dashboard.

## Requirements

- Docker Desktop with Docker Compose v2
- Python 3.11+ (optional, for generating data outside Docker)
- At least 8 GB RAM recommended; 12–16 GB is more comfortable for Hadoop + Spark
- Windows users: run commands in PowerShell from this directory

## Phase 1: Quick Start (Windows PowerShell)

### 1. Environment Configuration
Copy the configuration template:
```powershell
Copy-Item .env.example .env
```
Validate the Docker Compose configuration:
```powershell
docker compose config --quiet
```

### 2. Generate the Synthetic Dataset
You can generate the synthetic clinical observation dataset locally or via Docker:

**Option A — Local Python:**
```powershell
py -3.13 scripts\generate_synthetic_data.py --rows 10000 --seed 42 --output data\raw\patient_observations.csv
```

**Option B — Docker Compose Utility Container:**
```powershell
docker compose --profile tools run --build --rm data-generator
```

### 3. Run Automated Tests
Execute the comprehensive test suite (testing data generation, schema constraints, deterministic seeds, and starter API endpoints):
```powershell
py -3.13 -m pytest tests\ -v
```

### 4. Start Infrastructure and Starter Applications
Build and start the services in detached mode:
```powershell
docker compose up --build -d api dashboard
```
Or start all infrastructure services (Kafka, HDFS, Spark, API, Dashboard):
```powershell
docker compose up -d
```

### 5. Access Application and Service Web UIs
| Service | URL / Endpoint | Purpose |
|---|---|---|
| **Streamlit Dashboard** | [http://localhost:8501](http://localhost:8501) | Interactive clinical dataset explorer & architecture cards |
| **FastAPI Docs (Swagger)** | [http://localhost:8000/docs](http://localhost:8000/docs) | REST API endpoints (`/`, `/health`, `/metadata`) |
| **Hadoop NameNode UI** | [http://localhost:9870](http://localhost:9870) | HDFS distributed filesystem explorer |
| **Spark Master UI** | [http://localhost:8080](http://localhost:8080) | PySpark cluster execution monitor |
| **Kafka Broker** | `localhost:9092` | Event streaming broker listener |

### 6. Clean Up and Shutdown
Stop running containers:
```powershell
docker compose down
```
To also purge named Docker volumes (resets Kafka and HDFS storage):
```powershell
docker compose down -v
```

## Important notes

- This scaffold is the starting point. Later phases will implement Kafka ingestion, HDFS writing, Spark transformations, and model training.
- The current API/dashboard are starter surfaces and do not yet provide clinically valid predictions.
- The example target column is generated using transparent synthetic rules for software demonstration. It must not be interpreted as a medical risk model.
- Synthetic patient IDs and records are fictional. Do not load real patient-identifiable data into this demo.
- The threshold `0.7` is configurable and follows the project brief; it is not a clinically validated threshold.
- Hadoop and Spark containers can be resource-intensive. If startup fails, start the services in smaller groups and check `docker compose logs <service>`.


## Phase 2: Kafka Streaming and Data Ingestion (Windows PowerShell)

Phase 2 provides an end-to-end event-driven streaming pipeline:

### 1. Start Kafka Broker
```powershell
docker compose up -d kafka
docker compose ps
```

### 2. Generate Synthetic Observations (if not already present)
```powershell
py -3.13 scripts\generate_synthetic_data.py --rows 1000 --seed 42 --output data\raw\patient_observations.csv
```

### 3. Start the Kafka Consumer (Terminal 1)
**Local Execution:**
```powershell
py -3.13 scripts\kafka_consumer.py --output data\streamed\patient_events.jsonl
```
**Or Containerized:**
```powershell
docker compose --profile streaming up --build kafka-consumer
```

### 4. Stream Observations with the Producer (Terminal 2)
**Local Execution:**
```powershell
py -3.13 scripts\kafka_producer.py --input data\raw\patient_observations.csv --limit 100 --delay-ms 10
```
**Or Containerized:**
```powershell
docker compose --profile tools run --build --rm kafka-producer python scripts/kafka_producer.py --input /project/data/raw/patient_observations.csv --limit 100 --delay-ms 10
```

### 5. Inspect Captured Stream Events
```powershell
# Inspect the first 2 archived JSONL events
Get-Content .\data\streamed\patient_events.jsonl -TotalCount 2 | ConvertFrom-Json | Format-List

# Count total archived events
(Get-Content .\data\streamed\patient_events.jsonl).Count
```

Stop the consumer with `Ctrl+C` in Terminal 1. The consumer flushes remaining buffers, commits safe offsets, and closes cleanly. See [`docs/PHASE2_KAFKA.md`](file:///c:/Users/ANSHUL/BDA-PROJECT/emergency_admission_bda/docs/PHASE2_KAFKA.md) for full architecture and configuration options.


## Phase 3: Hadoop HDFS Ingestion & Raw Data Lake (Windows PowerShell)

Phase 3 provides an idempotent, staged HDFS ingestion pipeline that organizes the Phase 2 Kafka JSONL stream archive into a partitioned Big Data Lake:

### 1. Start Hadoop NameNode & DataNode
```powershell
docker compose up -d namenode datanode
docker compose ps
```

### 2. Verify Stream Archive Exists (from Phase 2)
```powershell
Test-Path .\data\streamed\patient_events.jsonl
Get-Content .\data\streamed\patient_events.jsonl -TotalCount 2
```

### 3. Ingest Stream Archive into HDFS
Run the PowerShell wrapper (or invoke `scripts/ingest_to_hdfs.py` directly):
```powershell
powershell -ExecutionPolicy Bypass -File scripts\ingest_to_hdfs.ps1
```
*Advanced Options (parameter overrides):*
```powershell
# Custom input path, explicit partition date, or forced overwrite:
powershell -ExecutionPolicy Bypass -File scripts\ingest_to_hdfs.ps1 -InputFile .\data\streamed\patient_events.jsonl -IngestionDate "2026-10-14" -Overwrite
```

### 4. Verify HDFS Health and Storage Structure
```powershell
powershell -ExecutionPolicy Bypass -File scripts\verify_hdfs.ps1
```

### 5. Inspect Files in HDFS
```powershell
# List recursive directory tree
docker compose exec namenode hdfs dfs -ls -R /emergency-admission-bda

# View published ingestion manifest
docker compose exec namenode hdfs dfs -cat /emergency-admission-bda/metadata/ingestion-manifests/ingestion_date=2026-10-13/manifest.json

# Check NameNode Web UI
# Open http://localhost:9870 in your browser
```

### Features & Safety Guarantees:
- **Date-Partitioned Landing Zone**: Raw files land under `/emergency-admission-bda/raw/kafka-events/ingestion_date=YYYY-MM-DD/patient_events.jsonl`.
- **Staged Atomic Upload**: Files are written to a `.tmp_<uuid>` path in HDFS, verified for exact byte length, and atomically renamed.
- **Idempotency**: Re-uploading an identical file returns an `idempotent_noop` without re-uploading; conflicting file sizes trigger safety aborts unless `-Overwrite` is passed.
- **Data Quality & Quarantine**: Corrupted records are logged to `quarantine/kafka-events/` with line numbers and reasons.
- **Machine-Readable Manifest**: Produces dynamic JSON manifests with SHA-256 hashes, valid/invalid record counts, and UTC timestamps.
- **Phase 4 Ready**: The raw JSONL layout matches the exact input contract expected by `scripts/process_hdfs_with_spark.py`.

See [`docs/PHASE3_HDFS.md`](file:///c:/Users/ANSHUL/BDA-PROJECT/emergency_admission_bda/docs/PHASE3_HDFS.md) for full architecture diagrams, schema specifications, and Linux/macOS workflows.


## Phase 4: Apache Spark Data Processing & Feature Engineering (Windows PowerShell)

Phase 4 reads the raw Kafka JSONL event archives from HDFS, applies data quality bounds, deduplicates events deterministically, performs batch median imputation, engineers time and window features, and writes a date-partitioned Parquet dataset to HDFS:

### 1. Execute Spark Processing Job
Run the PowerShell wrapper (or invoke Docker Compose directly):
```powershell
powershell -ExecutionPolicy Bypass -File scripts\process_hdfs_with_spark.ps1
```
*Advanced Options (with parameter overrides):*
```powershell
powershell -ExecutionPolicy Bypass -File scripts\process_hdfs_with_spark.ps1 -WriteMode overwrite -Master "local[*]"
```

### 2. Verify Output Parquet, Partitions & Manifests
```powershell
powershell -ExecutionPolicy Bypass -File scripts\verify_spark.ps1
```

### 3. Inspect Processed Dataset via Python / Pandas
```powershell
py -3.13 -c "import pandas as pd; df = pd.read_parquet('data/processed/features.parquet'); print(df.info()); print(df.head(2))"
```

### Features & Data Lake Deliverables:
- **Distributed Parquet Storage**: Written to HDFS at `/emergency-admission-bda/processed/patient-features/observation_date=YYYY-MM-DD/*.parquet` and exported to `data/processed/features.parquet` for immediate Phase 5 modeling.
- **Deterministic Deduplication**: Eliminates streaming duplicate events by `event_id` while preserving longitudinal patient visit history.
- **Data Quality & Imputation**: Bounds checks 10 clinical features; converts out-of-range values to NULL; applies batch median imputation.
- **Engineered Features**: Computes `observation_date`, `observation_hour` (arrival time signal), and `prior_observation_count` (strictly preceding patient visits).
- **Processing Manifest**: Publishes dynamic execution receipts detailing record counts, duplicates removed, and imputation summaries to HDFS and `data/metadata/spark_manifests/`.

See [`docs/PHASE4_SPARK.md`](file:///c:/Users/ANSHUL/BDA-PROJECT/emergency_admission_bda/docs/PHASE4_SPARK.md) for full architecture diagrams, schema specifications, and Linux/macOS workflows.


## Phase 5: Machine Learning Training, Evaluation & Model Registry (Windows PowerShell)

Phase 5 implements a robust, reproducible, leakage-safe machine learning training pipeline consuming the Phase 4 processed Parquet dataset (`data/processed/features.parquet`). It trains and compares three baseline classifiers, enforces patient-level group isolation, evaluates model performance using PR-AUC (Average Precision), generates audit reports and registry metadata, and saves a versioned model bundle compatible with Phase 6 FastAPI inference.

> [!CAUTION]
> **ACADEMIC DEMONSTRATION NOTICE**:
> All patient records and labels in this project are **SYNTHETIC**. The model is an academic big data software engineering demonstration and is **NOT clinically validated**. It must never be used for actual medical decisions or patient triage.

### 1. Install Machine Learning Dependencies
```powershell
py -3.13 -m pip install -r requirements-ml.txt
```

### 2. Run Model Training Pipeline
The training pipeline enforces strict statistical adequacy safeguards. By default, it requires at least 5 positive examples and aborts safely if the dataset cannot support valid multi-class evaluation:

```powershell
# Standard training (fails safely if input data is inadequate):
py -3.13 scripts\train_risk_model.py --input data\processed\features.parquet --output-dir models --seed 42 --threshold 0.70
```

To run a technical pipeline smoke test on an undersized test dataset (such as the current 50-row Spark output with 1 positive case), supply `--allow-small-dataset`. The resulting artifacts are explicitly flagged as experimental:

```powershell
# Experimental smoke-test mode:
py -3.13 scripts\train_risk_model.py --input data\processed\features.parquet --output-dir models --seed 42 --threshold 0.70 --allow-small-dataset
```

### 3. Validate Saved Model Artifacts and API Compatibility
Verify model bundle deserialization, feature ordering, metadata integrity, and inference probability contracts:

```powershell
py -3.13 scripts\validate_model_artifacts.py --model models\risk_model.joblib --metadata models\model_metadata.json
```

### 4. Run Automated ML and API Test Suite
```powershell
py -3.13 -m pytest tests\ -v
```

### Pipeline Architecture & Key Guarantees
- **Feature Allowlist & Leakage Prevention**: Identifiers (`event_id`, `patient_id`), observation timestamps, Kafka metadata, and synthetic flags are strictly excluded from predictors (`ml/features.py`).
- **Patient-Grouped Splitting**: Employs `GroupShuffleSplit` across `patient_id` so observations from the same patient never appear in both training and test/validation splits (`ml/training.py`).
- **Baseline Models Compared**:
  1. *Logistic Regression*: Scaled features with median imputation, balanced class weights, `max_iter=2000`.
  2. *Random Forest*: 100 estimators, balanced subsampling, tree depth controls.
  3. *HistGradientBoosting*: Native histogram binning with balanced class weights.
- **Selection Metric**: Validation PR-AUC (Average Precision) is used as the primary model selection metric to properly handle clinical class imbalance.
- **Published Artifacts**:
  - `models/risk_model.joblib`: Atomic bundle containing trained scikit-learn pipeline, feature list, model name, and threshold.
  - `models/model_metadata.json`: Model hyperparameters, feature schema, dataset SHA-256 fingerprint, environment versions, and synthetic disclaimer.
  - `models/evaluation_report.json`: Machine-readable validation comparison, held-out test metrics, and split statistics.
  - `models/evaluation_report.md`: Human-readable markdown summary with comparison tables and limitations.

See [`docs/PHASE5_ML.md`](file:///c:/Users/ANSHUL/BDA-PROJECT/emergency_admission_bda/docs/PHASE5_ML.md) for complete mathematical rationale, data adequacy analysis, and future dataset scaling instructions.


## Phase 6: FastAPI Model Inference API & Prediction Serving (Windows PowerShell)

Phase 6 implements a robust, asynchronous REST inference service using FastAPI. It loads the Phase 5 trained model bundle (`models/risk_model.joblib`), manages model lifecycles with startup caching, validates patient clinical observations against physiological ranges, prevents NaN/Infinity inputs, maps positive-class probabilities safely via `model.classes_`, and provides liveness (`/health`), readiness (`/ready`), model governance (`/model-info`), and single/batch prediction endpoints (`/predict`, `/predict/batch`).

> [!CAUTION]
> **ACADEMIC DEMONSTRATION NOTICE**:
> All predictions and patient telemetry values are **SYNTHETIC**. The model is an academic big data software engineering demonstration and is **NOT clinically validated**. Do not use for medical decisions or patient triage.

### 1. Run API in Docker
Build and start the container in detached mode:
```powershell
docker compose build api
docker compose up -d api
docker compose ps api
```

View live container logs:
```powershell
docker compose logs api --tail 50
```

### 2. Run API Locally with Python (Alternative)
```powershell
py -3.13 -m pip install -r requirements-api.txt
$env:MODEL_PATH="models/risk_model.joblib"
$env:RISK_THRESHOLD="0.70"
py -3.13 -m uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Test Endpoints via Windows PowerShell

**A. Liveness Probe (`GET /health`):**
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/health" -Method Get | ConvertTo-Json
```

**B. Inference Readiness Probe (`GET /ready`):**
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/ready" -Method Get | ConvertTo-Json
```

**C. Model Governance & Metrics (`GET /model-info`):**
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/model-info" -Method Get | ConvertTo-Json -Depth 4
```

**D. Single Patient Risk Prediction (`POST /predict`):**
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

**E. Batch Patient Risk Prediction (`POST /predict/batch`):**
```powershell
$batch = @(
    @{
        patient_id = "SYN-BATCH-01"; age = 55.0; heart_rate = 78.0; systolic_bp = 120.0; diastolic_bp = 80.0
        spo2 = 98.0; respiratory_rate = 16.0; temperature_c = 36.8; previous_admissions = 0.0
        chronic_conditions_count = 1.0; lactate = 1.1; observation_hour = 10.0; prior_observation_count = 0.0
    },
    @{
        patient_id = "SYN-BATCH-02"; age = 82.0; heart_rate = 110.0; systolic_bp = 90.0; diastolic_bp = 60.0
        spo2 = 90.0; respiratory_rate = 24.0; temperature_c = 38.5; previous_admissions = 4.0
        chronic_conditions_count = 4.0; lactate = 3.2; observation_hour = 22.0; prior_observation_count = 5.0
    }
) | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8000/predict/batch" -Method Post -ContentType "application/json" -Body $batch | ConvertTo-Json -Depth 4
```

### Interactive API Documentation
Open [http://localhost:8000/docs](http://localhost:8000/docs) in your browser to explore interactive Swagger UI documentation.

See [`docs/PHASE6_API.md`](file:///c:/Users/ANSHUL/BDA-PROJECT/emergency_admission_bda/docs/PHASE6_API.md) for full request/response schemas, security considerations, and error code specifications.


## Phase 7: Modern Streamlit Analytics Dashboard & FastAPI Integration

Phase 7 provides a modern, responsive healthcare analytics SaaS interface matching the project's visual design reference with a dark navy sidebar, live status cards, experimental model governance badges, single-patient 12-feature prediction form with presets, batch CSV evaluation (up to 500 records), and automated endpoint diagnostics.

Ensure Phase 5 artifacts exist (`models/risk_model.joblib`), then build and run via Docker Compose:

```powershell
docker compose build api dashboard
docker compose up -d api dashboard
docker compose ps
```

Access the web services:
- **Streamlit Analytics Dashboard:** [http://localhost:8501](http://localhost:8501)
- **FastAPI OpenAPI Documentation:** [http://localhost:8000/docs](http://localhost:8000/docs)

Run automated dashboard tests:
```powershell
py -3.13 -m pytest tests/test_dashboard_phase7.py -v
```

See [`docs/PHASE7_DASHBOARD.md`](file:///c:/Users/ANSHUL/BDA-PROJECT/emergency_admission_bda/docs/PHASE7_DASHBOARD.md) for architecture, schema validation, and full execution guidelines.


## Phase 8: Production-Style Academic UI, Dataset Explorer & End-to-End Demo

Phase 8 elevates the platform into a cohesive, demonstrable academic application:
- **Reproducible Synthetic Demo Dataset:** Generated via `scripts/generate_demo_dataset.py`, providing 500 rich patient records and pre-formatted batch prediction templates.
- **Interactive Dataset Explorer:** Paginated inspection, multi-attribute filtering (Age, Hour, Outcome, Patient ID), summary statistics, CSV export, and 1-click feature forwarding to the Single Prediction form.
- **Pipeline Observability Telemetry:** Real-time health scores, Spark & HDFS manifest inspection, quarantine monitoring, and live FastAPI endpoint probing.
- **Batch CSV Evaluation:** Upload and score up to 500 records via `POST /predict/batch` with immediate CSV export.

### Quick Verification & Demonstration Commands (Windows PowerShell)

```powershell
# 1. Generate Demo Dataset and Batch Template
py -3.13 scripts/generate_demo_dataset.py --rows 500 --seed 42

# 2. Build and Start Docker Services
docker compose build api dashboard
docker compose up -d api dashboard

# 3. Check Running Containers
docker compose ps

# 4. Run Automated Test Suite (114 passing tests)
py -3.13 -m pytest tests/ -v
```

Access Web Interfaces:
- **Streamlit Analytics Dashboard:** [http://localhost:8501](http://localhost:8501)
- **FastAPI OpenAPI Documentation:** [http://localhost:8000/docs](http://localhost:8000/docs)

See [`docs/PHASE8_DEMO_GUIDE.md`](file:///c:/Users/ANSHUL/BDA-PROJECT/emergency_admission_bda/docs/PHASE8_DEMO_GUIDE.md) for full walkthroughs, architecture diagrams, and demonstration steps.


## Phase 9: reproducibility and submission materials

Launch the local API/dashboard demo:

```powershell
.\scripts\run_demo.ps1
```

Run API contract tests (with the API and test dependencies installed in your Python environment):

```powershell
pip install -r requirements-api.txt
pip install -r requirements-test.txt
python -m pytest tests/test_api_contract.py -v
```

Use `docs/PROJECT_REPORT_TEMPLATE.md` as a report draft and replace every `[FILL IN]` field with actual results. The launcher does not run the entire Kafka/HDFS/Spark/model pipeline automatically. See `docs/PHASE9_REPRODUCIBILITY_REPORT.md`.


## Phase 10: configuration without editing operational settings

Copy `.env.example` to `.env`, adjust ports, thresholds, paths, data-generation settings and selected model-training parameters, then validate:

```powershell
Copy-Item .env.example .env
docker compose config --quiet
```

Recreate affected services after changes:

```powershell
docker compose up --build -d api dashboard
```

See `docs/PHASE10_CONFIGURATION.md` for configurable settings and the remaining intentional code contracts. The project does not claim that every literal has been removed; feature names, service discovery identifiers and validation rules remain explicit contracts.
