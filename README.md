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


## Phase 3: Hadoop HDFS storage

Phase 3 adds a repeatable HDFS ingestion workflow for the JSONL file created by the Phase 2 Kafka consumer.

1. Ensure `data/streamed/patient_events.jsonl` exists and contains events.
2. Start Hadoop:

   ```powershell
   docker compose up -d namenode datanode
   docker compose ps
   ```

3. Upload the stream output into a date-partitioned HDFS path:

   ```powershell
   .\scripts\ingest_to_hdfs.ps1
   ```

4. Verify files and storage usage:

   ```powershell
   .\scripts\verify_hdfs.ps1
   ```

5. Optional: open the NameNode UI at http://localhost:9870.

The Compose configuration mounts `./data` read-only into the NameNode container at `/shared-data`. The script uses `hdfs dfs -put -f` and stores the event batch under `/emergency-admission-bda/raw/kafka-events/ingestion_date=YYYY-MM-DD/`. See `docs/PHASE3_HDFS.md` for troubleshooting and manual commands.


## Phase 4: PySpark processing

After Phase 3 has uploaded Kafka JSONL events to HDFS, build and run the Spark processing job:

```powershell
docker compose --profile tools build spark-job
docker compose --profile tools run --rm spark-job
```

Verify the output in HDFS:

```powershell
docker compose exec namenode hdfs dfs -ls -R /emergency-admission-bda/processed/patient-features
docker compose exec namenode hdfs dfs -du -h -s /emergency-admission-bda/processed/patient-features
```

The job reads raw JSONL, parses nested events, quarantines malformed identifiers/timestamps, deduplicates event IDs, checks numeric bounds, imputes selected missing values, creates time/history features, and writes Parquet partitioned by observation date. See `docs/PHASE4_SPARK.md`. In Phase 5, model preprocessing will be fitted only on the training split to avoid leakage.


## Phase 5: machine learning

Phase 5 adds a reproducible scikit-learn training script. It compares Logistic Regression, Random Forest and HistGradientBoosting, selects the best using validation PR-AUC (F1 tie-breaker), evaluates once on the held-out test split, and writes a model bundle plus JSON evaluation report.

1. Export the processed Parquet dataset from HDFS to `data/processed/features.parquet` (instructions in `docs/PHASE5_ML.md`).
2. Install training dependencies:
   `pip install -r requirements-ml.txt`
   `pip install pyarrow`
3. Train:
   `python scripts/train_model.py --input data/processed/features.parquet --output-dir models --threshold 0.7`
4. Review `models/evaluation_report.json` and `models/risk_model.joblib`.

The script uses patient-grouped train/validation/test splits when `patient_id` is present. All data and labels are synthetic, so reported metrics are only a software demonstration and not clinical evidence.


## Phase 6: FastAPI model inference

After Phase 5 creates `models/risk_model.joblib`, start the API:

```powershell
docker compose up --build -d api
```

Open the API docs at http://localhost:8000/docs. The main endpoints are `GET /health`, `GET /metadata`, `POST /predict`, and `POST /predict/batch`. The API mounts `./models` read-only at `/app/models`; if the trained model is missing, prediction returns HTTP 503 rather than silently inventing a score. See `docs/PHASE6_API.md` for a sample request.


## Phase 7: Streamlit dashboard integration

Ensure Phase 5 has generated `models/risk_model.joblib`, then rebuild and start the API and dashboard:

```powershell
docker compose up --build -d api dashboard
```

Open the dashboard at http://localhost:8501. It includes summary metrics, patient search, age filters, observations, charts, and a single-record form that calls the trained model through `POST /predict`. The dashboard labels any legacy heuristic score as illustrative rather than a model prediction. See `docs/PHASE7_DASHBOARD.md`.


## Phase 8: integration and smoke testing

Phase 8 adds a PowerShell smoke-test script, a local-artifact validator, and a Compose configuration fix. These checks verify service reachability and expected files; they do not validate clinical performance.

```powershell
docker compose config --quiet
python scripts/validate_artifacts.py
docker compose up --build -d api dashboard
.\scripts\smoke_test.ps1
```

See `docs/PHASE8_INTEGRATION_TESTING.md` for the full demonstration order and troubleshooting.


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
