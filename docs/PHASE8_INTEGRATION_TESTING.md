# Phase 8 — Integration, smoke checks and reproducible demo

Phase 8 adds pre-demo checks and fixes a duplicate `environment` mapping in the API Compose service so `DATA_DIR`, `MODEL_PATH`, and `RISK_THRESHOLD` are all retained.

## 1. Validate Compose configuration

From the project root in PowerShell:

```powershell
docker compose config --quiet
```

No output and exit code zero means the Compose file parsed successfully. It does not prove every image can start.

## 2. Verify local artifacts

```powershell
python scripts/validate_artifacts.py
```

The script checks for:
- `data/raw/patient_observations.csv`
- `models/risk_model.joblib`
- `models/evaluation_report.json`

A missing artifact is reported, not generated automatically.

## 3. Start the API and dashboard

```powershell
docker compose up --build -d api dashboard
docker compose ps
```

Then run the smoke checks:

```powershell
.\scripts\smoke_test.ps1
```

The script checks Compose parsing, API health, metadata, API docs and dashboard reachability. If the model is missing, it reports a warning because `/health` may be healthy even when inference is unavailable.

## 4. Exercise prediction manually

Open `http://localhost:8000/docs`, expand `POST /predict`, and use a fictional record such as:

```json
{
  "patient_id": "SYN-DEMO-001",
  "age": 65,
  "heart_rate": 105,
  "systolic_bp": 100,
  "diastolic_bp": 65,
  "spo2": 94,
  "respiratory_rate": 22,
  "temperature_c": 37.5,
  "previous_admissions": 1,
  "chronic_conditions_count": 2,
  "lactate": 1.5,
  "observation_hour": 12,
  "prior_observation_count": 2
}
```

A valid prediction requires the Phase 5 model bundle to exist and to have compatible feature names. A missing model should produce HTTP 503; a feature incompatibility should produce a clear error rather than a fabricated score.

## 5. End-to-end demonstration order

1. Generate fictional data: `docker compose --profile tools run --build --rm data-generator`
2. Start Kafka: `docker compose up -d kafka`
3. Run the Kafka consumer in one terminal and the producer in another (see `docs/PHASE2_KAFKA.md`).
4. Upload JSONL to HDFS using `scripts/ingest_to_hdfs.ps1`.
5. Run the Spark processing job (see `docs/PHASE4_SPARK.md`).
6. Export processed Parquet and train the model (see `docs/PHASE5_ML.md`).
7. Start API and dashboard, then run `scripts/smoke_test.ps1`.

The complete path can require manual hand-offs between local files and HDFS; do not claim it is a single automatic pipeline unless you have configured and verified those hand-offs in your environment.

## Limitations and safety

- All patient records and target labels are synthetic. This project is not MIMIC-III and does not establish medical performance.
- Smoke checks test reachability and artifact presence only.
- Model metrics from synthetic data are software demonstrations, not clinical evidence.
- Never use the output to make real triage, diagnosis, or treatment decisions.
