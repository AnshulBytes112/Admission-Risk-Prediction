# Phase 10 — Configuration and hardcoded operational settings

## Goal
Operational settings are moved into environment variables and `.env.example` so ports, model location, thresholds, dataset generation size/seed, Kafka topic and selected training parameters can be changed without editing application code.

## Setup
Copy the template to `.env` in the project root:

```powershell
Copy-Item .env.example .env
```

Edit `.env` for your machine. Compose reads the `.env` file automatically for `${VARIABLE}` substitutions. Do not commit secrets or real patient data.

## Common configuration
- `API_PORT`, `DASHBOARD_PORT`, `KAFKA_EXTERNAL_PORT`: host ports.
- `MODEL_PATH`, `DATA_DIR`, `RISK_THRESHOLD`: API/model configuration.
- `DASHBOARD_API_URL`: dashboard-to-API URL from inside Docker; normally `http://api:8000`.
- `API_CORS_ORIGINS`: comma-separated allowed browser origins.
- `KAFKA_TOPIC`, `KAFKA_BOOTSTRAP_SERVERS`, `KAFKA_PARTITIONS`, `KAFKA_PUBLISH_DELAY_MS`: producer defaults.
- `SYNTHETIC_ROWS`, `SYNTHETIC_SEED`: repeatable data generation.
- `TRAIN_RANDOM_SEED`, `RF_N_ESTIMATORS`, `RF_MIN_SAMPLES_LEAF`, `LOGISTIC_MAX_ITER`, `HGB_MAX_ITER`, `HGB_LEARNING_RATE`: selected training parameters.
- `HDFS_RAW_URI`, `HDFS_PROCESSED_URI`, `HDFS_QUARANTINE_URI`: Spark job paths.

After changing environment values, recreate affected services:

```powershell
docker compose up --build -d api dashboard
```

## Important: configuration vs. code contracts
This phase removes configurable operational values from their previous fixed defaults. Some literals remain intentionally:
- Model feature names and API schema are contracts shared by training and inference.
- Internal Compose DNS names such as `api`, `kafka`, and `namenode` are service-discovery identifiers.
- Algorithm choices and schema validation bounds are code-level project behavior.
- Default values in code are safe fallbacks when a variable is omitted; override them in `.env`.

This is not a claim that every literal in the repository has disappeared. For a reliable system, feature/schema contracts should be versioned and validated rather than turned into arbitrary environment strings.

## Verify
```powershell
docker compose config --quiet
docker compose config
```

Review the resolved configuration before running. The resolved output can include environment values; do not paste secrets into public logs. Validate the final environment on your machine because image and service startup behavior depends on Docker Desktop and available resources.
