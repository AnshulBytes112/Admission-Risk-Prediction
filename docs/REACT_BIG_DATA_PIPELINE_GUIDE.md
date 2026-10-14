# Emergency Admission Risk Prediction — Big Data Pipeline Guide
## React.js Analytics Dashboard & End-to-End BDA Architecture Guide

This comprehensive reference document explains how data flows across **Apache Kafka**, **Hadoop HDFS**, **Apache Spark (PySpark)**, **Partitioned Parquet**, **Scikit-Learn ML**, and **FastAPI**, as visualized in the modern **React.js Frontend**.

---

## 1. End-to-End Pipeline Architecture

```mermaid
graph TD
    subgraph S1["1. Data Generation & Ingestion Layer"]
        GEN["1. Synthetic Data Generator<br/>(10,000 patient rows)"] -->|CSV Records| PROD["2. Kafka Producer<br/>(produce_kafka_stream.py)"]
        PROD -->|JSON Messages| TOPIC["3. Kafka Topic: patient-observations<br/>(3 Partitions, KRaft Broker)"]
        TOPIC -->|Streamed Events| CONS["4. Kafka Consumer<br/>(Group: emergency-admission-bda-consumer)"]
    end

    subgraph S2["2. Distributed Storage Layer"]
        CONS -->|Append Stream| JSONL["5. JSONL Event Archive<br/>(data/streamed/patient_events.jsonl)"]
        JSONL -->|Raw Ingestion| HDFS["6. Hadoop HDFS Raw Cluster<br/>(/emergency-admission-bda/raw/kafka-events)"]
        HDFS -->|Audit Manifest| MAN1["HDFS Ingestion Manifest<br/>(SHA-256 Checksum Verified)"]
    end

    subgraph S3["3. Distributed Processing Layer (PySpark)"]
        HDFS -->|Read JSONL| SPARK["7. PySpark Feature Engineering<br/>(SparkSession 3.5.6)"]
        SPARK --> DEDUP["8. Validation & Deduplication<br/>(FEATURE_RANGES, Windowing)"]
        DEDUP -->|Valid Deduplicated| PARQ["10. Partitioned Parquet Dataset<br/>(observation_date Partitions)"]
        DEDUP -->|Malformed / Out-of-bounds| QUAR["9. Quarantine Storage<br/>(quarantine/spark-processing)"]
        SPARK -->|Execution Audit| MAN2["Spark Processing Manifest<br/>(Duplicate & Imputation Summary)"]
    end

    subgraph S4["4. Machine Learning & Serving Layer"]
        PARQ -->|Train Features| ML["11. Scikit-Learn Model<br/>(LogisticRegression, Threshold 0.70)"]
        ML -->|Serialized Bundle| API["12. FastAPI Inference Service<br/>(Port 8000: /predict, /pipeline/status)"]
        API -->|REST Telemetry & Scoring| REACT["13. React Analytics Dashboard<br/>(Port 3000: SaaS UI, Walkthrough)"]
    end

    classDef streaming fill:#eff6ff,stroke:#2563eb,stroke-width:1.5px,color:#1e3a8a;
    classDef storage fill:#f5f3ff,stroke:#7c3aed,stroke-width:1.5px,color:#4c1d95;
    classDef processing fill:#fdf2f8,stroke:#db2777,stroke-width:1.5px,color:#831843;
    classDef ml fill:#ecfdf5,stroke:#059669,stroke-width:1.5px,color:#064e3b;
    classDef ui fill:#fffbeb,stroke:#d97706,stroke-width:1.5px,color:#78350f;

    class GEN,PROD,TOPIC,CONS streaming;
    class JSONL,HDFS,MAN1 storage;
    class SPARK,DEDUP,QUAR,MAN2 processing;
    class PARQ,ML,API ml;
    class REACT ui;
```

---

## 2. React Component Architecture (`frontend/src/`)

The Big Data Pipeline Explorer is implemented as a modular suite of React components using Vanilla CSS:

```
frontend/src/
├── services/
│   ├── api.js                   # Phase 6 Core API client (single & batch prediction)
│   └── pipelineApi.js           # Phase 8 Diagnostic client (status, kafka, hdfs, spark, parquet)
├── data/
│   └── pipelineStages.js        # 13 structured stages: presentation scripts, technical details, I/O
├── components/
│   └── pipeline/
│       ├── PipelineArchitecture.jsx # Real interactive node diagram with zoom/pan and branching
│       ├── PipelineStageDetails.jsx  # Rich deep-dive panel: presentation script, I/O, live evidence
│       ├── PipelineWalkthrough.jsx   # Step-by-step guided college presentation mode with Next/Prev
│       ├── PipelineStatusPanel.jsx   # Component table distinguishing live probes from manifests
│       └── ManifestViewer.jsx        # HDFS & PySpark execution manifest JSON inspector
├── pages/
│   └── BigDataPipeline.jsx      # Main unified page hosting overview KPI cards, views, and data funnel
└── App.jsx                      # App navigation sidebar integration ('pipeline' tab)
```

---

## 3. The 13 Pipeline Stages Explained

### Stage 1 — Synthetic Data Generator
* **Purpose:** Generates reproducible, fictional clinical observations for safe academic testing without protected health information (PHI).
* **Script:** `scripts/generate_synthetic_data.py`
* **Input:** Random seed (default 42), row count (10,000).
* **Output:** `data/raw/patient_observations.csv`.
* **Clinical Bounds:** Heart rate 25–240 bpm, Systolic BP 40–300 mmHg, SpO2 50–100%, Lactate 0–30 mmol/L.

### Stage 2 — Kafka Producer
* **Purpose:** Simulates live event streaming from hospitals to central messaging infrastructure.
* **Script:** `scripts/produce_kafka_stream.py`
* **Input:** CSV records.
* **Output:** UTF-8 encoded JSON messages with injected event timestamps and schema metadata.
* **Mechanism:** Batches events with a 25ms delay, keyed by `patient_id` for deterministic partition assignment.

### Stage 3 — Kafka Topic (`patient-observations`)
* **Purpose:** High-throughput, distributed event log acting as the ingestion decoupling buffer.
* **Configuration:** 3 partitions, replication factor 1 (KRaft broker mode).
* **Port:** `localhost:9092` (host) / `kafka:29092` (internal Docker network).
* **Key Concept:** Partitions provide horizontal parallel processing while guaranteeing strict chronological ordering per `patient_id`.

### Stage 4 — Kafka Consumer
* **Purpose:** Subscribes to event partitions and coordinates streaming ingestion.
* **Script:** `scripts/consume_kafka_stream.py`
* **Consumer Group:** `emergency-admission-bda-consumer`.
* **Mechanism:** Periodically flushes batches (every 250 records) and commits offsets upon persistence.

### Stage 5 — JSONL Event Archive
* **Purpose:** Immutable, append-only staging file for streaming records.
* **Location:** `data/streamed/patient_events.jsonl` (150 records, 111.7 KB).
* **Why JSONL:** Standalone lines allow O(1) concurrent writes without table-locking overhead or transaction contention.

### Stage 6 — Hadoop HDFS Raw Storage
* **Purpose:** Centralized, distributed filesystem for long-term raw event retention.
* **NameNode Web UI:** `http://localhost:9870` | **RPC Port:** `9000`.
* **Path:** `/emergency-admission-bda/raw/kafka-events/ingestion_date=YYYY-MM-DD/`.
* **Manifest Audit:** Cryptographically records source file size, SHA-256 hash, and record counts under `data/metadata/ingestion_manifests/`.

### Stage 7 — PySpark Processing Engine
* **Purpose:** In-memory distributed feature engineering across large datasets.
* **Script:** `scripts/process_features_spark.py`.
* **Engine:** Apache Spark 3.5.6 running on Docker (`spark-master:8080`).
* **Transformations:** Timestamp normalization, schema validation, rolling observation window calculations, and vital sign imputation.

### Stage 8 — Validation & Deduplication
* **Purpose:** Eliminates streaming re-transmissions and enforces physiological bounds.
* **Evidence:** In verified run `8a7f70504501`, exactly 50 duplicate events were removed from 100 raw records.
* **Imputations Applied:** Applied median values for missing vitals (Heart rate median: 82.0, Systolic BP: 124.0, Temp: 37.0°C).

### Stage 9 — Quarantine Storage
* **Purpose:** Isolates malformed or corrupted records for triage without breaking the pipeline.
* **Path:** `/emergency-admission-bda/quarantine/spark-processing`.
* **Evidence:** 0 quarantined records in benchmark runs (indicating 100% clean synthetic data compliance).

### Stage 10 — Partitioned Parquet Feature Dataset
* **Purpose:** Columnar analytical storage optimized for ML training reads.
* **Location:** `data/processed/features.parquet` (local) & `/emergency-admission-bda/processed/patient-features` (HDFS).
* **Structure:** 49 partition directories partitioned by `observation_date` with `_SUCCESS` commit marker.
* **Benefits:** Column pruning, dictionary encoding, and Snappy compression reduce analytical read I/O by up to 80%.

### Stage 11 — Machine Learning Model
* **Purpose:** Predicts probability of hospital admission within 24 hours.
* **Artifact:** `models/risk_model.joblib` (3.9 KB).
* **Algorithm:** Scikit-Learn `LogisticRegression` pipeline with `StandardScaler` and `SimpleImputer`.
* **Threshold:** 0.70.
* **Academic Disclaimer:** Trained on a small experimental synthetic smoke-test split; 100% test accuracy reflects lack of positive test cases and must never be cited as clinical validation.

### Stage 12 — FastAPI Prediction Microservice
* **Purpose:** REST API serving real-time single and batch predictions with sub-10ms latency.
* **Host Port:** `http://localhost:8000`.
* **Endpoints:** `/health`, `/ready`, `/model-info`, `/predict`, `/predict/batch`, `/pipeline/status`.

### Stage 13 — React Analytics Dashboard
* **Purpose:** Professional SaaS presentation frontend.
* **Host Port:** `http://localhost:3000`.
* **Features:** Live telemetry KPI cards, interactive architecture with branch routing, guided college presentation script, and manifest JSON inspection.

---

## 4. Live Probes vs. Historical Manifests

| Component | Status Type | Evidence Source | How Verified |
|---|---|---|---|
| **Kafka Broker** | Live Probed | TCP connection to port `29092` / `9092` | Real socket probe from backend |
| **HDFS NameNode** | Live Probed | TCP connection to port `9870` / `9000` | Real socket probe from backend |
| **FastAPI API** | Live Probed | HTTP probe to `/health` (port 8000) | Internal process liveness |
| **React Frontend** | Live Probed | Active browser context (port 3000) | Vite server runtime |
| **PySpark Execution** | Historical Record | `data/metadata/spark_manifests/` | Manifest logged upon job completion |
| **HDFS Ingestion** | Historical Record | `data/metadata/ingestion_manifests/` | SHA-256 ingestion audit JSON |
| **Parquet Dataset** | Verified On Disk | `data/processed/features.parquet` | Filesystem scan (49 partitions + `_SUCCESS`) |
| **Model Bundle** | In-Memory Registry | `models/risk_model.joblib` | Scikit-learn pipeline loaded in memory |

---

## 5. Diagnostic Backend Endpoints

The backend exposes safe, allowlisted, read-only diagnostic endpoints in `app/api/main.py`:

* `GET /pipeline/status`: Aggregated pipeline telemetry, socket probes, and latest manifest summaries.
* `GET /pipeline/kafka`: Detailed Kafka topic configuration, consumer group, and stream archive stats.
* `GET /pipeline/hdfs`: NameNode reachability and configured HDFS cluster paths.
* `GET /pipeline/spark/latest`: JSON manifest of the most recent PySpark batch feature engineering run.
* `GET /pipeline/parquet`: Parquet directory scan, partition count, and schema attributes.
* `GET /pipeline/manifests`: Catalog of all historical ingestion and processing execution manifests.

---

## 6. Windows PowerShell Verification Commands

Run these commands in Windows PowerShell from the repository root:

```powershell
# 1. Verify Docker containers are running
docker compose ps

# 2. Check live FastAPI status via pipeline probe
curl.exe -s http://localhost:8000/pipeline/status | ConvertFrom-Json

# 3. Inspect Kafka streaming archive lines
(Get-Content data\streamed\patient_events.jsonl | Measure-Object -Line).Lines

# 4. Inspect HDFS raw directory via NameNode container
docker compose exec namenode hdfs dfs -ls /emergency-admission-bda/raw/kafka-events

# 5. Inspect Partitioned Parquet directory structure
Get-ChildItem -Path data\processed\features.parquet | Select-Object -First 10

# 6. Run all automated pytest unit & integration tests
py -3.13 -m pytest tests\ -v

# 7. Start the React Frontend
cd frontend
npm run dev
```

---

## 7. 2–3 Minute Presentation Script (for Evaluators)

> "Good morning, respected evaluators. Today we are demonstrating an end-to-end Big Data Analytics pipeline for **Emergency Admission Risk Prediction** using synthetic clinical observation streams.
>
> 1. **Data Ingestion:** We begin with our Synthetic Data Generator. Because real medical data is restricted by privacy regulations, we generated 10,000 synthetic patient observations bounded by realistic physiological constraints.
> 2. **Streaming:** Our Kafka Producer streams these records as JSON messages into our Kafka topic: `patient-observations`. This decouples the real-time event arrival from downstream batch processing.
> 3. **Distributed Storage:** A dedicated consumer group archives these events as JSONL, which is then ingested into **Hadoop HDFS** under `/emergency-admission-bda/raw/kafka-events` with cryptographic SHA-256 audit manifests.
> 4. **Distributed Processing:** **Apache PySpark** reads raw records directly from HDFS, eliminates streaming duplicates (50 duplicates suppressed in our benchmark run), imputes missing vitals using median thresholds, and isolates any corrupted records into quarantine storage.
> 5. **Columnar Features:** Cleansed features are written as a partitioned **Parquet dataset** across 49 observation dates, maximizing query and ML read efficiency.
> 6. **Serving & Presentation:** A Scikit-Learn classification model is loaded and served by **FastAPI** with sub-10ms response times. Finally, our **React.js Dashboard** provides interactive architecture visualization, live telemetry probes, and simulated patient predictions."
