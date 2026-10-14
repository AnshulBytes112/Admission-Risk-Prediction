# Emergency Admission Risk Prediction — Big Data Analytics Platform

An enterprise-grade, academic Big Data Analytics (BDA) and Machine Learning platform that predicts emergency room patient admission risk using distributed streaming, distributed storage, large-scale data processing, gradient-boosted machine learning, and an interactive modern clinical dashboard.

---

> [!CAUTION]
> **ACADEMIC DEMONSTRATION NOTICE**:
> All patient records, clinical vitals, laboratory results, and admission risk labels in this project are **SYNTHETIC**. The platform is an academic Big Data software engineering demonstration and is **NOT clinically validated**. It must never be used for actual medical diagnosis, patient triage, or healthcare decisions.

---

## Table of Contents
1. [Project Overview & Problem Statement](#1-project-overview--problem-statement)
2. [End-to-End System Architecture](#2-end-to-end-system-architecture)
3. [Technology Stack](#3-technology-stack)
4. [Big Data Infrastructure in Detail](#4-big-data-infrastructure-in-detail)
   - [Apache Kafka — Distributed Event Streaming](#apache-kafka--distributed-event-streaming)
   - [Apache Hadoop HDFS — Distributed Data Lake](#apache-hadoop-hdfs--distributed-data-lake)
   - [Apache Spark (PySpark) — Distributed Feature Engineering](#apache-spark-pyspark--distributed-feature-engineering)
   - [Partitioned Parquet — Columnar Storage](#partitioned-parquet--columnar-storage)
5. [Machine Learning Pipeline & Model Governance](#5-machine-learning-pipeline--model-governance)
   - [12 Canonical Clinical Features](#12-canonical-clinical-features)
   - [Data Leakage Prevention](#data-leakage-prevention)
   - [Model Comparison & PR-AUC Selection](#model-comparison--pr-auc-selection)
   - [Dynamic Clinical Decision Threshold Controller](#dynamic-clinical-decision-threshold-controller)
6. [Frontend Application & Page-by-Page Feature Tour](#6-frontend-application--page-by-page-feature-tour)
   - [Page 1: Clinical Risk Dashboard](#page-1-clinical-risk-dashboard-dashboard)
   - [Page 2: Single Patient Risk Assessment](#page-2-single-patient-risk-assessment-single)
   - [Page 3: Batch Triage & CSV Evaluation](#page-3-batch-triage--csv-evaluation-batch)
   - [Page 4: Big Data Architecture & Live Trace](#page-4-big-data-architecture--live-trace-pipeline)
   - [Page 5: Model Governance & Performance Auditing](#page-5-model-governance--performance-auditing-model)
   - [Page 6: System Health & Infrastructure Observability](#page-6-system-health--infrastructure-observability-system)
   - [Page 7: Academic Project Reference & Documentation](#page-7-academic-project-reference--documentation-about)
7. [FastAPI Backend Service & API Contracts](#7-fastapi-backend-service--api-contracts)
8. [Quick Start & Step-by-Step Execution Guide](#8-quick-start--step-by-step-execution-guide)
9. [Project Report Writing Guide & Reference Formulas](#9-project-report-writing-guide--reference-formulas)
10. [Automated Testing & Quality Assurance](#10-automated-testing--quality-assurance)

---

## 1. Project Overview & Problem Statement

### 1.1 The Clinical & Big Data Challenge
Emergency Departments (ED) worldwide experience critical overcrowding, bed shortages, and resource bottlenecks. Unplanned emergency admissions represent a major operational and clinical challenge:
- **Triage Delay:** Overcrowded emergency rooms struggle to rapidly distinguish patients who require acute inpatient admission from those who can be safely stabilized and discharged.
- **Data Velocity & Volume:** Continuous patient vitals (heart rate, blood pressure, SpO2, respiratory rate, temperature) arrive as high-velocity time-series streams from bedside monitors and electronic health record (EHR) systems.
- **Data Heterogeneity & Quality:** Telemetry streams contain missing values, network jitter, out-of-range sensor artifacts, and duplicate packets.
- **Class Imbalance:** In general emergency admissions, acute admissions typically constitute an imbalanced positive class relative to routine discharges.

### 1.2 Project Objectives
1. **High-Throughput Ingestion:** Stream synthetic patient vitals through **Apache Kafka** with partition-key ordering.
2. **Fault-Tolerant Distributed Storage:** Ingest streamed observations into **Hadoop HDFS** with date partitioning, atomic staging, idempotency guarantees, and SHA-256 verification manifests.
3. **Scalable Batch Transformation:** Clean, deduplicate, impute, and engineer temporal/window features using **Apache Spark (PySpark)**, exporting to **Partitioned Parquet**.
4. **Reproducible Machine Learning:** Train and compare multiple ML algorithms (HistGradientBoosting, XGBoost, Random Forest, Logistic Regression) using **PR-AUC (Average Precision)** model selection and **patient-level group isolation** to eliminate data leakage.
5. **Real-Time Serving & Traceability:** Serve low-latency inference via **FastAPI** with live end-to-end Big Data pipeline telemetry.
6. **Clinical SaaS Interface:** Provide an interactive **React.js** frontend with dynamic threshold control, batch CSV processing, and pipeline visualization.

---

## 2. End-to-End System Architecture

```mermaid
graph TD
    subgraph S1["1. Ingestion & Streaming Layer"]
        GEN["Synthetic Data Generator<br/>(scripts/generate_synthetic_data.py)"] -->|Clinical CSV| PROD["Kafka Producer<br/>(scripts/kafka_producer.py)"]
        PROD -->|Key: patient_id| TOPIC["Kafka Broker: patient-observations<br/>(3 Partitions, KRaft Mode, Port 9092)"]
        TOPIC -->|Stream Consumer| CONS["Kafka Consumer<br/>(scripts/kafka_consumer.py)"]
    end

    subgraph S2["2. Distributed Storage Layer (Data Lake)"]
        CONS -->|Append Stream| JSONL["Stream Archive<br/>(data/streamed/patient_events.jsonl)"]
        JSONL -->|Atomic Upload (.tmp)| HDFS["Hadoop HDFS Cluster<br/>(/emergency-admission-bda/raw/kafka-events)"]
        HDFS -->|Integrity Receipt| MAN1["HDFS Ingestion Manifest<br/>(SHA-256 Checksum, Record Counts)"]
    end

    subgraph S3["3. Distributed Processing Layer (PySpark)"]
        HDFS -->|Read JSONL| SPARK["PySpark Processing Engine<br/>(SparkSession 3.5, Port 8080)"]
        SPARK --> DEDUP["Deduplication & Data Quality<br/>(FEATURE_RANGES, Windowing)"]
        DEDUP -->|Clean Feature Vector| PARQ["Partitioned Parquet Dataset<br/>(/emergency-admission-bda/processed/patient-features)"]
        DEDUP -->|Out-of-bounds/Corrupt| QUAR["HDFS Quarantine<br/>(quarantine/spark-processing)"]
        SPARK -->|Execution Audit| MAN2["Spark Processing Manifest<br/>(data/metadata/spark_manifests)"]
    end

    subgraph S4["4. Machine Learning & Model Registry Layer"]
        PARQ -->|Train Split| ML["Model Training Pipeline<br/>(scripts/train_risk_model.py)"]
        ML --> GSS["GroupShuffleSplit on patient_id<br/>(Zero Leakage Across Folds)"]
        GSS --> EVAL["PR-AUC Model Evaluation & Tuning<br/>(HistGradientBoosting, Threshold 0.70)"]
        EVAL --> BUNDLE["Model Bundle Registry<br/>(models/risk_model.joblib + metadata.json)"]
    end

    subgraph S5["5. Real-Time Inference & Clinical Dashboard Layer"]
        BUNDLE --> API["FastAPI Serving Engine<br/>(Port 8000: /predict, /config/threshold)"]
        API -->|Real-time REST + Live Pipeline Trace| REACT["React.js Clinical Dashboard<br/>(Port 3000: SaaS UI, 7 Interactive Pages)"]
    end

    classDef stream fill:#eff6ff,stroke:#2563eb,stroke-width:1.5px,color:#1e3a8a;
    classDef storage fill:#f5f3ff,stroke:#7c3aed,stroke-width:1.5px,color:#4c1d95;
    classDef processing fill:#fdf2f8,stroke:#db2777,stroke-width:1.5px,color:#831843;
    classDef ml fill:#ecfdf5,stroke:#059669,stroke-width:1.5px,color:#064e3b;
    classDef ui fill:#fffbeb,stroke:#d97706,stroke-width:1.5px,color:#78350f;

    class GEN,PROD,TOPIC,CONS stream;
    class JSONL,HDFS,MAN1 storage;
    class SPARK,DEDUP,PARQ,QUAR,MAN2 processing;
    class ML,GSS,EVAL,BUNDLE ml;
    class API,REACT ui;
```

---

## 3. Technology Stack

| Layer | Component | Version / Tech | Role & Architectural Purpose |
|---|---|---|---|
| **Frontend** | **React.js** | React 18 | Modern Single-Page Application (SPA) with 7 distinct clinical views |
| **Build Tool** | **Vite** | Vite 5.x | High-speed frontend bundling, hot module reloading (HMR), reverse proxy |
| **Styling** | **Vanilla CSS** | Modern CSS3 | Custom design system: dark sidebar, glassmorphic cards, responsive layouts |
| **Backend API** | **FastAPI** | FastAPI 0.115+ | High-performance asynchronous REST API for single/batch prediction & telemetry |
| **ASGI Server** | **Uvicorn** | Uvicorn 0.30+ | Asynchronous server implementation supporting workers and hot reload |
| **Data Validation** | **Pydantic** | Pydantic v2 | Strict schema validation, physiological bounds checking, alias reconciliation |
| **Streaming** | **Apache Kafka** | Kafka 3.7 (KRaft) | High-throughput distributed event streaming for patient observation telemetry |
| **Distributed Storage** | **Apache Hadoop** | Hadoop 3.3.6 (HDFS) | Fault-tolerant distributed data lake with NameNode and DataNode clustering |
| **Distributed Processing** | **Apache Spark** | PySpark 3.5.x | Cluster-scale data cleaning, deduplication, median imputation, and windowing |
| **File Format** | **Apache Parquet** | Parquet (Snappy) | Columnar storage format partitioned by `observation_date` |
| **Machine Learning** | **scikit-learn / XGBoost** | scikit-learn 1.5+ | HistGradientBoosting, XGBoost, Random Forest, Logistic Regression |
| **Model Registry** | **Joblib** | Joblib 1.4+ | Atomic serialized model bundle packaging pipeline, threshold, and feature schema |
| **Containerization** | **Docker** | Docker Engine 26+ | Multi-stage Dockerfiles for microservice isolation and reproducible execution |
| **Orchestration** | **Docker Compose** | Docker Compose v2 | Multi-container coordination (Kafka, HDFS, Spark, API, Dashboard, React) |
| **Testing** | **pytest** | pytest 8.x | Comprehensive test suite (114+ passing unit, integration, and contract tests) |

---

## 4. Big Data Infrastructure in Detail

### Apache Kafka — Distributed Event Streaming
- **Role:** Acts as the high-velocity ingestion buffer that decouples hospital patient monitors from downstream storage and analytical engines.
- **Broker Configuration:** Runs in KRaft (Kafka Raft metadata) mode, eliminating the external ZooKeeper dependency.
- **Topic Configuration:**
  - Topic Name: `patient-observations`
  - Partitions: 3 (allows parallel consumption across worker threads).
  - Partition Key: `patient_id` (guarantees that all observations for any individual patient land in the same partition in strict chronological sequence).
- **Producer Architecture (`scripts/kafka_producer.py`):**
  - Reads raw CSV observations and serializes each row into a structured JSON event record.
  - Injects ingestion metadata: `event_id` (UUIDv4), `ingestion_timestamp` (ISO 8601 UTC), and `schema_version` (1.0).
  - Implements configurable batching and delay parameters (`--delay-ms`) for rate simulation.
- **Non-Blocking FastAPI Fallback Check:**
  - The API inference endpoint includes a lightweight socket preflight check (300ms timeout) against `localhost:9092`.
  - If Kafka is unreachable (e.g. running outside Docker), it stages events into a local memory buffer and returns predictions in under 500ms without blocking.

### Apache Hadoop HDFS — Distributed Data Lake
- **Role:** Serves as the immutable, fault-tolerant raw data lake for longitudinal patient observation archives.
- **Cluster Components:**
  - **NameNode:** Master node managing filesystem namespace, block mapping, and metadata (Port 9870 Web UI, Port 9000 RPC).
  - **DataNode:** Worker node storing distributed block replicas (Port 9864 Web UI).
- **HDFS Directory Layout:**
  ```
  /emergency-admission-bda/
  ├── raw/
  │   └── kafka-events/
  │       └── ingestion_date=YYYY-MM-DD/
  │           └── patient_events.jsonl
  ├── processed/
  │   └── patient-features/
  │       └── observation_date=YYYY-MM-DD/
  │           └── part-*.parquet
  ├── metadata/
  │   └── ingestion-manifests/
  │       └── ingestion_date=YYYY-MM-DD/
  │           └── manifest.json
  └── quarantine/
      ├── kafka-events/
      └── spark-processing/
  ```
- **Enterprise Safety & Idempotency Features:**
  - **Staged Atomic Upload:** Files are uploaded to a temporary hidden path (`.tmp_<uuid>`), verified for byte completeness, and atomically renamed into the target directory.
  - **Idempotency Check:** Before writing, the ingestion script (`scripts/ingest_to_hdfs.py`) compares file size and cryptographic hashes. Re-uploading identical data results in an `idempotent_noop` rather than creating duplicate partitions.
  - **Quarantine Zone:** Malformed JSON strings or records violating schema constraints are diverted to `/emergency-admission-bda/quarantine/` with line-level failure annotations.
  - **Verification Manifest:** Every upload writes a machine-readable JSON manifest recording input SHA-256, HDFS file size, record count, and timestamp.

### Apache Spark (PySpark) — Distributed Feature Engineering
- **Role:** Distributed compute engine that transforms raw JSONL events into model-ready analytical feature vectors.
- **Execution Script:** `scripts/process_hdfs_with_spark.py`
- **Key Transformations:**
  1. **Schema Enforcement:** Applies a strict PySpark StructType schema across incoming JSON records.
  2. **Deterministic Deduplication:** Deduplicates streaming events on `event_id` while strictly preserving patient longitudinal visits.
  3. **Data Quality Clamping & Median Imputation:** Checks 10 vital signs against physiological clinical bounds (`FEATURE_RANGES`). Out-of-bounds artifacts are converted to nulls and imputed using batch median values.
  4. **Temporal Feature Engineering:** Extracts arrival `observation_date` (partition column) and `observation_hour` (0–23 circadian rhythm indicator).
  5. **Longitudinal Window Feature:** Computes `prior_observation_count` using a PySpark Window partitioned by `patient_id` and ordered by `observation_timestamp` (strictly counting preceding visits to avoid future leakage).

### Partitioned Parquet — Columnar Storage
- **Format:** Apache Parquet with Snappy compression.
- **Benefits:**
  - **Columnar Projection:** Downstream model training only reads the necessary predictor columns, cutting disk I/O by over 70%.
  - **Partition Pruning:** Date partitioning (`observation_date=YYYY-MM-DD`) enables queries to skip non-relevant date directories.
  - **Strong Typing:** Encodes strict numeric float/double types, eliminating schema drift between Spark and scikit-learn.

---

## 5. Machine Learning Pipeline & Model Governance

### 12 Canonical Clinical Features
The model operates on a strictly validated 12-feature vector:

| # | Feature Name | Description | Clinical Unit | Valid Bound |
|---|---|---|---|---|
| 1 | `age` | Patient chronological age | Years | [18.0, 100.0] |
| 2 | `heart_rate` | Pulse rate | Beats per minute (bpm) | [25.0, 240.0] |
| 3 | `systolic_bp` | Systolic arterial pressure | mmHg | [40.0, 300.0] |
| 4 | `diastolic_bp` | Diastolic arterial pressure | mmHg | [20.0, 200.0] |
| 5 | `spo2` / `oxygen_saturation` | Peripheral blood oxygen saturation | Percentage (%) | [50.0, 100.0] |
| 6 | `respiratory_rate` | Breathing frequency | Breaths / min | [3.0, 80.0] |
| 7 | `temperature_c` | Core body temperature | Degrees Celsius (°C) | [30.0, 44.0] |
| 8 | `previous_admissions` | Prior ED hospitalizations | Count | [0.0, 100.0] |
| 9 | `chronic_conditions_count` | Documented chronic comorbidities | Count | [0.0, 50.0] |
| 10 | `lactate` | Blood serum lactate concentration | mmol/L | [0.0, 30.0] |
| 11 | `observation_hour` | Hour of ED presentation (circadian cycle) | Hour (0–23) | [0.0, 23.0] |
| 12 | `prior_observation_count` | Strictly preceding visits for this patient | Count | [0.0, 100000.0] |

### Data Leakage Prevention
To guarantee valid generalization and prevent test set contamination:
- **Identifier Stripping:** Patient IDs (`patient_id`), record IDs (`event_id`), and timestamps (`observation_timestamp`) are stripped prior to feature matrix construction.
- **Patient-Level Group Isolation (`GroupShuffleSplit`):** Splitting is performed across unique `patient_id` values. Observations from the same patient never appear in both training and evaluation splits, preventing cross-observation patient identity leakage.
- **Preprocessing Encapsulation:** Imputation medians and scaling parameters are computed strictly inside training folds using scikit-learn Pipelines.

### Model Comparison & PR-AUC Selection
Because acute emergency admissions exhibit class imbalance, standard accuracy is misleading (a naive classifier predicting "No Admit" achieves high accuracy while missing critical emergencies). Therefore, models are compared using **PR-AUC (Precision-Recall Area Under Curve / Average Precision)**:

1. **HistGradientBoostingClassifier (Primary Champion):**
   - Employs histogram-based binning of continuous numerical features.
   - Handles nonlinear physiological interactions (e.g. shock index: low BP combined with elevated heart rate).
   - Provides native class weighting (`class_weight='balanced'`).
2. **Random Forest Classifier:**
   - Ensemble of 100 decision trees with balanced subsampling.
3. **Logistic Regression (Linear Baseline):**
   - Robust L2-regularized linear model with standardized features.

### Dynamic Clinical Decision Threshold Controller
- **The Concept:** Traditional ML pipelines hardcode a classification threshold of `0.50`. In emergency medicine, decision thresholds must be calibrated according to hospital capacity, bed availability, and epidemiological alerts.
- **Implementation:**
  - Default threshold: `0.70` (configured in model bundle metadata).
  - Runtime override via `POST /config/threshold`: Allows hospital administrators to adjust the global decision cutoff between `0.01` and `0.99`.
  - Immediate feedback: If threshold is increased to `0.90`, moderate patients (e.g. 87% risk) shift from **ADMIT** to **OBSERVE / DISCHARGE**.
  - Dual control: Supports global runtime threshold, request-level query parameter, and payload-level override.

---

## 6. Frontend Application & Page-by-Page Feature Tour

The React frontend (`frontend/`) is a clinical SaaS analytics platform accessible at `http://localhost:3000`. It contains 7 dedicated navigation pages:

```
[ Navigation Sidebar ]
  ├── 1. Dashboard         -> Clinical Overview & KPI Metrics
  ├── 2. Single Predict    -> 12-Feature Patient Risk Form + Presets + Pipeline Trace
  ├── 3. Batch Triage      -> CSV Upload, Batch Inference, Risk Distribution & Export
  ├── 4. BDA Pipeline      -> Interactive 13-Stage Architecture Diagram & Trace
  ├── 5. Model Governance  -> PR-AUC Metrics, Confusion Matrix, Threshold Curves
  ├── 6. System Health     -> Real-Time Probes (FastAPI, Kafka, HDFS, Spark)
  └── 7. About & Docs      -> Academic Documentation, Data Dictionary & Theory
```

### Page 1: Clinical Risk Dashboard (`dashboard`)
- **Header & Live Status Banner:** Displays backend connection status, active model name (`hist_gradient_boosting`), active decision threshold (`0.70`), and an experimental badge.
- **Top Metric Cards:**
  - *Active Triage Census:* Total evaluated patient encounters.
  - *Predicted Admission Rate:* Percentage of scored patients crossing the admission threshold.
  - *Mean Clinical Risk Score:* Average predicted probability across recent cases.
  - *System Health Index:* End-to-end API response latency and broker readiness.
- **Dynamic Decision Threshold Slider:** Global UI controller with range `0.01` to `0.99`. Moving the slider dynamically calls `POST /config/threshold` on the backend and updates all live prediction cards in real time.
- **Quick Evaluation Panel:** Provides quick-start presets to evaluate representative clinical cases with one click.
- **Recent Patient Activity Table:** Chronological log of recent patient evaluations with risk badges (Red: High/Admit, Yellow: Moderate, Green: Low).

### Page 2: Single Patient Risk Assessment (`single`)
- **12 Canonical Input Controls:** Interactive numeric inputs for all 12 physiological and clinical history features, each equipped with clinical valid-range hints and unit labels.
- **One-Click Clinical Preset Buttons:**
  - **Healthy Adult Preset:** (Age 28, HR 72, BP 118/76, SpO2 98.5%, RR 16, Temp 36.8°C, 0 admissions, 0 conditions, Lactate 1.5) $\rightarrow$ Produces **0.02% risk (NO ADMIT)**.
  - **Septic Shock / Critical Preset:** (Age 75, HR 130, BP 78/45, SpO2 82.0%, RR 32, Temp 39.5°C, 4 admissions, 3 conditions, Lactate 5.5) $\rightarrow$ Produces **99.65% risk (ADMIT)**.
  - **Borderline / Elevated Preset:** (Age 62, HR 108, BP 145/92, SpO2 93%, RR 22, Temp 38.1°C, 2 admissions, 2 conditions, Lactate 3.2) $\rightarrow$ Produces **87.97% risk**.
- **Real-Time Client-Side Validation:** Validates numbers against `FEATURE_RANGES` on keystroke, preventing non-finite or out-of-range payloads.
- **Detailed Prediction Result Card:**
  - Large Risk Probability Gauge (`0.0% – 100.0%`).
  - Clear Clinical Recommendation Banner (`ADMIT TO INPATIENT` vs `OBSERVE / DISCHARGE`).
  - Comparative threshold indicator showing whether risk exceeded the active threshold.
- **Live Big Data Execution Trace Panel (`LivePipelineExecutionTrace.jsx`):**
  - Renders a 5-step trace card directly in response to the prediction:
    1. *Stage 1 — Patient Ingestion:* Schema verification and normalization.
    2. *Stage 2 — Apache Kafka Producer:* Message stream confirmation to topic `patient-observations` with partition and offset telemetry.
    3. *Stage 3 — Hadoop HDFS Archive:* Date-partitioned raw JSONL persistence path and replication factor.
    4. *Stage 4 — Apache Spark & Parquet:* Schema alignment against partitioned Parquet lake.
    5. *Stage 5 — Gradient Boosted Inference:* Final model inference probability and decision cutoff.

### Page 3: Batch Triage & CSV Evaluation (`batch`)
- **CSV Template Download:** Generates and downloads a standardized CSV template containing header columns for all 12 features.
- **Drag-and-Drop Batch File Upload:** Accepts `.csv` files up to 500 patient records.
- **Pre-Flight Parsing & Schema Verification:** Parses rows client-side, flags missing or malformed values, and displays record counts before API submission.
- **Batch Inference Execution:** Sends parsed records to `POST /predict/batch` with progress tracking.
- **Batch Results Summary:**
  - High / Moderate / Low risk count distribution cards.
  - Filterable, sortable interactive results table.
  - Search by `patient_id`.
- **Export Evaluated Cohort:** Exports scored records with computed `risk_probability`, `threshold`, and `admission_decision` back to CSV for clinical report generation.

### Page 4: Big Data Architecture & Live Trace (`pipeline`)
- **Interactive Multi-Stage Node Diagram (`PipelineArchitecture.jsx`):** Visualizes the complete data flow across Kafka, HDFS, Spark, Parquet, ML, and FastAPI with zoom/pan controls and color-coded stages.
- **Guided Presentation Walkthrough Mode (`PipelineWalkthrough.jsx`):** College presentation mode with Previous/Next buttons, structured speaking points, and technical deep-dives for each of the 13 pipeline stages.
- **Component Status Panel (`PipelineStatusPanel.jsx`):** Differentiates live socket/HTTP probes from disk manifests, displaying latency and connection health.
- **JSON Manifest Inspector (`ManifestViewer.jsx`):** In-browser viewer for HDFS ingestion manifests and PySpark feature processing manifests (showing SHA-256 hashes, deduplication counts, and imputation logs).

### Page 5: Model Governance & Performance Auditing (`model`)
- **Model Metadata Card:** Loaded algorithm name, scikit-learn version, feature count, dataset SHA-256 fingerprint, and training date.
- **Validation Evaluation Metrics:** Table displaying PR-AUC (Average Precision), ROC-AUC, Precision, Recall, F1 Score, and Accuracy.
- **Confusion Matrix Visualization:** Interactive 2x2 matrix displaying True Positives, False Positives, True Negatives, and False Negatives.
- **Threshold Sensitivity Curve:** Visual plot demonstrating how precision, recall, and admission rates trade off as the decision threshold moves from 0.1 to 0.9.
- **Feature Importance Breakdown:** Horizontal bar chart illustrating the relative weight of vital signs and historical features in driving admission decisions.
- **Governance Notice:** Full synthetic disclaimer detailing academic demonstration limitations.

### Page 6: System Health & Infrastructure Observability (`system`)
- **Live Infrastructure Probes:**
  - *FastAPI Backend:* Liveness (`/health`) and readiness (`/ready`) response latency.
  - *Apache Kafka Broker:* TCP socket reachability on `localhost:9092`.
  - *Hadoop HDFS NameNode:* HTTP probe on `http://localhost:9870`.
  - *Apache Spark Master:* Cluster status on `http://localhost:8080`.
- **Endpoint Diagnostics Suite:** Interactive test buttons to trigger live probe calls to `/health`, `/ready`, `/model-info`, and `/config/threshold`.
- **Environment Telemetry:** Displays Python version, operating system, working directory, and active model file path.

### Page 7: Academic Project Reference & Documentation (`about`)
- **Theoretical Foundations:** Explains the clinical rationale for emergency risk stratification and BDA architectural choices.
- **Data Dictionary:** Reference table of all 12 features with physical units, normal clinical ranges, and boundary clamping limits.
- **Big Data Engineering Concepts:** Detailed explanations of Kafka partitioning, HDFS block distribution, Spark distributed transformations, and columnar Parquet storage.
- **Citation & Project Scope:** Academic citation guidelines and scope definitions for viva voce presentation.

---

## 7. FastAPI Backend Service & API Contracts

The FastAPI inference service (`app/api/main.py`) runs on port 8000:

### 7.1 Complete API Endpoints Table

| Method | Endpoint | Description | Request Body | Response Codes |
|---|---|---|---|---|
| `GET` | `/` | Service root and welcome banner | None | `200 OK` |
| `GET` | `/health` | Liveness probe for container orchestrators | None | `200 OK` |
| `GET` | `/ready` | Readiness probe; verifies model bundle is loaded | None | `200 OK`, `503 Unavailable` |
| `GET` | `/model-info` | Governance metadata, features, and evaluation metrics | None | `200 OK` |
| `GET` | `/config/threshold` | Retrieve current active decision threshold | None | `200 OK` |
| `POST` | `/config/threshold` | Dynamically update global runtime threshold | `{"threshold": 0.75}` | `200 OK`, `422 Unprocessable` |
| `DELETE`| `/config/threshold` | Reset runtime threshold back to model default | None | `200 OK` |
| `POST` | `/predict` | Predict admission risk for a single patient | `PatientFeatures` JSON | `200 OK`, `422 Unprocessable` |
| `POST` | `/predict/batch` | Predict admission risk for up to 500 patients | Array of `PatientFeatures` | `200 OK`, `422 Unprocessable` |

### 7.2 Sample Single Prediction Payload (`POST /predict`)
```json
{
  "patient_id": "PT-10492",
  "age": 72.0,
  "heart_rate": 115.0,
  "systolic_bp": 95.0,
  "diastolic_bp": 60.0,
  "spo2": 91.0,
  "respiratory_rate": 26.0,
  "temperature_c": 38.6,
  "previous_admissions": 2.0,
  "chronic_conditions_count": 2.0,
  "lactate": 3.8,
  "observation_hour": 14.0,
  "prior_observation_count": 2.0,
  "threshold": 0.70
}
```

### 7.3 Sample Prediction Response
```json
{
  "patient_id": "PT-10492",
  "prediction": 1,
  "risk_probability": 0.941258,
  "risk_percent": 94.13,
  "threshold": 0.7,
  "review_flag": true,
  "model_name": "hist_gradient_boosting",
  "dataset_type": "synthetic",
  "is_experimental": true,
  "pipeline_trace": {
    "stages": [
      {
        "step": 1,
        "name": "Patient Clinical Ingestion",
        "component": "FastAPI Validation Gateway",
        "status": "VALIDATED",
        "color": "#10b981",
        "details": "Normalized 12 canonical features; patient_id PT-10492 isolated."
      },
      {
        "step": 2,
        "name": "Real-Time Event Streaming",
        "component": "Apache Kafka Producer",
        "status": "STREAMED",
        "color": "#3b82f6",
        "details": "Staged to Kafka topic patient-observations (key PT-10492)."
      },
      {
        "step": 3,
        "name": "Distributed Raw Storage",
        "component": "Hadoop HDFS Ingestion",
        "status": "STORED",
        "color": "#8b5cf6",
        "details": "Appended to HDFS raw zone (/emergency-admission-bda/raw/kafka-events/ingestion_date=2026-10-15)."
      },
      {
        "step": 4,
        "name": "Distributed Feature Processing",
        "component": "Apache Spark & Parquet",
        "status": "TRANSFORMED",
        "color": "#f59e0b",
        "details": "Aligned schema to 12 canonical model inputs."
      },
      {
        "step": 5,
        "name": "Hist Gradient Boosting Risk Inference",
        "component": "Hist Gradient Boosting Engine",
        "status": "DECISION_EMITTED",
        "color": "#ef4444",
        "details": "Risk Score: 94.13% vs Threshold 0.7 -> ADMIT"
      }
    ],
    "kafka": {
      "status": "staged_for_kafka",
      "topic": "patient-observations",
      "partition_key": "PT-10492",
      "broker": "localhost:9092"
    },
    "hdfs": {
      "raw_path": "/emergency-admission-bda/raw/kafka-events/ingestion_date=2026-10-15/patient_events.jsonl",
      "namenode_rpc": "hdfs://namenode:9000",
      "namenode_ui": "http://localhost:9870",
      "status": "persisted"
    }
  },
  "notice": "Synthetic academic demonstration only. Not a clinically validated prediction; do not use for medical decisions or triage."
}
```

---

## 8. Quick Start & Step-by-Step Execution Guide

### 8.1 System Requirements
- **Operating System:** Windows 10/11, macOS, or Ubuntu Linux.
- **Python:** Python 3.10 – 3.13.
- **Node.js:** Node 18+ and npm (for React frontend).
- **Docker:** Docker Desktop with Docker Compose v2 (recommended for full Big Data cluster).
- **RAM:** Minimum 8 GB (16 GB recommended for running Kafka, Hadoop NameNode/DataNode, and PySpark simultaneously).

### 8.2 Local Quick Start (FastAPI + React Frontend)

**Terminal 1 — Start FastAPI Backend:**
```powershell
# Navigate to project root
cd C:\Users\ANSHUL\BDA-PROJECT\emergency_admission_bda

# Set Python Path and start Uvicorn
$env:PYTHONPATH = "."
py -3 -m uvicorn app.api.main:app --host 0.0.0.0 --port 8000
```
Backend Swagger API documentation will be available at: **`http://localhost:8000/docs`**.

**Terminal 2 — Start React Clinical Dashboard:**
```powershell
# Navigate to frontend folder
cd C:\Users\ANSHUL\BDA-PROJECT\emergency_admission_bda\frontend

# Install dependencies and launch Vite development server
npm install
npm run dev
```
Frontend will open at: **`http://localhost:3000`**.

---

### 8.3 Full Big Data Pipeline Execution via Docker

To execute the entire end-to-end Big Data lifecycle across Docker containers:

#### Step 1: Start All Infrastructure Services
```powershell
docker compose up -d
docker compose ps
```
*Web Consoles:*
- Hadoop NameNode UI: [http://localhost:9870](http://localhost:9870)
- Spark Master UI: [http://localhost:8080](http://localhost:8080)
- FastAPI Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
- React Frontend: [http://localhost:3000](http://localhost:3000)

#### Step 2: Generate Synthetic Observations (Phase 1)
```powershell
py -3 scripts/generate_synthetic_data.py --rows 10000 --seed 42 --output data/raw/patient_observations.csv
```

#### Step 3: Stream Observations via Kafka Producer & Consumer (Phase 2)
```powershell
# In Terminal A: Run Kafka consumer to write stream archive
py -3 scripts/kafka_consumer.py --output data/streamed/patient_events.jsonl

# In Terminal B: Run Kafka producer to publish records
py -3 scripts/kafka_producer.py --input data/raw/patient_observations.csv --limit 500 --delay-ms 10
```

#### Step 4: Ingest Stream Archive into Hadoop HDFS (Phase 3)
```powershell
powershell -ExecutionPolicy Bypass -File scripts/ingest_to_hdfs.ps1
# Verify HDFS health:
powershell -ExecutionPolicy Bypass -File scripts/verify_hdfs.ps1
```

#### Step 5: Process with Apache Spark (Phase 4)
```powershell
powershell -ExecutionPolicy Bypass -File scripts/process_hdfs_with_spark.ps1
# Verify Spark Parquet output:
powershell -ExecutionPolicy Bypass -File scripts/verify_spark.ps1
```

#### Step 6: Train Machine Learning Model (Phase 5)
```powershell
py -3 scripts/train_risk_model.py --input data/processed/features.parquet --output-dir models --seed 42 --threshold 0.70
```

---

## 9. Project Report Writing Guide & Reference Formulas

This section provides academic formulations, architectural justifications, and statistical equations ready to incorporate into your academic project report, thesis, or viva presentation:

### 9.1 Mathematical Formulations

#### 1. Precision-Recall Area Under the Curve (PR-AUC / Average Precision)
For class-imbalanced clinical prediction, Average Precision summarizes the trade-off across varying decision cutoffs:
$$\text{AP} = \sum_{n} (R_n - R_{n-1}) P_n$$
Where $P_n$ and $R_n$ are precision and recall at the $n$-th threshold.

#### 2. Clinical Decision Rule
Given predicted positive class probability $\hat{p} = P(Y = 1 \mid \mathbf{x})$ and decision threshold $\tau \in [0, 1]$:
$$\hat{Y} = \begin{cases} 1 \quad (\text{ADMIT TO INPATIENT}), & \text{if } \hat{p} \ge \tau \\ 0 \quad (\text{OBSERVE / DISCHARGE}), & \text{if } \hat{p} < \tau \end{cases}$$

#### 3. Standard Clinical Classification Metrics
- **Sensitivity (Recall):** $\text{Recall} = \frac{TP}{TP + FN}$ (fraction of true admissions correctly flagged).
- **Positive Predictive Value (Precision):** $\text{Precision} = \frac{TP}{TP + FP}$ (fraction of flagged patients who truly require admission).
- **F1 Score:** $F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$.
- **False Alarm Rate (Fall-out):** $\text{FPR} = \frac{FP}{FP + TN}$.

---

### 9.2 Big Data Tools Academic Justification

| Technology | Why It Was Chosen Over Alternatives | Academic Justification for Project Report |
|---|---|---|
| **Apache Kafka** | Chosen over REST batching / RabbitMQ | Provides distributed, persistent, partitioned append-only event logging. Keying by `patient_id` ensures chronological telemetry order per patient while enabling horizontal multi-worker consumption. |
| **Hadoop HDFS** | Chosen over standard POSIX filesystem | Delivers distributed fault tolerance through block replication (default factor 3). Decouples storage from compute, serving as the scalable data lake foundation for petabyte-scale healthcare archives. |
| **Apache Spark** | Chosen over standalone Python/Pandas | In-memory distributed computing framework utilizing Directed Acyclic Graphs (DAGs). Capable of parallelizing cleaning, deduplication, and window-based longitudinal aggregations across large clusters without memory bottlenecks. |
| **Apache Parquet** | Chosen over CSV / JSON | Columnar binary format offering dictionary encoding and Snappy compression. Reduces disk footprint by up to 75% and dramatically speeds up machine learning feature reads through column pruning. |
| **HistGradientBoosting** | Chosen over standard Decision Trees | Uses integer-based histogram binning of continuous vital signs (reducing split complexity from $O(n \log n)$ to $O(n)$). Handles nonlinear interactions (e.g. shock vitals) and provides native support for class weighting. |

---

## 10. Automated Testing & Quality Assurance

The repository includes a comprehensive test suite covering schema integrity, data generator determinism, HDFS ingestion staging, PySpark transformations, model evaluation metrics, and FastAPI HTTP contracts:

```powershell
# Run the entire automated test suite:
py -3 -m pytest tests\ -v
```

### Test Suite Structure
- `tests/test_data_generator.py`: Verifies deterministic seed reproducibility and physiological bounds clamping.
- `tests/test_kafka_streaming.py`: Tests Kafka producer serialization, message keying, and consumer buffering.
- `tests/test_hdfs_ingestion.py`: Verifies staged atomic uploads (`.tmp`), idempotent re-runs, and SHA-256 manifests.
- `tests/test_spark_processing.py`: Tests PySpark schema application, windowed visit counts, and median imputation.
- `tests/test_model_selection_and_audit.py`: Tests group isolation (`patient_id`), leakage exclusion, and PR-AUC ranking.
- `tests/test_api_contract.py`: Probes `/health`, `/ready`, `/model-info`, `/config/threshold`, and `/predict` response schemas.

---

## License & Citation
This project is developed for academic coursework in **Big Data Analytics (BDA)**. 
- **Author:** Anshul
- **Repository:** `emergency_admission_bda`
- **Topic:** Emergency Admission Risk Prediction — React + FastAPI + scikit-learn + Kafka + Hadoop + PySpark
