# Project Scope: Emergency Admission Risk Prediction Using Big Data Analytics

## 1. Problem Statement

Emergency departments (EDs) and acute monitoring units manage continuous influxes of patients presenting with diverse conditions and volatile vital signs. Early recognition of clinical deterioration and imminent need for emergency hospital admission within 24 hours of triage allows care teams to proactively allocate inpatient beds, triage critical resources, and prevent adverse outcomes.

However, clinical data streams arrive at high velocity, exhibit variable measurement frequencies, contain missing values from sensor detachment, and require distributed infrastructure to process at scale. This academic project explores an end-to-end Big Data Analytics (BDA) architecture designed to simulate, stream, store, process, and classify acute monitoring records.

---

## 2. Project Objectives

1. **Scalable Big Data Architecture**: Design and implement a containerized big-data pipeline using Docker Compose, integrating Apache Kafka for streaming, Apache Hadoop HDFS for distributed persistence, and Apache Spark / PySpark for distributed feature engineering.
2. **Reproducible Foundation**: Establish clean, modular Python components with configuration driven by environment variables, robust type annotations, and automated testing suites.
3. **Synthetic Clinical Observation Generation**: Provide a deterministic, configurable synthetic clinical generator that produces physiologically plausible vital signs and historical attributes without risking patient privacy.
4. **Machine Learning Pipeline**: Compare baseline and gradient-boosted classification algorithms trained on patient-grouped partitions to avoid data leakage across observations.
5. **Interactive Serving & Observability**: Expose predictions via a RESTful FastAPI backend and an interactive Streamlit analytics dashboard.

---

## 3. Scope and Non-Goals

### In Scope
- Generation of synthetic clinical observation records with controlled missingness.
- Dockerized local development infrastructure (Kafka, Zookeeper/KRaft, Hadoop NameNode, DataNode, Spark Master, Spark Worker, API, Dashboard).
- Automated unit and integration testing without requiring proprietary clinical data or pre-existing models.
- Modular code structure ready for incremental expansion across 10 distinct phases.

### Non-Goals (Out of Scope)
- **No Clinical Decision Support**: The software is strictly an academic demonstration and must **never** be used to guide patient triage, diagnosis, or treatment.
- **No Real Protected Health Information (PHI)**: The project does not ingest, process, or scrape real patient databases (e.g., MIMIC-III, PhysioNet, eICU).
- **No Standalone High-Availability Cloud Deployment**: Local containerization via Docker Compose v2 is the target deployment baseline for Phase 1.
- **No Automatic Model Fabrication**: Phase 1 establishes the infrastructure foundation; machine learning model training is formally scheduled for Phase 5.

---

## 4. Synthetic Data Policy

- All records are fictional and contain `synthetic_data=True`.
- The binary admission target (`admission_within_24h`) is generated via a transparent mathematical logistic function combined with binomial sampling. It does not represent genuine clinical risk.
- The dataset supports configurable row count (`SYNTHETIC_ROWS`), random seed (`SYNTHETIC_SEED`), and output destination (`SYNTHETIC_OUTPUT_PATH`).

---

## 5. System Architecture

```text
+-----------------------------------------------------------------------------------+
|                                BIG DATA PIPELINE                                   |
|                                                                                   |
|  +--------------------+       +-------------------+       +--------------------+  |
|  |  Synthetic Data    | ----> |   Apache Kafka    | ----> |    Hadoop HDFS     |  |
|  |     Generator      |       |  (Event Stream)   |       |  (Raw Data Lake)   |  |
|  +--------------------+       +-------------------+       +--------------------+  |
|                                                                     |             |
|                                                                     v             |
|  +--------------------+       +-------------------+       +--------------------+  |
|  | Streamlit Dashboard| <---> |  FastAPI Backend  | <---- |    Apache Spark    |  |
|  | (UI & Explorer)    |       |  (Inference API)  |       | (Batch Processing) |  |
|  +--------------------+       +-------------------+       +--------------------+  |
+-----------------------------------------------------------------------------------+
```

---

## 6. Technology Stack

- **Execution Environment**: Python 3.11 / Python 3.13 on Windows PowerShell / Docker Desktop Linux Containers
- **Container Orchestration**: Docker Compose v2 with multi-stage Dockerfiles and profiles
- **Message Broker**: Apache Kafka (Bitnami 3.8 KRaft mode)
- **Distributed Storage**: Apache Hadoop HDFS 3.2.1 (NameNode & DataNode)
- **Distributed Analytics**: Apache Spark 3.5 / PySpark
- **Backend API**: FastAPI 0.115+, Uvicorn, Pydantic v2
- **Frontend Dashboard**: Streamlit 1.39+, Plotly Express
- **Machine Learning & Math**: scikit-learn, pandas, NumPy, joblib
- **Testing**: pytest, HTTPX

---

## 7. Ethical and Legal Limitations

1. **Non-Clinical Software**: This project was developed as an educational coursework demonstration in Big Data Analytics. It has not undergone clinical validation, CE-marking, or FDA clearance.
2. **Deterministic Bias**: Synthetic formulas are artificial approximations and must not be interpreted as medical evidence.
3. **Threshold Calibration**: The default risk threshold (`0.7`) is an operational project baseline for software demonstration and alerts, not an established medical cutoff.

---

## 8. Planned Implementation Phases

1. **Phase 1: Project Foundation and Infrastructure** (Completed)
2. **Phase 2: Kafka Streaming Ingestion** (Planned)
3. **Phase 3: Hadoop HDFS Data Lake Storage** (Planned)
4. **Phase 4: Apache Spark Distributed Feature Engineering** (Planned)
5. **Phase 5: Machine Learning Model Training and Evaluation** (Planned)
6. **Phase 6: FastAPI Model Inference Service** (Planned)
7. **Phase 7: Streamlit Interactive Analytics Dashboard** (Planned)
8. **Phase 8: End-to-End Integration and Smoke Testing** (Planned)
9. **Phase 9: Reproducibility, Benchmark Reports, and Submission Package** (Planned)
10. **Phase 10: Configuration Hardening and Multi-Environment Deployment** (Planned)
