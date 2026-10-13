# Master Phase Implementation Plan

**Project**: Emergency Admission Risk Prediction Using Big Data Analytics  
**Current Milestone**: Phase 1: Project Foundation and Infrastructure  
**Authoritative Status**: Only Phase 1 is verified and active. Subsequent phases represent the incremental roadmap.

---

## 1. Phase Breakdown & Status Matrix

| Phase | Title | Status | Primary Deliverables |
|---|---|---|---|
| **Phase 1** | **Foundation & Synthetic Data Generation** | **COMPLETED** | Repo structure, `.env.example`, generator script, Docker Compose, starter API & dashboard, pytest suite. |
| **Phase 2** | **Kafka Event Streaming** | Planned (Deferred to Phase 2) | Topic creation, Kafka producer/consumer scripts, event serialization, partition routing, replay mode. |
| **Phase 3** | **HDFS Ingestion & Storage** | Planned (Deferred to Phase 3) | Date-partitioned landing zone, HDFS CLI automation, data integrity checks, persistence verification. |
| **Phase 4** | **Spark Processing & Feature Engineering** | Planned (Deferred to Phase 4) | PySpark batch job, missingness imputation, patient longitudinal features, partitioned Parquet output. |
| **Phase 5** | **Model Training & Evaluation** | Planned (Deferred to Phase 5) | Patient-grouped train/test split, Logistic Regression vs. Gradient Boosting, PR-AUC evaluation, model bundle export. |
| **Phase 6** | **FastAPI Model Inference** | Planned (Deferred to Phase 6) | Full prediction endpoints (`/predict`, `/predict/batch`), schema enforcement, threshold review flagging. |
| **Phase 7** | **Streamlit Dashboard Integration** | Planned (Deferred to Phase 7) | End-to-end patient explorer with live model inference invocations, telemetry charts, alert banners. |
| **Phase 8** | **Integration & Smoke Testing** | Planned (Deferred to Phase 8) | End-to-end pipeline test script, multi-service resilience, automated smoke test validation. |
| **Phase 9** | **Reproducibility & Report Preparation** | Planned (Deferred to Phase 9) | Repeatable demo launcher, metric summary reports, academic project documentation package. |
| **Phase 10** | **Configuration Consistency & Hardening** | Planned (Deferred to Phase 10) | Multi-environment validation, configuration auditing, production security hardening guidelines. |

---

## 2. Phase 1 Accomplishments

In Phase 1, the core architectural foundation has been established:

- **Repository Layout**: Clean modular structure separating source code (`app/`), configuration (`config/`), raw/streamed/processed datasets (`data/`), models (`models/`), operational scripts (`scripts/`), test suites (`tests/`), and documentation (`docs/`).
- **Configuration Management**: Created `.env.example` as the authoritative template with deduplicated settings, port declarations, and environment variable fallbacks.
- **Synthetic Data Generation**: Implemented `scripts/generate_synthetic_data.py` producing 15 required clinical features and synthetic target labels using deterministic random seeds and controlled vital sign missingness.
- **Docker Compose Infrastructure**: Verified `docker-compose.yml` declaring Bitnami Kafka, Hadoop NameNode/DataNode, Spark Master/Worker, FastAPI backend, Streamlit dashboard, and on-demand tools profiles.
- **FastAPI Starter**: Developed `app/api/main.py` exposing `GET /`, `GET /health`, and `GET /metadata`, with clear disclaimers and explicit 503 error handling when a model is not yet trained.
- **Streamlit Starter Dashboard**: Implemented `app/dashboard/Home.py` providing an overview, architecture diagram cards, API health checking, dataset exploration, vital sign distributions, and empty-state guidance.
- **Automated Testing Suite**: Implemented `tests/test_data_generator.py` and `tests/test_api.py` covering 22 test scenarios with 100% pass rate.
