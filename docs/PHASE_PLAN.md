# Master Phase Implementation Plan

**Project**: Emergency Admission Risk Prediction Using Big Data Analytics  
**Current Milestone**: Phase 2: Kafka Event Streaming and Data Ingestion  
**Authoritative Status**: Phase 1 and Phase 2 are verified and completed. Subsequent phases represent the incremental roadmap.

---

## 1. Phase Breakdown & Status Matrix

| Phase | Title | Status | Primary Deliverables |
|---|---|---|---|
| **Phase 1** | **Foundation & Synthetic Data Generation** | **COMPLETED** | Repo structure, `.env.example`, generator script, Docker Compose, starter API & dashboard, pytest suite. |
| **Phase 2** | **Kafka Event Streaming & Data Ingestion** | **COMPLETED** | Topic management utility, schema-validated Kafka producer, at-least-once consumer, quarantine handling, JSONL stream archive, pytest suite. |
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

- **Repository Layout**: Clean modular structure separating source code (`app/`), configuration (`config/`), raw/streamed/processed datasets (`data/`), models (`models/`), operational scripts (`scripts/`), test suites (`tests/`), and documentation (`docs/`).
- **Configuration Management**: Created `.env.example` as the authoritative template with deduplicated settings, port declarations, and environment variable fallbacks.
- **Synthetic Data Generation**: Implemented `scripts/generate_synthetic_data.py` producing 15 required clinical features and synthetic target labels using deterministic random seeds and controlled vital sign missingness.
- **Docker Compose Infrastructure**: Verified `docker-compose.yml` declaring Bitnami Kafka, Hadoop NameNode/DataNode, Spark Master/Worker, FastAPI backend, Streamlit dashboard, and on-demand tools profiles.
- **FastAPI Starter**: Developed `app/api/main.py` exposing `GET /`, `GET /health`, and `GET /metadata`, with clear disclaimers and explicit 503 error handling when a model is not yet trained.
- **Streamlit Starter Dashboard**: Implemented `app/dashboard/Home.py` providing an overview, architecture diagram cards, API health checking, dataset exploration, vital sign distributions, and empty-state guidance.
- **Automated Testing Suite**: Implemented `tests/test_data_generator.py` and `tests/test_api.py` covering 22 test scenarios with 100% pass rate.

---

## 3. Phase 2 Accomplishments

- **Topic Management Utility**: Implemented `scripts/kafka_topic_manager.py` with idempotent topic creation, concurrent race handling (`TopicAlreadyExistsError`), partition mismatch detection, and broker polling.
- **Schema-Validated Producer**: Implemented `scripts/kafka_producer.py` with mandatory 15-column header validation, type casting (floats, ints, booleans, nulls), partition keying by `patient_id`, standardized event envelopes with UTC timestamps, delivery acknowledgment tracking, rate limiting, and graceful termination.
- **Resilient Consumer**: Implemented `scripts/kafka_consumer.py` with manual offset management (`enable_auto_commit=False`), at-least-once delivery guarantees (offsets committed strictly after successful disk write and `fsync`), corrupted message quarantine, and graceful shutdown handling.
- **Phase 3 Hand-Off Contract**: Standardized UTF-8 JSON Lines stream archive at `data/streamed/patient_events.jsonl` matching the input schema required for Hadoop HDFS ingestion and PySpark processing.
- **Docker Compose Integration**: Added `kafka-consumer` (`--profile streaming`) and `kafka-producer` (`--profile tools`) container definitions with health-check dependencies on Kafka.
- **Automated Testing**: Added `tests/test_kafka_pipeline.py` with 13 unit and component tests validating serialization, envelopes, headers, quarantine, offset commit sequencing, and topic race handling without requiring external brokers.
