# Master Phase Implementation Plan

**Project**: Emergency Admission Risk Prediction Using Big Data Analytics  
**Current Milestone**: Phase 4: Apache Spark Data Processing & Feature Engineering  
**Authoritative Status**: Phases 1, 2, 3, and 4 are verified and completed. Subsequent phases represent the incremental roadmap.

---

## 1. Phase Breakdown & Status Matrix

| Phase | Title | Status | Primary Deliverables |
|---|---|---|---|
| **Phase 1** | **Foundation & Synthetic Data Generation** | **COMPLETED** | Repo structure, `.env.example`, generator script, Docker Compose, starter API & dashboard, pytest suite. |
| **Phase 2** | **Kafka Event Streaming & Data Ingestion** | **COMPLETED** | Topic management utility, schema-validated Kafka producer, at-least-once consumer, quarantine handling, JSONL stream archive, pytest suite. |
| **Phase 3** | **HDFS Ingestion & Raw Data Lake** | **COMPLETED** | Date-partitioned raw lake, staged atomic upload, idempotency, SHA-256 verification, dynamic JSON manifests, cross-platform CLI, pytest suite. |
| **Phase 4** | **Spark Processing & Feature Engineering** | **COMPLETED** | PySpark batch processing, schema parsing, bounds enforcement, deduplication, median imputation, time and window features, partitioned Parquet output, quarantine & manifest, pytest suite. |
| **Phase 5** | **Model Training & Evaluation** | Planned (Next Phase) | Patient-grouped train/test split, Logistic Regression vs. Gradient Boosting, PR-AUC evaluation, model bundle export. |
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

---

## 4. Phase 3 Accomplishments

- **Core Ingestion Engine**: Implemented `scripts/ingest_to_hdfs.py` supporting CLI overrides, SHA-256 integrity calculation, schema envelope verification, and SafeMode readiness polling.
- **Data Lake Directory Layout**: Established date-partitioned HDFS layout:
  - Raw lake: `/emergency-admission-bda/raw/kafka-events/ingestion_date=YYYY-MM-DD/patient_events.jsonl`
  - Quarantine: `/emergency-admission-bda/quarantine/kafka-events/ingestion_date=YYYY-MM-DD/quarantine_events.jsonl`
  - Metadata: `/emergency-admission-bda/metadata/ingestion-manifests/ingestion_date=YYYY-MM-DD/manifest.json`
- **Safe Staged Upload & Atomic Commit**: Implemented staged uploads to temporary HDFS paths (`.tmp_<uuid>`), byte size verification against source, and atomic rename (`hdfs dfs -mv`) upon verification.
- **Idempotency & Overwrite Guard**: Re-running ingestion on identical files produces an `idempotent_noop` without re-uploading; conflicting file sizes trigger safety aborts unless explicitly overridden with `--overwrite`.
- **Dynamic Ingestion Manifests**: Publishes machine-readable JSON manifests containing dynamically computed SHA-256 hashes, exact byte counts, valid/invalid record totals, UTC timestamps, and explicit synthetic provenance.
- **Cross-Platform Wrappers**: Implemented `scripts/ingest_to_hdfs.ps1` and `scripts/verify_hdfs.ps1` for Windows, along with `scripts/ingest_to_hdfs.sh` and `scripts/verify_hdfs.sh` for Linux/macOS.
- **Automated Testing**: Implemented `tests/test_hdfs_ingestion.py` with 9 mock unit tests verifying validation, checksums, staging, idempotency, and error handling without requiring live Docker containers.
- **Live Verification**: Successfully verified live HDFS upload, SafeMode health check, recursive listing, and disk usage on Docker NameNode/DataNode containers.

---

## 5. Phase 4 Accomplishments

- **Spark Processing Engine**: Implemented production `scripts/process_hdfs_with_spark.py` using PySpark DataFrame API with explicit `ENVELOPE_SCHEMA` and permissive parsing for corrupt JSON record capture.
- **Data Quality & Bounds Enforcement**: Configured 10 physiological data quality bounds converting out-of-range measurements to NULL for subsequent imputation rather than discarding patient records.
- **Deterministic Deduplication**: Implemented distributed window deduplication partitioned by `event_id`, ordered by `kafka_partition`, `kafka_offset`, and `observation_ts` to eliminate streaming duplicates while preserving patient visit history.
- **Missing-Value Imputation**: Implemented batch median imputation via `approxQuantile` across all 7 vital sign and lab measurements with dynamic tracking of pre-impute null counts.
- **Feature Engineering**: Derived `observation_date` (date partition), `observation_hour` (circadian arrival signal), and `prior_observation_count` (window function partitioned by `patient_id` strictly excluding current visit).
- **Partitioned Parquet Output**: Stored processed features in Apache Parquet partitioned by `observation_date` in HDFS at `/emergency-admission-bda/processed/patient-features`, with automatic export to `data/processed/features.parquet` for immediate Phase 5 modeling.
- **Quarantine & Processing Manifest**: Implemented quarantine output routing for invalid identifiers or unparseable records, and dynamic JSON manifest publication to HDFS and local metadata storage.
- **Cross-Platform Automation**: Added `scripts/process_hdfs_with_spark.ps1`, `scripts/process_hdfs_with_spark.sh`, `scripts/verify_spark.ps1`, and `scripts/verify_spark.sh`.
- **Comprehensive Testing Suite**: Added `tests/test_spark_processing.py` (12 unit tests) bringing total passing tests to 56/56 (100% pass rate).
- **Live Verification**: Executed live Spark job in Docker on real HDFS data lake, verifying 100 raw events -> 50 valid deduplicated rows -> 50+ date partitions in HDFS and local Parquet export.
