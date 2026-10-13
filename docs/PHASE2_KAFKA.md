# Phase 2 — Kafka Event Streaming and Data Ingestion

**Project Title**: Emergency Admission Risk Prediction Using Big Data Analytics  
**Component**: Apache Kafka Event-Driven Ingestion Pipeline  
**Phase Status**: Implemented and Validated

---

## 1. Architecture and Event Flow

The Phase 2 streaming architecture ingests synthetic clinical observations row-by-row into an event-driven Kafka pipeline:

```text
+-----------------------+
|  Synthetic Data CSV   |  (data/raw/patient_observations.csv)
+-----------------------+
            |
            v
+-----------------------+
|    Kafka Producer     |  (scripts/kafka_producer.py)
| - Validates CSV cols  |
| - Keys on patient_id  |
| - Attaches envelope   |
+-----------------------+
            |
            v  [TCP: localhost:9092 or kafka:29092]
+------------------------------------------------------+
|                     Apache Kafka                     |
| Topic: 'patient-observations'                        |
| Partitions: 3  |  Replication Factor: 1              |
+------------------------------------------------------+
            |
            v  [Consumer Group: emergency-admission-bda-consumer]
+-----------------------+
|    Kafka Consumer     |  (scripts/kafka_consumer.py)
| - Validates JSON/keys |
| - Flushes to disk     |
| - Commits offsets     |
+-----------------------+
       /        \
      /          \
     v            v
+------------------------+    +---------------------------+
| JSONL Stream Archive   |    | JSONL Quarantine Log      |
| (data/streamed/        |    | (data/streamed/           |
|  patient_events.jsonl) |    |  quarantine_events.jsonl) |
| [Phase 3 Hand-off]     |    +---------------------------+
+------------------------+
```

---

## 2. Producer and Consumer Responsibilities

### Producer (`scripts/kafka_producer.py`):
1. **Schema Validation**: Validates that all 15 required clinical features are present in the CSV header before streaming.
2. **Type Casting**: Robustly parses strings into typed integers, floats, booleans, and `None` (JSON `null`).
3. **Partition Ordering**: Uses `patient_id` as the Kafka message key. All observations for a patient map deterministically to the same partition, preserving longitudinal order.
4. **Envelope Packaging**: Wraps the clinical record in an audit envelope with UTC timestamps, unique event keys, and provenance markers.
5. **Delivery Reliability**: Uses `acks="all"`, `retries=10`, and synchronous delivery acknowledgments (`future.get()`).
6. **Graceful Shutdown**: Catches `SIGINT`/`SIGTERM`, flushes pending buffers, and safely closes the producer.

### Consumer (`scripts/kafka_consumer.py`):
1. **Manual Offset Management**: Disables auto-commit (`enable_auto_commit=False`).
2. **At-Least-Once Delivery**: Commits offsets **only after** records are safely written and flushed to disk (`fsync`).
3. **Quarantine Handling**: Catches malformed JSON or records missing `event_id`/`patient_id` without crashing; routes them to a quarantine log.
4. **JSONL Format Enforcement**: Appends single-line UTF-8 JSON records with transport metadata.
5. **Clean Teardown**: Flushes buffers and commits verified offsets upon shutdown signals.

---

## 3. Event Envelope Schema

Events are published to Kafka in a standardized envelope:

```json
{
  "event": {
    "event_id": "EVT000000001",
    "patient_id": "SYN000001",
    "observation_timestamp": "2025-01-01T10:00:00",
    "age": 65,
    "heart_rate": 88.0,
    "systolic_bp": 124.0,
    "diastolic_bp": 78,
    "spo2": 97.0,
    "respiratory_rate": 17,
    "temperature_c": 36.9,
    "previous_admissions": 1,
    "chronic_conditions_count": 2,
    "lactate": 1.4,
    "admission_within_24h": 0,
    "synthetic_data": true,
    "_stream": {
      "event_key": "EVT000000001",
      "published_at_utc": "2026-10-14T00:30:00Z",
      "schema_version": "1.0",
      "data_origin": "synthetic"
    }
  },
  "metadata": {
    "schema_version": "1.0",
    "data_origin": "synthetic",
    "published_at_utc": "2026-10-14T00:30:00Z",
    "event_key": "EVT000000001"
  }
}
```

---

## 4. Topic and Partition Configuration

- **Topic Name**: `patient-observations`
- **Partitions**: `3` (enables parallel consumption and partition scaling)
- **Replication Factor**: `1` (for local single-broker Docker development; a higher factor requires a multi-broker cluster)
- **Topic Initialization**: Automatic via [`scripts/kafka_topic_manager.py`](file:///c:/Users/ANSHUL/BDA-PROJECT/emergency_admission_bda/scripts/kafka_topic_manager.py). Handles concurrent creation races (`TopicAlreadyExistsError`) and logs partition mismatches without destroying existing topic data.

---

## 5. Environment Variables

All parameters are configurable via `.env` / environment variables:

| Variable | Default Value | Description |
|---|---|---|
| `KAFKA_BOOTSTRAP_SERVERS` | `localhost:9092` | Host bootstrap address for local Python scripts. |
| `KAFKA_CONTAINER_BOOTSTRAP` | `kafka:29092` | Internal Docker DNS address for containerized services. |
| `KAFKA_TOPIC` | `patient-observations` | Target topic for clinical observation records. |
| `KAFKA_PARTITIONS` | `3` | Number of topic partitions. |
| `KAFKA_REPLICATION_FACTOR` | `1` | Replication factor. |
| `KAFKA_CONSUMER_GROUP` | `emergency-admission-bda-consumer` | Consumer group identifier. |
| `INPUT_CSV` | `data/raw/patient_observations.csv` | Input synthetic clinical observations file. |
| `KAFKA_PUBLISH_DELAY_MS` | `25` | Delay in milliseconds between streamed events. |
| `KAFKA_BATCH_SIZE` | `16384` | Producer batch buffer size in bytes. |
| `KAFKA_REQUEST_TIMEOUT_MS` | `30000` | Producer request timeout in ms. |
| `KAFKA_RETRY_ATTEMPTS` | `10` | Producer retry count on transient connection failures. |
| `KAFKA_RETRY_BACKOFF_SECONDS`| `3` | Seconds between producer retries. |
| `STREAM_OUTPUT_PATH` | `data/streamed/patient_events.jsonl` | Hand-off destination file for valid consumed events. |
| `STREAM_ERROR_PATH` | `data/streamed/quarantine_events.jsonl` | Quarantine file for malformed or invalid messages. |
| `STREAM_MAX_EVENTS` | `0` | Event limit (0 = process all available records). |
| `STREAM_FLUSH_INTERVAL` | `250` | Records processed per disk flush and offset commit. |
| `KAFKA_AUTO_OFFSET_RESET` | `earliest` | Consumer offset reset policy (`earliest` or `latest`). |
| `KAFKA_POLL_TIMEOUT_MS` | `1000` | Consumer poll wait time in milliseconds. |

---

## 6. Local Windows PowerShell Execution

### Step 1: Start Kafka Broker in Docker
```powershell
docker compose up -d kafka
docker compose ps
```

### Step 2: Ensure Synthetic Dataset Exists
```powershell
py -3.13 scripts\generate_synthetic_data.py --rows 1000 --seed 42 --output data\raw\patient_observations.csv
```

### Step 3: Run the Consumer in Terminal 1
```powershell
py -3.13 scripts\kafka_consumer.py --output data\streamed\patient_events.jsonl
```

### Step 4: Run the Producer in Terminal 2
```powershell
py -3.13 scripts\kafka_producer.py --input data\raw\patient_observations.csv --limit 100 --delay-ms 10
```

---

## 7. Docker-Based Execution

Using Docker Compose profiles to run the containerized pipeline:

### Step 1: Start Kafka Broker
```powershell
docker compose up -d kafka
```

### Step 2: Start Containerized Consumer (Terminal 1)
```powershell
docker compose --profile streaming up --build kafka-consumer
```

### Step 3: Publish Records Using Containerized Producer (Terminal 2)
```powershell
docker compose --profile tools run --build --rm kafka-producer python scripts/kafka_producer.py --input /project/data/raw/patient_observations.csv --limit 100 --delay-ms 5
```

---

## 8. Inspecting Messages and Output Files

### View Archived JSONL Events
```powershell
# Inspect the first 2 JSONL records
Get-Content .\data\streamed\patient_events.jsonl -TotalCount 2 | ConvertFrom-Json | Format-List
```

### Count Total Archived Events
```powershell
(Get-Content .\data\streamed\patient_events.jsonl).Count
```

### Check for Quarantined Records
```powershell
if (Test-Path .\data\streamed\quarantine_events.jsonl) {
    Get-Content .\data\streamed\quarantine_events.jsonl -TotalCount 5
}
```

---

## 9. Stopping Services Safely

- **Consumer**: Send `Ctrl+C` (SIGINT) in the active terminal. The consumer flushes remaining records to disk, commits the highest safely written offset to Kafka, and closes connections cleanly.
- **Producer**: Send `Ctrl+C`. The producer flushes buffered records and prints final publish metrics.
- **Docker Teardown**:
  ```powershell
  docker compose down
  ```

---

## 10. Delivery Guarantees, Offsets, and Duplicates

- **Semantics**: At-least-once delivery.
- **Failure Window**: If the system crashes after writing records to `patient_events.jsonl` but before the consumer offset commit is recorded by Kafka, restarted consumers will re-read those messages from the previous committed offset.
- **Downstream Deduplication**: Every event contains a persistent `event_id`. Downstream Phase 4 PySpark jobs perform deduplication (`dropDuplicates(["event_id"])`) to ensure data integrity.
- **Offset Safety**: Offsets are committed **only after** `out.flush()` and `os.fsync()` successfully write records to host storage.

---

## 11. Quarantine Policy

Messages are routed to `quarantine_events.jsonl` without halting consumer operation when:
1. Message payload cannot be decoded as valid UTF-8 JSON.
2. The JSON payload lacks the outer `event` dictionary.
3. The event dictionary lacks mandatory identifiers (`event_id` or `patient_id`).

Each quarantine record captures the Kafka partition, offset, timestamp, failure reason, and raw preview.

---

## 12. Troubleshooting

| Symptom | Probable Cause | Resolution |
|---|---|---|
| `TimeoutError: Kafka broker at localhost:9092 unavailable` | Kafka container is not running or still initializing. | Run `docker compose up -d kafka` and verify health with `docker compose ps`. |
| `NoBrokersAvailable` inside container | Container trying to connect to `localhost:9092`. | Use internal listener address `kafka:29092` inside Docker networks. |
| `FileNotFoundError: Input CSV not found` | Synthetic data not generated yet. | Run `py -3.13 scripts\generate_synthetic_data.py` first. |
| `Replication factor greater than broker count` | `KAFKA_REPLICATION_FACTOR` set > 1 in single-node setup. | Set `KAFKA_REPLICATION_FACTOR=1`. |

---

## 13. Phase 3 Hand-Off Contract

The Phase 2 consumer produces the canonical hand-off file for Phase 3:
- **Location**: `data/streamed/patient_events.jsonl` (configurable via `STREAM_OUTPUT_PATH`)
- **Format**: Valid JSON Lines (UTF-8, one object per line)
- **Top-Level Keys**: `kafka_topic`, `kafka_partition`, `kafka_offset`, `kafka_key`, `consumed_at_utc`, `event`, `metadata`
- **Nested Event Keys**: Contains all 15 clinical columns plus `_stream` provenance metadata, matching PySpark's expected schema for Phase 4.
