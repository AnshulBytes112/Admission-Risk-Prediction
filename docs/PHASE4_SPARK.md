# Phase 4 — PySpark Data Processing

## Objective
Use Spark to read the raw Kafka JSONL event batches from HDFS, validate records, deduplicate events, handle missing/out-of-range values, engineer time/history features, and write a Parquet feature dataset back to HDFS.

## Data flow

```text
HDFS raw JSONL
    → Spark JSON reader
    → schema/type conversion
    → quarantine invalid IDs/timestamps
    → deduplicate event_id
    → quality bounds and median imputation
    → timestamp and patient-history features
    → HDFS Parquet partitioned by observation_date
```

## Run in Docker

1. Complete Phase 2 and Phase 3 first.
2. Make sure HDFS contains raw events. Verify:
   ```powershell
   docker compose exec namenode hdfs dfs -ls -R /emergency-admission-bda/raw
   ```
3. Build the Spark image:
   ```powershell
   docker compose --profile tools build spark-job
   ```
4. Run the Spark job:
   ```powershell
   docker compose --profile tools run --rm spark-job
   ```
5. Verify output:
   ```powershell
   docker compose exec namenode hdfs dfs -ls -R /emergency-admission-bda/processed/patient-features
   ```
6. Inspect size:
   ```powershell
   docker compose exec namenode hdfs dfs -du -h -s /emergency-admission-bda/processed/patient-features
   ```

## What is written
- Parquet output, partitioned by `observation_date`.
- Quarantine JSON records for invalid patient/event identifiers or invalid timestamps.
- Data quality corrections where values outside broad validation bounds become null.
- Median imputation for selected numeric features, computed over the current batch.

## Important modelling note
Batch medians are used here only to demonstrate the data-cleaning stage. In Phase 5, preprocessing must be fitted on the training set only and applied unchanged to validation/test data to avoid leakage. The label `admission_within_24h` is synthetic and generated from simulation rules; it is not a clinical endpoint.

## Troubleshooting
- **Input path missing:** complete Phase 3 and verify raw HDFS path.
- **HDFS connection refused:** run `docker compose up -d namenode datanode` and check logs.
- **Permission errors:** inspect container logs and HDFS directory ownership.
- **No output rows:** check input JSONL structure; it must have an outer `event` field.
- **Spark startup fails:** check available RAM and `docker compose logs spark-job`; Hadoop/Spark containers can be resource-intensive.
