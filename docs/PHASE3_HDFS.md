# Phase 3 — Hadoop HDFS Integration

## Goal
Persist the Kafka consumer output in HDFS, giving the project a distributed-storage layer. This phase uploads the JSONL event batch to HDFS and verifies the destination. Spark processing will be implemented in Phase 4.

## HDFS destination layout

```text
/emergency-admission-bda/
└── raw/
    └── kafka-events/
        └── ingestion_date=YYYY-MM-DD/
            └── patient_events.jsonl
```

The date partition is the UTC upload date. The local `data/streamed/patient_events.jsonl` is mounted read-only into the NameNode container at `/shared-data`, allowing the Hadoop CLI inside that container to copy the file into HDFS.

## Steps (Windows PowerShell)

1. Start Hadoop:

   ```powershell
   docker compose up -d namenode datanode
   docker compose ps
   ```

2. Complete Phase 2 and ensure the consumer has written data:

   ```powershell
   Test-Path .\data\streamed\patient_events.jsonl
   Get-Content .\data\streamed\patient_events.jsonl -TotalCount 2
   ```

3. Upload the captured stream to HDFS:

   ```powershell
   .\scripts\ingest_to_hdfs.ps1
   ```

4. Verify the whole project HDFS directory:

   ```powershell
   .\scripts\verify_hdfs.ps1
   ```

5. Open the NameNode web UI at http://localhost:9870 and inspect the filesystem browser if available.

## Manual commands

```powershell
docker compose exec namenode hdfs dfs -mkdir -p /emergency-admission-bda/raw/kafka-events
docker compose exec namenode hdfs dfs -put -f /shared-data/streamed/patient_events.jsonl /emergency-admission-bda/raw/kafka-events/patient_events.jsonl
docker compose exec namenode hdfs dfs -ls -R /emergency-admission-bda
docker compose exec namenode hdfs dfs -du -h -s /emergency-admission-bda
docker compose exec namenode hdfs dfs -cat /emergency-admission-bda/raw/kafka-events/patient_events.jsonl | head
```

The manual path is not date-partitioned; use the PowerShell script for the partitioned path.

## Troubleshooting

- **NameNode is not ready:** `docker compose logs -f namenode`; wait for its health check to pass.
- **DataNode is not ready:** `docker compose logs -f datanode`; check the `SERVICE_PRECONDITION` and container health.
- **Shared file not found:** confirm the Compose `namenode` service mounts `./data:/shared-data:ro`, and the file exists at `data/streamed/patient_events.jsonl`.
- **No events in the file:** keep the Kafka consumer running, run the producer in a second terminal, and inspect both logs.
- **HDFS SafeMode / not writable:** check `docker compose exec namenode hdfs dfsadmin -safemode get`; allow startup to finish before retrying.
- **Repeated upload:** the script uses `-put -f`, so it replaces the destination file for that ingestion date. The local consumer output remains append-only.

## Big data concepts to explain in the report

- HDFS separates metadata management (NameNode) from block storage (DataNodes).
- HDFS is the distributed-storage layer; Kafka is the event-streaming layer.
- `ingestion_date` partitions help organize batches by arrival date.
- Replication is set to 1 in this single-DataNode local setup to reduce resource usage. It demonstrates HDFS APIs and workflow, not fault tolerance.
- All records remain synthetic and are not MIMIC-III or clinical evidence.
