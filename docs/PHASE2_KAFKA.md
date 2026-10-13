# Phase 2 — Kafka Streaming

## What this phase adds
- CSV-to-Kafka producer with event keys based on `patient_id`.
- Topic creation (`patient-observations`, 3 partitions, replication factor 1).
- JSON event envelope with schema version, synthetic-data marker, and publish timestamp.
- Consumer group that writes Kafka metadata and event content to JSON Lines.
- Replay, rate limiting, and limited-message options for demos.

## Run on Windows PowerShell
1. Start Docker Desktop.
2. Generate the CSV if needed:
   `docker compose --profile tools run --build --rm data-generator`
3. Start Kafka:
   `docker compose up -d kafka`
4. In terminal 1, start the consumer:
   `docker compose --profile streaming up --build kafka-consumer`
5. In terminal 2, publish the dataset:
   `docker compose --profile tools run --build --rm kafka-producer`
6. Inspect captured events:
   `Get-Content .\data\streamed\patient_events.jsonl -TotalCount 3`

For a short test, override the producer command:
`docker compose --profile tools run --build --rm kafka-producer python scripts/kafka_producer.py --input /project/data/raw/patient_observations.csv --limit 100 --delay-ms 0`

Stop the consumer with Ctrl+C. Output is append-only; for a clean replay, stop the consumer and remove or rename `data/streamed/patient_events.jsonl`.

## Notes
- The producer uses `kafka:29092` inside Docker. Scripts run directly on the host should use `localhost:9092`.
- The consumer writes JSONL as a staging artifact. Phase 3 will move this stream output into HDFS.
- This local configuration is for an academic demo, not production security.
- Data is synthetic; scores/labels are not clinically valid.
