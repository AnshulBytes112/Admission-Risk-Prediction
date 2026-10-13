#!/usr/bin/env bash
set -euo pipefail

LOCAL_FILE="${1:-./data/streamed/patient_events.jsonl}"
HDFS_BASE="${2:-/emergency-admission-bda/raw/kafka-events}"

if [[ ! -f "$LOCAL_FILE" ]]; then
  echo "Input file not found: $LOCAL_FILE. Complete Phase 2 first." >&2
  exit 2
fi

# The project ./data directory is mounted read-only in the NameNode at /shared-data.
ABS_FILE="$(cd "$(dirname "$LOCAL_FILE")" && pwd)/$(basename "$LOCAL_FILE")"
PROJECT_ROOT="$(pwd)"
RELATIVE="${ABS_FILE#"$PROJECT_ROOT"/}"
if [[ "$RELATIVE" == "$ABS_FILE" ]]; then
  echo "Input file must be inside the project data directory so the container can read it." >&2
  exit 2
fi
CONTAINER_FILE="/shared-data/${RELATIVE#./}"
PARTITION="$(date -u +%F)"
HDFS_DIR="${HDFS_BASE}/ingestion_date=${PARTITION}"

docker compose exec namenode hdfs dfs -mkdir -p "$HDFS_DIR"
docker compose exec namenode hdfs dfs -put -f "$CONTAINER_FILE" "${HDFS_DIR}/patient_events.jsonl"
docker compose exec namenode hdfs dfs -ls -h "$HDFS_DIR"
docker compose exec namenode hdfs dfs -du -h "${HDFS_DIR}/patient_events.jsonl"
echo "Phase 3 upload completed: ${HDFS_DIR}/patient_events.jsonl"
