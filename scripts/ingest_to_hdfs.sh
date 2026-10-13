#!/usr/bin/env bash
# Ingests Phase 2 JSONL stream archive into Hadoop HDFS (Phase 3).
set -euo pipefail

INPUT_FILE="${1:-${STREAM_OUTPUT_PATH:-./data/streamed/patient_events.jsonl}}"
HDFS_BASE="${2:-${HDFS_BASE_PATH:-/emergency-admission-bda}}"
INGESTION_DATE="${3:-${HDFS_INGESTION_DATE:-}}"

if [[ ! -f "$INPUT_FILE" ]]; then
  echo "ERROR: Input file not found: $INPUT_FILE. Complete Phase 2 first." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_CMD="python3"
if ! command -v python3 &>/dev/null; then
  PYTHON_CMD="python"
fi

ARGS=("$SCRIPT_DIR/ingest_to_hdfs.py" "--input" "$INPUT_FILE" "--hdfs-base" "$HDFS_BASE")
if [[ -n "$INGESTION_DATE" ]]; then
  ARGS+=("--ingestion-date" "$INGESTION_DATE")
fi

"$PYTHON_CMD" "${ARGS[@]}"
