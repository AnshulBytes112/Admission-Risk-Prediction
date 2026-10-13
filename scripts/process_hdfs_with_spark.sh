#!/usr/bin/env bash
# Runs Phase 4 Apache Spark processing inside Docker Compose.
set -euo pipefail

INPUT_URI="${1:-${HDFS_RAW_URI:-hdfs://namenode:9000/emergency-admission-bda/raw/kafka-events}}"
OUTPUT_URI="${2:-${HDFS_PROCESSED_URI:-hdfs://namenode:9000/emergency-admission-bda/processed/patient-features}}"
QUARANTINE_URI="${3:-${HDFS_QUARANTINE_URI:-hdfs://namenode:9000/emergency-admission-bda/quarantine/spark-processing}}"
MANIFEST_URI="${4:-${HDFS_MANIFEST_URI:-hdfs://namenode:9000/emergency-admission-bda/metadata/spark-processing}}"
MASTER="${5:-${SPARK_MASTER_URL:-local[*]}}"

echo "================================================================="
echo "  PHASE 4: APACHE SPARK PROCESSING & FEATURE ENGINEERING"
echo "================================================================="
echo "Input HDFS path:      $INPUT_URI"
echo "Output HDFS path:     $OUTPUT_URI"
echo "Quarantine HDFS path: $QUARANTINE_URI"
echo "Spark Master:         $MASTER"
echo "================================================================="

docker compose --profile tools run --rm spark-job \
  /opt/bda/process_hdfs_with_spark.py \
  --master "$MASTER" \
  --input "$INPUT_URI" \
  --output "$OUTPUT_URI" \
  --quarantine "$QUARANTINE_URI" \
  --manifest "$MANIFEST_URI" \
  --write-mode overwrite

echo -e "\n================================================================="
echo "  SPARK PROCESSING COMPLETED SUCCESSFULLY"
echo "================================================================="
