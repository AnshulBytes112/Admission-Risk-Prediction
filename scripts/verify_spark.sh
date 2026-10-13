#!/usr/bin/env bash
# Verifies Phase 4 Spark Parquet feature output, quarantine, and manifests.
set -euo pipefail

HDFS_BASE="${1:-${HDFS_BASE_PATH:-/emergency-admission-bda}}"

echo "================================================================="
echo "  PHASE 4: APACHE SPARK OUTPUT & PARQUET VERIFICATION"
echo "================================================================="

echo -e "\n1. Listing Processed Parquet Features in HDFS ($HDFS_BASE/processed/patient-features):"
docker compose exec -T namenode hdfs dfs -ls "$HDFS_BASE/processed/patient-features"

echo -e "\n2. Disk Usage for Processed Dataset:"
docker compose exec -T namenode hdfs dfs -du -h -s "$HDFS_BASE/processed/patient-features"

echo -e "\n3. Checking Processing Manifests in HDFS:"
docker compose exec -T namenode hdfs dfs -ls "$HDFS_BASE/metadata/spark-processing"

echo -e "\n================================================================="
echo "  VERIFICATION COMPLETE"
echo "================================================================="
