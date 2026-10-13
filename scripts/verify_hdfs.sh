#!/usr/bin/env bash
# Verifies Hadoop HDFS health, storage layout, and disk usage (Phase 3).
set -euo pipefail

HDFS_PATH="${1:-${HDFS_BASE_PATH:-/emergency-admission-bda}}"

echo "================================================================="
echo "  HADOOP HDFS STORAGE & HEALTH VERIFICATION"
echo "================================================================="

echo -e "\n1. Checking HDFS SafeMode Status..."
docker compose exec -T namenode hdfs dfsadmin -safemode get

echo -e "\n2. HDFS Directory Tree ($HDFS_PATH):"
docker compose exec -T namenode hdfs dfs -ls -R "$HDFS_PATH"

echo -e "\n3. HDFS Disk Usage Summary:"
docker compose exec -T namenode hdfs dfs -du -h -s "$HDFS_PATH"

echo -e "\n4. Detailed Partition Usage:"
docker compose exec -T namenode hdfs dfs -du -h "$HDFS_PATH/raw/kafka-events" || true

echo "================================================================="
echo "  VERIFICATION COMPLETE"
echo "================================================================="
