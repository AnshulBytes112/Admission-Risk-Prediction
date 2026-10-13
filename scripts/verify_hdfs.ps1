<#
.SYNOPSIS
    Verifies Hadoop HDFS health, storage layout, manifests, and disk usage (Phase 3).
#>
param(
    [string]$HdfsPath = $env:HDFS_BASE_PATH
)

$ErrorActionPreference = "Stop"

if (-not $HdfsPath) {
    $HdfsPath = "/emergency-admission-bda"
}

Write-Host "================================================================="
Write-Host "  HADOOP HDFS STORAGE & HEALTH VERIFICATION"
Write-Host "================================================================="

Write-Host "`n1. Checking HDFS SafeMode Status..."
docker compose exec -T namenode hdfs dfsadmin -safemode get
if ($LASTEXITCODE -ne 0) {
    throw "NameNode is unavailable or HDFS is unhealthy. Verify container status with 'docker compose ps'."
}

Write-Host "`n2. HDFS Directory Tree ($HdfsPath):"
docker compose exec -T namenode hdfs dfs -ls -R $HdfsPath
if ($LASTEXITCODE -ne 0) {
    throw "Failed to list HDFS path '$HdfsPath'."
}

Write-Host "`n3. HDFS Disk Usage Summary:"
docker compose exec -T namenode hdfs dfs -du -h -s $HdfsPath

Write-Host "`n4. Detailed Partition Usage:"
docker compose exec -T namenode hdfs dfs -du -h $HdfsPath/raw/kafka-events

Write-Host "`n================================================================="
Write-Host "  VERIFICATION COMPLETE"
Write-Host "================================================================="
