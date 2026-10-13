<#
.SYNOPSIS
    Runs the Apache Spark batch processing job inside Docker (Phase 4).
.DESCRIPTION
    Launches the Spark container to read raw Kafka JSONL records from HDFS,
    perform validation, deduplication, median imputation, feature engineering,
    and write date-partitioned Parquet datasets to HDFS.
#>
param(
    [string]$InputUri = $env:HDFS_RAW_URI,
    [string]$OutputUri = $env:HDFS_PROCESSED_URI,
    [string]$QuarantineUri = $env:HDFS_QUARANTINE_URI,
    [string]$ManifestUri = $env:HDFS_MANIFEST_URI,
    [string]$Master = $env:SPARK_MASTER_URL,
    [string]$WriteMode = "overwrite"
)

$ErrorActionPreference = "Stop"

if (-not $InputUri) {
    $InputUri = "hdfs://namenode:9000/emergency-admission-bda/raw/kafka-events"
}
if (-not $OutputUri) {
    $OutputUri = "hdfs://namenode:9000/emergency-admission-bda/processed/patient-features"
}
if (-not $QuarantineUri) {
    $QuarantineUri = "hdfs://namenode:9000/emergency-admission-bda/quarantine/spark-processing"
}
if (-not $ManifestUri) {
    $ManifestUri = "hdfs://namenode:9000/emergency-admission-bda/metadata/spark-processing"
}
if (-not $Master) {
    $Master = "local[*]"
}

Write-Host "================================================================="
Write-Host "  PHASE 4: APACHE SPARK PROCESSING & FEATURE ENGINEERING"
Write-Host "================================================================="
Write-Host "Input HDFS path:      $InputUri"
Write-Host "Output HDFS path:     $OutputUri"
Write-Host "Quarantine HDFS path: $QuarantineUri"
Write-Host "Spark Master:         $Master"
Write-Host "Write Mode:           $WriteMode"
Write-Host "================================================================="

# 1. Verify HDFS NameNode & DataNode are running
Write-Host "`n1. Checking Hadoop HDFS readiness..."
$nnStatus = docker compose ps --format json namenode | ConvertFrom-Json
if (-not $nnStatus -or $nnStatus.State -ne "running") {
    Write-Host "Hadoop NameNode is not running. Starting namenode and datanode..."
    docker compose up -d namenode datanode
    Start-Sleep -Seconds 10
}

# 2. Check SafeMode
docker compose exec -T namenode hdfs dfsadmin -safemode get
if ($LASTEXITCODE -ne 0) {
    throw "NameNode is unavailable or HDFS is unhealthy. Verify container status with 'docker compose ps'."
}

# 3. Build & Run the Spark processing job
Write-Host "`n2. Executing Spark Job in Docker..."
$cmdArgs = @(
    "compose", "--profile", "tools", "run", "--rm", "spark-job",
    "/opt/bda/process_hdfs_with_spark.py",
    "--master", $Master,
    "--input", $InputUri,
    "--output", $OutputUri,
    "--quarantine", $QuarantineUri,
    "--manifest", $ManifestUri,
    "--write-mode", $WriteMode
)

& docker @cmdArgs

if ($LASTEXITCODE -ne 0) {
    Write-Error "Spark processing job failed with exit code $LASTEXITCODE."
    exit $LASTEXITCODE
}

Write-Host "`n================================================================="
Write-Host "  SPARK PROCESSING COMPLETED SUCCESSFULLY"
Write-Host "================================================================="
Write-Host "Run .\scripts\verify_spark.ps1 to inspect output Parquet files and manifests."
