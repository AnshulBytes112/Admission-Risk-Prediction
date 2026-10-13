<#
.SYNOPSIS
    Verifies Phase 4 Spark Parquet feature output, quarantine, and manifests.
#>
param(
    [string]$HdfsBasePath = $env:HDFS_BASE_PATH
)

$ErrorActionPreference = "Stop"

if (-not $HdfsBasePath) {
    $HdfsBasePath = "/emergency-admission-bda"
}

Write-Host "================================================================="
Write-Host "  PHASE 4: APACHE SPARK OUTPUT & PARQUET VERIFICATION"
Write-Host "================================================================="

Write-Host "`n1. Listing Processed Parquet Features in HDFS ($HdfsBasePath/processed/patient-features):"
docker compose exec -T namenode hdfs dfs -ls "$HdfsBasePath/processed/patient-features"

Write-Host "`n2. Sample Date Partitions & Parquet Files:"
docker compose exec -T namenode hdfs dfs -ls -R "$HdfsBasePath/processed/patient-features" | Select-Object -First 15

Write-Host "`n3. Checking Disk Usage for Processed Dataset:"
docker compose exec -T namenode hdfs dfs -du -h -s "$HdfsBasePath/processed/patient-features"

Write-Host "`n4. Checking Processing Manifest in HDFS:"
docker compose exec -T namenode hdfs dfs -ls "$HdfsBasePath/metadata/spark-processing"

Write-Host "`n5. Latest Spark Processing Manifest (Local View):"
$manifestFiles = Get-ChildItem -Path .\data\metadata\spark_manifests\ -Filter "processing_manifest.json" -Recurse | Sort-Object LastWriteTime -Descending
if ($manifestFiles -and $manifestFiles.Count -gt 0) {
    $latestManifest = $manifestFiles[0]
    Write-Host "Reading $($latestManifest.FullName):"
    Get-Content $latestManifest.FullName | ConvertFrom-Json | Format-List
} else {
    Write-Host "No local manifest copy found in data/metadata/spark_manifests/."
}

Write-Host "`n6. Checking Local Parquet Features Export (for Phase 5):"
if (Test-Path .\data\processed\features.parquet) {
    $parquetItems = Get-ChildItem -Path .\data\processed\features.parquet
    Write-Host "Found $($parquetItems.Count) partitions/files in data\processed\features.parquet"
} else {
    Write-Host "Local export data\processed\features.parquet not created yet."
}

Write-Host "`n================================================================="
Write-Host "  VERIFICATION COMPLETE"
Write-Host "================================================================="
