param(
    [string]$LocalFile = ".\data\streamed\patient_events.jsonl",
    [string]$HdfsBase = "/emergency-admission-bda/raw/kafka-events"
)
$ErrorActionPreference = "Stop"

if (-not (Test-Path $LocalFile)) {
    throw "Input file not found: $LocalFile. Complete Phase 2 first and ensure the Kafka consumer wrote events."
}

# The data directory is mounted read-only into the NameNode container at /shared-data.
$resolved = (Resolve-Path $LocalFile).Path
$relative = [System.IO.Path]::GetRelativePath((Get-Location).Path, $resolved).Replace("\", "/")
$containerPath = "/shared-data/" + $relative.TrimStart("./")
$partition = (Get-Date).ToUniversalTime().ToString("yyyy-MM-dd")
$hdfsDir = "$HdfsBase/ingestion_date=$partition"

Write-Host "Checking NameNode..."
docker compose exec namenode hdfs dfs -mkdir -p $hdfsDir
if ($LASTEXITCODE -ne 0) { throw "Could not create HDFS directory. Check that Hadoop is healthy." }

$hdfsTarget = "$hdfsDir/patient_events.jsonl"
Write-Host "Copying $containerPath to $hdfsTarget ..."
docker compose exec namenode hdfs dfs -put -f $containerPath $hdfsTarget
if ($LASTEXITCODE -ne 0) { throw "HDFS upload failed. Check the shared mount and input file." }

Write-Host "`nHDFS directory listing:"
docker compose exec namenode hdfs dfs -ls -h $hdfsDir
if ($LASTEXITCODE -ne 0) { throw "Upload may have failed; unable to list destination." }

Write-Host "`nApproximate file size:"
docker compose exec namenode hdfs dfs -du -h $hdfsTarget

Write-Host "`nPhase 3 upload completed. Destination: $hdfsTarget"
