<#
.SYNOPSIS
    Ingests Phase 2 JSONL event stream archive into Hadoop HDFS (Phase 3).
.DESCRIPTION
    Wraps scripts/ingest_to_hdfs.py to execute validation, readiness polling,
    idempotent staged upload, manifest publication, and verification.
#>
param(
    [string]$InputFile = $env:STREAM_OUTPUT_PATH,
    [string]$HdfsBase = $env:HDFS_BASE_PATH,
    [string]$IngestionDate = $env:HDFS_INGESTION_DATE,
    [switch]$Overwrite,
    [int]$TimeoutSeconds = 60
)

$ErrorActionPreference = "Stop"

if (-not $InputFile) {
    $InputFile = ".\data\streamed\patient_events.jsonl"
}
if (-not $HdfsBase) {
    $HdfsBase = "/emergency-admission-bda"
}

if (-not (Test-Path $InputFile)) {
    Write-Error "Input file not found: $InputFile. Complete Phase 2 first and verify that patient_events.jsonl exists."
    exit 1
}

$scriptPath = Join-Path $PSScriptRoot "ingest_to_hdfs.py"
$cmdArgs = @($scriptPath, "--input", $InputFile, "--hdfs-base", $HdfsBase, "--timeout", $TimeoutSeconds)

if ($IngestionDate) {
    $cmdArgs += @("--ingestion-date", $IngestionDate)
}
if ($Overwrite) {
    $cmdArgs += "--overwrite"
}

# Execute via Python launcher or fallback
if (Get-Command py -ErrorAction SilentlyContinue) {
    py -3.13 @cmdArgs
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    python @cmdArgs
} else {
    Write-Error "Python executable not found in PATH. Please install Python 3.11+."
    exit 1
}

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
