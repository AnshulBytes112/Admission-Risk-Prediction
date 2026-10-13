param(
    [string]$ApiBase = "http://localhost:8000",
    [string]$DashboardBase = "http://localhost:8501"
)
$ErrorActionPreference = "Stop"

function Test-HttpGet($Url, $Name) {
    try {
        $r = Invoke-WebRequest -Uri $Url -Method Get -TimeoutSec 8
        Write-Host "[PASS] $Name (HTTP $($r.StatusCode))" -ForegroundColor Green
        return $r
    } catch {
        Write-Host "[FAIL] $Name — $($_.Exception.Message)" -ForegroundColor Red
        return $null
    }
}

Write-Host "Emergency Admission BDA — Phase 8 smoke checks"
Write-Host "This checks service reachability; it does not validate clinical performance."

$compose = docker compose config --quiet 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "[PASS] docker compose config" -ForegroundColor Green
} else {
    Write-Host "[FAIL] docker compose config: $compose" -ForegroundColor Red
    exit 1
}

$health = Test-HttpGet "$ApiBase/health" "API health"
if ($health) {
    try {
        $h = $health.Content | ConvertFrom-Json
        if ($h.status -eq "ok") {
            Write-Host "[PASS] API health payload" -ForegroundColor Green
        } else {
            Write-Host "[FAIL] Unexpected health payload" -ForegroundColor Red
        }
    } catch { Write-Host "[FAIL] Health response was not valid JSON" -ForegroundColor Red }
}

$metaResponse = Test-HttpGet "$ApiBase/metadata" "API metadata"
if ($metaResponse) {
    try {
        $meta = $metaResponse.Content | ConvertFrom-Json
        if ($meta.model_loaded -eq $true) {
            Write-Host "[PASS] Model is loaded: $($meta.model_name)" -ForegroundColor Green
        } else {
            Write-Host "[WARN] API is reachable but no trained model is loaded. Complete Phase 5." -ForegroundColor Yellow
        }
    } catch { Write-Host "[FAIL] Metadata response was not valid JSON" -ForegroundColor Red }
}

$docs = Test-HttpGet "$ApiBase/docs" "FastAPI docs"
$dash = Test-HttpGet $DashboardBase "Streamlit dashboard"

if (-not $health -or -not $docs -or -not $dash) {
    Write-Host "At least one service is unavailable. Inspect: docker compose ps; docker compose logs --tail=100 api dashboard" -ForegroundColor Yellow
    exit 1
}
Write-Host "Smoke checks completed. A passing smoke test does not establish model validity." -ForegroundColor Cyan
