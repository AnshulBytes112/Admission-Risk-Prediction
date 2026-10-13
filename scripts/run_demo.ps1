param(
    [switch]$SkipBuild,
    [switch]$SkipSmokeTest
)
$ErrorActionPreference = "Stop"
Write-Host "Emergency Admission BDA — local demo launcher" -ForegroundColor Cyan
Write-Host "Reminder: this project uses synthetic data only; not for clinical use." -ForegroundColor Yellow

docker compose config --quiet
if ($LASTEXITCODE -ne 0) { throw "docker compose configuration is invalid." }

if (-not (Test-Path ".\data\raw\patient_observations.csv")) {
    Write-Host "Generating synthetic observation data..." -ForegroundColor Cyan
    docker compose --profile tools run --build --rm data-generator
    if ($LASTEXITCODE -ne 0) { throw "Synthetic data generation failed." }
}

if (-not (Test-Path ".\models\risk_model.joblib")) {
    Write-Host "Model bundle not found. The dashboard's model prediction form will return an error until Phase 5 training is completed." -ForegroundColor Yellow
}

if ($SkipBuild) {
    docker compose up -d api dashboard
} else {
    docker compose up --build -d api dashboard
}
if ($LASTEXITCODE -ne 0) { throw "Could not start API/dashboard services." }

Write-Host "Waiting briefly for services to initialize..." -ForegroundColor Cyan
Start-Sleep -Seconds 8
docker compose ps

if (-not $SkipSmokeTest) {
    if (Test-Path ".\scripts\smoke_test.ps1") {
        & ".\scripts\smoke_test.ps1"
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Some smoke checks failed. Inspect: docker compose logs --tail=100 api dashboard" -ForegroundColor Yellow
        }
    }
}

Write-Host "Demo launch attempted. Open http://localhost:8501 and http://localhost:8000/docs." -ForegroundColor Green
