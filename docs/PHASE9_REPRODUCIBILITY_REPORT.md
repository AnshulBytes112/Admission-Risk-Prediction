# Phase 9 — Reproducibility, automated tests and report template

## Demo launcher

From the project root in Windows PowerShell:

```powershell
.\scripts\run_demo.ps1
```

Options:

```powershell
.\scripts\run_demo.ps1 -SkipBuild
.\scripts\run_demo.ps1 -SkipSmokeTest
```

The script validates Compose, generates synthetic data if missing, starts the API and dashboard, and attempts the existing smoke tests. It does not run the full Kafka → HDFS → Spark → ML workflow automatically; the documented hand-off steps are still required.

## API contract tests

Install dependencies in a Python environment that already has `requirements-api.txt` installed:

```powershell
pip install -r requirements-test.txt
pip install -r requirements-api.txt
python -m pytest tests/test_api_contract.py -v
```

Tests cover basic API response contracts, synthetic-data safety metadata, missing required fields and extra-field rejection. They do not require the trained model for these checks, although the metadata test assumes the API can answer without a model file.

If `httpx` or FastAPI versions conflict in your environment, use the project dependency pins and record the actual versions in your report rather than silently skipping failures.

## Model evaluation visualization

After Phase 5 training, inspect `models/evaluation_report.json`. Record:
- split strategy and patient-group isolation,
- target prevalence for each split,
- precision, recall, F1, PR-AUC and ROC-AUC if provided,
- selected estimator and threshold,
- limitations caused by synthetic labels and synthetic feature distributions.

Do not invent metric values. If a metric is absent from the JSON report, report it as unavailable.

## Report template

Use `docs/PROJECT_REPORT_TEMPLATE.md` as the submission draft. Replace each `[FILL IN]` field with results from your actual run. Include screenshots of the running dashboard and API docs, Docker service status, and actual evaluation report. Clearly state that the dataset is synthetic and is not MIMIC-III.

## Repeatability checklist

1. Record the OS, Docker Desktop version, Docker Compose version and Python version.
2. Record the synthetic-data generation command and seed.
3. Save a copy of `models/evaluation_report.json`.
4. Run API tests and retain the terminal output.
5. Run smoke tests and retain the terminal output.
6. Capture screenshots only after the services have started successfully.
7. Never report a failed or unrun test as passing.
