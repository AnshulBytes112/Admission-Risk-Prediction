"""Robust HTTP client for FastAPI inference backend integration.

Provides:
- Configurable endpoint base URL resolution (Docker vs. Local)
- Bounded HTTP request timeouts
- Precise latency measurement (milliseconds)
- Safe error handling and response formatting without leaking secrets
"""
from __future__ import annotations

import os
import time
from typing import Any

import requests


class BackendAPIClient:
    """HTTP client communicating with the Phase 6 FastAPI inference microservice."""

    def __init__(self, base_url: str | None = None, timeout: float = 5.0) -> None:
        if base_url:
            self.base_url = base_url.rstrip("/")
        else:
            # Check container environment first, then fallback to localhost
            env_url = os.getenv("DASHBOARD_API_URL") or os.getenv("API_BASE_URL")
            self.base_url = (env_url or "http://localhost:8000").rstrip("/")
        self.timeout = timeout
        self.timeout_seconds = timeout

    def _get(self, path: str) -> tuple[bool, dict[str, Any], float]:
        """Perform a GET request with latency measurement."""
        url = f"{self.base_url}{path}"
        start_time = time.perf_counter()
        try:
            resp = requests.get(url, timeout=self.timeout)
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 1)
            try:
                data = resp.json()
            except Exception:
                data = {"raw_text": resp.text, "status_code": resp.status_code}
            if resp.status_code >= 400 and "error" not in data:
                detail = data.get("detail", resp.text)
                if resp.status_code == 422:
                    data["error"] = f"Validation Error (422): {detail}"
                elif resp.status_code == 503:
                    data["error"] = f"Service Unavailable (503): {detail}"
                else:
                    data["error"] = f"HTTP Error ({resp.status_code}): {detail}"
            return resp.status_code < 400, data, elapsed_ms
        except requests.exceptions.Timeout:
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 1)
            return False, {"error": "Connection timed out", "status_code": 504}, elapsed_ms
        except requests.exceptions.ConnectionError:
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 1)
            return False, {
                "error": f"Failed to connect to backend API at {self.base_url}. Ensure the service is running.",
                "status_code": 503,
            }, elapsed_ms
        except Exception as exc:
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 1)
            return False, {"error": f"Request failed: {exc}", "status_code": 500}, elapsed_ms

    def _post(self, path: str, json_data: Any) -> tuple[bool, dict[str, Any], float]:
        """Perform a POST request with latency measurement."""
        url = f"{self.base_url}{path}"
        start_time = time.perf_counter()
        try:
            resp = requests.post(url, json=json_data, timeout=self.timeout)
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 1)
            try:
                data = resp.json()
            except Exception:
                data = {"raw_text": resp.text, "status_code": resp.status_code}
            if resp.status_code >= 400 and "error" not in data:
                detail = data.get("detail", resp.text)
                if resp.status_code == 422:
                    data["error"] = f"Validation Error (422): {detail}"
                elif resp.status_code == 503:
                    data["error"] = f"Service Unavailable (503): {detail}"
                else:
                    data["error"] = f"HTTP Error ({resp.status_code}): {detail}"
            return resp.status_code < 400, data, elapsed_ms
        except requests.exceptions.Timeout:
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 1)
            return False, {"error": "Connection timed out", "status_code": 504}, elapsed_ms
        except requests.exceptions.ConnectionError:
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 1)
            return False, {
                "error": f"Failed to connect to backend API at {self.base_url}.",
                "status_code": 503,
            }, elapsed_ms
        except Exception as exc:
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 1)
            return False, {"error": f"Request failed: {exc}", "status_code": 500}, elapsed_ms

    def check_health(self) -> tuple[bool, dict[str, Any], float]:
        """Call GET /health for process liveness."""
        return self._get("/health")

    def check_readiness(self) -> tuple[bool, dict[str, Any], float]:
        """Call GET /ready for model inference readiness."""
        return self._get("/ready")

    def get_model_info(self) -> tuple[bool, dict[str, Any], float]:
        """Call GET /model-info for governance metadata."""
        return self._get("/model-info")

    def get_metadata(self) -> tuple[bool, dict[str, Any], float]:
        """Call GET /metadata for registry configuration."""
        return self._get("/metadata")

    def predict_single(self, patient_features: dict[str, Any]) -> tuple[bool, dict[str, Any], float]:
        """Call POST /predict with a single patient observation vector."""
        return self._post("/predict", patient_features)

    def predict_batch(self, patient_records: list[dict[str, Any]]) -> tuple[bool, dict[str, Any], float]:
        """Call POST /predict/batch with a list of up to 500 patient records."""
        return self._post("/predict/batch", patient_records)


api_client = BackendAPIClient()
DashboardApiClient = BackendAPIClient
