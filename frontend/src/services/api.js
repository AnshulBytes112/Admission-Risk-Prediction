/**
 * API client service for communicating with Phase 6 FastAPI backend.
 * Provides live latency measurement, bounded timeouts, and typed response parsing.
 */

const API_BASE_URL = import.meta.env.VITE_API_URL !== undefined
  ? import.meta.env.VITE_API_URL
  : (import.meta.env.DEV ? '/api' : 'http://localhost:8000');

const DISPLAY_API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

/**
 * Measure round-trip execution latency and execute HTTP fetch.
 */
async function fetchWithTiming(endpoint, options = {}) {
  const url = `${API_BASE_URL}${endpoint}`;
  const start = performance.now();

  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), options.timeout || 6000);

    const response = await fetch(url, {
      ...options,
      signal: controller.signal,
      headers: {
        'Content-Type': 'application/json',
        ...(options.headers || {}),
      },
    });

    clearTimeout(timeoutId);
    const latencyMs = Math.round(performance.now() - start);

    let data;
    try {
      data = await response.json();
    } catch {
      data = { error: 'Invalid JSON response from server' };
    }

    if (!response.ok) {
      const errorMsg = data.detail
        ? (typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail))
        : (data.error || `HTTP error ${response.status}`);

      return {
        ok: false,
        status: response.status,
        data: null,
        error: errorMsg,
        latencyMs,
      };
    }

    return {
      ok: true,
      status: response.status,
      data,
      error: null,
      latencyMs,
    };
  } catch (err) {
    const latencyMs = Math.round(performance.now() - start);
    let errorMessage = err.message || 'Connection failed';
    if (err.name === 'AbortError') {
      errorMessage = 'Request timed out (server did not respond in 6s)';
    }

    return {
      ok: false,
      status: 0,
      data: null,
      error: `${errorMessage}. Ensure FastAPI backend is running on ${API_BASE_URL}.`,
      latencyMs,
    };
  }
}

export const api = {
  checkHealth: () => fetchWithTiming('/health'),
  checkReadiness: () => fetchWithTiming('/ready'),
  getModelInfo: () => fetchWithTiming('/model-info'),
  getMetadata: () => fetchWithTiming('/metadata'),
  getThreshold: () => fetchWithTiming('/config/threshold'),
  setThreshold: (threshold) =>
    fetchWithTiming('/config/threshold', {
      method: 'POST',
      body: JSON.stringify({ threshold }),
    }),
  resetThreshold: () =>
    fetchWithTiming('/config/threshold', {
      method: 'DELETE',
    }),
  predictSingle: (patientData, threshold) => {
    const q = threshold !== undefined && threshold !== null ? `?threshold=${threshold}` : '';
    return fetchWithTiming(`/predict${q}`, {
      method: 'POST',
      body: JSON.stringify(patientData),
    });
  },
  predictBatch: (records, threshold) => {
    const q = threshold !== undefined && threshold !== null ? `?threshold=${threshold}` : '';
    return fetchWithTiming(`/predict/batch${q}`, {
      method: 'POST',
      body: JSON.stringify(records),
    });
  },
  baseUrl: API_BASE_URL,
  displayUrl: DISPLAY_API_URL,
};
