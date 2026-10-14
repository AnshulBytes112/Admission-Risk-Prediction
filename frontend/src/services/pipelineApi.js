/**
 * Pipeline API client service for communicating with FastAPI pipeline diagnostics endpoints.
 * Provides live latency timing, bounded timeouts, and defensive fallback parsing.
 */

const API_BASE_URL = import.meta.env.VITE_API_URL !== undefined
  ? import.meta.env.VITE_API_URL
  : (import.meta.env.DEV ? '/api' : 'http://localhost:8000');

async function fetchPipeline(endpoint, options = {}) {
  const url = `${API_BASE_URL}${endpoint}`;
  const start = performance.now();

  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), options.timeout || 5000);

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
      data = { error: 'Invalid JSON response from pipeline diagnostic server' };
    }

    if (!response.ok) {
      return {
        ok: false,
        status: response.status,
        data: null,
        error: data.detail || data.error || `HTTP ${response.status}`,
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
    let errorMessage = err.message || 'Diagnostic probe connection failed';
    if (err.name === 'AbortError') {
      errorMessage = 'Diagnostic request timed out (server did not respond in 5s)';
    }

    return {
      ok: false,
      status: 0,
      data: null,
      error: errorMessage,
      latencyMs,
    };
  }
}

export const pipelineApi = {
  getStatus: () => fetchPipeline('/pipeline/status'),
  getKafkaInfo: () => fetchPipeline('/pipeline/kafka'),
  getHdfsInfo: () => fetchPipeline('/pipeline/hdfs'),
  getSparkLatest: () => fetchPipeline('/pipeline/spark/latest'),
  getParquetInfo: () => fetchPipeline('/pipeline/parquet'),
  getManifests: () => fetchPipeline('/pipeline/manifests'),
  getHealth: () => fetchPipeline('/health'),
  getReady: () => fetchPipeline('/ready'),
};
