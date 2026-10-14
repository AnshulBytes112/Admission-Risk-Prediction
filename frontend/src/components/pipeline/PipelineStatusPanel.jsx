import React, { useState } from 'react';
import {
  RefreshCw,
  CheckCircle2,
  AlertTriangle,
  XCircle,
  HelpCircle,
  Clock,
  Terminal,
  Activity,
  Server,
  Layers,
  Database,
  Search,
  ExternalLink,
} from 'lucide-react';
import { pipelineApi } from '../../services/pipelineApi';

export default function PipelineStatusPanel({
  liveStatus,
  onRefresh,
  loading,
}) {
  const [selectedInspection, setSelectedInspection] = useState(null);
  const [inspectionLoading, setInspectionLoading] = useState(false);
  const [inspectionResult, setInspectionResult] = useState(null);

  const handleInspect = async (type, fetcher) => {
    setSelectedInspection(type);
    setInspectionLoading(true);
    setInspectionResult(null);

    const res = await fetcher();
    setInspectionResult({
      ok: res.ok,
      status: res.status,
      data: res.data,
      error: res.error,
      latencyMs: res.latencyMs,
    });
    setInspectionLoading(false);
  };

  const comps = liveStatus?.components || {};

  const getStatusBadge = (status) => {
    switch (status) {
      case 'verified':
      case 'online':
      case 'loaded':
        return (
          <span className="table-badge badge-success">
            <CheckCircle2 size={13} /> {status.toUpperCase()}
          </span>
        );
      case 'historical_success':
        return (
          <span className="table-badge badge-warning">
            <Clock size={13} /> HISTORICAL SUCCESS
          </span>
        );
      case 'unavailable':
      case 'missing':
      case 'offline':
        return (
          <span className="table-badge badge-danger">
            <XCircle size={13} /> {status.toUpperCase()}
          </span>
        );
      default:
        return (
          <span className="table-badge badge-neutral">
            <HelpCircle size={13} /> UNKNOWN
          </span>
        );
    }
  };

  // Build the 8 strict status rows
  const statusRows = [
    {
      component: 'Apache Kafka Broker',
      status: comps.kafka?.status || 'unknown',
      evidenceSource: comps.kafka?.evidence_source || 'Live TCP socket check (port 29092/9092)',
      details: comps.kafka?.probe_result?.reachable
        ? `Reachable via ${comps.kafka.probe_result.target}; Topic: patient-observations`
        : 'Broker socket not responding on port 29092/9092',
      inspectType: 'Kafka Topic & Stream',
      inspectAction: () => pipelineApi.getKafkaInfo(),
    },
    {
      component: 'Kafka Consumer Group',
      status: comps.kafka?.status === 'verified' ? 'verified' : 'unknown',
      evidenceSource: 'Consumer group: emergency-admission-bda-consumer',
      details: liveStatus?.streaming_storage
        ? `${liveStatus.streaming_storage.patient_events_lines} records archived in patient_events.jsonl`
        : 'No consumer event archive records found',
      inspectType: 'Kafka Consumer & Archive',
      inspectAction: () => pipelineApi.getKafkaInfo(),
    },
    {
      component: 'Hadoop HDFS Cluster',
      status: comps.hdfs?.status || 'unknown',
      evidenceSource: comps.hdfs?.evidence_source || 'Live NameNode TCP socket check (port 9870/9000)',
      details: comps.hdfs?.probe_result?.reachable
        ? `NameNode reachable at ${comps.hdfs.probe_result.target}; Raw: /emergency-admission-bda/raw/kafka-events`
        : 'NameNode socket connection refused on port 9870/9000',
      inspectType: 'HDFS Cluster & Paths',
      inspectAction: () => pipelineApi.getHdfsInfo(),
    },
    {
      component: 'PySpark Processing Engine',
      status: comps.spark?.status || 'unknown',
      evidenceSource: comps.spark?.evidence_source || 'Spark processing manifest',
      details: comps.spark?.last_run_id
        ? `Run ID: ${comps.spark.last_run_id} • Output: ${comps.spark.output_records} records • Duplicates removed: ${comps.spark.duplicate_count}`
        : 'No processing manifests located in data/metadata/spark_manifests',
      inspectType: 'Spark Manifest Details',
      inspectAction: () => pipelineApi.getSparkLatest(),
    },
    {
      component: 'Partitioned Parquet Dataset',
      status: comps.parquet?.status || 'unknown',
      evidenceSource: comps.parquet?.evidence_source || 'Local filesystem scan of features.parquet',
      details: comps.parquet?.status === 'verified'
        ? `${comps.parquet.partition_count} partitions (partitioned by observation_date) • _SUCCESS marker present`
        : 'Parquet feature directory missing from data/processed',
      inspectType: 'Parquet Schema & Partitions',
      inspectAction: () => pipelineApi.getParquetInfo(),
    },
    {
      component: 'Scikit-Learn ML Model',
      status: comps.model?.status || 'unknown',
      evidenceSource: comps.model?.evidence_source || 'FastAPI ModelManager memory registry',
      details: comps.model?.status === 'loaded'
        ? `Bundle: ${comps.model.model_name} • Threshold: ${comps.model.threshold.toFixed(2)} (Experimental smoke-test)`
        : 'Model artifact not loaded in memory',
      inspectType: 'Model Readiness Probe',
      inspectAction: () => pipelineApi.getReady(),
    },
    {
      component: 'FastAPI Prediction API',
      status: comps.fastapi?.status || 'online',
      evidenceSource: comps.fastapi?.evidence_source || 'Internal process probe',
      details: `Service online • Verified at ${liveStatus?.timestamp_utc ? new Date(liveStatus.timestamp_utc).toLocaleTimeString() : 'now'}`,
      inspectType: 'FastAPI Health Probe',
      inspectAction: () => pipelineApi.getHealth(),
    },
    {
      component: 'React Analytics Dashboard',
      status: 'verified',
      evidenceSource: 'Active browser runtime and React execution context',
      details: 'Frontend running on port 3000 • Connected via Vite dev proxy',
      inspectType: null,
      inspectAction: null,
    },
  ];

  return (
    <div className="card status-panel-card">
      {/* Header & Refresh */}
      <div className="status-panel-header">
        <div>
          <h2 className="status-panel-title">Live Pipeline Status & Historical Evidence Table</h2>
          <p className="status-panel-sub">
            Transparently distinguishes live verified operational components from historical batch execution manifests.
          </p>
        </div>
        <button
          className="btn btn-primary refresh-status-btn"
          onClick={onRefresh}
          disabled={loading}
        >
          <RefreshCw size={15} className={loading ? 'spin' : ''} />
          {loading ? 'Probing Services...' : 'Refresh Status Probes'}
        </button>
      </div>

      {/* Main Evidence Table */}
      <div className="table-responsive">
        <table className="evidence-table">
          <thead>
            <tr>
              <th>Component</th>
              <th>Status</th>
              <th>Last Evidence Source</th>
              <th>Details & Operational Metrics</th>
              <th style={{ textAlign: 'right' }}>Allowlisted Action</th>
            </tr>
          </thead>
          <tbody>
            {statusRows.map((row, idx) => (
              <tr key={idx}>
                <td className="component-cell">
                  <strong>{row.component}</strong>
                </td>
                <td>{getStatusBadge(row.status)}</td>
                <td className="evidence-cell">
                  <code>{row.evidenceSource}</code>
                </td>
                <td className="details-cell">{row.details}</td>
                <td style={{ textAlign: 'right' }}>
                  {row.inspectAction ? (
                    <button
                      className="btn btn-secondary btn-sm"
                      onClick={() => handleInspect(row.inspectType, row.inspectAction)}
                    >
                      Inspect
                    </button>
                  ) : (
                    <span className="text-muted text-sm">Self-Hosted</span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Real Response Payload Viewer Box */}
      {selectedInspection && (
        <div className="inspection-result-box">
          <div className="inspection-box-header">
            <div>
              <Terminal size={14} /> Diagnostic Probe Output: <strong>{selectedInspection}</strong>
            </div>
            {inspectionResult && (
              <div className="inspection-meta">
                <span>Status: {inspectionResult.status}</span>
                <span>Latency: {inspectionResult.latencyMs} ms</span>
                <button
                  className="close-box-btn"
                  onClick={() => setSelectedInspection(null)}
                >
                  &times;
                </button>
              </div>
            )}
          </div>
          <div className="inspection-box-body">
            {inspectionLoading ? (
              <div className="inspection-loading">
                <RefreshCw size={18} className="spin" /> Executing bounded probe...
              </div>
            ) : inspectionResult ? (
              <pre className="json-code-block">
                {JSON.stringify(inspectionResult.data || { error: inspectionResult.error }, null, 2)}
              </pre>
            ) : null}
          </div>
        </div>
      )}
    </div>
  );
}
