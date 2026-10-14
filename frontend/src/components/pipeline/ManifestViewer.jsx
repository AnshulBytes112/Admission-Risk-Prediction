import React, { useState } from 'react';
import {
  FileText,
  Copy,
  Check,
  Database,
  Cpu,
  Layers,
  Clock,
  ShieldCheck,
  AlertCircle,
} from 'lucide-react';

export default function ManifestViewer({ liveStatus }) {
  const [activeTab, setActiveTab] = useState('spark');
  const [copied, setCopied] = useState(false);

  const sparkManifest = liveStatus?.latest_spark_manifest;
  const ingestManifest = liveStatus?.latest_ingest_manifest;

  const currentManifest = activeTab === 'spark' ? sparkManifest : ingestManifest;
  const currentJsonString = currentManifest
    ? JSON.stringify(currentManifest, null, 2)
    : '// No manifest located for this category.';

  const handleCopy = () => {
    navigator.clipboard.writeText(currentJsonString);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="card manifest-viewer-card">
      {/* Header */}
      <div className="manifest-header">
        <div>
          <div className="manifest-badge">
            <Clock size={13} /> HISTORICAL AUDIT TRAILS
          </div>
          <h2 className="manifest-title">HDFS Ingestion & PySpark Execution Manifest Explorer</h2>
          <p className="manifest-sub">
            Inspect the exact cryptographic and batch execution logs that document how raw data was stored and transformed.
          </p>
        </div>

        {/* Tab Toggle */}
        <div className="manifest-tabs">
          <button
            className={`manifest-tab-btn ${activeTab === 'spark' ? 'active' : ''}`}
            onClick={() => setActiveTab('spark')}
          >
            <Cpu size={14} /> PySpark Manifest ({sparkManifest?.run_id || 'Latest'})
          </button>
          <button
            className={`manifest-tab-btn ${activeTab === 'ingest' ? 'active' : ''}`}
            onClick={() => setActiveTab('ingest')}
          >
            <Database size={14} /> HDFS Ingestion Manifest ({ingestManifest?.ingestion_date || 'Latest'})
          </button>
        </div>
      </div>

      {/* Historical Label Banner */}
      <div className="historical-warning-banner">
        <AlertCircle size={15} />
        <span>
          <strong>Audit Record Notice:</strong> These manifests are immutable historical execution records logged to <code>data/metadata/</code> upon job completion. They prove data lineage and do not simulate an active in-flight execution.
        </span>
      </div>

      {/* Key Metrics Quick Cards */}
      {activeTab === 'spark' && sparkManifest && (
        <div className="manifest-metrics-row">
          <div className="mini-stat-card">
            <span className="stat-label">Execution Status</span>
            <span className="stat-val text-success">{sparkManifest.status}</span>
            <span className="stat-sub">Spark {sparkManifest.spark_version}</span>
          </div>
          <div className="mini-stat-card">
            <span className="stat-label">Raw Ingested Records</span>
            <span className="stat-val">{sparkManifest.raw_record_count}</span>
            <span className="stat-sub">From HDFS raw path</span>
          </div>
          <div className="mini-stat-card">
            <span className="stat-label">Duplicates Suppressed</span>
            <span className="stat-val text-amber">{sparkManifest.duplicate_count}</span>
            <span className="stat-sub">Exact 50% duplicate suppression</span>
          </div>
          <div className="mini-stat-card">
            <span className="stat-label">Clean Parquet Output</span>
            <span className="stat-val text-blue">{sparkManifest.output_record_count}</span>
            <span className="stat-sub">Partitioned by observation_date</span>
          </div>
        </div>
      )}

      {activeTab === 'ingest' && ingestManifest && (
        <div className="manifest-metrics-row">
          <div className="mini-stat-card">
            <span className="stat-label">Verification Status</span>
            <span className="stat-val text-success">{ingestManifest.verification_status}</span>
            <span className="stat-sub">SHA-256 Checksum Verified</span>
          </div>
          <div className="mini-stat-card">
            <span className="stat-label">Ingested Records</span>
            <span className="stat-val">{ingestManifest.record_count}</span>
            <span className="stat-sub">{ingestManifest.valid_record_count} valid, {ingestManifest.invalid_record_count} invalid</span>
          </div>
          <div className="mini-stat-card">
            <span className="stat-label">Source File Size</span>
            <span className="stat-val">{(ingestManifest.source_size_bytes / 1024).toFixed(1)} KB</span>
            <span className="stat-sub">{ingestManifest.source_filename}</span>
          </div>
          <div className="mini-stat-card">
            <span className="stat-label">Ingestion Date</span>
            <span className="stat-val" style={{ fontSize: '18px' }}>{ingestManifest.ingestion_date}</span>
            <span className="stat-sub">UTC: {ingestManifest.ingested_at_utc}</span>
          </div>
        </div>
      )}

      {/* Imputation Summary Table (if Spark tab) */}
      {activeTab === 'spark' && sparkManifest?.imputation_summary && (
        <div className="imputation-table-section">
          <div className="imputation-title">
            <Layers size={14} /> PySpark Imputation Summary (Applied Medians for Missing Vitals)
          </div>
          <div className="imputation-tags-grid">
            {Object.entries(sparkManifest.imputation_summary).map(([feature, info]) => (
              <div key={feature} className="imputation-tag-card">
                <span className="feature-name">{feature}</span>
                <span className="feature-median">Median: {info.median_applied}</span>
                <span className="feature-nulls">{info.pre_impute_null_count} nulls imputed</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Raw JSON Code Block with Copy Button */}
      <div className="raw-json-container">
        <div className="json-toolbar">
          <span className="json-path">
            File: <code>{currentManifest?._filepath || 'Unknown'}</code>
          </span>
          <button className="copy-json-btn" onClick={handleCopy}>
            {copied ? <Check size={14} className="text-success" /> : <Copy size={14} />}
            {copied ? 'Copied to Clipboard' : 'Copy Manifest JSON'}
          </button>
        </div>
        <pre className="json-manifest-viewer">
          <code>{currentJsonString}</code>
        </pre>
      </div>
    </div>
  );
}
