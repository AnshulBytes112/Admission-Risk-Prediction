import React from 'react';
import {
  FileText,
  Terminal,
  Database,
  ArrowRight,
  AlertTriangle,
  Lightbulb,
  CheckCircle2,
  Layers,
  HardDrive,
  Cpu,
  Info,
} from 'lucide-react';

export default function PipelineStageDetails({
  stage,
  onOpenWalkthrough,
  liveStatus,
}) {
  if (!stage) {
    return (
      <div className="card stage-details-card">
        <div className="empty-selection-placeholder">
          <Info size={32} className="placeholder-icon" />
          <p>Select any node in the architecture diagram above to view its technical details, inputs, outputs, and live evidence.</p>
        </div>
      </div>
    );
  }

  // Get dynamic evidence value if available
  let dynamicEvidence = null;
  if (liveStatus) {
    if (stage.id === 'stage-3' && liveStatus.components?.kafka) {
      const k = liveStatus.components.kafka;
      dynamicEvidence = `Live Kafka Broker probe: ${k.probe_result?.reachable ? 'CONNECTED' : 'UNREACHABLE'} (${k.probe_result?.target})`;
    } else if (stage.id === 'stage-5' && liveStatus.streaming_storage) {
      const s = liveStatus.streaming_storage;
      dynamicEvidence = `Current file on disk: ${s.patient_events_lines} lines (${(s.patient_events_bytes / 1024).toFixed(1)} KB)`;
    } else if (stage.id === 'stage-6' && liveStatus.latest_ingest_manifest) {
      const m = liveStatus.latest_ingest_manifest;
      dynamicEvidence = `Verified Manifest: ${m.record_count} records ingested on ${m.ingestion_date} (SHA-256: ${m.source_sha256?.slice(0, 12)}...)`;
    } else if (stage.id === 'stage-7' && liveStatus.latest_spark_manifest) {
      const sp = liveStatus.latest_spark_manifest;
      dynamicEvidence = `Spark Manifest run_id=${sp.run_id}: raw=${sp.raw_record_count}, valid=${sp.valid_record_count}, duplicates=${sp.duplicate_count}, output=${sp.output_record_count}`;
    } else if (stage.id === 'stage-10' && liveStatus.components?.parquet) {
      const pq = liveStatus.components.parquet;
      dynamicEvidence = `Local Parquet dataset: ${pq.partition_count} observation_date partition folders, _SUCCESS marker=${pq.success_marker ? 'Yes' : 'No'}`;
    } else if (stage.id === 'stage-12' && liveStatus.components?.fastapi) {
      dynamicEvidence = `FastAPI Status: ONLINE (live internal process check at ${liveStatus.timestamp_utc})`;
    }
  }

  return (
    <div className="card stage-details-card">
      {/* Header */}
      <div className="stage-detail-header">
        <div>
          <div className="stage-badge-row">
            <span className="stage-number-tag">Stage {stage.step} of 13</span>
            <span className="stage-group-tag">{stage.group.toUpperCase()}</span>
            <span className="stage-tech-tag">{stage.technology}</span>
          </div>
          <h2 className="stage-detail-title">{stage.name}</h2>
          <p className="stage-detail-sub">{stage.shortDescription}</p>
        </div>
        <button
          className="btn btn-secondary walkthrough-jump-btn"
          onClick={() => onOpenWalkthrough(stage.step)}
        >
          View in Presentation Walkthrough <ArrowRight size={15} />
        </button>
      </div>

      {/* Presentation Script Banner */}
      <div className="presentation-script-banner">
        <div className="script-banner-title">
          <Terminal size={14} /> Suggested Presentation Script (for Evaluators):
        </div>
        <p className="script-banner-text">"{stage.presentationScript}"</p>
      </div>

      {/* Grid: Inputs, Processing, Outputs */}
      <div className="io-spec-grid">
        <div className="io-card input-io">
          <div className="io-card-title">
            <ArrowRight size={14} className="io-icon in" /> Input Specification
          </div>
          <div className="io-card-content">{stage.input}</div>
        </div>

        <div className="io-card proc-io">
          <div className="io-card-title">
            <Cpu size={14} className="io-icon proc" /> Transformations & Logic
          </div>
          <div className="io-card-content">{stage.processing}</div>
        </div>

        <div className="io-card output-io">
          <div className="io-card-title">
            <CheckCircle2 size={14} className="io-icon out" /> Output & Destination
          </div>
          <div className="io-card-content">{stage.output}</div>
        </div>
      </div>

      {/* Technical Deep Dive */}
      <div className="tech-deep-dive-section">
        <div className="detail-section-title">
          <Layers size={16} /> Technical Architecture Breakdown
        </div>
        <p className="tech-deep-dive-text">{stage.technicalExplanation}</p>
      </div>

      {/* Evidence & Verification Box */}
      <div className="evidence-verification-box">
        <div className="evidence-box-header">
          <Database size={15} /> Verified Project Evidence & Storage Target
        </div>
        <div className="evidence-box-body">
          <div className="evidence-row">
            <span className="ev-label">Target / Location:</span>
            <code className="ev-code">{stage.location}</code>
          </div>
          <div className="evidence-row">
            <span className="ev-label">Baseline Evidence:</span>
            <span className="ev-text">{stage.evidenceSource}</span>
          </div>
          {dynamicEvidence && (
            <div className="evidence-row live-match">
              <span className="ev-label">Live Probed Evidence:</span>
              <span className="ev-live-badge">{dynamicEvidence}</span>
            </div>
          )}
        </div>
      </div>

      {/* Common Failure Modes & Key Takeaway */}
      <div className="stage-footer-grid">
        <div className="failure-scenario-card">
          <div className="card-sub-header danger">
            <AlertTriangle size={15} /> Common Failure Scenarios
          </div>
          <p className="failure-text">{stage.failureScenarios}</p>
        </div>

        <div className="key-takeaway-card">
          <div className="card-sub-header success">
            <Lightbulb size={15} /> One Key Presentation Takeaway
          </div>
          <p className="takeaway-text">{stage.keyTakeaway}</p>
        </div>
      </div>
    </div>
  );
}
