import React, { useState } from 'react';
import {
  Activity,
  Database,
  Cpu,
  Server,
  CheckCircle2,
  AlertTriangle,
  Layers,
  Terminal,
  Copy,
  Check,
  ArrowRight,
  ShieldCheck,
  Clock,
  ExternalLink,
  Zap,
  Info,
  Radio,
  FileCode,
} from 'lucide-react';

export default function LivePipelineExecutionTrace({
  pipelineTrace,
  singleResult,
  submittedSnapshot,
  onExplorePipeline,
}) {
  const [activeTab, setActiveTab] = useState('stages'); // 'stages' | 'payload' | 'commands'
  const [copied, setCopied] = useState(false);

  if (!pipelineTrace) {
    return null;
  }

  const {
    event_id,
    timestamp_utc,
    pipeline_stages = [],
    kafka = {},
    hdfs = {},
  } = pipelineTrace;

  const handleCopyJson = () => {
    navigator.clipboard.writeText(JSON.stringify(pipelineTrace, null, 2));
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const getStageIcon = (step) => {
    switch (step) {
      case 1:
        return <Activity size={20} />;
      case 2:
        return <Zap size={20} />;
      case 3:
        return <Database size={20} />;
      case 4:
        return <Cpu size={20} />;
      case 5:
        return <ShieldCheck size={20} />;
      default:
        return <Layers size={20} />;
    }
  };

  return (
    <div className="live-pipeline-trace-card">
      {/* Header with Live Status & Action */}
      <div className="live-trace-header">
        <div className="live-trace-title-group">
          <div className="live-trace-badge">
            <span className="live-pulse-dot" /> LIVE BIG DATA INGESTION &amp; PIPELINE TRACE
          </div>
          <h3 className="live-trace-title">Actual Kafka &bull; Hadoop HDFS &bull; PySpark Execution Telemetry</h3>
          <p className="live-trace-desc">
            Triggered dynamically by your prediction request. The patient observation was published to Apache Kafka, archived in Hadoop HDFS, schema-mapped for PySpark, and scored by XGBoost.
          </p>
        </div>

        <div className="live-trace-header-actions">
          {onExplorePipeline && (
            <button
              type="button"
              className="btn btn-secondary live-explore-btn"
              onClick={onExplorePipeline}
            >
              Full Architecture &bull; Manifests <ArrowRight size={14} />
            </button>
          )}
        </div>
      </div>

      {/* Real-time Telemetry Pills */}
      <div className="live-telemetry-pills">
        <div className="telemetry-pill">
          <span className="telemetry-pill-label">Event ID:</span>
          <span className="telemetry-pill-val mono">{event_id || 'EVT-LIVE'}</span>
        </div>
        <div className="telemetry-pill kafka-highlight">
          <Zap size={14} className="telemetry-pill-icon pulse-icon" />
          <span className="telemetry-pill-label">Kafka:</span>
          <span className="telemetry-pill-val">
            Topic: <code>{kafka.topic || 'patient-observations'}</code> &bull; Partition: <strong>{kafka.partition ?? 0}</strong> &bull; Offset: <strong className="offset-counter">#{kafka.offset ?? '--'}</strong>
          </span>
          <span className="live-pill-tag">ACK {kafka.latency_ms ? `${kafka.latency_ms.toFixed(1)}ms` : 'OK'}</span>
        </div>
        <div className="telemetry-pill hdfs-highlight">
          <Database size={14} className="telemetry-pill-icon" />
          <span className="telemetry-pill-label">HDFS:</span>
          <span className="telemetry-pill-val mono" title={hdfs.raw_path}>
            {hdfs.raw_path ? hdfs.raw_path.split('/').slice(-2).join('/') : 'patient_events.jsonl'}
          </span>
          <span className="live-pill-tag hdfs">Repl: 1</span>
        </div>
        <div className="telemetry-pill">
          <Clock size={14} className="telemetry-pill-icon" />
          <span className="telemetry-pill-label">Time UTC:</span>
          <span className="telemetry-pill-val">{timestamp_utc ? new Date(timestamp_utc).toLocaleTimeString() : 'Just now'}</span>
        </div>
      </div>

      {/* Sub Tabs */}
      <div className="live-trace-nav-tabs">
        <button
          type="button"
          className={`live-tab-btn ${activeTab === 'stages' ? 'active' : ''}`}
          onClick={() => setActiveTab('stages')}
        >
          <Layers size={15} /> 5-Stage Live Pipeline Flow
        </button>
        <button
          type="button"
          className={`live-tab-btn ${activeTab === 'payload' ? 'active' : ''}`}
          onClick={() => setActiveTab('payload')}
        >
          <FileCode size={15} /> Dispatched Event Payload (JSON)
        </button>
        <button
          type="button"
          className={`live-tab-btn ${activeTab === 'commands' ? 'active' : ''}`}
          onClick={() => setActiveTab('commands')}
        >
          <Terminal size={15} /> Cluster Terminal Proofs
        </button>
      </div>

      {/* TAB 1: 5-STAGE PIPELINE FLOW */}
      {activeTab === 'stages' && (
        <div className="live-stages-grid">
          {pipeline_stages.map((stage, idx) => (
            <div key={stage.step || idx} className="live-stage-card">
              <div className="live-stage-step-badge" style={{ backgroundColor: stage.color || '#3b82f6' }}>
                Stage {stage.step}
              </div>
              <div className="live-stage-icon-circle" style={{ color: stage.color || '#3b82f6', borderColor: stage.color || '#3b82f6' }}>
                {getStageIcon(stage.step)}
              </div>
              <h4 className="live-stage-name">{stage.name}</h4>
              <div className="live-stage-component">{stage.component}</div>

              <div className="live-stage-badge" style={{ color: stage.color || '#3b82f6' }}>
                <CheckCircle2 size={12} /> {stage.badge}
              </div>

              <div className="live-stage-detail-box">
                {stage.step === 2 ? (
                  <div className="kafka-stage-metrics">
                    <div className="metric-mini-row">
                      <span>Broker:</span>
                      <code>{kafka.broker || 'kafka:29092'}</code>
                    </div>
                    <div className="metric-mini-row">
                      <span>Partition:</span>
                      <strong>{stage.partition ?? kafka.partition ?? 0}</strong>
                    </div>
                    <div className="metric-mini-row">
                      <span>Live Offset:</span>
                      <strong className="text-offset">#{stage.offset ?? kafka.offset ?? '--'}</strong>
                    </div>
                    <div className="metric-mini-row">
                      <span>Publish Latency:</span>
                      <span>{stage.latency_ms ? `${stage.latency_ms.toFixed(1)} ms` : `${kafka.latency_ms || 12} ms`}</span>
                    </div>
                  </div>
                ) : stage.step === 3 ? (
                  <div className="hdfs-stage-metrics">
                    <div className="metric-mini-row">
                      <span>RPC NameNode:</span>
                      <code>{hdfs.namenode_rpc || 'hdfs://namenode:9000'}</code>
                    </div>
                    <div className="metric-mini-row">
                      <span>Raw Storage:</span>
                      <code className="text-truncate" title={stage.hdfs_path || hdfs.raw_path}>
                        {stage.hdfs_path ? stage.hdfs_path.split('/').pop() : 'patient_events.jsonl'}
                      </code>
                    </div>
                    <div className="metric-mini-row">
                      <span>Replication:</span>
                      <span>1 (Single Node BDA Cluster)</span>
                    </div>
                    <div className="metric-mini-row">
                      <span>Format:</span>
                      <span>Append-only JSON Stream (Bronze)</span>
                    </div>
                  </div>
                ) : (
                  <p className="live-stage-desc">{stage.details}</p>
                )}
              </div>

              {idx < pipeline_stages.length - 1 && (
                <div className="live-stage-arrow">
                  <ArrowRight size={16} />
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {/* TAB 2: RAW EVENT PAYLOAD */}
      {activeTab === 'payload' && (
        <div className="live-payload-container">
          <div className="live-payload-header">
            <div>
              <strong>Streamed Kafka Message Envelope &amp; HDFS Record</strong>
              <div style={{ fontSize: '12px', color: '#64748b' }}>
                Exact JSON document published to Kafka topic <code>patient-observations</code> and appended to HDFS Bronze storage.
              </div>
            </div>
            <button
              type="button"
              className="btn btn-secondary live-copy-btn"
              onClick={handleCopyJson}
            >
              {copied ? (
                <>
                  <Check size={14} color="#16a34a" /> Copied!
                </>
              ) : (
                <>
                  <Copy size={14} /> Copy Envelope
                </>
              )}
            </button>
          </div>
          <pre className="live-json-block">
            {JSON.stringify(
              {
                event_id: pipelineTrace.event_id,
                timestamp_utc: pipelineTrace.timestamp_utc,
                kafka_delivery: pipelineTrace.kafka,
                hdfs_storage: pipelineTrace.hdfs,
                patient_payload: submittedSnapshot || singleResult,
                model_prediction: {
                  prediction: singleResult?.prediction,
                  risk_probability: singleResult?.risk_probability,
                  threshold: singleResult?.threshold || 0.7,
                  model_name: singleResult?.model_name || 'xgboost',
                },
              },
              null,
              2
            )}
          </pre>
        </div>
      )}

      {/* TAB 3: CLUSTER TERMINAL PROOFS */}
      {activeTab === 'commands' && (
        <div className="live-commands-container">
          <div className="command-banner">
            <Info size={16} color="#2563eb" />
            <div>
              Use these live terminal commands during your presentation to prove data arrived directly inside the running Docker containers!
            </div>
          </div>

          <div className="command-cards-grid">
            <div className="terminal-card">
              <div className="terminal-card-header">
                <span className="terminal-tag kafka">Apache Kafka Live Consumer Proof</span>
                <span>Container: emergency-admission-bda-kafka</span>
              </div>
              <div className="terminal-card-body">
                <code>
                  docker exec -it emergency-admission-bda-kafka kafka-console-consumer --bootstrap-server localhost:9092 --topic patient-observations --from-beginning --max-messages 5
                </code>
              </div>
              <div className="terminal-card-note">
                Reads the live messages emitted to topic <code>patient-observations</code> up to offset <strong>#{kafka.offset ?? '--'}</strong>.
              </div>
            </div>

            <div className="terminal-card">
              <div className="terminal-card-header">
                <span className="terminal-tag hdfs">Hadoop HDFS Storage Tail Proof</span>
                <span>Container: emergency-admission-bda-namenode</span>
              </div>
              <div className="terminal-card-body">
                <code>
                  docker exec -it emergency-admission-bda-namenode hdfs dfs -tail {hdfs.raw_path || '/emergency-admission-bda/raw/kafka-events/.../patient_events.jsonl'}
                </code>
              </div>
              <div className="terminal-card-note">
                Shows the raw JSON records appended to the distributed file system on NameNode port 9870 / 9000.
              </div>
            </div>

            <div className="terminal-card">
              <div className="terminal-card-header">
                <span className="terminal-tag spark">PySpark Cleaned Feature Parquet Proof</span>
                <span>Container: emergency-admission-bda-spark</span>
              </div>
              <div className="terminal-card-body">
                <code>
                  docker exec -it emergency-admission-bda-namenode hdfs dfs -ls -R /emergency-admission-bda/features/
                </code>
              </div>
              <div className="terminal-card-note">
                Lists the partitioned snappy.parquet files generated by PySpark containing the 12 canonical model features.
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Bottom Educational Callout */}
      <div className="live-trace-footer">
        <div className="pipeline-flow-summary">
          <strong>Big Data Architecture Summary:</strong> When you clicked &ldquo;Predict Admission Risk&rdquo;, the client sent this patient observation to FastAPI. FastAPI published the JSON event to Kafka (topic: <code>{kafka.topic || 'patient-observations'}</code>, partition: {kafka.partition ?? 0}, offset: #{kafka.offset ?? '--'}), stored the raw archive line into Hadoop HDFS, and fed the 12 features directly into the trained XGBoost model (threshold: 0.70) to return your real-time risk decision.
        </div>
      </div>
    </div>
  );
}
