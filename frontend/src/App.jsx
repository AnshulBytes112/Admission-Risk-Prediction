import React, { useState, useEffect } from 'react';
import {
  Heart,
  Activity,
  Brain,
  Database,
  FlaskConical,
  CheckCircle2,
  XCircle,
  AlertTriangle,
  FileText,
  Upload,
  Download,
  RotateCcw,
  Sparkles,
  Search,
  ShieldCheck,
  Server,
  Layers,
  ArrowRight,
  Info,
  GitBranch,
  Sliders,
} from 'lucide-react';
import { api } from './services/api';
import BigDataPipeline from './pages/BigDataPipeline';
import LivePipelineExecutionTrace from './components/pipeline/LivePipelineExecutionTrace';
import {
  buildPredictionPayload,
  areInputsStale,
  FEATURE_CONFIG,
  MODEL_FEATURE_ORDER,
} from './utils/featureMapping';

const PRESETS = {
  'Typical / Low Risk Case': {
    patient_id: 'SYNTH-PT-1001',
    age: 65,
    heart_rate: 82,
    systolic_bp: 124,
    diastolic_bp: 78,
    spo2: 97,
    respiratory_rate: 15,
    temperature_c: 37.0,
    previous_admissions: 1,
    chronic_conditions_count: 2,
    lactate: 1.3,
    observation_hour: 14,
    prior_observation_count: 4,
  },
  'Elevated / Critical Case': {
    patient_id: 'SYNTH-PT-9042',
    age: 78,
    heart_rate: 128,
    systolic_bp: 88,
    diastolic_bp: 54,
    spo2: 89,
    respiratory_rate: 32,
    temperature_c: 39.2,
    previous_admissions: 4,
    chronic_conditions_count: 5,
    lactate: 4.6,
    observation_hour: 3,
    prior_observation_count: 12,
  },
  'Borderline / Moderate Case': {
    patient_id: 'SYNTH-PT-5021',
    age: 70,
    heart_rate: 102,
    systolic_bp: 105,
    diastolic_bp: 68,
    spo2: 93,
    respiratory_rate: 22,
    temperature_c: 38.1,
    previous_admissions: 2,
    chronic_conditions_count: 3,
    lactate: 2.2,
    observation_hour: 18,
    prior_observation_count: 6,
  },
};

export default function App() {
  const [activeTab, setActiveTab] = useState('dashboard');

  // Backend Health States
  const [apiStatus, setApiStatus] = useState({ online: false, latencyMs: 0 });
  const [modelStatus, setModelStatus] = useState({ ready: false, modelName: 'xgboost', threshold: 0.70 });
  const [loadingStatus, setLoadingStatus] = useState(true);
  const [decisionThreshold, setDecisionThreshold] = useState(0.70);

  // Dynamic threshold change handler (propagates to backend and recalculates UI instantly)
  const handleThresholdChange = (val) => {
    const newTh = parseFloat(Number(val).toFixed(2));
    setDecisionThreshold(newTh);
    setModelStatus((prev) => ({ ...prev, threshold: newTh }));

    // Persist to backend runtime threshold
    api.setThreshold(newTh);

    // If single prediction is displayed, immediately recalculate decision in UI
    if (singleResult) {
      const isAdmit = singleResult.risk_probability >= newTh;
      setSingleResult((prev) => ({
        ...prev,
        threshold: newTh,
        review_flag: isAdmit,
        prediction: isAdmit ? 1 : 0,
      }));
    }

    // If batch predictions are displayed, immediately recalculate all predictions
    if (batchResult && batchResult.predictions) {
      setBatchResult((prev) => ({
        ...prev,
        predictions: prev.predictions.map((p) => ({
          ...p,
          threshold: newTh,
          review_flag: p.risk_probability >= newTh,
          prediction: p.risk_probability >= newTh ? 1 : 0,
        })),
      }));
    }
  };

  // Single Prediction Form State
  const [presetKey, setPresetKey] = useState('Typical / Low Risk Case');
  const [formInputs, setFormInputs] = useState(PRESETS['Typical / Low Risk Case']);
  const [submittedSnapshot, setSubmittedSnapshot] = useState(null);
  const [resultTimestamp, setResultTimestamp] = useState(null);
  const [singleResult, setSingleResult] = useState(null);
  const [singleLoading, setSingleLoading] = useState(false);
  const [singleError, setSingleError] = useState(null);

  // Check if current inputs differ from the snapshot that generated the displayed result
  const isResultStale = Boolean(singleResult && submittedSnapshot && areInputsStale(formInputs, submittedSnapshot));

  // Batch Prediction State
  const [batchFile, setBatchFile] = useState(null);
  const [batchRecords, setBatchRecords] = useState([]);
  const [batchResult, setBatchResult] = useState(null);
  const [batchLoading, setBatchLoading] = useState(false);
  const [batchError, setBatchError] = useState(null);

  // Dataset Explorer Filter State
  const [searchQuery, setSearchQuery] = useState('');
  const [filterOutcome, setFilterOutcome] = useState('all');

  // Fetch API status on mount
  const checkBackend = async () => {
    setLoadingStatus(true);
    const health = await api.checkHealth();
    const ready = await api.checkReadiness();

    setApiStatus({
      online: health.ok,
      latencyMs: health.latencyMs || 0,
      details: health.data,
    });

    if (ready.ok && ready.data) {
      const th = ready.data.model_threshold || 0.70;
      setModelStatus({
        ready: true,
        modelName: ready.data.model_name || 'xgboost',
        threshold: th,
        featureCount: ready.data.feature_count || 12,
      });
      setDecisionThreshold(th);
    } else {
      setModelStatus({
        ready: false,
        modelName: 'xgboost',
        threshold: 0.70,
        featureCount: 12,
      });
    }
    setLoadingStatus(false);
  };

  useEffect(() => {
    checkBackend();
  }, []);

  // Update input field and clear input-level error
  const handleInputChange = (field, value) => {
    setFormInputs((prev) => ({ ...prev, [field]: value }));
    setSingleError(null);
  };

  // Handle preset switcher
  const handlePresetChange = (name) => {
    setPresetKey(name);
    setFormInputs(PRESETS[name]);
    setSingleError(null);
  };

  const handleRandomize = () => {
    const randomCase = {
      patient_id: `SYNTH-${Math.floor(1000 + Math.random() * 9000)}`,
      age: Math.floor(25 + Math.random() * 65),
      heart_rate: Math.floor(60 + Math.random() * 70),
      systolic_bp: Math.floor(90 + Math.random() * 75),
      diastolic_bp: Math.floor(55 + Math.random() * 45),
      spo2: Math.floor(88 + Math.random() * 11),
      respiratory_rate: Math.floor(12 + Math.random() * 22),
      temperature_c: +(36.2 + Math.random() * 3.3).toFixed(1),
      previous_admissions: Math.floor(Math.random() * 6),
      chronic_conditions_count: Math.floor(Math.random() * 5),
      lactate: +(0.7 + Math.random() * 4.3).toFixed(1),
      observation_hour: Math.floor(Math.random() * 24),
      prior_observation_count: Math.floor(1 + Math.random() * 15),
    };
    setPresetKey('Custom Random');
    setFormInputs(randomCase);
    setSingleError(null);
  };

  // Submit Single Prediction with authoritative validation & stale result tracking
  const handleSingleSubmit = async (e) => {
    if (e && e.preventDefault) e.preventDefault();
    if (singleLoading) return; // Prevent duplicate in-flight requests

    setSingleError(null);

    // Validate using authoritative feature mapping
    const validation = buildPredictionPayload(formInputs);
    if (!validation.isValid) {
      setSingleError(validation.errors[0] || 'Please provide valid clinical inputs.');
      return;
    }

    setSingleLoading(true);

    const res = await api.predictSingle(validation.payload, decisionThreshold);

    if (res.ok && res.data) {
      if (typeof res.data.risk_probability !== 'number' || !Number.isFinite(res.data.risk_probability)) {
        setSingleError('Inference service returned malformed prediction probability.');
        setSingleLoading(false);
        return;
      }

      setSingleResult({
        ...res.data,
        latencyMs: res.latencyMs,
      });
      setSubmittedSnapshot({ ...formInputs });
      setResultTimestamp(new Date().toLocaleTimeString());
      setSingleError(null);
    } else {
      setSingleError(res.error || 'Prediction request failed. Ensure FastAPI is running.');
    }
    setSingleLoading(false);
  };

  // Parse CSV file for Batch Prediction
  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setBatchFile(file);
    setBatchError(null);
    setBatchResult(null);

    const reader = new FileReader();
    reader.onload = (event) => {
      try {
        const text = event.target.result;
        const lines = text.split('\n').map((l) => l.trim()).filter(Boolean);
        if (lines.length < 2) {
          setBatchError('CSV file is empty or missing data rows.');
          return;
        }

        const headers = lines[0].split(',').map((h) => h.trim());
        const required = ['age', 'heart_rate', 'systolic_bp', 'diastolic_bp', 'spo2', 'respiratory_rate', 'temperature_c', 'previous_admissions', 'chronic_conditions_count', 'lactate', 'observation_hour', 'prior_observation_count'];
        const missing = required.filter((r) => !headers.includes(r));

        if (missing.length > 0) {
          setBatchError(`Missing required columns: ${missing.join(', ')}`);
          return;
        }

        const parsed = [];
        for (let i = 1; i < lines.length; i++) {
          const values = lines[i].split(',').map((v) => v.trim());
          if (values.length !== headers.length) continue;

          const row = {};
          headers.forEach((h, idx) => {
            row[h] = h === 'patient_id' ? values[idx] : Number(values[idx]);
          });
          if (!row.patient_id) row.patient_id = `BATCH-PT-${i}`;
          parsed.push(row);
        }

        if (parsed.length > 500) {
          setBatchError(`Batch file has ${parsed.length} rows. Maximum allowed is 500.`);
          return;
        }

        setBatchRecords(parsed);
      } catch (err) {
        setBatchError(`Failed to parse CSV: ${err.message}`);
      }
    };
    reader.readAsText(file);
  };

  // Submit Batch Prediction
  const handleBatchSubmit = async () => {
    if (batchRecords.length === 0) return;
    setBatchLoading(true);
    setBatchError(null);

    const res = await api.predictBatch(batchRecords, decisionThreshold);

    if (res.ok && res.data) {
      setBatchResult({
        ...res.data,
        latencyMs: res.latencyMs,
      });
    } else {
      setBatchError(res.error || 'Batch prediction failed');
    }
    setBatchLoading(false);
  };

  // Download Sample Batch CSV
  const downloadSampleBatchCSV = () => {
    const csvContent =
      'patient_id,age,heart_rate,systolic_bp,diastolic_bp,spo2,respiratory_rate,temperature_c,previous_admissions,chronic_conditions_count,lactate,observation_hour,prior_observation_count\n' +
      'SYNTH-PT-001,65,82,124,78,97,15,37.0,1,2,1.3,14,4\n' +
      'SYNTH-PT-002,78,128,88,54,89,32,39.2,4,5,4.6,3,12\n' +
      'SYNTH-PT-003,70,102,105,68,93,22,38.1,2,3,2.2,18,6\n' +
      'SYNTH-PT-004,52,72,118,74,99,14,36.7,0,1,1.1,10,2\n' +
      'SYNTH-PT-005,84,115,92,60,91,26,38.8,3,4,3.8,22,9\n';

    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'batch_prediction_sample.csv';
    a.click();
  };

  // Download Scored Batch Results CSV
  const downloadScoredCSV = () => {
    if (!batchResult || !batchResult.predictions) return;
    const headers = 'patient_id,prediction,risk_probability,risk_percent,review_flag,model_name\n';
    const rows = batchResult.predictions
      .map(
        (p) =>
          `${p.patient_id},${p.prediction},${p.risk_probability},${p.risk_percent || (p.risk_probability * 100).toFixed(1)},${p.review_flag},${p.model_name}`
      )
      .join('\n');

    const blob = new Blob([headers + rows], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `scored_batch_${batchResult.predictions.length}_records.csv`;
    a.click();
  };

  return (
    <div className="app-container">
      {/* -------------------------------------------------------------------
          SIDEBAR
         ------------------------------------------------------------------- */}
      <aside className="sidebar">
        <div className="brand-header">
          <div className="brand-icon">
            <Heart size={22} fill="currentColor" />
          </div>
          <div>
            <div className="brand-title">Emergency<br />Admission Risk</div>
            <div className="brand-subtitle">BDA Project</div>
          </div>
        </div>

        <nav className="nav-menu">
          <button
            className={`nav-item ${activeTab === 'dashboard' ? 'active' : ''}`}
            onClick={() => setActiveTab('dashboard')}
          >
            <Activity size={17} /> Dashboard
          </button>
          <button
            className={`nav-item ${activeTab === 'single' ? 'active' : ''}`}
            onClick={() => setActiveTab('single')}
          >
            <CheckCircle2 size={17} /> Single Prediction
          </button>
          <button
            className={`nav-item ${activeTab === 'batch' ? 'active' : ''}`}
            onClick={() => setActiveTab('batch')}
          >
            <FileText size={17} /> Batch Prediction
          </button>
          <button
            className={`nav-item ${activeTab === 'pipeline' ? 'active' : ''}`}
            onClick={() => setActiveTab('pipeline')}
          >
            <GitBranch size={17} /> Big Data Pipeline
          </button>
          <button
            className={`nav-item ${activeTab === 'model' ? 'active' : ''}`}
            onClick={() => setActiveTab('model')}
          >
            <Brain size={17} /> Model Information
          </button>
          <button
            className={`nav-item ${activeTab === 'system' ? 'active' : ''}`}
            onClick={() => setActiveTab('system')}
          >
            <Server size={17} /> System Status
          </button>
          <button
            className={`nav-item ${activeTab === 'about' ? 'active' : ''}`}
            onClick={() => setActiveTab('about')}
          >
            <Info size={17} /> About
          </button>
        </nav>

        {/* Bottom Academic Disclaimer Card */}
        <div className="sidebar-disclaimer-card">
          <div className="sidebar-disclaimer-title">
            <Info size={14} /> Academic Demonstration Only
          </div>
          <div className="sidebar-disclaimer-text">
            All patient records and clinical data are <strong>SYNTHETIC</strong>. This application is an academic demonstration and must <strong>never</strong> be used for actual medical decisions, patient diagnosis, or emergency department triage.
          </div>
        </div>
      </aside>

      {/* -------------------------------------------------------------------
          MAIN WORKSPACE
         ------------------------------------------------------------------- */}
      <main className="workspace">
        {/* Header Bar */}
        <header className="workspace-header">
          <div>
            <h1 className="page-title">Emergency Admission Risk Prediction</h1>
            <p className="page-subtitle">Predict the risk of hospital admission within 24 hours using machine learning</p>
          </div>
          <div className="badge-experimental">
            <AlertTriangle size={15} /> EXPERIMENTAL MODEL (SYNTHETIC DATA)
          </div>
        </header>

        {/* Top 4 Metrics Status Cards (matching reference image) */}
        <section className="metrics-row">
          <div className="metric-card">
            <div className="metric-icon-circle green">
              {apiStatus.online ? <CheckCircle2 size={24} /> : <XCircle size={24} />}
            </div>
            <div>
              <div className="metric-label">API Status</div>
              <div className="metric-value">{apiStatus.online ? 'Online' : 'Offline'}</div>
              <div className="metric-subtext">
                {apiStatus.online ? `FastAPI service is running (${apiStatus.latencyMs} ms)` : 'Backend unreachable'}
              </div>
            </div>
          </div>

          <div className="metric-card">
            <div className="metric-icon-circle teal">
              <Brain size={24} />
            </div>
            <div>
              <div className="metric-label">Model Status</div>
              <div className="metric-value">{modelStatus.ready ? 'Ready' : 'Not Ready'}</div>
              <div className="metric-subtext">Model loaded and operational</div>
            </div>
          </div>

          <div className="metric-card">
            <div className="metric-icon-circle purple">
              <Database size={24} />
            </div>
            <div>
              <div className="metric-label">Model Version</div>
              <div className="metric-value" style={{ fontSize: '16px' }}>{modelStatus.modelName}</div>
              <div className="metric-subtext">12 features • Threshold: {modelStatus.threshold.toFixed(2)}</div>
            </div>
          </div>

          <div className="metric-card">
            <div className="metric-icon-circle yellow">
              <FlaskConical size={24} />
            </div>
            <div>
              <div className="metric-label">Data Origin</div>
              <div className="metric-value">Synthetic</div>
              <div className="metric-subtext">Academic demonstration only</div>
            </div>
          </div>
        </section>

        {/* -----------------------------------------------------------------
            VIEW DISPATCHER
           ----------------------------------------------------------------- */}

        {/* DASHBOARD OR SINGLE PREDICTION VIEW */}
        {(activeTab === 'dashboard' || activeTab === 'single') && (
          <>
            {/* Dynamic Clinical Decision Threshold Controller */}
            <div className="threshold-controller-card">
              <div className="threshold-card-header">
                <div className="threshold-title-group">
                  <div className="threshold-icon-circle">
                    <Sliders size={18} />
                  </div>
                  <div>
                    <div className="threshold-title">Clinical Decision Threshold (Operating Cutoff)</div>
                    <div className="threshold-subtitle">
                      Dynamically calibrate the admission cutoff in real time across the ML model, API, and live pipeline.
                    </div>
                  </div>
                </div>

                <div className="threshold-live-badge">
                  Operating Cutoff: <strong>{decisionThreshold.toFixed(2)}</strong> ({Math.round(decisionThreshold * 100)}%)
                </div>
              </div>

              <div className="threshold-slider-container">
                <input
                  type="range"
                  min="0.10"
                  max="0.90"
                  step="0.01"
                  value={decisionThreshold}
                  onChange={(e) => handleThresholdChange(e.target.value)}
                  className="threshold-range-slider"
                />
                <div className="threshold-range-limits">
                  <span>0.10 &bull; High Sensitivity (Flag More Admissions)</span>
                  <span>0.50 &bull; Standard Balanced Cutoff</span>
                  <span>0.90 &bull; High Specificity (Conserve ED Beds)</span>
                </div>
              </div>

              <div className="threshold-presets-row">
                <span className="threshold-presets-label">Operating Presets:</span>
                <button
                  type="button"
                  className={`threshold-preset-btn ${decisionThreshold === 0.35 ? 'active' : ''}`}
                  onClick={() => handleThresholdChange(0.35)}
                >
                  0.35 (Sensitive / Sepsis Alert)
                </button>
                <button
                  type="button"
                  className={`threshold-preset-btn ${decisionThreshold === 0.50 ? 'active' : ''}`}
                  onClick={() => handleThresholdChange(0.50)}
                >
                  0.50 (Standard Balanced)
                </button>
                <button
                  type="button"
                  className={`threshold-preset-btn ${decisionThreshold === 0.70 ? 'active' : ''}`}
                  onClick={() => handleThresholdChange(0.70)}
                >
                  0.70 (Model Default Cutoff)
                </button>
                <button
                  type="button"
                  className={`threshold-preset-btn ${decisionThreshold === 0.85 ? 'active' : ''}`}
                  onClick={() => handleThresholdChange(0.85)}
                >
                  0.85 (Strict / Bed Conservation)
                </button>
              </div>

              {singleResult && (
                <div className={`threshold-insight-banner ${singleResult.prediction === 1 ? 'admission' : 'no-admission'}`}>
                  {singleResult.prediction === 1 ? (
                    <>
                      <AlertTriangle size={16} />
                      <span>
                        Current Patient Risk (<strong>{(singleResult.risk_probability * 100).toFixed(1)}%</strong>) &ge; Threshold (<strong>{decisionThreshold.toFixed(2)}</strong>) &rarr; <strong style={{ textDecoration: 'underline' }}>Admission Predicted (1)</strong>. Drag slider above {singleResult.risk_probability.toFixed(2)} to classify as Observe / Discharge.
                      </span>
                    </>
                  ) : (
                    <>
                      <CheckCircle2 size={16} />
                      <span>
                        Current Patient Risk (<strong>{(singleResult.risk_probability * 100).toFixed(1)}%</strong>) &lt; Threshold (<strong>{decisionThreshold.toFixed(2)}</strong>) &rarr; <strong style={{ textDecoration: 'underline' }}>No Admission Required (0)</strong>. Drag slider below {singleResult.risk_probability.toFixed(2)} to trigger an admission decision.
                      </span>
                    </>
                  )}
                </div>
              )}
            </div>

            {/* Middle Row: Single Patient Prediction & Result */}
            <div className="prediction-grid">
              {/* Left Column: Form */}
              <div className="card">
                <div className="card-header">
                  <div className="card-icon">
                    <Activity size={18} />
                  </div>
                  <div>
                    <div className="card-title">Single Patient Prediction</div>
                    <div className="card-subtitle">Enter patient vital signs and clinical features to predict admission risk</div>
                  </div>
                </div>

                {/* Preset switcher */}
                <div style={{ display: 'flex', gap: '8px', marginBottom: '16px', alignItems: 'center' }}>
                  <select
                    className="form-input"
                    value={presetKey}
                    onChange={(e) => handlePresetChange(e.target.value)}
                    style={{ flex: 1, padding: '6px 10px', fontSize: '13px' }}
                  >
                    {Object.keys(PRESETS).map((k) => (
                      <option key={k} value={k}>
                        Preset: {k}
                      </option>
                    ))}
                    {presetKey === 'Custom Random' && <option value="Custom Random">Custom Random</option>}
                  </select>
                  <button type="button" className="btn btn-secondary" onClick={handleRandomize} style={{ padding: '6px 12px', fontSize: '12px' }}>
                    <Sparkles size={14} /> Random
                  </button>
                </div>

                <form onSubmit={handleSingleSubmit}>
                  {/* Row 1: Vitals */}
                  <div className="form-grid-4">
                    <div className="form-group">
                      <label className="form-label">Heart Rate (bpm)</label>
                      <input
                        type="number"
                        className="form-input"
                        value={formInputs.heart_rate}
                        onChange={(e) => handleInputChange('heart_rate', e.target.value)}
                        required
                        step="1"
                        min="25"
                        max="240"
                      />
                      <span style={{ fontSize: '10px', color: '#94a3b8' }}>30 – 220</span>
                    </div>

                    <div className="form-group">
                      <label className="form-label">Systolic BP (mmHg)</label>
                      <input
                        type="number"
                        className="form-input"
                        value={formInputs.systolic_bp}
                        onChange={(e) => handleInputChange('systolic_bp', e.target.value)}
                        required
                        step="1"
                        min="40"
                        max="300"
                      />
                      <span style={{ fontSize: '10px', color: '#94a3b8' }}>70 – 250</span>
                    </div>

                    <div className="form-group">
                      <label className="form-label">Diastolic BP (mmHg)</label>
                      <input
                        type="number"
                        className="form-input"
                        value={formInputs.diastolic_bp}
                        onChange={(e) => handleInputChange('diastolic_bp', e.target.value)}
                        required
                        step="1"
                        min="20"
                        max="200"
                      />
                      <span style={{ fontSize: '10px', color: '#94a3b8' }}>40 – 150</span>
                    </div>

                    <div className="form-group">
                      <label className="form-label">SpO₂ (%)</label>
                      <input
                        type="number"
                        className="form-input"
                        value={formInputs.spo2}
                        onChange={(e) => handleInputChange('spo2', e.target.value)}
                        required
                        step="1"
                        min="50"
                        max="100"
                      />
                      <span style={{ fontSize: '10px', color: '#94a3b8' }}>70 – 100</span>
                    </div>
                  </div>

                  {/* Row 2: Secondary Vitals & Age */}
                  <div className="form-grid-4">
                    <div className="form-group">
                      <label className="form-label">Resp Rate (breaths/min)</label>
                      <input
                        type="number"
                        className="form-input"
                        value={formInputs.respiratory_rate}
                        onChange={(e) => handleInputChange('respiratory_rate', e.target.value)}
                        required
                        step="1"
                        min="3"
                        max="80"
                      />
                      <span style={{ fontSize: '10px', color: '#94a3b8' }}>5 – 60</span>
                    </div>

                    <div className="form-group">
                      <label className="form-label">Temperature (°C)</label>
                      <input
                        type="number"
                        className="form-input"
                        value={formInputs.temperature_c}
                        onChange={(e) => handleInputChange('temperature_c', e.target.value)}
                        required
                        step="0.1"
                        min="30"
                        max="44"
                      />
                      <span style={{ fontSize: '10px', color: '#94a3b8' }}>30.0 – 42.0</span>
                    </div>

                    <div className="form-group">
                      <label className="form-label">Lactate (mmol/L)</label>
                      <input
                        type="number"
                        className="form-input"
                        value={formInputs.lactate}
                        onChange={(e) => handleInputChange('lactate', e.target.value)}
                        required
                        step="0.1"
                        min="0"
                        max="30"
                      />
                      <span style={{ fontSize: '10px', color: '#94a3b8' }}>0.0 – 20.0</span>
                    </div>

                    <div className="form-group">
                      <label className="form-label">Age (years)</label>
                      <input
                        type="number"
                        className="form-input"
                        value={formInputs.age}
                        onChange={(e) => handleInputChange('age', e.target.value)}
                        required
                        step="1"
                        min="18"
                        max="100"
                      />
                      <span style={{ fontSize: '10px', color: '#94a3b8' }}>18 – 100</span>
                    </div>
                  </div>

                  {/* Row 3: History & Context */}
                  <div className="form-grid-4">
                    <div className="form-group">
                      <label className="form-label">Prior Admissions (1 yr)</label>
                      <input
                        type="number"
                        className="form-input"
                        value={formInputs.previous_admissions}
                        onChange={(e) => handleInputChange('previous_admissions', e.target.value)}
                        required
                        min="0"
                        max="50"
                      />
                      <span style={{ fontSize: '10px', color: '#94a3b8' }}>0 – 50</span>
                    </div>

                    <div className="form-group">
                      <label className="form-label">Chronic Conditions</label>
                      <input
                        type="number"
                        className="form-input"
                        value={formInputs.chronic_conditions_count}
                        onChange={(e) => handleInputChange('chronic_conditions_count', e.target.value)}
                        required
                        min="0"
                        max="20"
                      />
                      <span style={{ fontSize: '10px', color: '#94a3b8' }}>0 – 20</span>
                    </div>

                    <div className="form-group">
                      <label className="form-label">Arrival Hour (0–23)</label>
                      <input
                        type="number"
                        className="form-input"
                        value={formInputs.observation_hour}
                        onChange={(e) => handleInputChange('observation_hour', e.target.value)}
                        required
                        min="0"
                        max="23"
                      />
                      <span style={{ fontSize: '10px', color: '#94a3b8' }}>0 – 23</span>
                    </div>

                    <div className="form-group">
                      <label className="form-label">Prior Obs Count</label>
                      <input
                        type="number"
                        className="form-input"
                        value={formInputs.prior_observation_count}
                        onChange={(e) => handleInputChange('prior_observation_count', e.target.value)}
                        required
                        min="0"
                        max="1000"
                      />
                      <span style={{ fontSize: '10px', color: '#94a3b8' }}>0 – 1000</span>
                    </div>
                  </div>

                  {singleError && (
                    <div style={{ color: '#b91c1c', background: '#fee2e2', padding: '10px 14px', borderRadius: '6px', fontSize: '12px', marginTop: '10px' }}>
                      <strong>Validation Error:</strong> {singleError}
                    </div>
                  )}

                  <div className="form-actions">
                    <button type="submit" className="btn btn-primary" disabled={singleLoading}>
                      <Activity size={16} /> {singleLoading ? 'Executing Inference...' : 'Predict Admission Risk'}
                    </button>
                    <button
                      type="button"
                      className="btn btn-secondary"
                      onClick={() => handlePresetChange('Typical / Low Risk Case')}
                    >
                      <RotateCcw size={16} /> Reset
                    </button>
                  </div>
                </form>
              </div>

              {/* Right Column: Prediction Result (matching reference image) */}
              <div className="card">
                <div className="card-header">
                  <div className="card-icon">
                    <ShieldCheck size={18} />
                  </div>
                  <div>
                    <div className="card-title">Prediction Result</div>
                    <div className="card-subtitle">Real-time model inference from FastAPI</div>
                  </div>
                </div>

                {singleResult ? (
                  <div>
                    {/* Stale Warning Banner if inputs were edited since inference */}
                    {isResultStale && (
                      <div style={{ background: '#fffbeb', border: '1px solid #fde68a', borderRadius: '8px', padding: '10px 12px', marginBottom: '14px', color: '#92400e' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontWeight: '700', fontSize: '12.5px', marginBottom: '2px' }}>
                          <AlertTriangle size={15} color="#d97706" /> Form Inputs Modified — Result is Stale
                        </div>
                        <div style={{ fontSize: '11.5px', lineHeight: '1.4' }}>
                          Displayed risk was calculated for previous values (Patient: {submittedSnapshot?.patient_id}, HR: {submittedSnapshot?.heart_rate} bpm, SBP: {submittedSnapshot?.systolic_bp} mmHg). Click <strong>Predict Admission Risk</strong> to recalculate for current inputs.
                        </div>
                      </div>
                    )}

                    <div className={`result-box ${singleResult.prediction === 1 || singleResult.review_flag ? 'high' : 'low'}`}>
                      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px', fontSize: '16px', fontWeight: '700' }}>
                        {singleResult.prediction === 1 || singleResult.review_flag ? (
                          <>
                            <AlertTriangle size={20} /> High Risk (Admission Predicted)
                          </>
                        ) : (
                          <>
                            <ShieldCheck size={20} /> Low Risk (No Admission)
                          </>
                        )}
                      </div>
                      <div style={{ fontSize: '12px', color: '#64748b', marginTop: '2px' }}>
                        Admission Risk Probability within 24 Hours
                      </div>
                      <div className="result-percentage">
                        {(singleResult.risk_probability * 100).toFixed(1)}%
                      </div>
                      <div className="progress-track">
                        <div
                          className={`progress-fill ${singleResult.prediction === 1 || singleResult.review_flag ? 'high' : 'low'}`}
                          style={{ width: `${Math.min(Math.max(singleResult.risk_probability * 100, 4), 100)}%` }}
                        />
                      </div>
                      <div style={{ fontSize: '12px', fontWeight: '500', color: '#64748b' }}>
                        {singleResult.risk_probability >= (singleResult.threshold || 0.7)
                          ? `Exceeds threshold (${(singleResult.threshold || 0.7).toFixed(2)})`
                          : `Below threshold (${(singleResult.threshold || 0.7).toFixed(2)})`}
                      </div>
                    </div>

                    <table className="data-table">
                      <tbody>
                        <tr>
                          <td className="label">Risk Probability</td>
                          <td className="value" style={{ fontFamily: 'monospace', fontWeight: '700' }}>
                            {singleResult.risk_probability.toFixed(4)}
                          </td>
                        </tr>
                        <tr>
                          <td className="label">Predicted Class</td>
                          <td className="value" style={{ color: singleResult.prediction === 1 ? '#dc2626' : '#16a34a', fontWeight: '700' }}>
                            {singleResult.prediction === 1 ? '1 (Admission Required)' : '0 (No Admission)'}
                          </td>
                        </tr>
                        <tr>
                          <td className="label">Decision Threshold</td>
                          <td className="value">{(singleResult.threshold || 0.7).toFixed(2)}</td>
                        </tr>
                        <tr>
                          <td className="label">Active Model</td>
                          <td className="value" style={{ fontFamily: 'monospace' }}>
                            {singleResult.model_name || 'logistic_regression'}
                          </td>
                        </tr>
                        <tr>
                          <td className="label">Inference Latency</td>
                          <td className="value">{singleResult.latencyMs} ms</td>
                        </tr>
                        <tr>
                          <td className="label">Evaluated At</td>
                          <td className="value">{resultTimestamp || 'Just now'}</td>
                        </tr>
                        {submittedSnapshot && (
                          <tr>
                            <td className="label">Input Snapshot</td>
                            <td className="value" style={{ fontSize: '11px', color: '#475569' }}>
                              {submittedSnapshot.patient_id} (Age: {submittedSnapshot.age}, HR: {submittedSnapshot.heart_rate}, BP: {submittedSnapshot.systolic_bp}/{submittedSnapshot.diastolic_bp})
                            </td>
                          </tr>
                        )}
                        <tr>
                          <td className="label">Data Nature</td>
                          <td className="value" style={{ color: '#d97706' }}>Synthetic Demonstration</td>
                        </tr>
                      </tbody>
                    </table>

                    {singleResult.pipeline_trace && (
                      <div className="kafka-hdfs-quick-badge">
                        <span className="live-pulse-dot" />
                        <span>
                          <strong>Live Pipeline ACK:</strong> Kafka <code>{singleResult.pipeline_trace.kafka?.topic || 'patient-observations'}</code> (Offset <strong style={{ color: '#2563eb' }}>#{singleResult.pipeline_trace.kafka?.offset ?? '--'}</strong>) &bull; HDFS Streamed
                        </span>
                      </div>
                    )}

                    <div style={{ fontSize: '11px', color: '#94a3b8', fontStyle: 'italic', borderTop: '1px dashed #e2e8f0', paddingTop: '8px' }}>
                      * Model score generated live by FastAPI using XGBoost. Patient observation streamed to Apache Kafka &amp; archived in Hadoop HDFS. All patient records are synthetic.
                    </div>
                  </div>
                ) : (
                  <div style={{ textAlign: 'center', padding: '60px 20px', color: '#94a3b8' }}>
                    <Activity size={48} style={{ opacity: 0.3, marginBottom: '12px' }} />
                    <div style={{ fontWeight: '600', color: '#64748b' }}>No Active Prediction</div>
                    <div style={{ fontSize: '12px', marginTop: '4px' }}>
                      Click "Predict Admission Risk" on the left to evaluate this patient profile through the Phase 6 FastAPI backend.
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Live Big Data Pipeline Execution Trace (Kafka -> Hadoop HDFS -> PySpark -> XGBoost) */}
            {singleResult && singleResult.pipeline_trace && (
              <LivePipelineExecutionTrace
                pipelineTrace={singleResult.pipeline_trace}
                singleResult={singleResult}
                submittedSnapshot={submittedSnapshot}
                onExplorePipeline={() => setActiveTab('pipeline')}
              />
            )}

            {/* Bottom Row: Batch Prediction & Model Information (matching reference image) */}
            {activeTab === 'dashboard' && (
              <div className="bottom-grid">
                {/* Bottom Col 1: Batch Prediction Form */}
                <div className="card">
                  <div className="card-header">
                    <div className="card-icon">
                      <FileText size={18} />
                    </div>
                    <div>
                      <div className="card-title">Batch Prediction</div>
                      <div className="card-subtitle">Upload a CSV file with multiple patient records (max 500)</div>
                    </div>
                  </div>

                  <label className="dropzone" style={{ display: 'block' }}>
                    <Upload size={28} style={{ color: '#3b82f6', marginBottom: '8px' }} />
                    <div style={{ fontSize: '13px', fontWeight: '600', color: '#1e293b' }}>
                      {batchFile ? batchFile.name : 'Drag and drop a CSV file here'}
                    </div>
                    <div style={{ fontSize: '11px', color: '#64748b', marginTop: '2px' }}>
                      {batchFile ? `${batchRecords.length} records verified` : 'or click to browse'}
                    </div>
                    <input
                      type="file"
                      accept=".csv"
                      onChange={handleFileUpload}
                      style={{ display: 'none' }}
                    />
                  </label>

                  <div style={{ fontSize: '11px', color: '#64748b', marginBottom: '14px', textAlign: 'center' }}>
                    Maximum 500 records • Must contain the 12 required features
                  </div>

                  {batchError && (
                    <div style={{ color: '#b91c1c', background: '#fee2e2', padding: '8px 12px', borderRadius: '6px', fontSize: '11px', marginBottom: '12px' }}>
                      {batchError}
                    </div>
                  )}

                  <div style={{ display: 'flex', gap: '8px' }}>
                    <button
                      type="button"
                      className="btn btn-primary"
                      onClick={handleBatchSubmit}
                      disabled={batchLoading || batchRecords.length === 0}
                      style={{ flex: 1 }}
                    >
                      {batchLoading ? 'Processing...' : '▶ Run Batch Prediction'}
                    </button>
                    <button
                      type="button"
                      className="btn btn-secondary"
                      onClick={downloadSampleBatchCSV}
                      title="Download a verified 5-record CSV template"
                    >
                      <Download size={14} /> Template
                    </button>
                  </div>
                </div>

                {/* Bottom Col 2: Batch Results Preview */}
                <div className="card">
                  <div className="card-header" style={{ justifyContent: 'space-between' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                      <div className="card-icon">
                        <Activity size={18} />
                      </div>
                      <div>
                        <div className="card-title">Batch Results (Preview)</div>
                        <div className="card-subtitle">
                          {batchResult ? `${batchResult.predictions.length} records scored in ${batchResult.latencyMs} ms` : 'Upload CSV to view preview'}
                        </div>
                      </div>
                    </div>
                    {batchResult && (
                      <button className="btn btn-secondary" onClick={downloadScoredCSV} style={{ padding: '4px 10px', fontSize: '12px' }}>
                        <Download size={13} /> Export CSV
                      </button>
                    )}
                  </div>

                  {batchResult && batchResult.predictions ? (
                    <div>
                      <table className="preview-table">
                        <thead>
                          <tr>
                            <th>#</th>
                            <th>Risk Probability</th>
                            <th>Prediction</th>
                            <th>Risk Level</th>
                          </tr>
                        </thead>
                        <tbody>
                          {batchResult.predictions.slice(0, 5).map((p, idx) => {
                            const isHigh = p.prediction === 1 || p.review_flag;
                            return (
                              <tr key={idx}>
                                <td style={{ fontWeight: '600' }}>{idx + 1}</td>
                                <td style={{ fontFamily: 'monospace' }}>{p.risk_probability.toFixed(4)}</td>
                                <td>{p.prediction === 1 ? 'Admission' : 'No Admission'}</td>
                                <td>
                                  <span className={`pill-badge ${isHigh ? 'high' : 'low'}`}>
                                    {isHigh ? 'High' : 'Low'}
                                  </span>
                                </td>
                              </tr>
                            );
                          })}
                        </tbody>
                      </table>
                      <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '12px', fontSize: '12px', color: '#64748b' }}>
                        <span>Showing 5 of {batchResult.predictions.length} records</span>
                        <span style={{ color: '#2563eb', cursor: 'pointer', fontWeight: '600' }} onClick={() => setActiveTab('batch')}>
                          View All Results →
                        </span>
                      </div>
                    </div>
                  ) : (
                    <div style={{ textAlign: 'center', padding: '36px 16px', color: '#64748b', background: '#f8fafc', borderRadius: '8px', border: '1px dashed #cbd5e1' }}>
                      <FileText size={28} style={{ color: '#94a3b8', marginBottom: '8px' }} />
                      <div style={{ fontWeight: '600', fontSize: '13px', color: '#334155' }}>No Batch Results Yet</div>
                      <div style={{ fontSize: '12px', color: '#64748b', marginTop: '2px' }}>
                        Upload a clinical CSV file above or switch to the Batch Prediction tab to generate real-time model scores.
                      </div>
                    </div>
                  )}
                </div>

                {/* Bottom Col 3: Compact Model Info & System Status */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                  <div className="card" style={{ flex: 1 }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '10px' }}>
                      <Info size={16} color="#2563eb" />
                      <div style={{ fontSize: '13px', fontWeight: '700' }}>Model Information</div>
                    </div>
                    <table className="data-table" style={{ fontSize: '12px' }}>
                      <tbody>
                        <tr>
                          <td className="label">Model Name</td>
                          <td className="value" style={{ color: '#16a34a' }}>● {modelStatus.modelName}</td>
                        </tr>
                        <tr>
                          <td className="label">Version</td>
                          <td className="value">1.0</td>
                        </tr>
                        <tr>
                          <td className="label">Features</td>
                          <td className="value">12</td>
                        </tr>
                        <tr>
                          <td className="label">Trained On</td>
                          <td className="value">50 records (synthetic)</td>
                        </tr>
                        <tr>
                          <td className="label">Decision Threshold</td>
                          <td className="value">{modelStatus.threshold.toFixed(2)}</td>
                        </tr>
                        <tr>
                          <td className="label">Model Status</td>
                          <td className="value" style={{ color: '#16a34a' }}>● Ready</td>
                        </tr>
                        <tr>
                          <td className="label">Is Experimental</td>
                          <td className="value" style={{ color: '#dc2626' }}>⚠️ Yes</td>
                        </tr>
                      </tbody>
                    </table>
                    <button
                      className="btn btn-secondary"
                      onClick={() => setActiveTab('model')}
                      style={{ width: '100%', padding: '5px', fontSize: '11px' }}
                    >
                      📖 View Full Model Info
                    </button>
                  </div>

                  <div className="card" style={{ padding: '14px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '13px', fontWeight: '700' }}>
                        <Server size={15} color="#2563eb" /> System Status
                      </div>
                      <button className="btn btn-secondary" onClick={checkBackend} style={{ padding: '3px 8px', fontSize: '11px' }}>
                        <RotateCcw size={11} /> Check Status
                      </button>
                    </div>
                    <div style={{ fontSize: '12px', display: 'flex', justifyContent: 'space-between', padding: '3px 0' }}>
                      <span style={{ color: '#64748b' }}>API Status:</span>
                      <span style={{ fontWeight: '600', color: apiStatus.online ? '#16a34a' : '#dc2626' }}>
                        ● {apiStatus.online ? 'Online' : 'Offline'}
                      </span>
                    </div>
                    <div style={{ fontSize: '12px', display: 'flex', justifyContent: 'space-between', padding: '3px 0' }}>
                      <span style={{ color: '#64748b' }}>Response Time:</span>
                      <span style={{ fontWeight: '600', fontFamily: 'monospace' }}>{apiStatus.latencyMs} ms</span>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </>
        )}

        {/* -----------------------------------------------------------------
            DEDICATED BATCH PREDICTION VIEW
           ----------------------------------------------------------------- */}
        {activeTab === 'batch' && (
          <div className="card">
            <div className="card-header">
              <div className="card-icon">
                <FileText size={20} />
              </div>
              <div>
                <div className="card-title">Batch Patient Risk Inference</div>
                <div className="card-subtitle">Upload synthetic clinical CSV files up to 500 rows for atomic scoring</div>
              </div>
            </div>

            <div style={{ display: 'flex', gap: '16px', marginBottom: '20px' }}>
              <div style={{ flex: 1 }}>
                <label className="dropzone" style={{ display: 'block' }}>
                  <Upload size={32} style={{ color: '#2563eb', marginBottom: '10px' }} />
                  <div style={{ fontSize: '15px', fontWeight: '700', color: '#0f172a' }}>
                    {batchFile ? batchFile.name : 'Select or drop a CSV file here'}
                  </div>
                  <div style={{ fontSize: '12px', color: '#64748b', marginTop: '4px' }}>
                    {batchFile ? `${batchRecords.length} patient records loaded` : 'Supports up to 500 records'}
                  </div>
                  <input type="file" accept=".csv" onChange={handleFileUpload} style={{ display: 'none' }} />
                </label>
              </div>

              <div style={{ width: '320px', background: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: '10px', padding: '16px' }}>
                <div style={{ fontSize: '13px', fontWeight: '700', marginBottom: '6px' }}>Instructions & Template</div>
                <div style={{ fontSize: '12px', color: '#64748b', lineHeight: '1.5', marginBottom: '12px' }}>
                  Must include the 12 model features (age, heart_rate, systolic_bp, diastolic_bp, spo2, respiratory_rate, temperature_c, previous_admissions, chronic_conditions_count, lactate, observation_hour, prior_observation_count).
                </div>
                <button className="btn btn-secondary" onClick={downloadSampleBatchCSV} style={{ width: '100%' }}>
                  <Download size={14} /> Download Sample Template CSV
                </button>
              </div>
            </div>

            {batchError && (
              <div style={{ background: '#fee2e2', border: '1px solid #fecaca', color: '#b91c1c', padding: '12px', borderRadius: '8px', fontSize: '13px', marginBottom: '16px' }}>
                {batchError}
              </div>
            )}

            <button
              className="btn btn-primary"
              onClick={handleBatchSubmit}
              disabled={batchLoading || batchRecords.length === 0}
              style={{ marginBottom: '24px' }}
            >
              {batchLoading ? 'Executing Inference...' : `🚀 Run Batch Prediction (${batchRecords.length} records)`}
            </button>

            {batchResult && (
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
                  <div style={{ fontSize: '15px', fontWeight: '700' }}>
                    Scored Results ({batchResult.predictions.length} records in {batchResult.latencyMs} ms)
                  </div>
                  <button className="btn btn-secondary" onClick={downloadScoredCSV}>
                    <Download size={14} /> Export All as CSV
                  </button>
                </div>

                <div style={{ maxHeight: '450px', overflowY: 'auto', border: '1px solid #e2e8f0', borderRadius: '8px' }}>
                  <table className="preview-table">
                    <thead style={{ position: 'sticky', top: 0, background: '#f8fafc', zIndex: 1 }}>
                      <tr>
                        <th>#</th>
                        <th>Patient ID</th>
                        <th>Risk Probability</th>
                        <th>Risk %</th>
                        <th>Prediction</th>
                        <th>Review Flag</th>
                      </tr>
                    </thead>
                    <tbody>
                      {batchResult.predictions.map((p, idx) => {
                        const isHigh = p.prediction === 1 || p.review_flag;
                        return (
                          <tr key={idx}>
                            <td>{idx + 1}</td>
                            <td style={{ fontFamily: 'monospace', fontWeight: '600' }}>{p.patient_id}</td>
                            <td style={{ fontFamily: 'monospace' }}>{p.risk_probability.toFixed(4)}</td>
                            <td style={{ fontWeight: '700', color: isHigh ? '#dc2626' : '#16a34a' }}>
                              {(p.risk_probability * 100).toFixed(1)}%
                            </td>
                            <td>{p.prediction === 1 ? 'Admission (1)' : 'No Admission (0)'}</td>
                            <td>
                              <span className={`pill-badge ${isHigh ? 'high' : 'low'}`}>
                                {isHigh ? 'Flagged High' : 'Standard'}
                              </span>
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>
        )}

        {/* -----------------------------------------------------------------
            MODEL INFORMATION VIEW
           ----------------------------------------------------------------- */}
        {activeTab === 'model' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
            <div className="card">
              <div className="card-header">
                <div className="card-icon">
                  <Brain size={20} />
                </div>
                <div>
                  <div className="card-title">Model Governance & Architecture</div>
                  <div className="card-subtitle">Technical details from Phase 5 scikit-learn registry</div>
                </div>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '14px', marginBottom: '20px' }}>
                <div style={{ background: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: '8px', padding: '14px' }}>
                  <div style={{ fontSize: '11px', textTransform: 'uppercase', color: '#64748b', fontWeight: '700' }}>Algorithm</div>
                  <div style={{ fontSize: '16px', fontWeight: '700', marginTop: '2px' }}>LogisticRegression</div>
                  <div style={{ fontSize: '12px', color: '#64748b', marginTop: '2px' }}>L2 penalty, lbfgs solver, C=1.0</div>
                </div>
                <div style={{ background: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: '8px', padding: '14px' }}>
                  <div style={{ fontSize: '11px', textTransform: 'uppercase', color: '#64748b', fontWeight: '700' }}>Cutoff Threshold</div>
                  <div style={{ fontSize: '16px', fontWeight: '700', marginTop: '2px' }}>0.70 (Review Flag)</div>
                  <div style={{ fontSize: '12px', color: '#64748b', marginTop: '2px' }}>Target: admission_within_24h</div>
                </div>
                <div style={{ background: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: '8px', padding: '14px' }}>
                  <div style={{ fontSize: '11px', textTransform: 'uppercase', color: '#64748b', fontWeight: '700' }}>Dataset Maturity</div>
                  <div style={{ fontSize: '16px', fontWeight: '700', color: '#d97706', marginTop: '2px' }}>Smoke-Test (50 rows)</div>
                  <div style={{ fontSize: '12px', color: '#64748b', marginTop: '2px' }}>Academic prototype, non-clinical</div>
                </div>
              </div>

              <div style={{ background: '#fffbeb', border: '1px solid #fde68a', borderRadius: '8px', padding: '16px', color: '#92400e', fontSize: '13px', lineHeight: '1.6' }}>
                <strong>⚠️ Statistical Evaluation Notice Regarding Smoke-Test Metrics:</strong><br />
                The active model was trained on an initial smoke-test dataset (50 rows total) containing 1 positive admission case, which was grouped into the training split. Consequently, the held-out test split (11 observations) contained <strong>zero positive cases (0% prevalence)</strong>.
                While test accuracy is mathematically reported as 1.0 (100%), this is merely an artifact of predicting the majority negative class. <strong>PR-AUC and ROC-AUC are mathematically undefined</strong> due to the absence of positive test labels.
              </div>
            </div>
          </div>
        )}

        {/* -----------------------------------------------------------------
            SYSTEM STATUS & OBSERVABILITY VIEW
           ----------------------------------------------------------------- */}
        {activeTab === 'system' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
            <div className="card">
              <div className="card-header" style={{ justifyContent: 'space-between' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                  <div className="card-icon">
                    <Server size={20} />
                  </div>
                  <div>
                    <div className="card-title">Live API Endpoint Diagnostics</div>
                    <div className="card-subtitle">Real-time health probes targeting {api.displayUrl}</div>
                  </div>
                </div>
                <button className="btn btn-primary" onClick={checkBackend}>
                  <RotateCcw size={15} /> Probe All Endpoints
                </button>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '14px', marginBottom: '20px' }}>
                {[
                  { name: 'GET /health', desc: 'Process Liveness', ok: apiStatus.online, latency: apiStatus.latencyMs },
                  { name: 'GET /ready', desc: 'Model Readiness', ok: modelStatus.ready, latency: apiStatus.latencyMs },
                  { name: 'GET /model-info', desc: 'Model Specifications', ok: modelStatus.ready, latency: apiStatus.latencyMs },
                  { name: 'GET /metadata', desc: 'Registry Governance', ok: apiStatus.online, latency: apiStatus.latencyMs },
                ].map((ep, i) => (
                  <div key={i} style={{ background: ep.ok ? '#f0fdf4' : '#fef2f2', border: `1px solid ${ep.ok ? '#bbf7d0' : '#fecaca'}`, borderRadius: '10px', padding: '14px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <span style={{ fontFamily: 'monospace', fontWeight: '700', fontSize: '13px' }}>{ep.name}</span>
                      {ep.ok ? <CheckCircle2 size={16} color="#16a34a" /> : <XCircle size={16} color="#dc2626" />}
                    </div>
                    <div style={{ fontSize: '11px', color: '#64748b', marginTop: '2px' }}>{ep.desc}</div>
                    <div style={{ fontSize: '14px', fontWeight: '700', color: ep.ok ? '#16a34a' : '#dc2626', marginTop: '10px' }}>
                      {ep.ok ? 'Operational' : 'Unavailable'}
                    </div>
                    <div style={{ fontSize: '11px', color: '#64748b' }}>Latency: {ep.latency} ms</div>
                  </div>
                ))}
              </div>

              <div style={{ background: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: '8px', padding: '16px' }}>
                <div style={{ fontSize: '14px', fontWeight: '700', marginBottom: '8px' }}>Troubleshooting Checklist</div>
                <ul style={{ fontSize: '13px', color: '#475569', lineHeight: '1.8', paddingLeft: '20px' }}>
                  <li><strong>Docker Compose:</strong> Confirm container is running via <code>docker compose ps</code>.</li>
                  <li><strong>Local Host:</strong> Ensure FastAPI is running on port 8000: <code>uvicorn app.api.main:app --port 8000</code>.</li>
                  <li><strong>Artifact Missing (503):</strong> Verify <code>models/risk_model.joblib</code> exists.</li>
                </ul>
              </div>
            </div>
          </div>
        )}

        {/* -----------------------------------------------------------------
            BIG DATA PIPELINE VIEW
           ----------------------------------------------------------------- */}
        {activeTab === 'pipeline' && <BigDataPipeline />}

        {/* -----------------------------------------------------------------
            ABOUT VIEW
           ----------------------------------------------------------------- */}
        {activeTab === 'about' && (
          <div className="card">
            <div className="card-header">
              <div className="card-icon">
                <Info size={20} />
              </div>
              <div>
                <div className="card-title">About the Big Data Analytics Platform</div>
                <div className="card-subtitle">End-to-End Emergency Admission Risk Prediction Pipeline</div>
              </div>
            </div>

            <div className="alert-banner">
              <AlertTriangle size={18} style={{ flexShrink: 0 }} />
              <div>
                <strong>ACADEMIC SOFTWARE ENGINEERING DEMONSTRATION ONLY:</strong><br />
                All patient profiles, telemetry observations, laboratory markers, and hospital admission labels are <strong>100% synthetically generated</strong>.
                This software is an educational engineering showcase and must <strong>never</strong> be used for actual patient care, diagnosis, or triage.
              </div>
            </div>

            <div style={{ fontSize: '14px', fontWeight: '700', marginBottom: '12px' }}>8-Phase Big Data Analytics Lifecycle</div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
              {[
                { phase: 'Phase 1', title: 'Synthetic Data Generator', desc: 'Simulates realistic ED vital signs without Protected Health Information (PHI).' },
                { phase: 'Phase 2', title: 'Apache Kafka Streaming', desc: 'Publishes streaming observations to the patient-observations topic.' },
                { phase: 'Phase 3', title: 'Hadoop HDFS Ingestion', desc: 'Archives raw events into distributed HDFS storage with checksum manifests.' },
                { phase: 'Phase 4', title: 'PySpark Feature Engineering', desc: 'Calculates rolling window aggregations, deltas, and writes partitioned Parquet.' },
                { phase: 'Phase 5', title: 'Scikit-Learn ML Training', desc: 'Trains and registers versioned classification models and governance metadata.' },
                { phase: 'Phase 6', title: 'FastAPI Serving Microservice', desc: 'Exposes /health, /ready, /metadata, and /predict with Pydantic validation.' },
                { phase: 'Phase 7/8', title: 'Modern React & Streamlit Analytics', desc: 'Interactive SaaS dashboard, single & batch inference, and pipeline observability.' },
              ].map((p, idx) => (
                <div key={idx} style={{ display: 'flex', gap: '12px', padding: '10px 0', borderBottom: '1px solid #f1f5f9' }}>
                  <span style={{ background: '#eff6ff', color: '#2563eb', fontWeight: '700', fontSize: '11px', padding: '3px 8px', borderRadius: '4px', height: 'fit-content' }}>
                    {p.phase}
                  </span>
                  <div>
                    <div style={{ fontSize: '13.5px', fontWeight: '600' }}>{p.title}</div>
                    <div style={{ fontSize: '12px', color: '#64748b' }}>{p.desc}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
