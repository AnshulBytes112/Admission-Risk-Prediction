import React, { useState } from 'react';
import {
  ChevronLeft,
  ChevronRight,
  Play,
  RotateCcw,
  Terminal,
  Layers,
  ArrowRight,
  CheckCircle2,
  Lightbulb,
  FileCode,
  ChevronDown,
  ChevronUp,
  Cpu,
  Database,
  ExternalLink,
} from 'lucide-react';

export default function PipelineWalkthrough({
  stages,
  currentStep,
  onStepChange,
  liveStatus,
}) {
  const [showTechnicalAccordion, setShowTechnicalAccordion] = useState(false);

  const stageIndex = Math.max(0, Math.min(currentStep - 1, stages.length - 1));
  const currentStage = stages[stageIndex];

  const handlePrev = () => {
    if (currentStep > 1) {
      onStepChange(currentStep - 1);
    }
  };

  const handleNext = () => {
    if (currentStep < stages.length) {
      onStepChange(currentStep + 1);
    }
  };

  const handleReset = () => {
    onStepChange(1);
  };

  return (
    <div className="card walkthrough-container">
      {/* Walkthrough Header & Controls */}
      <div className="walkthrough-header">
        <div>
          <div className="walkthrough-badge">
            <Play size={13} /> GUIDED COLLEGE PRESENTATION MODE
          </div>
          <h2 className="walkthrough-title">Step-by-Step Big Data Pipeline Demonstration</h2>
          <p className="walkthrough-sub">
            Follow the exact data journey from synthetic generation to streaming, distributed Hadoop HDFS storage, PySpark batch processing, and live inference.
          </p>
        </div>

        {/* Navigation Controls */}
        <div className="walkthrough-controls">
          <button
            className="btn btn-secondary nav-btn"
            onClick={handlePrev}
            disabled={currentStep === 1}
          >
            <ChevronLeft size={16} /> Previous Step
          </button>
          <span className="step-counter-pill">
            Step {currentStep} of {stages.length}
          </span>
          <button
            className="btn btn-primary nav-btn"
            onClick={handleNext}
            disabled={currentStep === stages.length}
          >
            Next Step <ChevronRight size={16} />
          </button>
          <button
            className="btn btn-secondary icon-only"
            title="Restart Walkthrough"
            onClick={handleReset}
          >
            <RotateCcw size={15} />
          </button>
        </div>
      </div>

      {/* Progress Dots Bar */}
      <div className="walkthrough-progress-track">
        {stages.map((st) => (
          <div
            key={st.id}
            className={`progress-step-indicator ${
              st.step === currentStep
                ? 'active'
                : st.step < currentStep
                ? 'completed'
                : 'upcoming'
            }`}
            onClick={() => onStepChange(st.step)}
            title={`Step ${st.step}: ${st.name}`}
          >
            <span className="step-num">{st.step}</span>
            <span className="step-label">{st.name.split(' ')[0]}</span>
          </div>
        ))}
      </div>

      {/* Main Walkthrough Card for Current Stage */}
      <div className="walkthrough-stage-body">
        {/* Stage Identity Bar */}
        <div className="walkthrough-stage-identity">
          <div className="stage-top-meta">
            <span className="stage-step-tag">Step {currentStage.step} of 13</span>
            <span className="stage-group-tag">{currentStage.group.toUpperCase()} LAYER</span>
            <span className="stage-tech-tag">{currentStage.technology}</span>
          </div>
          <h3 className="walkthrough-current-name">{currentStage.name}</h3>
          <p className="walkthrough-current-sub">{currentStage.shortDescription}</p>
        </div>

        {/* Presentation Script Banner */}
        <div className="presentation-script-banner highlight">
          <div className="script-banner-title">
            <Terminal size={14} /> Recommended Presentation Narrative (Read aloud to evaluators):
          </div>
          <p className="script-banner-text">"{currentStage.presentationScript}"</p>
        </div>

        {/* Two Column Explanation: College High-Level vs Deep Technical */}
        <div className="walkthrough-dual-columns">
          {/* Column 1: Presentation Summary */}
          <div className="dual-card summary-card">
            <div className="dual-card-header">
              <Lightbulb size={16} className="text-amber" /> College Presentation Explanation
            </div>
            <p className="dual-card-body">{currentStage.shortDescription}</p>
            <div className="takeaway-pill">
              <strong>Key Takeaway:</strong> {currentStage.keyTakeaway}
            </div>
          </div>

          {/* Column 2: Technical Explanation */}
          <div className="dual-card technical-card">
            <div className="dual-card-header">
              <Cpu size={16} className="text-blue" /> In-Depth Technical Mechanism
            </div>
            <p className="dual-card-body">{currentStage.technicalExplanation}</p>
          </div>
        </div>

        {/* Input, Transformation, Output Grid */}
        <div className="walkthrough-io-row">
          <div className="walkthrough-io-box">
            <span className="io-tag in">INPUT</span>
            <div className="io-text">{currentStage.input}</div>
          </div>
          <div className="io-divider">
            <ArrowRight size={20} />
          </div>
          <div className="walkthrough-io-box">
            <span className="io-tag proc">PROCESSING</span>
            <div className="io-text">{currentStage.processing}</div>
          </div>
          <div className="io-divider">
            <ArrowRight size={20} />
          </div>
          <div className="walkthrough-io-box">
            <span className="io-tag out">OUTPUT</span>
            <div className="io-text">{currentStage.output}</div>
          </div>
        </div>

        {/* Technical Details Accordion (File paths, commands, code) */}
        <div className="technical-details-accordion">
          <button
            className="accordion-toggle-btn"
            onClick={() => setShowTechnicalAccordion(!showTechnicalAccordion)}
          >
            <span className="toggle-label">
              <FileCode size={15} /> Show Technical Details, Target Paths & Verified Evidence
            </span>
            {showTechnicalAccordion ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
          </button>

          {showTechnicalAccordion && (
            <div className="accordion-content-panel">
              <div className="meta-detail-row">
                <span className="meta-key">Target Path / Identifier:</span>
                <code className="meta-val">{currentStage.location}</code>
              </div>
              <div className="meta-detail-row">
                <span className="meta-key">Baseline Evidence Source:</span>
                <span className="meta-val">{currentStage.evidenceSource}</span>
              </div>
              <div className="meta-detail-row">
                <span className="meta-key">Common Failure Scenarios:</span>
                <span className="meta-val text-muted">{currentStage.failureScenarios}</span>
              </div>
              <div className="historical-notice-box">
                <em>Notice:</em> This walkthrough references genuine verified artifacts and historical execution manifests stored in your repository. It does not re-trigger cluster processing during page inspection.
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
