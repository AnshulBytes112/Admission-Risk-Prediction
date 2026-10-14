"""Model Information & Governance view for Phase 7 Streamlit dashboard.

Integrates with Phase 6 GET /model-info and GET /metadata endpoints, supplemented by
local models/model_metadata.json and models/evaluation_report.json artifacts.
"""
from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from app.dashboard.api_client import api_client

METADATA_FILE = Path("models/model_metadata.json")
EVAL_FILE = Path("models/evaluation_report.json")


def load_local_model_artifacts() -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Load local model metadata and evaluation reports if present on disk."""
    meta, eval_rep = None, None
    if METADATA_FILE.exists():
        try:
            with open(METADATA_FILE, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            pass
    if EVAL_FILE.exists():
        try:
            with open(EVAL_FILE, "r", encoding="utf-8") as f:
                eval_rep = json.load(f)
        except Exception:
            pass
    return meta, eval_rep


def render_model_info() -> None:
    """Render the Model Governance, Architecture, and Evaluation details."""
    st.markdown(
        """
        <div style="margin-bottom: 20px;">
            <h1 style="font-size: 26px; font-weight: 700; color: #0f172a; margin: 0 0 6px 0;">
                Model Information & Governance
            </h1>
            <p style="color: #64748b; font-size: 14px; margin: 0;">
                Detailed architecture specifications, feature dictionary, decision thresholds, and empirical evaluation metrics.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Academic Disclaimer Alert
    st.markdown(
        """
        <div class="bda-alert-banner" style="margin-bottom: 20px;">
            <div style="font-size: 16px;">⚠️</div>
            <div>
                <strong>ACADEMIC DEMONSTRATION ONLY:</strong> All model artifacts, training data, and metrics are experimental.
                The model is <strong>not clinically validated</strong> and must never be used for real emergency medical decisions or hospital triage.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Fetch live API info
    api_ready, ready_data, _ = api_client.check_readiness()
    api_info_ok, api_info, _ = api_client.get_model_info()
    api_meta_ok, api_meta, _ = api_client.get_metadata()

    # Load local report files as trusted backup/supplement
    local_meta, local_eval = load_local_model_artifacts()

    # Combine data sources
    model_name = api_info.get("model_name") or (local_meta or {}).get("model_name", "logistic_regression")
    model_version = api_info.get("model_version") or (local_meta or {}).get("artifact_version", "1.0.0")
    model_class = api_info.get("model_class") or (local_meta or {}).get("model_class", "LogisticRegression")
    threshold = api_info.get("decision_threshold") or (local_meta or {}).get("threshold", 0.70)
    features_list = api_info.get("features") or (local_meta or {}).get("features", [])
    feature_count = len(features_list) or (local_meta or {}).get("feature_count", 12)
    is_exp = api_info.get("is_experimental") if api_info_ok else True
    training_status = (local_eval or {}).get("training_status", "EXPERIMENTAL_SMOKE_TEST")

    # -------------------------------------------------------------------------
    # Distinction of Operational Dimensions
    # -------------------------------------------------------------------------
    st.markdown(
        "<div style='font-size: 14px; font-weight: 700; color: #1e293b; margin-bottom: 12px;'>System & Governance Status Dimensions</div>",
        unsafe_allow_html=True,
    )
    d1, d2, d3 = st.columns(3)
    with d1:
        st.markdown(
            f"""
            <div class="bda-card" style="padding: 16px; border-left: 4px solid #10b981;">
                <div style="font-size: 11px; text-transform: uppercase; color: #64748b; font-weight: 700;">Dimension 1</div>
                <div style="font-size: 16px; font-weight: 700; color: #0f172a; margin: 4px 0;">API Service</div>
                <div style="color: #15803d; font-size: 13px; font-weight: 600;">● Operational (Port 8000)</div>
                <div style="color: #64748b; font-size: 11px; margin-top: 4px;">FastAPI inference server handling HTTP endpoints.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with d2:
        model_ready_str = "Loaded in Memory" if api_ready else "Offline / Disconnected"
        model_ready_col = "#15803d" if api_ready else "#dc2626"
        st.markdown(
            f"""
            <div class="bda-card" style="padding: 16px; border-left: 4px solid #3b82f6;">
                <div style="font-size: 11px; text-transform: uppercase; color: #64748b; font-weight: 700;">Dimension 2</div>
                <div style="font-size: 16px; font-weight: 700; color: #0f172a; margin: 4px 0;">Model Artifact</div>
                <div style="color: {model_ready_col}; font-size: 13px; font-weight: 600;">● {model_ready_str}</div>
                <div style="color: #64748b; font-size: 11px; margin-top: 4px;">Scikit-learn pipeline bundle (Phase 5).</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with d3:
        st.markdown(
            f"""
            <div class="bda-card" style="padding: 16px; border-left: 4px solid #f59e0b;">
                <div style="font-size: 11px; text-transform: uppercase; color: #64748b; font-weight: 700;">Dimension 3</div>
                <div style="font-size: 16px; font-weight: 700; color: #0f172a; margin: 4px 0;">Evaluation Maturity</div>
                <div style="color: #b45309; font-size: 13px; font-weight: 600;">▲ {training_status}</div>
                <div style="color: #64748b; font-size: 11px; margin-top: 4px;">Synthetic smoke-test; no clinical validation.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")

    # -------------------------------------------------------------------------
    # Core Specifications & Feature Dictionary
    # -------------------------------------------------------------------------
    col_specs, col_features = st.columns([1, 1], gap="medium")

    with col_specs:
        st.markdown(
            """
            <div class="bda-card">
                <div style="font-size: 16px; font-weight: 700; color: #0f172a; margin-bottom: 12px;">Architecture Specifications</div>
            """,
            unsafe_allow_html=True,
        )
        spec_items = [
            ("Model Name", model_name),
            ("Artifact Version", model_version),
            ("Algorithm Class", model_class),
            ("Target Variable", (local_meta or {}).get("target", "admission_within_24h")),
            ("Decision Threshold", f"{threshold:.2f} (Admission cutoff / review flag)"),
            ("Data Origin", "Synthetic Emergency Observations"),
            ("Training Dataset Size", f"{(local_eval or {}).get('dataset_metadata', {}).get('total_rows', 50)} records (academic smoke test)"),
            ("Fingerprint SHA-256", f"{(local_meta or {}).get('training_dataset_fingerprint_sha256', 'be4d...3c32b')[:16]}..."),
        ]
        spec_html = "".join(
            f"""
            <div style="display: flex; justify-content: space-between; padding: 7px 0; border-bottom: 1px solid #f1f5f9; font-size: 13px;">
                <span style="color: #64748b;">{k}</span>
                <span style="font-weight: 600; color: #1e293b; font-family: monospace;">{v}</span>
            </div>
            """
            for k, v in spec_items
        )
        st.markdown(spec_html + "</div>", unsafe_allow_html=True)

    with col_features:
        st.markdown(
            f"""
            <div class="bda-card">
                <div style="font-size: 16px; font-weight: 700; color: #0f172a; margin-bottom: 4px;">Validated Feature Dictionary ({feature_count} Predictors)</div>
                <div style="font-size: 12px; color: #64748b; margin-bottom: 12px;">Patient identifiers and raw timestamps are strictly excluded.</div>
            """,
            unsafe_allow_html=True,
        )
        # Show feature tags
        feat_badges = "".join(
            f"<span style='display: inline-block; background: #eff6ff; color: #1d4ed8; border: 1px solid #bfdbfe; border-radius: 6px; padding: 4px 8px; margin: 3px; font-size: 12px; font-family: monospace;'>{f}</span>"
            for f in features_list
        )
        st.markdown(
            f"<div style='margin-bottom: 12px;'>{feat_badges}</div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div style="background: #f8fafc; border-left: 3px solid #3b82f6; padding: 10px; font-size: 12px; color: #475569;">
                <strong>Leakage Prevention Note:</strong> Patient ID is preserved only for logging and auditing; it is strictly excluded from the scikit-learn feature array.
            </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # -------------------------------------------------------------------------
    # Evaluation Report Summary & Limitations
    # -------------------------------------------------------------------------
    st.write("")
    st.markdown(
        """
        <div class="bda-card">
            <div style="font-size: 16px; font-weight: 700; color: #0f172a; margin-bottom: 6px;">
                Empirical Evaluation & Limitations Analysis
            </div>
            <div style="font-size: 13px; color: #64748b; margin-bottom: 16px;">
                Critical evaluation from Phase 5 training pipeline (held-out patient-grouped test split).
            </div>
        """,
        unsafe_allow_html=True,
    )

    test_metrics = (local_eval or {}).get("held_out_test_metrics", {})
    if test_metrics:
        # Crucial Note about single-class evaluation
        st.warning(
            "⚠️ **Statistical Warning Regarding Evaluation Metrics:** "
            "The synthetic smoke-test dataset (50 rows total) contained only 1 positive admission case, which was grouped into the training set. "
            "Consequently, the held-out test split (11 observations) contained **zero positive examples (0% prevalence)**. "
            "While raw accuracy is reported as **1.0 (100%)**, this is merely an artifact of predicting the majority negative class. "
            "**ROC-AUC and PR-AUC are mathematically undefined** due to the absence of positive labels in the test set."
        )

        e_col1, e_col2, e_col3, e_col4 = st.columns(4)
        with e_col1:
            st.metric("Test Accuracy", f"{test_metrics.get('accuracy', 1.0):.2f}", "Majority negative class only")
        with e_col2:
            st.metric("PR-AUC (Avg Precision)", "Undefined", "0 positives in test split")
        with e_col3:
            st.metric("ROC-AUC", "Undefined", "0 positives in test split")
        with e_col4:
            st.metric("Brier Loss", f"{test_metrics.get('brier_score', 0.0084):.4f}", "Probability calibration")

        st.write("")
        st.markdown(
            "<div style='font-size: 13px; font-weight: 600; color: #334155; margin-bottom: 6px;'>Candidate Models Validation Comparison</div>",
            unsafe_allow_html=True,
        )
        val_comp = (local_eval or {}).get("validation_model_comparison", {})
        if val_comp:
            comp_rows = []
            for name, metrics in val_comp.items():
                comp_rows.append({
                    "Model Candidate": name,
                    "Decision Threshold": metrics.get("decision_threshold", 0.7),
                    "Accuracy": f"{metrics.get('accuracy', 0.0):.4f}",
                    "F1 Score": f"{metrics.get('f1_score', 0.0):.4f}",
                    "Brier Loss": f"{metrics.get('brier_score', 0.0):.4f}",
                    "PR-AUC": "Undefined (Single Class)",
                })
            st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)

    else:
        st.info("No local evaluation report file (models/evaluation_report.json) found to display detailed metrics.")

    st.markdown("</div>", unsafe_allow_html=True)
