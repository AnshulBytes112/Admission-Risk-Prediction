"""About & Academic Context view for Phase 7 Streamlit dashboard.

Details the Big Data Analytics (BDA) project architecture, pipeline stages,
and non-negotiable synthetic-data academic limitations.
"""
from __future__ import annotations

import streamlit as st


def render_about() -> None:
    """Render the About, Pipeline Architecture, and Academic Context page."""
    st.markdown(
        """
        <div style="margin-bottom: 20px;">
            <h1 style="font-size: 26px; font-weight: 700; color: #0f172a; margin: 0 0 6px 0;">
                About the BDA Emergency Admission Project
            </h1>
            <p style="color: #64748b; font-size: 14px; margin: 0;">
                Comprehensive Big Data Analytics pipeline demonstration for real-time telemetry processing and risk scoring.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Prominent Disclaimer Alert
    st.markdown(
        """
        <div class="bda-alert-banner" style="margin-bottom: 24px;">
            <div style="font-size: 18px;">⚠️</div>
            <div>
                <strong>ACADEMIC SOFTWARE ENGINEERING DEMONSTRATION ONLY</strong><br>
                All patient records, physiological measurements, laboratory markers, and admission outcomes are entirely <strong>synthetic</strong>.
                The underlying machine learning model is experimental, trained on limited synthetic data, and is <strong>not clinically validated</strong>.
                It must <strong>never</strong> be used for actual medical decisions, patient diagnosis, clinical triaging, or emergency room operations.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -------------------------------------------------------------------------
    # End-to-End Pipeline Architecture
    # -------------------------------------------------------------------------
    st.markdown(
        """
        <div class="bda-card" style="margin-bottom: 24px;">
            <div style="font-size: 18px; font-weight: 700; color: #0f172a; margin-bottom: 12px;">
                End-to-End Big Data Analytics Architecture (Phases 1 – 7)
            </div>
            <div style="font-size: 13px; color: #475569; line-height: 1.6; margin-bottom: 16px;">
                The project implements an industry-grade, reproducible big data workflow spanning real-time ingestion, distributed batch feature engineering, model registry, containerized REST serving, and an interactive UI.
            </div>
        """,
        unsafe_allow_html=True,
    )

    phases = [
        ("Phase 1", "Synthetic Data Generation", "Simulates realistic emergency department physiological telemetry streams and patient admission labels without using protected health information (PHI)."),
        ("Phase 2", "Distributed Kafka Streaming", "Publishes streaming patient observation events into fault-tolerant Apache Kafka event topics with JSON schema enforcement."),
        ("Phase 3", "HDFS Data Lake Ingestion", "Persists raw immutable streaming events to simulated/cluster Hadoop Distributed File System storage partitions."),
        ("Phase 4", "PySpark Distributed Feature Engineering", "Executes patient-level rolling window aggregations, delta calculations, and exports clean, partitioned Parquet datasets."),
        ("Phase 5", "Machine Learning Training & Registry", "Trains scikit-learn classification pipelines, evaluates validation metrics, and serializes versioned joblib bundles and governance metadata."),
        ("Phase 6", "FastAPI Low-Latency Serving API", "Exposes /health, /ready, /model-info, /metadata, and /predict endpoints with strict Pydantic schema validation."),
        ("Phase 7", "Streamlit Analytics Dashboard", "Provides this modern, responsive, healthcare analytics SaaS dashboard integrated with the FastAPI inference microservice."),
    ]

    for p_id, p_title, p_desc in phases:
        st.markdown(
            f"""
            <div style="display: flex; gap: 14px; padding: 10px 0; border-bottom: 1px solid #f1f5f9;">
                <div style="background: #eff6ff; color: #2563eb; font-weight: 700; font-size: 12px; border-radius: 6px; padding: 4px 8px; height: fit-content; white-space: nowrap;">
                    {p_id}
                </div>
                <div>
                    <div style="font-size: 14px; font-weight: 600; color: #0f172a;">{p_title}</div>
                    <div style="font-size: 12px; color: #64748b; margin-top: 2px;">{p_desc}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # Governance & Technology Stack
    # -------------------------------------------------------------------------
    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.markdown(
            """
            <div class="bda-card">
                <div style="font-size: 16px; font-weight: 700; color: #0f172a; margin-bottom: 12px;">Core Technology Stack</div>
                <ul style="font-size: 13px; color: #475569; line-height: 1.8; padding-left: 20px;">
                    <li><strong>Backend Serving:</strong> FastAPI, Uvicorn, Pydantic v2, Starlette</li>
                    <li><strong>Analytics & Frontend:</strong> Streamlit, Plotly Express, Pandas, PyArrow</li>
                    <li><strong>Machine Learning:</strong> Scikit-Learn 1.7.1, NumPy, Joblib</li>
                    <li><strong>Big Data Ingestion & Processing:</strong> Apache Kafka, Apache Spark (PySpark), PyArrow Parquet</li>
                    <li><strong>Containerization:</strong> Docker Compose, multi-stage Linux containers</li>
                    <li><strong>Testing:</strong> Pytest, HTTPX TestClient, Scikit-learn test runners</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            """
            <div class="bda-card">
                <div style="font-size: 16px; font-weight: 700; color: #0f172a; margin-bottom: 12px;">Academic Project Governance</div>
                <ul style="font-size: 13px; color: #475569; line-height: 1.8; padding-left: 20px;">
                    <li><strong>Course:</strong> Big Data Analytics (BDA)</li>
                    <li><strong>Project Scope:</strong> Emergency Admission Risk Prediction</li>
                    <li><strong>Data Privacy:</strong> Zero Protected Health Information (Zero PHI); 100% synthetic generation</li>
                    <li><strong>Model Governance:</strong> Strict separation of operational status, model artifact readiness, and evaluation validation</li>
                    <li><strong>Leakage Prevention:</strong> Grouped patient-level train/validation/test splits with explicit exclusion of patient identifiers</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
