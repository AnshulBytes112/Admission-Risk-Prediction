"""System Status & Health Diagnostics view for Phase 7 Streamlit dashboard.

Probes the FastAPI /health, /ready, /model-info, and /metadata endpoints, measuring
latencies, validating payloads, and providing actionable troubleshooting guidance.
"""
from __future__ import annotations

from datetime import datetime
import html
from typing import Any

import streamlit as st

from app.dashboard.api_client import api_client


def render_system_status() -> None:
    """Render the System Diagnostics and Endpoint Health Monitoring interface."""
    st.markdown(
        """
        <div style="margin-bottom: 20px;">
            <h1 style="font-size: 26px; font-weight: 700; color: #0f172a; margin: 0 0 6px 0;">
                System Status & API Diagnostics
            </h1>
            <p style="color: #64748b; font-size: 14px; margin: 0;">
                Live health, readiness, and connectivity diagnostics for the Phase 6 FastAPI inference microservice.
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
                <strong>ACADEMIC DEMONSTRATION ONLY:</strong> This diagnostic interface monitors local development microservices.
                No clinical systems or protected health information (PHI) are connected.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Top Bar: Connection details and Refresh Action
    col_url, col_btn = st.columns([3, 1])
    with col_url:
        st.markdown(
            f"""
            <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px 16px; display: flex; align-items: center; justify-content: space-between;">
                <div>
                    <span style="font-size: 12px; color: #64748b; font-weight: 600; text-transform: uppercase;">Configured API Base Target:</span>
                    <span style="font-family: monospace; font-size: 14px; color: #0f172a; font-weight: 600; margin-left: 8px;">{html.escape(api_client.base_url)}</span>
                </div>
                <div style="font-size: 11px; color: #64748b;">Timeout: {api_client.timeout_seconds}s bounded</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_btn:
        if st.button("🔄 Probe All Endpoints", use_container_width=True, type="primary"):
            st.rerun()

    st.write("")

    # Execute Probes
    with st.spinner("Probing backend endpoints..."):
        h_ok, h_data, h_ms = api_client.check_health()
        r_ok, r_data, r_ms = api_client.check_readiness()
        m_ok, m_data, m_ms = api_client.get_model_info()
        meta_ok, meta_data, meta_ms = api_client.get_metadata()

    # 4 Status Cards for the 4 core endpoints
    c1, c2, c3, c4 = st.columns(4)

    endpoints = [
        ("GET /health", "Liveness Probe", h_ok, h_ms, h_data.get("status", "error" if not h_ok else "ok")),
        ("GET /ready", "Model Readiness", r_ok, r_ms, "ready" if r_ok else "not ready"),
        ("GET /model-info", "Architecture Specs", m_ok, m_ms, m_data.get("model_name", "error" if not m_ok else "loaded")),
        ("GET /metadata", "Governance Contract", meta_ok, meta_ms, "valid" if meta_ok else "error"),
    ]

    for col, (route, desc, ok, lat, val) in zip([c1, c2, c3, c4], endpoints):
        with col:
            bg_col = "#f0fdf4" if ok else "#fef2f2"
            border_col = "#bbf7d0" if ok else "#fecaca"
            txt_col = "#15803d" if ok else "#991b1b"
            icon = "✓" if ok else "✕"
            status_text = "Operational" if ok else "Unavailable"

            st.markdown(
                f"""
                <div style="background: {bg_col}; border: 1px solid {border_col}; border-radius: 10px; padding: 14px; text-align: left; height: 140px; display: flex; flex-direction: column; justify-content: space-between;">
                    <div>
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span style="font-family: monospace; font-size: 13px; font-weight: 700; color: #0f172a;">{route}</span>
                            <span style="background: {txt_col}; color: #ffffff; border-radius: 50%; width: 18px; height: 18px; display: inline-flex; align-items: center; justify-content: center; font-size: 11px; font-weight: 700;">{icon}</span>
                        </div>
                        <div style="font-size: 11px; color: #64748b; margin-top: 2px;">{desc}</div>
                    </div>
                    <div>
                        <div style="font-size: 15px; font-weight: 700; color: {txt_col};">{status_text}</div>
                        <div style="font-size: 11px; color: #64748b;">Latency: {lat:.1f} ms</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.write("")

    # Detailed Inspection Tabs
    st.markdown(
        "<div style='font-size: 14px; font-weight: 700; color: #1e293b; margin-bottom: 8px;'>Endpoint Response Payloads</div>",
        unsafe_allow_html=True,
    )
    tab_health, tab_ready, tab_info, tab_meta = st.tabs([
        "/health Response",
        "/ready Response",
        "/model-info Response",
        "/metadata Response",
    ])

    with tab_health:
        st.json(h_data)
    with tab_ready:
        st.json(r_data)
    with tab_info:
        st.json(m_data)
    with tab_meta:
        st.json(meta_data)

    st.write("")

    # Troubleshooting & Remediation Section
    st.markdown(
        """
        <div class="bda-card">
            <div style="font-size: 16px; font-weight: 700; color: #0f172a; margin-bottom: 8px;">
                Diagnostics & Troubleshooting Guide
            </div>
            <div style="font-size: 13px; color: #475569; line-height: 1.6;">
                If any endpoint reports <strong style="color: #b91c1c;">Unavailable</strong>:
                <ul style="margin-top: 6px; padding-left: 20px;">
                    <li><strong>Docker Execution:</strong> Ensure the <code>api</code> service is running via <code>docker compose ps</code>. Inside the Docker network, the dashboard communicates using <code>http://api:8000</code>.</li>
                    <li><strong>Local Host Execution:</strong> Ensure FastAPI is started on port 8000: <code>uvicorn app.api.main:app --host 0.0.0.0 --port 8000</code>.</li>
                    <li><strong>Model Missing Error (503):</strong> Verify that Phase 5 artifacts (<code>models/risk_model.joblib</code> and <code>models/model_metadata.json</code>) are generated and readable.</li>
                    <li><strong>Network Timeout:</strong> Verify that local firewall or antivirus software is not intercepting port 8000 connections.</li>
                </ul>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
