"""Dashboard Overview page displaying live system cards, dataset analytics, and recent activity.

Follows the layout and color palette of the UI reference image.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

from app.dashboard.api_client import api_client
from app.dashboard.pipeline_inspector import get_pipeline_telemetry

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
PARQUET_PATH = DATA_DIR / "processed" / "features.parquet"
DEMO_CSV_PATH = DATA_DIR / "demo" / "synthetic_demo_patients.csv"
CSV_PATH = DATA_DIR / "raw" / "patient_observations.csv"


@st.cache_data(ttl=60)
def load_exploratory_data() -> tuple[pd.DataFrame | None, str]:
    """Load local processed Parquet, synthetic demo CSV, or raw CSV data for exploratory analytics."""
    if PARQUET_PATH.exists():
        try:
            df = pd.read_parquet(PARQUET_PATH)
            return df, f"Processed Spark Parquet dataset ({PARQUET_PATH})"
        except Exception:
            pass
    if DEMO_CSV_PATH.exists():
        try:
            df = pd.read_csv(DEMO_CSV_PATH)
            return df, f"Synthetic Demo Dataset CSV ({DEMO_CSV_PATH})"
        except Exception:
            pass
    if CSV_PATH.exists():
        try:
            df = pd.read_csv(CSV_PATH)
            return df, f"Raw Synthetic CSV ({CSV_PATH})"
        except Exception:
            pass
    return None, "No dataset found"


def render_overview() -> None:
    """Render the main dashboard overview page."""
    # Fetch live API and Model status
    health_ok, health_data, health_ms = api_client.check_health()
    ready_ok, ready_data, _ = api_client.check_readiness()
    info_ok, info_data, _ = api_client.get_model_info()

    model_name = ready_data.get("model_name") or info_data.get("model_name") or "logistic_regression"
    is_experimental = ready_data.get("is_experimental") if ready_ok else True
    threshold = ready_data.get("model_threshold", 0.70)
    feature_count = ready_data.get("feature_count", 12)

    # -------------------------------------------------------------------------
    # Header Section
    # -------------------------------------------------------------------------
    h_col1, h_col2 = st.columns([3, 1])
    with h_col1:
        st.markdown(
            """
            <h1 style="font-size: 28px; font-weight: 700; color: #0f172a; margin: 0 0 4px 0;">
                Emergency Admission Risk Prediction
            </h1>
            <p style="font-size: 15px; color: #64748b; margin: 0 0 16px 0;">
                Predict the risk of hospital admission within 24 hours using machine learning
            </p>
            """,
            unsafe_allow_html=True,
        )
    with h_col2:
        if is_experimental:
            st.markdown(
                """
                <div style="text-align: right; padding-top: 6px;">
                    <span class="bda-badge-experimental">
                        ⚠️ EXPERIMENTAL MODEL (SYNTHETIC DATA)
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Persistent Academic Warning Banner
    st.markdown(
        """
        <div class="bda-banner-warning">
            <strong>ACADEMIC DEMONSTRATION ONLY:</strong> All patient profiles, vital signs, and admission flags are <strong>100% synthetic</strong>.
            Predictions are experimental software demonstration outputs, are <strong>not clinically validated</strong>, and must never be used for real medical decisions or patient triage.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -------------------------------------------------------------------------
    # 4 Top Status Cards (matches visual reference image)
    # -------------------------------------------------------------------------
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        status_text = "Online" if health_ok else "Offline"
        icon_class = "icon-green" if health_ok else "icon-red"
        icon_symbol = "✓" if health_ok else "✕"
        subtext = f"FastAPI is running ({health_ms} ms)" if health_ok else "Backend unreachable"
        st.markdown(
            f"""
            <div class="bda-status-card">
                <div class="bda-icon-circle {icon_class}">{icon_symbol}</div>
                <div>
                    <div style="font-size: 12px; font-weight: 500; color: #64748b;">API Status</div>
                    <div style="font-size: 18px; font-weight: 700; color: #0f172a;">{status_text}</div>
                    <div style="font-size: 11px; color: #64748b;">{subtext}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        model_ready_text = "Ready" if ready_ok else "Not Ready"
        icon_class = "icon-teal" if ready_ok else "icon-yellow"
        icon_symbol = "🧠"
        subtext = "Model loaded and operational" if ready_ok else (ready_data.get("error_reason") or "Model unloaded")
        st.markdown(
            f"""
            <div class="bda-status-card">
                <div class="bda-icon-circle {icon_class}">{icon_symbol}</div>
                <div>
                    <div style="font-size: 12px; font-weight: 500; color: #64748b;">Model Status</div>
                    <div style="font-size: 18px; font-weight: 700; color: #0f172a;">{model_ready_text}</div>
                    <div style="font-size: 11px; color: #64748b;">{subtext[:35]}...</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            f"""
            <div class="bda-status-card">
                <div class="bda-icon-circle icon-purple">📦</div>
                <div>
                    <div style="font-size: 12px; font-weight: 500; color: #64748b;">Model Version</div>
                    <div style="font-size: 16px; font-weight: 700; color: #0f172a;">{model_name}</div>
                    <div style="font-size: 11px; color: #64748b;">{feature_count} features • Threshold: {threshold:.2f}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c4:
        st.markdown(
            """
            <div class="bda-status-card">
                <div class="bda-icon-circle icon-yellow">🧪</div>
                <div>
                    <div style="font-size: 12px; font-weight: 500; color: #64748b;">Data Origin</div>
                    <div style="font-size: 18px; font-weight: 700; color: #0f172a;">Synthetic</div>
                    <div style="font-size: 11px; color: #64748b;">Academic demonstration only</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # Dataset Exploration and Statistical Summary
    # -------------------------------------------------------------------------
    df, source_label = load_exploratory_data()

    t_col1, t_col2 = st.columns([3, 1])
    with t_col1:
        st.markdown(
            f"""
            <div style="margin-bottom:6px;">
                <h3 style="font-size:18px; font-weight:700; color:#0f172a; margin:0;">
                    📊 Processed Big Data Analytics Dataset
                </h3>
                <span style="font-size:12px; color:#64748b;">Source: {source_label}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with t_col2:
        if st.button("🔍 Open Dataset Explorer", use_container_width=True):
            st.session_state["nav_page"] = "Dataset Explorer"
            st.rerun()

    if df is not None and not df.empty:
        total_rows = len(df)
        unique_pts = df["patient_id"].nunique() if "patient_id" in df else total_rows
        target_col = "admission_within_24h"
        pos_count = int(df[target_col].sum()) if target_col in df else 0
        neg_count = total_rows - pos_count
        pos_rate = (pos_count / total_rows * 100) if total_rows > 0 else 0.0

        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Total Records", f"{total_rows:,}")
        d2.metric("Unique Patients", f"{unique_pts:,}")
        d3.metric("Simulated Admissions", f"{pos_count} ({pos_rate:.1f}%)")
        d4.metric("Non-Admissions", f"{neg_count:,}")

        if total_rows <= 100 and pos_count <= 2:
            st.info(
                f"ℹ️ **Dataset Note ({source_label})**:\n"
                f"The current processed dataset contains **{total_rows} records** and **{pos_count} positive case**. "
                "This small sample size represents an academic pipeline smoke-test. It is statistically insufficient "
                "for clinical generalization, and metrics derived from it reflect technical validation only."
            )

        # Visualizations
        ch1, ch2 = st.columns(2)
        with ch1:
            if target_col in df:
                target_df = pd.DataFrame({
                    "Class": ["Non-Admission (0)", "Admission (1)"],
                    "Count": [neg_count, pos_count],
                })
                fig_t = px.bar(
                    target_df,
                    x="Class",
                    y="Count",
                    text="Count",
                    title="Target Distribution (admission_within_24h)",
                    color="Class",
                    color_discrete_map={"Non-Admission (0)": "#64748b", "Admission (1)": "#dc2626"},
                )
                fig_t.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20), showlegend=False)
                st.plotly_chart(fig_t, use_container_width=True)

        with ch2:
            if "heart_rate" in df:
                fig_hr = px.histogram(
                    df.dropna(subset=["heart_rate"]),
                    x="heart_rate",
                    nbins=20,
                    title="Heart Rate Distribution (bpm)",
                    color_discrete_sequence=["#2563eb"],
                )
                fig_hr.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_hr, use_container_width=True)

        ch3, ch4 = st.columns(2)
        with ch3:
            if "systolic_bp" in df and "diastolic_bp" in df:
                fig_bp = px.scatter(
                    df.dropna(subset=["systolic_bp", "diastolic_bp"]),
                    x="systolic_bp",
                    y="diastolic_bp",
                    title="Blood Pressure Distribution (Systolic vs Diastolic)",
                    color=target_col if target_col in df else None,
                    color_discrete_map={0: "#3b82f6", 1: "#ef4444"} if target_col in df else None,
                    labels={"systolic_bp": "Systolic BP (mmHg)", "diastolic_bp": "Diastolic BP (mmHg)"},
                )
                fig_bp.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_bp, use_container_width=True)

        with ch4:
            if "observation_hour" in df:
                fig_h = px.histogram(
                    df.dropna(subset=["observation_hour"]),
                    x="observation_hour",
                    nbins=24,
                    title="Patient Arrival Hour Distribution (0 - 23)",
                    color_discrete_sequence=["#0d9488"],
                )
                fig_h.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_h, use_container_width=True)
    else:
        st.info(
            "ℹ️ No local dataset found in `data/processed/` or `data/raw/`.\n\n"
            "Generate synthetic observations via PowerShell:\n"
            "`py -3.13 scripts/generate_synthetic_data.py --rows 1000 --seed 42 --output data/raw/patient_observations.csv`"
        )

    # -------------------------------------------------------------------------
    # Recent Session Predictions Table
    # -------------------------------------------------------------------------
    if "recent_predictions" in st.session_state and st.session_state["recent_predictions"]:
        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
        st.subheader("⏱️ Recent Session Predictions")
        st.caption("Predictions stored in browser session state; cleared on page refresh.")
        recent_df = pd.DataFrame(st.session_state["recent_predictions"])
        st.dataframe(recent_df, use_container_width=True, hide_index=True)
