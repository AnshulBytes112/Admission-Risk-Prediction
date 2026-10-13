"""Streamlit Starter Dashboard for Emergency Admission Risk Prediction (Phase 1).

Academic Big Data Analytics software demonstration.
Provides:
- Project overview & architecture summary
- Clear synthetic-data disclaimer
- Live backend API status check with graceful offline fallback
- Local dataset inspection with informative empty state
- Physiological vitals distribution visualizations
- Phase 1 foundation scope notes and next steps
"""
import os
from pathlib import Path
import pandas as pd
import plotly.express as px
import requests
import streamlit as st

# Configure page layout and metadata
st.set_page_config(
    page_title="Emergency Admission Risk Prediction | BDA Platform",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Configuration from environment variables
# -----------------------------------------------------------------------------
API_BASE = os.getenv("DASHBOARD_API_URL", os.getenv("API_BASE_URL", "http://localhost:8000"))
DATA_DIR = os.getenv("DATA_DIR", "data")
THRESHOLD_DEFAULT = float(os.getenv("RISK_THRESHOLD", "0.7"))
RAW_DATA_PATH = Path(DATA_DIR) / "raw" / "patient_observations.csv"

# -----------------------------------------------------------------------------
# Header and Academic Disclaimer
# -----------------------------------------------------------------------------
st.title("🏥 Emergency Admission Risk Prediction")
st.markdown("### Big Data Analytics Academic Platform · Phase 1: Foundation & Infrastructure")

st.warning(
    "⚠️ **ACADEMIC SOFTWARE DEMONSTRATION DISCLAIMER**\n\n"
    "All patient profiles, vital signs, lab observations, and admission target flags in this project are "
    "**100% synthetically generated** for software and big-data architecture demonstration purposes. "
    "This system is **NOT** a clinical decision-support tool, is not derived from real hospital patients, and "
    "must **NEVER** be used for medical diagnosis, triage, treatment, or clinical decisions."
)

# -----------------------------------------------------------------------------
# Sidebar: System Settings & Health Check
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ System Status & Controls")
    st.caption("Phase 1: Project Foundation")
    
    st.markdown("---")
    st.markdown("**Backend API Connectivity**")
    
    # Graceful API status check
    api_online = False
    api_info = {}
    try:
        r = requests.get(f"{API_BASE}/health", timeout=3)
        if r.ok:
            api_online = True
            api_info = r.json()
    except Exception:
        api_online = False

    if api_online:
        st.success(f"API Online ({API_BASE})")
        st.write(f"• Application Health: `{api_info.get('application_health', 'healthy')}`")
        st.write(f"• Model Status: `{api_info.get('model_readiness', 'not_loaded')}`")
    else:
        st.error(f"API Offline ({API_BASE})")
        st.caption("Start the API container using Docker Compose or `uvicorn app.api.main:app`.")

    st.markdown("---")
    st.markdown("**Configuration**")
    st.write(f"• Risk Threshold: `{THRESHOLD_DEFAULT}`")
    st.write(f"• Raw Data Path: `{RAW_DATA_PATH}`")
    
    if st.button("🔄 Refresh View"):
        st.rerun()

# -----------------------------------------------------------------------------
# Section 1: Architecture Summary
# -----------------------------------------------------------------------------
st.markdown("---")
st.subheader("📐 Big Data Analytics Architecture")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(
        """
        <div style="padding:12px; border-radius:8px; border:1px solid #3b82f6; background-color:#eff6ff; color:#1e3a8a;">
            <h4 style="margin:0; color:#1e40af;">Phase 1: Foundation</h4>
            <p style="margin:4px 0 0 0; font-size:13px; color:#1e3a8a;"><strong>Status: Active</strong><br/>
            Synthetic data generator, Docker infrastructure, starter API & dashboard.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col2:
    st.markdown(
        """
        <div style="padding:12px; border-radius:8px; border:1px solid #94a3b8; background-color:#f8fafc; color:#334155;">
            <h4 style="margin:0; color:#475569;">Phases 2–4: Ingestion & Storage</h4>
            <p style="margin:4px 0 0 0; font-size:13px; color:#334155;"><strong>Status: Deferred</strong><br/>
            Kafka streaming producer/consumer, HDFS ingestion, and PySpark batch cleaning.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col3:
    st.markdown(
        """
        <div style="padding:12px; border-radius:8px; border:1px solid #94a3b8; background-color:#f8fafc; color:#334155;">
            <h4 style="margin:0; color:#475569;">Phase 5: Machine Learning</h4>
            <p style="margin:4px 0 0 0; font-size:13px; color:#334155;"><strong>Status: Deferred</strong><br/>
            Patient-grouped split, baseline model comparison, PR-AUC evaluation.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col4:
    st.markdown(
        """
        <div style="padding:12px; border-radius:8px; border:1px solid #94a3b8; background-color:#f8fafc; color:#334155;">
            <h4 style="margin:0; color:#475569;">Phases 6–10: Inference & Ops</h4>
            <p style="margin:4px 0 0 0; font-size:13px; color:#334155;"><strong>Status: Deferred</strong><br/>
            REST model inference, real-time dashboard alerts, integration testing.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# -----------------------------------------------------------------------------
# Section 2: Local Dataset Inspection
# -----------------------------------------------------------------------------
st.markdown("---")
st.subheader("📊 Synthetic Clinical Dataset Explorer")

if not RAW_DATA_PATH.exists():
    # Empty State Display
    st.info(
        f"ℹ️ **No dataset found at `{RAW_DATA_PATH}`.**\n\n"
        "Generate synthetic patient observations to explore records and summary statistics."
    )
    st.markdown("#### How to Generate the Dataset:")
    st.markdown("**Option A: Using Python directly (PowerShell):**")
    st.code("py -3.13 scripts/generate_synthetic_data.py --rows 10000 --seed 42 --output data/raw/patient_observations.csv")
    st.markdown("**Option B: Using Docker Compose utility service:**")
    st.code("docker compose --profile tools run --build --rm data-generator")
else:
    # Load and cache dataset
    @st.cache_data
    def load_data(filepath: str) -> pd.DataFrame:
        return pd.read_csv(filepath)

    try:
        df = load_data(str(RAW_DATA_PATH))
        
        # Summary KPI Metrics
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Total Observations", f"{len(df):,}")
        m2.metric("Unique Patients", f"{df['patient_id'].nunique():,}" if "patient_id" in df else "N/A")
        
        prevalence = df["admission_within_24h"].mean() * 100 if "admission_within_24h" in df else 0
        m3.metric("Simulated Admission Rate", f"{prevalence:.2f}%")
        
        synthetic_flag = df["synthetic_data"].all() if "synthetic_data" in df else False
        m4.metric("Synthetic Marker", "True (100%)" if synthetic_flag else "Mixed")
        
        missing_count = int(df.isna().sum().sum())
        m5.metric("Controlled Missing Values", f"{missing_count:,}")

        # Data Exploration and Filtering
        st.markdown("##### Filter Observations")
        fcol1, fcol2, fcol3 = st.columns([2, 1, 1])
        with fcol1:
            search_query = st.text_input("Search Patient ID", placeholder="e.g. SYN000123")
        with fcol2:
            min_age = st.number_input("Min Age", min_value=18, max_value=100, value=18)
        with fcol3:
            max_age = st.number_input("Max Age", min_value=18, max_value=100, value=100)

        filtered_df = df.copy()
        if search_query:
            filtered_df = filtered_df[filtered_df["patient_id"].str.contains(search_query, case=False, na=False)]
        filtered_df = filtered_df[(filtered_df["age"] >= min_age) & (filtered_df["age"] <= max_age)]

        st.dataframe(filtered_df.head(100), use_container_width=True, hide_index=True)
        st.caption(f"Showing {min(100, len(filtered_df))} of {len(filtered_df):,} matching rows.")

        # Vitals Distribution Visualizations
        st.markdown("##### Physiological Vitals Distributions")
        chart1, chart2 = st.columns(2)
        with chart1:
            if "heart_rate" in filtered_df:
                fig_hr = px.histogram(
                    filtered_df.dropna(subset=["heart_rate"]),
                    x="heart_rate",
                    nbins=35,
                    title="Heart Rate Distribution (bpm)",
                    color_discrete_sequence=["#2563eb"],
                )
                fig_hr.update_layout(margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_hr, use_container_width=True)

        with chart2:
            if "systolic_bp" in filtered_df:
                fig_sbp = px.histogram(
                    filtered_df.dropna(subset=["systolic_bp"]),
                    x="systolic_bp",
                    nbins=35,
                    title="Systolic Blood Pressure Distribution (mmHg)",
                    color_discrete_sequence=["#059669"],
                )
                fig_sbp.update_layout(margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_sbp, use_container_width=True)

        chart3, chart4 = st.columns(2)
        with chart3:
            if "spo2" in filtered_df:
                fig_spo2 = px.histogram(
                    filtered_df.dropna(subset=["spo2"]),
                    x="spo2",
                    nbins=25,
                    title="Oxygen Saturation SpO₂ (%)",
                    color_discrete_sequence=["#d97706"],
                )
                fig_spo2.update_layout(margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_spo2, use_container_width=True)

        with chart4:
            if "admission_within_24h" in filtered_df:
                target_counts = filtered_df["admission_within_24h"].value_counts().reset_index()
                target_counts.columns = ["Target (0=No, 1=Yes)", "Count"]
                fig_target = px.bar(
                    target_counts,
                    x="Target (0=No, 1=Yes)",
                    y="Count",
                    title="Synthetic Admission Within 24h Distribution",
                    color="Target (0=No, 1=Yes)",
                    color_discrete_map={0: "#64748b", 1: "#dc2626"},
                )
                fig_target.update_layout(margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_target, use_container_width=True)

    except Exception as exc:
        st.error(f"Error loading dataset: {exc}")

# -----------------------------------------------------------------------------
# Section 3: Phase 1 Scope Notice
# -----------------------------------------------------------------------------
st.markdown("---")
st.subheader("ℹ️ Phase 1 Scope & Next Steps")
st.markdown(
    """
    - **Model Training Notice**: In accordance with the project roadmap, machine learning model training 
      is scheduled for **Phase 5**. No predictive models are fabricated or loaded in Phase 1.
    - **Streaming Notice**: Kafka streaming pipelines and consumers will be integrated in **Phase 2**.
    - **HDFS & Spark Notice**: Distributed storage upload and batch feature engineering will be configured in **Phases 3 & 4**.
    """
)
