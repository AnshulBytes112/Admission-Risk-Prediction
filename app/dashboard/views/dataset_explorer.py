"""Dataset Explorer view for Phase 8 Streamlit dashboard.

Allows interactive inspection, filtering, pagination, statistics, and downloading
of the processed Spark Parquet dataset or the synthetic demonstration CSV.
Enables 1-click transferring of a selected patient profile into Single Prediction.
"""
from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
PARQUET_PATH = DATA_DIR / "processed" / "features.parquet"
DEMO_CSV_PATH = DATA_DIR / "demo" / "synthetic_demo_patients.csv"
TEMPLATE_CSV_PATH = DATA_DIR / "demo" / "batch_prediction_template.csv"

# Exact 12 predictor features for model inference
MODEL_FEATURES = [
    "age",
    "heart_rate",
    "systolic_bp",
    "diastolic_bp",
    "spo2",
    "respiratory_rate",
    "temperature_c",
    "previous_admissions",
    "chronic_conditions_count",
    "lactate",
    "observation_hour",
    "prior_observation_count",
]


@st.cache_data(ttl=60)
def load_datasets() -> dict[str, tuple[pd.DataFrame | None, str]]:
    """Load both primary processed Parquet and demo CSV datasets."""
    datasets: dict[str, tuple[pd.DataFrame | None, str]] = {}

    if PARQUET_PATH.exists():
        try:
            df_pq = pd.read_parquet(PARQUET_PATH)
            datasets["Processed Spark Parquet"] = (df_pq, str(PARQUET_PATH))
        except Exception as e:
            datasets["Processed Spark Parquet"] = (None, f"Error reading {PARQUET_PATH}: {e}")

    if DEMO_CSV_PATH.exists():
        try:
            df_csv = pd.read_csv(DEMO_CSV_PATH)
            datasets["Synthetic Demo CSV"] = (df_csv, str(DEMO_CSV_PATH))
        except Exception as e:
            datasets["Synthetic Demo CSV"] = (None, f"Error reading {DEMO_CSV_PATH}: {e}")

    return datasets


def render_dataset_explorer() -> None:
    """Render the interactive Dataset Explorer interface."""
    st.markdown(
        """
        <div style="margin-bottom: 20px;">
            <h1 style="font-size: 26px; font-weight: 700; color: #0f172a; margin: 0 0 6px 0;">
                Dataset Explorer & Cohort Analytics
            </h1>
            <p style="color: #64748b; font-size: 14px; margin: 0;">
                Explore, filter, inspect, and export synthetic emergency department observations and model feature matrices.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Mandatory Academic Safety Banner
    st.markdown(
        """
        <div class="bda-alert-banner" style="margin-bottom: 20px;">
            <div style="font-size: 16px;">⚠️</div>
            <div>
                <strong>ACADEMIC DEMONSTRATION ONLY:</strong> All patient records, physiological measurements, and admission labels are synthetic.
                Predictions are experimental, not clinically validated, and must <strong>never</strong> be used for real medical decisions or emergency triage.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    available_data = load_datasets()

    if not available_data:
        st.error(
            "❌ **No Datasets Found.** Neither `data/processed/features.parquet` nor `data/demo/synthetic_demo_patients.csv` could be located.\n\n"
            "To generate a rich demo dataset immediately, run:\n"
            "```powershell\n"
            "py -3.13 scripts/generate_demo_dataset.py --rows 500 --seed 42\n"
            "```"
        )
        return

    # Source Selector & Quick Download Bar
    bar_col1, bar_col2 = st.columns([3, 2])
    with bar_col1:
        source_options = list(available_data.keys())
        selected_source = st.selectbox(
            "Select Active Dataset Source",
            options=source_options,
            index=0,
            help="Switch between Spark processed Parquet output and the extended synthetic demonstration CSV.",
        )
    with bar_col2:
        st.write("")
        st.write("")
        if TEMPLATE_CSV_PATH.exists():
            with open(TEMPLATE_CSV_PATH, "r", encoding="utf-8") as f:
                template_data = f.read()
            st.download_button(
                label="📥 Download Batch Test Template CSV",
                data=template_data,
                file_name="batch_prediction_template.csv",
                mime="text/csv",
                use_container_width=True,
                help="Download pre-formatted CSV containing only the 12 model features ready for Batch Prediction.",
            )

    df_active, source_path = available_data[selected_source]

    if df_active is None:
        st.warning(f"Could not load data from {selected_source}: {source_path}")
        return

    # Active Source Provenance Tag
    source_badge_color = "#2563eb" if "Parquet" in selected_source else "#0d9488"
    st.markdown(
        f"""
        <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 10px 16px; margin-bottom: 20px; display: flex; align-items: center; justify-content: space-between;">
            <div>
                <span style="font-size: 11px; text-transform: uppercase; font-weight: 700; color: #64748b;">Active Source:</span>
                <span style="background: {source_badge_color}; color: #ffffff; padding: 3px 8px; border-radius: 6px; font-size: 12px; font-weight: 600; margin-left: 6px;">{selected_source}</span>
                <span style="font-family: monospace; font-size: 12px; color: #64748b; margin-left: 8px;">({source_path})</span>
            </div>
            <div style="font-size: 12px; font-weight: 600; color: #d97706;">
                Provenance: 100% Synthetic (Zero PHI)
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -------------------------------------------------------------------------
    # Top Metrics Cards
    # -------------------------------------------------------------------------
    total_rows = len(df_active)
    unique_patients = df_active["patient_id"].nunique() if "patient_id" in df_active.columns else total_rows
    col_count = len(df_active.columns)

    target_col = "admission_within_24h"
    if target_col in df_active.columns:
        pos_count = int(df_active[target_col].sum())
        neg_count = total_rows - pos_count
        pos_pct = (pos_count / max(total_rows, 1)) * 100.0
    else:
        pos_count, neg_count, pos_pct = 0, total_rows, 0.0

    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("Total Records", f"{total_rows:,}")
    with m2:
        st.metric("Unique Patients", f"{unique_patients:,}")
    with m3:
        st.metric("Admissions (Class 1)", f"{pos_count}", f"{pos_pct:.1f}%")
    with m4:
        st.metric("Non-Admitted (Class 0)", f"{neg_count}")
    with m5:
        st.metric("Total Columns", f"{col_count}", "12 model predictors")

    st.write("")

    # -------------------------------------------------------------------------
    # Interactive Search & Filtering Controls
    # -------------------------------------------------------------------------
    with st.expander("🔍 Search & Filter Records", expanded=True):
        f_col1, f_col2, f_col3, f_col4 = st.columns([2, 1, 1, 1])

        with f_col1:
            search_query = st.text_input(
                "Search by Patient ID",
                value="",
                placeholder="e.g. SYN-DEMO-001 or SYN000",
                help="Filter rows whose Patient ID contains this text.",
            )

        with f_col2:
            target_filter = st.selectbox(
                "Admission Outcome",
                options=["All Outcomes", "Admitted (1)", "Non-Admitted (0)"],
                index=0,
            )

        with f_col3:
            min_age = int(df_active["age"].min()) if "age" in df_active.columns else 18
            max_age = int(df_active["age"].max()) if "age" in df_active.columns else 100
            selected_age_range = st.slider("Age Range", min_value=min_age, max_value=max_age, value=(min_age, max_age))

        with f_col4:
            if "observation_hour" in df_active.columns:
                hour_range = st.slider("Arrival Hour", min_value=0, max_value=23, value=(0, 23))
            else:
                hour_range = (0, 23)

    # Apply filters
    filtered_df = df_active.copy()

    if search_query.strip() and "patient_id" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["patient_id"].astype(str).str.contains(search_query.strip(), case=False, na=False)]

    if target_filter == "Admitted (1)" and target_col in filtered_df.columns:
        filtered_df = filtered_df[filtered_df[target_col] == 1]
    elif target_filter == "Non-Admitted (0)" and target_col in filtered_df.columns:
        filtered_df = filtered_df[filtered_df[target_col] == 0]

    if "age" in filtered_df.columns:
        filtered_df = filtered_df[
            (filtered_df["age"] >= selected_age_range[0]) & (filtered_df["age"] <= selected_age_range[1])
        ]

    if "observation_hour" in filtered_df.columns:
        filtered_df = filtered_df[
            (filtered_df["observation_hour"] >= hour_range[0]) & (filtered_df["observation_hour"] <= hour_range[1])
        ]

    st.markdown(
        f"<div style='font-size: 13px; font-weight: 600; color: #475569; margin: 8px 0;'>Showing {len(filtered_df):,} matching rows (out of {total_rows:,} total)</div>",
        unsafe_allow_html=True,
    )

    # -------------------------------------------------------------------------
    # Paginated Table & Row Inspector
    # -------------------------------------------------------------------------
    tab_data, tab_stats, tab_charts = st.tabs(["📋 Data Records & Inspector", "📊 Summary Statistics", "📈 Cohort Distributions"])

    with tab_data:
        p_col1, p_col2 = st.columns([1, 4])
        with p_col1:
            rows_per_page = st.selectbox("Rows per page", options=[10, 25, 50, 100], index=1)
        with p_col2:
            total_pages = max(1, (len(filtered_df) + rows_per_page - 1) // rows_per_page)
            page_num = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)

        start_idx = (page_num - 1) * rows_per_page
        end_idx = min(start_idx + rows_per_page, len(filtered_df))
        page_df = filtered_df.iloc[start_idx:end_idx]

        st.dataframe(
            page_df,
            use_container_width=True,
            height=320,
        )

        # Row Selection for Structured Inspection & Prefill to Single Prediction
        if not page_df.empty:
            st.markdown(
                """
                <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 18px; margin-top: 14px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
                    <div style="font-size: 15px; font-weight: 700; color: #0f172a; margin-bottom: 4px;">
                        Structured Record Inspector & Inference Pre-fill
                    </div>
                    <div style="font-size: 12px; color: #64748b; margin-bottom: 12px;">
                        Select a patient from the current page to inspect full physiological telemetry or forward directly to the Single Prediction form.
                    </div>
                """,
                unsafe_allow_html=True,
            )

            pt_options = [
                f"Row #{start_idx + i + 1}: {row.get('patient_id', 'Unknown')} (Age {row.get('age', 'N/A')}, Outcome: {row.get('admission_within_24h', 'N/A')})"
                for i, (_, row) in enumerate(page_df.iterrows())
            ]
            selected_pt_desc = st.selectbox("Select Patient to Inspect", options=pt_options, index=0)
            selected_pt_idx = pt_options.index(selected_pt_desc)
            selected_record = page_df.iloc[selected_pt_idx]

            # Detail Display Grid
            d_col1, d_col2, d_col3 = st.columns(3)
            with d_col1:
                st.markdown(
                    f"""
                    <div style="font-size: 12px; color: #64748b; line-height: 1.8;">
                        <strong>Patient ID:</strong> <span style="font-family: monospace;">{selected_record.get('patient_id', 'N/A')}</span><br>
                        <strong>Age:</strong> {selected_record.get('age', 'N/A')} yrs<br>
                        <strong>Heart Rate:</strong> {selected_record.get('heart_rate', 'N/A')} bpm<br>
                        <strong>Blood Pressure:</strong> {selected_record.get('systolic_bp', 'N/A')} / {selected_record.get('diastolic_bp', 'N/A')} mmHg
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with d_col2:
                st.markdown(
                    f"""
                    <div style="font-size: 12px; color: #64748b; line-height: 1.8;">
                        <strong>SpO₂:</strong> {selected_record.get('spo2', 'N/A')}%<br>
                        <strong>Resp Rate:</strong> {selected_record.get('respiratory_rate', 'N/A')} /min<br>
                        <strong>Temperature:</strong> {selected_record.get('temperature_c', 'N/A')} °C<br>
                        <strong>Lactate:</strong> {selected_record.get('lactate', 'N/A')} mmol/L
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with d_col3:
                target_val = selected_record.get("admission_within_24h", "N/A")
                target_str = "Admitted (1)" if target_val == 1 else ("Not Admitted (0)" if target_val == 0 else str(target_val))
                target_col_str = "#dc2626" if target_val == 1 else "#16a34a"
                st.markdown(
                    f"""
                    <div style="font-size: 12px; color: #64748b; line-height: 1.8;">
                        <strong>Prior Admissions:</strong> {selected_record.get('previous_admissions', 'N/A')}<br>
                        <strong>Chronic Conditions:</strong> {selected_record.get('chronic_conditions_count', 'N/A')}<br>
                        <strong>Arrival Hour:</strong> {selected_record.get('observation_hour', 'N/A')}:00<br>
                        <strong>Simulated Label:</strong> <span style="color: {target_col_str}; font-weight: 700;">{target_str}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.write("")
            act_col1, act_col2 = st.columns([2, 3])
            with act_col1:
                if st.button("🚀 Send to Single Prediction Form", use_container_width=True, type="primary"):
                    # Store selected record in session state for pre-filling
                    st.session_state["prefill_patient"] = {
                        "patient_id": str(selected_record.get("patient_id", "SYNTH-SELECT")),
                        "age": int(selected_record.get("age", 65)),
                        "heart_rate": float(selected_record.get("heart_rate", 82.0)),
                        "systolic_bp": float(selected_record.get("systolic_bp", 124.0)),
                        "diastolic_bp": float(selected_record.get("diastolic_bp", 78.0)),
                        "spo2": float(selected_record.get("spo2", 97.0)),
                        "respiratory_rate": float(selected_record.get("respiratory_rate", 15.0)),
                        "temperature_c": float(selected_record.get("temperature_c", 37.0)),
                        "previous_admissions": int(selected_record.get("previous_admissions", 1)),
                        "chronic_conditions_count": int(selected_record.get("chronic_conditions_count", 2)),
                        "lactate": float(selected_record.get("lactate", 1.3)),
                        "observation_hour": int(selected_record.get("observation_hour", 14)),
                        "prior_observation_count": int(selected_record.get("prior_observation_count", 4)),
                    }
                    st.session_state["nav_page"] = "Single Prediction"
                    st.rerun()

            with act_col2:
                # CSV Download of filtered rows
                filtered_csv = filtered_df.to_csv(index=False)
                st.download_button(
                    label=f"📥 Download Filtered Rows as CSV ({len(filtered_df):,} records)",
                    data=filtered_csv,
                    file_name="filtered_patient_records.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            st.markdown("</div>", unsafe_allow_html=True)

    with tab_stats:
        st.markdown(
            "<div style='font-size: 14px; font-weight: 700; color: #0f172a; margin-bottom: 8px;'>Numerical Features Summary Statistics</div>",
            unsafe_allow_html=True,
        )
        avail_num = [c for c in MODEL_FEATURES if c in df_active.columns]
        if avail_num:
            stats_df = df_active[avail_num].describe().T[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]]
            stats_df["null_count"] = df_active[avail_num].isna().sum()
            st.dataframe(stats_df.round(2), use_container_width=True)

    with tab_charts:
        ch_col1, ch_col2 = st.columns(2)
        with ch_col1:
            if target_col in df_active.columns:
                target_counts = df_active[target_col].value_counts().reset_index()
                target_counts.columns = ["Target Class", "Count"]
                target_counts["Label"] = target_counts["Target Class"].map({0: "Non-Admitted (0)", 1: "Admitted (1)"})
                fig_target = px.pie(
                    target_counts,
                    names="Label",
                    values="Count",
                    title="Simulated Admission Target Distribution",
                    color="Label",
                    color_discrete_map={"Non-Admitted (0)": "#3b82f6", "Admitted (1)": "#ef4444"},
                    hole=0.45,
                )
                fig_target.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=300)
                st.plotly_chart(fig_target, use_container_width=True)
        with ch_col2:
            if "age" in df_active.columns:
                fig_age = px.histogram(
                    df_active,
                    x="age",
                    nbins=20,
                    title="Patient Age Distribution",
                    color_discrete_sequence=["#2563eb"],
                )
                fig_age.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=300)
                st.plotly_chart(fig_age, use_container_width=True)
