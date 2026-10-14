"""Batch Patient Prediction view for Phase 7 Streamlit dashboard.

Integrates with Phase 6 POST /predict/batch API, validates CSV inputs (max 500 records),
previews uploaded data, and displays/downloads scored predictions.
"""
from __future__ import annotations

import html
import io
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from app.dashboard.api_client import api_client

REQUIRED_NUMERIC_FEATURES = [
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

MAX_BATCH_SIZE = 500


def generate_sample_batch_csv(n_rows: int = 15) -> str:
    """Generate synthetic CSV string adhering to required 12-feature schema."""
    rng = np.random.default_rng(42)
    records = []
    for i in range(1, n_rows + 1):
        records.append({
            "patient_id": f"SYNTH-BATCH-{i:04d}",
            "age": int(rng.integers(22, 92)),
            "heart_rate": round(float(rng.uniform(62.0, 135.0)), 1),
            "systolic_bp": round(float(rng.uniform(92.0, 168.0)), 1),
            "diastolic_bp": round(float(rng.uniform(58.0, 102.0)), 1),
            "spo2": round(float(rng.uniform(88.0, 99.5)), 1),
            "respiratory_rate": round(float(rng.uniform(12.0, 32.0)), 1),
            "temperature_c": round(float(rng.uniform(36.4, 39.4)), 1),
            "previous_admissions": int(rng.integers(0, 7)),
            "chronic_conditions_count": int(rng.integers(0, 6)),
            "lactate": round(float(rng.uniform(0.8, 5.2)), 2),
            "observation_hour": int(rng.integers(0, 24)),
            "prior_observation_count": int(rng.integers(1, 20)),
        })
    df = pd.DataFrame(records)
    return df.to_csv(index=False)


def render_batch_prediction() -> None:
    """Render the Batch CSV upload and inference interface."""
    st.markdown(
        """
        <div style="margin-bottom: 20px;">
            <h1 style="font-size: 26px; font-weight: 700; color: #0f172a; margin: 0 0 6px 0;">
                Batch Risk Inference
            </h1>
            <p style="color: #64748b; font-size: 14px; margin: 0;">
                Upload a CSV file with synthetic patient observations (up to 500 records) to score in a single atomic API call.
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
                <strong>ACADEMIC DEMONSTRATION ONLY:</strong> Batch predictions are experimental and derived from synthetic test records.
                They must <strong>never</strong> be used for real emergency admissions or clinical triaging.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Sample template download banner
    t_col1, t_col2 = st.columns([3, 1])
    with t_col1:
        st.info("💡 **Need a test file?** Download a pre-validated synthetic batch CSV (15 patient records) ready to upload.")
    with t_col2:
        sample_csv = generate_sample_batch_csv(15)
        st.download_button(
            label="📥 Download Sample CSV",
            data=sample_csv,
            file_name="synthetic_patients_batch_sample.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # Main workflow cards: Uploader & Inspector (Left) vs Results (Right)
    col_upload, col_preview = st.columns([5, 7], gap="large")

    with col_upload:
        st.markdown(
            """
            <div style="background: #ffffff; border-radius: 12px; border: 1px solid #e2e8f0; padding: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); margin-bottom: 20px;">
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 12px;">
                    <div style="background: #eff6ff; color: #2563eb; width: 32px; height: 32px; border-radius: 8px; display: flex; align-items: center; justify-content: center; font-weight: 700;">📄</div>
                    <div>
                        <div style="font-size: 16px; font-weight: 700; color: #0f172a;">Upload CSV File</div>
                        <div style="font-size: 12px; color: #64748b;">Maximum 500 records • Must contain 12 required features</div>
                    </div>
                </div>
            """,
            unsafe_allow_html=True,
        )

        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=["csv"],
            help="Upload a CSV with patient records matching the 12 model features.",
            label_visibility="collapsed",
        )

        valid_df: pd.DataFrame | None = None
        validation_errors: list[str] = []

        if uploaded_file is not None:
            try:
                raw_df = pd.read_csv(uploaded_file)
                if raw_df.empty:
                    validation_errors.append("Uploaded CSV file is completely empty.")
                elif len(raw_df) > MAX_BATCH_SIZE:
                    validation_errors.append(f"Batch size of {len(raw_df)} exceeds maximum allowed limit of {MAX_BATCH_SIZE} records.")
                else:
                    # Check required columns
                    missing_cols = [c for c in REQUIRED_NUMERIC_FEATURES if c not in raw_df.columns]
                    if missing_cols:
                        validation_errors.append(f"Missing required feature columns: {', '.join(missing_cols)}")
                    else:
                        # Validate and convert numeric types
                        df_clean = raw_df.copy()
                        if "patient_id" not in df_clean.columns:
                            df_clean["patient_id"] = [f"SYNTH-ROW-{i+1}" for i in range(len(df_clean))]
                        else:
                            df_clean["patient_id"] = df_clean["patient_id"].astype(str)

                        # Check for non-numeric or NaN values in feature columns
                        has_nan = False
                        for col in REQUIRED_NUMERIC_FEATURES:
                            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")
                            if df_clean[col].isna().any():
                                has_nan = True
                                validation_errors.append(f"Column '{col}' contains non-numeric or missing (NaN) values.")

                        if not has_nan:
                            valid_df = df_clean

            except Exception as e:
                validation_errors.append(f"Failed to parse CSV file: {e}")

        # Render Validation feedback
        if validation_errors:
            for err in validation_errors:
                st.error(f"❌ {err}")
        elif valid_df is not None:
            st.success(f"✓ Valid CSV: **{len(valid_df)} records** verified with all 12 required features.")

        st.write("")
        b_col1, b_col2 = st.columns([3, 2])
        with b_col1:
            run_batch_btn = st.button(
                "🚀 Run Batch Prediction",
                disabled=(valid_df is None),
                use_container_width=True,
                type="primary",
            )
        with b_col2:
            clear_batch_btn = st.button("↺ Clear", use_container_width=True)
            if clear_batch_btn:
                if "last_batch_result" in st.session_state:
                    del st.session_state["last_batch_result"]
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

        # File preview if uploaded
        if valid_df is not None:
            st.markdown(
                f"<div style='font-size: 13px; font-weight: 600; color: #334155; margin-bottom: 6px;'>Uploaded File Preview (First {min(len(valid_df), 5)} of {len(valid_df)} rows)</div>",
                unsafe_allow_html=True,
            )
            display_cols = ["patient_id", "age", "heart_rate", "systolic_bp", "spo2", "respiratory_rate", "lactate"]
            avail_disp = [c for c in display_cols if c in valid_df.columns]
            st.dataframe(valid_df[avail_disp].head(5), use_container_width=True)

    # Handle Batch Prediction Trigger
    if run_batch_btn and valid_df is not None:
        records_payload = []
        for _, row in valid_df.iterrows():
            rec = {"patient_id": str(row["patient_id"])}
            for col in REQUIRED_NUMERIC_FEATURES:
                rec[col] = float(row[col]) if col not in ("age", "previous_admissions", "chronic_conditions_count", "observation_hour", "prior_observation_count") else int(row[col])
            records_payload.append(rec)

        with st.spinner(f"Scoring {len(records_payload)} synthetic patient records via FastAPI..."):
            success, result_data, latency_ms = api_client.predict_batch(records_payload)

        if success:
            st.session_state["last_batch_result"] = {
                "response": result_data,
                "count": len(records_payload),
                "latency_ms": latency_ms,
            }
        else:
            st.session_state["last_batch_error"] = result_data.get("error", "Batch inference failed")

    # -------------------------------------------------------------------------
    # Right Column: Batch Results Table & Summary
    # -------------------------------------------------------------------------
    with col_preview:
        st.markdown(
            """
            <div style="background: #ffffff; border-radius: 12px; border: 1px solid #e2e8f0; padding: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 16px;">
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <div style="background: #eff6ff; color: #2563eb; width: 32px; height: 32px; border-radius: 8px; display: flex; align-items: center; justify-content: center; font-weight: 700;">📊</div>
                        <div>
                            <div style="font-size: 16px; font-weight: 700; color: #0f172a;">Batch Results</div>
                            <div style="font-size: 12px; color: #64748b;">Inference outcomes and risk distributions</div>
                        </div>
                    </div>
                </div>
            """,
            unsafe_allow_html=True,
        )

        if "last_batch_error" in st.session_state and ("last_batch_result" not in st.session_state):
            st.markdown(
                f"""
                <div style="background: #fef2f2; border: 1px solid #fecaca; border-radius: 8px; padding: 14px; color: #991b1b; margin-bottom: 16px;">
                    <strong>Batch Request Failed:</strong><br>
                    {html.escape(str(st.session_state['last_batch_error']))}
                </div>
                """,
                unsafe_allow_html=True,
            )

        if "last_batch_result" in st.session_state:
            batch_obj = st.session_state["last_batch_result"]
            batch_resp = batch_obj["response"]
            pred_list = batch_resp.get("predictions", [])
            total_records = batch_resp.get("total_records", len(pred_list))
            model_name = batch_resp.get("model_name", "logistic_regression")
            latency = batch_obj.get("latency_ms", 0.0)

            # Build Results DataFrame
            results_rows = []
            for idx, p in enumerate(pred_list, 1):
                prob = float(p.get("risk_probability", 0.0))
                pred_val = int(p.get("prediction", 0))
                thresh_val = float(p.get("threshold", 0.70))
                rev_flag = p.get("review_flag", prob >= thresh_val)

                risk_level = "High" if (pred_val == 1 or rev_flag) else "Low"
                results_rows.append({
                    "#": idx,
                    "Patient ID": p.get("patient_id", f"PT-{idx}"),
                    "Risk Probability": round(prob, 4),
                    "Prediction": "Admission (1)" if pred_val == 1 else "No Admission (0)",
                    "Risk Level": risk_level,
                    "Threshold": thresh_val,
                })

            res_df = pd.DataFrame(results_rows)

            # Summary Metric Cards
            high_count = sum(1 for r in results_rows if r["Risk Level"] == "High")
            low_count = total_records - high_count
            avg_prob = res_df["Risk Probability"].mean() if not res_df.empty else 0.0

            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            with m_col1:
                st.metric("Total Scored", f"{total_records}")
            with m_col2:
                st.metric("High Risk / Flagged", f"{high_count}", f"{(high_count / max(total_records, 1)) * 100:.1f}%")
            with m_col3:
                st.metric("Low Risk", f"{low_count}")
            with m_col4:
                st.metric("Batch Latency", f"{latency:.0f} ms")

            st.write("")

            # Result Data Table
            st.dataframe(
                res_df,
                use_container_width=True,
                hide_index=True,
                height=320,
            )

            # CSV Download button
            csv_output = res_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Scored Batch CSV",
                data=csv_output,
                file_name=f"scored_admissions_{model_name}_{total_records}records.csv",
                mime="text/csv",
                use_container_width=True,
                type="primary",
            )

        else:
            st.markdown(
                """
                <div style="text-align: center; padding: 60px 20px; color: #94a3b8;">
                    <div style="font-size: 40px; margin-bottom: 12px;">📁</div>
                    <div style="font-size: 14px; font-weight: 600; color: #64748b;">No Scored Batch Yet</div>
                    <div style="font-size: 12px; margin-top: 4px;">Upload a CSV file and click "Run Batch Prediction" to review multi-patient risk scores here.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)
