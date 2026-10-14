"""Single Patient Prediction view for Phase 7 Streamlit dashboard.

Integrates with Phase 6 POST /predict API and adheres strictly to the 12-feature schema.
"""
from __future__ import annotations

import html
import random
from typing import Any

import streamlit as st

from app.dashboard.api_client import api_client

# Default synthetic demonstration profiles
PROFILES = {
    "Typical / Low Risk Case": {
        "age": 65,
        "heart_rate": 82.0,
        "systolic_bp": 124.0,
        "diastolic_bp": 78.0,
        "spo2": 97.0,
        "respiratory_rate": 15.0,
        "temperature_c": 37.0,
        "previous_admissions": 1,
        "chronic_conditions_count": 2,
        "lactate": 1.3,
        "observation_hour": 14,
        "prior_observation_count": 4,
    },
    "Elevated / Critical Case": {
        "age": 78,
        "heart_rate": 128.0,
        "systolic_bp": 88.0,
        "diastolic_bp": 54.0,
        "spo2": 89.0,
        "respiratory_rate": 32.0,
        "temperature_c": 39.2,
        "previous_admissions": 4,
        "chronic_conditions_count": 5,
        "lactate": 4.6,
        "observation_hour": 3,
        "prior_observation_count": 12,
    },
    "Borderline / Moderate Case": {
        "age": 70,
        "heart_rate": 102.0,
        "systolic_bp": 105.0,
        "diastolic_bp": 68.0,
        "spo2": 93.0,
        "respiratory_rate": 22.0,
        "temperature_c": 38.1,
        "previous_admissions": 2,
        "chronic_conditions_count": 3,
        "lactate": 2.2,
        "observation_hour": 18,
        "prior_observation_count": 6,
    },
}


def render_single_prediction() -> None:
    """Render the single patient inference interface."""
    st.markdown(
        """
        <div style="margin-bottom: 20px;">
            <h1 style="font-size: 26px; font-weight: 700; color: #0f172a; margin: 0 0 6px 0;">
                Single Patient Risk Inference
            </h1>
            <p style="color: #64748b; font-size: 14px; margin: 0;">
                Evaluate experimental 24-hour emergency admission probability using the active FastAPI model service.
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
                <strong>ACADEMIC DEMONSTRATION ONLY:</strong> All patient records, physiological measurements, and admission labels are synthetic.
                Predictions are experimental, not clinically validated, and must <strong>never</strong> be used for real medical decisions, diagnosis, or emergency triage.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Check for pre-filled patient from Dataset Explorer
    prefilled = st.session_state.get("prefill_patient")
    if prefilled:
        p_col_info, p_col_clear = st.columns([4, 1])
        with p_col_info:
            st.info(f"📋 **Prefilled from Dataset Explorer:** Observation record for **{prefilled.get('patient_id')}** loaded into input form.")
        with p_col_clear:
            if st.button("✕ Clear Prefill", use_container_width=True):
                del st.session_state["prefill_patient"]
                st.rerun()

    # Initialize preset selection in session state
    if "single_profile_name" not in st.session_state:
        st.session_state["single_profile_name"] = "Typical / Low Risk Case"

    # Profile switcher bar
    p_col1, p_col2 = st.columns([3, 1])
    with p_col1:
        selected_profile = st.selectbox(
            "Load Demonstration Preset Profile",
            options=list(PROFILES.keys()),
            index=list(PROFILES.keys()).index(st.session_state["single_profile_name"]),
            key="preset_profile_selector",
            help="Pre-fills inputs with synthetic profiles for demonstration testing.",
        )
        if selected_profile != st.session_state["single_profile_name"]:
            st.session_state["single_profile_name"] = selected_profile
            if "prefill_patient" in st.session_state:
                del st.session_state["prefill_patient"]
            st.rerun()

    with p_col2:
        st.write("")
        st.write("")
        if st.button("🎲 Random Synthetic", use_container_width=True):
            if "prefill_patient" in st.session_state:
                del st.session_state["prefill_patient"]
            st.session_state["random_pt_id"] = f"SYNTH-{random.randint(1000, 9999)}"
            st.session_state["age"] = random.randint(25, 88)
            st.session_state["heart_rate"] = float(random.randint(60, 130))
            st.session_state["systolic_bp"] = float(random.randint(90, 165))
            st.session_state["diastolic_bp"] = float(random.randint(55, 100))
            st.session_state["spo2"] = float(random.randint(88, 99))
            st.session_state["respiratory_rate"] = float(random.randint(12, 34))
            st.session_state["temperature_c"] = round(random.uniform(36.2, 39.5), 1)
            st.session_state["lactate"] = round(random.uniform(0.7, 5.0), 1)
            st.session_state["previous_admissions"] = random.randint(0, 6)
            st.session_state["chronic_conditions_count"] = random.randint(0, 5)
            st.session_state["observation_hour"] = random.randint(0, 23)
            st.session_state["prior_observation_count"] = random.randint(1, 15)
            st.rerun()

    profile_data = prefilled if prefilled else PROFILES[st.session_state["single_profile_name"]]

    # Two column layout: Input Form (left) vs Prediction Result (right)
    col_form, col_result = st.columns([7, 5], gap="large")

    with col_form:
        st.markdown(
            """
            <div style="background: #ffffff; border-radius: 12px; border: 1px solid #e2e8f0; padding: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); margin-bottom: 24px;">
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 16px;">
                    <div style="background: #eff6ff; color: #2563eb; width: 32px; height: 32px; border-radius: 8px; display: flex; align-items: center; justify-content: center; font-weight: 700;">👤</div>
                    <div>
                        <div style="font-size: 16px; font-weight: 700; color: #0f172a;">Patient Clinical & Observational Inputs</div>
                        <div style="font-size: 12px; color: #64748b;">Specify the 12 features validated by the Phase 5/6 model schema.</div>
                    </div>
                </div>
            """,
            unsafe_allow_html=True,
        )

        with st.form("single_prediction_form"):
            # Header info
            pt_col1, pt_col2 = st.columns([2, 1])
            with pt_col1:
                patient_id = st.text_input(
                    "Synthetic Patient ID (Required identifier, not a predictor)",
                    value=st.session_state.get("random_pt_id", "SYNTH-PT-1001"),
                    help="Identifier forwarded to API response logging. Not used as a predictive feature.",
                )
            with pt_col2:
                age = st.number_input(
                    "Age (years)",
                    min_value=18,
                    max_value=100,
                    value=int(st.session_state.get("age", profile_data["age"])),
                    step=1,
                    help="Patient age [18 - 100].",
                )

            st.markdown(
                "<div style='font-size: 13px; font-weight: 600; color: #334155; margin-top: 12px; margin-bottom: 6px;'>Physiological & Vital Signs</div>",
                unsafe_allow_html=True,
            )
            v_col1, v_col2, v_col3, v_col4 = st.columns(4)
            with v_col1:
                heart_rate = st.number_input(
                    "Heart Rate (bpm)",
                    min_value=25.0,
                    max_value=240.0,
                    value=float(st.session_state.get("heart_rate", profile_data["heart_rate"])),
                    step=1.0,
                    help="Resting pulse rate [25 - 240 bpm].",
                )
            with v_col2:
                systolic_bp = st.number_input(
                    "Systolic BP (mmHg)",
                    min_value=40.0,
                    max_value=300.0,
                    value=float(st.session_state.get("systolic_bp", profile_data["systolic_bp"])),
                    step=1.0,
                    help="Systolic pressure [40 - 300 mmHg].",
                )
            with v_col3:
                diastolic_bp = st.number_input(
                    "Diastolic BP (mmHg)",
                    min_value=20.0,
                    max_value=200.0,
                    value=float(st.session_state.get("diastolic_bp", profile_data["diastolic_bp"])),
                    step=1.0,
                    help="Diastolic pressure [20 - 200 mmHg].",
                )
            with v_col4:
                spo2 = st.number_input(
                    "SpO₂ (%)",
                    min_value=50.0,
                    max_value=100.0,
                    value=float(st.session_state.get("spo2", profile_data["spo2"])),
                    step=0.5,
                    help="Blood oxygen saturation [50 - 100%].",
                )

            v2_col1, v2_col2, v2_col3 = st.columns(3)
            with v2_col1:
                respiratory_rate = st.number_input(
                    "Resp Rate (breaths/min)",
                    min_value=3.0,
                    max_value=80.0,
                    value=float(st.session_state.get("respiratory_rate", profile_data["respiratory_rate"])),
                    step=1.0,
                    help="Respiratory frequency [3 - 80 /min].",
                )
            with v2_col2:
                temperature_c = st.number_input(
                    "Temp (°C)",
                    min_value=30.0,
                    max_value=44.0,
                    value=float(st.session_state.get("temperature_c", profile_data["temperature_c"])),
                    step=0.1,
                    format="%.1f",
                    help="Core temperature [30.0 - 44.0 °C].",
                )
            with v2_col3:
                lactate = st.number_input(
                    "Lactate (mmol/L)",
                    min_value=0.0,
                    max_value=30.0,
                    value=float(st.session_state.get("lactate", profile_data["lactate"])),
                    step=0.1,
                    format="%.1f",
                    help="Blood lactate marker [0.0 - 30.0 mmol/L].",
                )

            st.markdown(
                "<div style='font-size: 13px; font-weight: 600; color: #334155; margin-top: 12px; margin-bottom: 6px;'>Clinical History & Observation Context</div>",
                unsafe_allow_html=True,
            )
            h_col1, h_col2, h_col3, h_col4 = st.columns(4)
            with h_col1:
                previous_admissions = st.number_input(
                    "Prior Admissions",
                    min_value=0,
                    max_value=100,
                    value=int(st.session_state.get("previous_admissions", profile_data["previous_admissions"])),
                    step=1,
                    help="Prior hospital admissions count [0 - 100].",
                )
            with h_col2:
                chronic_conditions_count = st.number_input(
                    "Chronic Conditions",
                    min_value=0,
                    max_value=50,
                    value=int(st.session_state.get("chronic_conditions_count", profile_data["chronic_conditions_count"])),
                    step=1,
                    help="Documented co-morbidities [0 - 50].",
                )
            with h_col3:
                observation_hour = st.number_input(
                    "Arrival Hour (0-23)",
                    min_value=0,
                    max_value=23,
                    value=int(st.session_state.get("observation_hour", profile_data["observation_hour"])),
                    step=1,
                    help="Hour of emergency triage observation [0 - 23].",
                )
            with h_col4:
                prior_observation_count = st.number_input(
                    "Prior Obs Count",
                    min_value=0,
                    max_value=100000,
                    value=int(st.session_state.get("prior_observation_count", profile_data["prior_observation_count"])),
                    step=1,
                    help="Cumulative prior observation telemetry points [0 - 100,000].",
                )

            st.write("")
            btn_col1, btn_col2 = st.columns([2, 1])
            with btn_col1:
                submit_btn = st.form_submit_button(
                    "▶ Predict Admission Risk",
                    use_container_width=True,
                    type="primary",
                )
            with btn_col2:
                reset_btn = st.form_submit_button("↺ Clear to Preset", use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

        if reset_btn:
            for k in [
                "random_pt_id",
                "age",
                "heart_rate",
                "systolic_bp",
                "diastolic_bp",
                "spo2",
                "respiratory_rate",
                "temperature_c",
                "lactate",
                "previous_admissions",
                "chronic_conditions_count",
                "observation_hour",
                "prior_observation_count",
            ]:
                if k in st.session_state:
                    del st.session_state[k]
            st.rerun()

    # Form Submission Logic
    if submit_btn:
        payload = {
            "patient_id": patient_id.strip() or "SYNTH-PT-1001",
            "age": age,
            "heart_rate": heart_rate,
            "systolic_bp": systolic_bp,
            "diastolic_bp": diastolic_bp,
            "spo2": spo2,
            "respiratory_rate": respiratory_rate,
            "temperature_c": temperature_c,
            "previous_admissions": previous_admissions,
            "chronic_conditions_count": chronic_conditions_count,
            "lactate": lactate,
            "observation_hour": observation_hour,
            "prior_observation_count": prior_observation_count,
        }

        with st.spinner("Executing model inference via FastAPI backend..."):
            success, result_data, latency_ms = api_client.predict_single(payload)

        if success:
            st.session_state["last_single_result"] = {
                "response": result_data,
                "input": payload,
                "latency_ms": latency_ms,
            }
            # Add to recent predictions in session state
            if "recent_predictions" not in st.session_state:
                st.session_state["recent_predictions"] = []
            st.session_state["recent_predictions"].insert(
                0,
                {
                    "patient_id": payload["patient_id"],
                    "probability": result_data.get("risk_probability", 0.0),
                    "prediction": result_data.get("prediction", 0),
                    "model": result_data.get("model_name", "model"),
                    "latency_ms": latency_ms,
                },
            )
            # Keep max 10
            st.session_state["recent_predictions"] = st.session_state["recent_predictions"][:10]
        else:
            st.session_state["last_single_error"] = result_data.get("error", "Unknown inference error")
            if "last_single_result" in st.session_state:
                del st.session_state["last_single_result"]

    # -------------------------------------------------------------------------
    # Right Column: Prediction Result Card (matching UI Reference design)
    # -------------------------------------------------------------------------
    with col_result:
        st.markdown(
            """
            <div style="background: #ffffff; border-radius: 12px; border: 1px solid #e2e8f0; padding: 22px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 16px;">
                    <div style="background: #eff6ff; color: #2563eb; width: 32px; height: 32px; border-radius: 8px; display: flex; align-items: center; justify-content: center; font-weight: 700;">📊</div>
                    <div>
                        <div style="font-size: 16px; font-weight: 700; color: #0f172a;">Inference Result</div>
                        <div style="font-size: 12px; color: #64748b;">Model prediction and experimental risk score</div>
                    </div>
                </div>
            """,
            unsafe_allow_html=True,
        )

        if "last_single_error" in st.session_state:
            st.markdown(
                f"""
                <div style="background: #fef2f2; border: 1px solid #fecaca; border-radius: 8px; padding: 14px; color: #991b1b; margin-bottom: 16px;">
                    <strong>Inference Request Failed:</strong><br>
                    {html.escape(str(st.session_state['last_single_error']))}
                </div>
                """,
                unsafe_allow_html=True,
            )

        if "last_single_result" in st.session_state:
            res_obj = st.session_state["last_single_result"]
            resp = res_obj["response"]
            prob = float(resp.get("risk_probability", 0.0))
            pred = int(resp.get("prediction", 0))
            thresh = float(resp.get("threshold", 0.70))
            model_n = resp.get("model_name", "logistic_regression")
            lat = res_obj.get("latency_ms", 0.0)
            is_exp = resp.get("is_experimental", True)
            review_flag = resp.get("review_flag", False) or (prob >= thresh)

            pct_str = f"{prob * 100:.1f}%"

            if pred == 1 or review_flag:
                badge_bg = "#fef2f2"
                badge_border = "#fecaca"
                badge_color = "#b91c1c"
                badge_icon = "🚨"
                status_title = "High Risk (Review Flag)"
                thresh_text = f"Exceeds decision threshold ({thresh:.2f})"
                bar_color = "#ef4444"
            else:
                badge_bg = "#f0fdf4"
                badge_border = "#bbf7d0"
                badge_color = "#15803d"
                badge_icon = "🛡️"
                status_title = "Low Risk (Standard Flow)"
                thresh_text = f"Below decision threshold ({thresh:.2f})"
                bar_color = "#10b981"

            # Render Result Box
            st.markdown(
                f"""
                <div style="background: {badge_bg}; border: 1px solid {badge_border}; border-radius: 12px; padding: 18px; text-align: center; margin-bottom: 20px;">
                    <div style="font-size: 28px; margin-bottom: 4px;">{badge_icon}</div>
                    <div style="font-size: 16px; font-weight: 700; color: {badge_color};">{status_title}</div>
                    <div style="font-size: 12px; color: #64748b; margin-top: 2px;">Admission Risk (within 24 hours)</div>
                    <div style="font-size: 38px; font-weight: 800; color: {badge_color}; margin: 8px 0;">{pct_str}</div>
                    <div style="background: #e2e8f0; border-radius: 9999px; height: 8px; width: 85%; margin: 0 auto 8px auto; overflow: hidden;">
                        <div style="background: {bar_color}; height: 100%; width: {min(max(prob * 100, 2), 100)}%;"></div>
                    </div>
                    <div style="font-size: 12px; color: #64748b; font-weight: 500;">{thresh_text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Metadata Table
            exp_label = "Yes (Synthetic Data)" if is_exp else "No"
            pred_label = "Admission (1)" if pred == 1 else "No Admission (0)"

            st.markdown(
                f"""
                <table style="width: 100%; border-collapse: collapse; font-size: 13px; color: #334155; margin-bottom: 16px;">
                    <tr style="border-bottom: 1px solid #f1f5f9;">
                        <td style="padding: 8px 0; color: #64748b;">Risk Probability</td>
                        <td style="padding: 8px 0; font-weight: 600; text-align: right; font-family: monospace;">{prob:.4f}</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #f1f5f9;">
                        <td style="padding: 8px 0; color: #64748b;">Model Output</td>
                        <td style="padding: 8px 0; font-weight: 600; text-align: right;">{pred_label}</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #f1f5f9;">
                        <td style="padding: 8px 0; color: #64748b;">Decision Threshold</td>
                        <td style="padding: 8px 0; font-weight: 600; text-align: right; font-family: monospace;">{thresh:.2f}</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #f1f5f9;">
                        <td style="padding: 8px 0; color: #64748b;">Active Model</td>
                        <td style="padding: 8px 0; font-weight: 600; text-align: right; font-family: monospace;">{html.escape(model_n)}</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #f1f5f9;">
                        <td style="padding: 8px 0; color: #64748b;">Inference Latency</td>
                        <td style="padding: 8px 0; font-weight: 600; text-align: right;">{lat:.1f} ms</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px 0; color: #64748b;">Is Experimental</td>
                        <td style="padding: 8px 0; font-weight: 600; text-align: right; color: #d97706;">{exp_label}</td>
                    </tr>
                </table>
                """,
                unsafe_allow_html=True,
            )

            # Disclaimer Note under results
            st.markdown(
                """
                <div style="background: #f8fafc; border: 1px dashed #cbd5e1; border-radius: 8px; padding: 10px; font-size: 11px; color: #64748b; line-height: 1.4;">
                    <strong>Academic Software Note:</strong> The probability score represents an experimental logistic output derived from synthetic data, not a clinically established probability of patient outcome.
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div style="text-align: center; padding: 40px 20px; color: #94a3b8;">
                    <div style="font-size: 40px; margin-bottom: 12px;">🩺</div>
                    <div style="font-size: 14px; font-weight: 600; color: #64748b;">No Active Prediction</div>
                    <div style="font-size: 12px; margin-top: 4px;">Submit the form on the left to evaluate patient risk through the Phase 6 FastAPI backend.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)
