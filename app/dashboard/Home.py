"""Modern Streamlit Analytics Dashboard — Phase 7 Main Entry Point.

Emergency Admission Risk Prediction — Big Data Analytics Platform.
Integrates modern responsive UI matching reference layout with Phase 6 FastAPI inference backend.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if "/app" not in sys.path:
    sys.path.insert(0, "/app")

import streamlit as st

# Configure page layout and visual identity
st.set_page_config(
    page_title="Emergency Admission Risk Prediction | BDA Platform",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply unified design system tokens, fonts, and dark sidebar styling
from app.dashboard.styles import apply_custom_styles
apply_custom_styles()

# Import page views
from app.dashboard.views.overview import render_overview
from app.dashboard.views.dataset_explorer import render_dataset_explorer
from app.dashboard.views.single_prediction import render_single_prediction
from app.dashboard.views.batch_prediction import render_batch_prediction
from app.dashboard.views.model_info import render_model_info
from app.dashboard.views.system_status import render_system_status
from app.dashboard.views.about import render_about

# Define navigation pages
NAV_PAGES = [
    "Dashboard",
    "Dataset Explorer",
    "Single Prediction",
    "Batch Prediction",
    "Model Information",
    "System Status",
    "About",
]

# Initialize navigation session state
if "nav_page" not in st.session_state:
    st.session_state["nav_page"] = "Dashboard"

# -----------------------------------------------------------------------------
# Persistent Sidebar
# -----------------------------------------------------------------------------
with st.sidebar:
    # Brand Header matching UI Reference design
    st.markdown(
        """
        <div style="display: flex; align-items: center; gap: 12px; padding: 12px 4px 20px 4px;">
            <div style="background: #e11d48; width: 38px; height: 38px; border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 20px; box-shadow: 0 4px 6px -1px rgba(225, 29, 72, 0.4);">
                ❤️
            </div>
            <div>
                <div style="font-size: 15px; font-weight: 700; color: #ffffff; line-height: 1.2;">
                    Emergency<br>Admission Risk
                </div>
                <div style="font-size: 11px; color: #94a3b8; font-weight: 500; margin-top: 2px;">
                    BDA Project
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Navigation Radio Selector
    # Map icon representations
    icon_map = {
        "Dashboard": "📊  Dashboard",
        "Dataset Explorer": "🔍  Dataset Explorer",
        "Single Prediction": "👤  Single Prediction",
        "Batch Prediction": "📄  Batch Prediction",
        "Model Information": "ℹ️  Model Information",
        "System Status": "⚙️  System Status",
        "About": "📖  About",
    }

    display_options = [icon_map[p] for p in NAV_PAGES]
    current_index = NAV_PAGES.index(st.session_state["nav_page"])

    selected_display = st.radio(
        "Navigation",
        options=display_options,
        index=current_index,
        label_visibility="collapsed",
    )

    # Update active page from selection
    for page_name, disp in icon_map.items():
        if disp == selected_display:
            st.session_state["nav_page"] = page_name
            break

    # Spacing before bottom persistent disclaimer card
    st.markdown("<div style='margin-top: 36px;'></div>", unsafe_allow_html=True)

    # Bottom Academic Demonstration Card (matches reference image sidebar card)
    st.markdown(
        """
        <div style="background: rgba(30, 41, 59, 0.85); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 10px; padding: 14px; margin-top: auto;">
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
                <span style="font-size: 16px;">ℹ️</span>
                <span style="font-size: 12px; font-weight: 700; color: #f8fafc;">Academic Demonstration Only</span>
            </div>
            <div style="font-size: 11px; color: #94a3b8; line-height: 1.45;">
                All patient records and clinical data are <strong>SYNTHETIC</strong>. This application is an academic demonstration and must <strong>never</strong> be used for actual medical decisions, patient diagnosis, or emergency department triage.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# -----------------------------------------------------------------------------
# Main View Router
# -----------------------------------------------------------------------------
active_page = st.session_state["nav_page"]

if active_page == "Dashboard":
    render_overview()
elif active_page == "Dataset Explorer":
    render_dataset_explorer()
elif active_page == "Single Prediction":
    render_single_prediction()
elif active_page == "Batch Prediction":
    render_batch_prediction()
elif active_page == "Model Information":
    render_model_info()
elif active_page == "System Status":
    render_system_status()
elif active_page == "About":
    render_about()
else:
    render_overview()
