"""Design system, color tokens, and polished SaaS CSS for Phase 8 Streamlit dashboard.

Implements the modern healthcare analytics SaaS design from the UI reference:
- Dark navy/indigo sidebar (#0f172a) with pill navigation
- Light neutral workspace (#f8fafc)
- Clean, bordered white cards (#ffffff, border #e2e8f0, radius 12px)
- Professional typography (Inter, sans-serif)
- Primary action blue (#2563eb)
- Availability teal (#0d9488), experimental amber (#d97706), and alert red (#dc2626)
"""
from __future__ import annotations

import streamlit as st


def apply_custom_styles() -> None:
    """Inject custom CSS tokens and component styles into the Streamlit app."""
    css = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    /* Global Typography & Reset */
    html, body, [class*="css"], [class*="st-"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
    }

    /* Streamlit Main Workspace Container */
    .stApp {
        background-color: #f8fafc !important;
        color: #0f172a !important;
    }

    /* Reduce default large Streamlit top padding */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
        max-width: 1400px !important;
    }

    /* -------------------------------------------------------------------------
       Sidebar Styling (Dark Navy #0f172a)
       ------------------------------------------------------------------------- */
    [data-testid="stSidebar"] {
        background-color: #0f172a !important;
        border-right: 1px solid #1e293b !important;
    }

    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        gap: 0.35rem !important;
    }

    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] div {
        color: #f8fafc !important;
    }

    /* Radio buttons transformed into sleek SaaS navigation pills */
    [data-testid="stSidebar"] div[role="radiogroup"] {
        gap: 4px !important;
        display: flex !important;
        flex-direction: column !important;
    }

    [data-testid="stSidebar"] div[role="radiogroup"] > label {
        padding: 10px 14px !important;
        margin: 0 !important;
        border-radius: 8px !important;
        transition: all 0.15s ease-in-out !important;
        color: #94a3b8 !important;
        font-weight: 500 !important;
        font-size: 13.5px !important;
        cursor: pointer !important;
        background-color: transparent !important;
        display: flex !important;
        align-items: center !important;
    }

    [data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
        background-color: #1e293b !important;
        color: #ffffff !important;
    }

    [data-testid="stSidebar"] div[role="radiogroup"] > label[data-checked="true"],
    [data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {
        background-color: #2563eb !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        box-shadow: 0 2px 4px rgba(37, 99, 235, 0.25) !important;
    }

    /* Hide the radio dot */
    [data-testid="stSidebar"] div[role="radiogroup"] input[type="radio"],
    [data-testid="stSidebar"] div[role="radiogroup"] [data-testid="stWidgetLabel"] {
        display: none !important;
    }

    /* -------------------------------------------------------------------------
       Card Styling (st.container(border=True) Native Hook)
       ------------------------------------------------------------------------- */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 12px !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04) !important;
        padding: 1.25rem !important;
        margin-bottom: 0.75rem !important;
    }

    /* Status card styling */
    .bda-status-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 14px 16px;
        display: flex;
        align-items: center;
        gap: 14px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
        height: 100%;
    }

    .bda-icon-circle {
        width: 42px;
        height: 42px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 19px;
        flex-shrink: 0;
    }

    .icon-green { background: #dcfce7; color: #16a34a; }
    .icon-teal { background: #ccfbf1; color: #0d9488; }
    .icon-purple { background: #f3e8ff; color: #9333ea; }
    .icon-yellow { background: #fef9c3; color: #ca8a04; }
    .icon-red { background: #fee2e2; color: #dc2626; }

    /* -------------------------------------------------------------------------
       Badges & Banners
       ------------------------------------------------------------------------- */
    .bda-badge-experimental {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background-color: #fee2e2;
        color: #b91c1c;
        border: 1px solid #fecaca;
        border-radius: 6px;
        padding: 5px 12px;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.3px;
    }

    .bda-alert-banner {
        background-color: #fffbeb;
        border-left: 4px solid #f59e0b;
        color: #92400e;
        padding: 12px 16px;
        border-radius: 8px;
        font-size: 13px;
        display: flex;
        align-items: flex-start;
        gap: 10px;
        line-height: 1.5;
        box-shadow: 0 1px 2px rgba(0,0,0,0.02);
    }

    /* -------------------------------------------------------------------------
       Form Inputs & Buttons
       ------------------------------------------------------------------------- */
    .stNumberInput input, .stTextInput input, .stSelectbox [data-baseweb="select"] {
        border-radius: 8px !important;
        border: 1px solid #cbd5e1 !important;
        font-size: 13.5px !important;
    }

    .stNumberInput input:focus, .stTextInput input:focus {
        border-color: #2563eb !important;
        box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.2) !important;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 13.5px !important;
        padding: 8px 16px !important;
        transition: all 0.15s ease-in-out !important;
    }

    .stButton > button[kind="primary"] {
        background-color: #2563eb !important;
        border-color: #2563eb !important;
        color: #ffffff !important;
        box-shadow: 0 2px 4px rgba(37, 99, 235, 0.2) !important;
    }

    .stButton > button[kind="primary"]:hover {
        background-color: #1d4ed8 !important;
        border-color: #1d4ed8 !important;
    }

    .stButton > button[kind="secondary"] {
        background-color: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        color: #334155 !important;
    }

    .stButton > button[kind="secondary"]:hover {
        background-color: #f1f5f9 !important;
        border-color: #94a3b8 !important;
    }

    /* DataFrames */
    [data-testid="stDataFrame"] {
        border: 1px solid #e2e8f0 !important;
        border-radius: 8px !important;
        overflow: hidden !important;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
