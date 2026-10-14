"""Design system, color tokens, and custom CSS for the Streamlit dashboard.

Matches the healthcare analytics SaaS aesthetic from the UI reference image:
- Dark navy/indigo sidebar (#0f172a / #1e293b)
- Light neutral workspace (#f8fafc)
- Clean, rounded cards (#ffffff with subtle border and shadow)
- Primary blue actions (#2563eb)
- Green/teal availability accents, amber/red warning badges
"""
from __future__ import annotations

import streamlit as st


def apply_custom_styles() -> None:
    """Inject custom CSS tokens and component styles into the Streamlit app."""
    css = """
    <style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Main background */
    .stApp {
        background-color: #f8fafc;
        color: #0f172a;
    }

    /* Sidebar container styling */
    [data-testid="stSidebar"] {
        background-color: #0f172a !important;
        color: #f8fafc !important;
    }

    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #f8fafc !important;
    }

    /* Sidebar navigation radio buttons styled as SaaS nav pills */
    [data-testid="stSidebar"] div[role="radiogroup"] > label {
        padding: 8px 12px;
        margin-bottom: 4px;
        border-radius: 8px;
        transition: all 0.15s ease-in-out;
        color: #94a3b8 !important;
        font-weight: 500;
        cursor: pointer;
    }

    [data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
        background-color: #1e293b;
        color: #ffffff !important;
    }

    [data-testid="stSidebar"] div[role="radiogroup"] > label[data-checked="true"],
    [data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {
        background-color: #2563eb !important;
        color: #ffffff !important;
        font-weight: 600;
    }

    /* Hide standard radio circle in sidebar */
    [data-testid="stSidebar"] div[role="radiogroup"] input[type="radio"] {
        display: none;
    }

    /* White rounded card */
    .bda-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }

    /* Status metric card */
    .bda-status-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 16px;
        display: flex;
        align-items: center;
        gap: 14px;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04);
        height: 100%;
    }

    .bda-icon-circle {
        width: 44px;
        height: 44px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 20px;
        flex-shrink: 0;
    }

    .icon-green { background-color: #dcfce7; color: #16a34a; }
    .icon-teal { background-color: #ccfbf1; color: #0d9488; }
    .icon-purple { background-color: #f3e8ff; color: #9333ea; }
    .icon-yellow { background-color: #fef9c3; color: #ca8a04; }
    .icon-red { background-color: #fee2e2; color: #dc2626; }

    /* Experimental Badge */
    .bda-badge-experimental {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background-color: #fee2e2;
        color: #991b1b;
        border: 1px solid #fecaca;
        border-radius: 6px;
        padding: 6px 12px;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.5px;
    }

    /* Persistent Synthetic Banner */
    .bda-banner-warning {
        background-color: #fffbeb;
        border-left: 4px solid #f59e0b;
        color: #92400e;
        padding: 12px 16px;
        border-radius: 6px;
        font-size: 13px;
        margin-bottom: 16px;
        line-height: 1.5;
    }

    /* Risk Score Results Badge */
    .bda-risk-badge-low {
        background-color: #dcfce7;
        color: #15803d;
        border: 1px solid #bbf7d0;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 14px;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }

    .bda-risk-badge-high {
        background-color: #fee2e2;
        color: #b91c1c;
        border: 1px solid #fecaca;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 14px;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }

    /* Sidebar Footer Disclaimer Card */
    .sidebar-disclaimer-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 12px;
        font-size: 12px;
        color: #cbd5e1 !important;
        line-height: 1.4;
        margin-top: 20px;
    }

    /* Buttons styling */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.15s ease-in-out;
    }

    .stButton > button[kind="primary"] {
        background-color: #2563eb;
        border-color: #2563eb;
        color: #ffffff;
    }

    .stButton > button[kind="primary"]:hover {
        background-color: #1d4ed8;
        border-color: #1d4ed8;
    }

    /* Streamlit forms cleaner styling */
    div[data-testid="stForm"] {
        border: none;
        padding: 0;
    }

    /* Compact inputs */
    .stNumberInput input {
        border-radius: 6px;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
