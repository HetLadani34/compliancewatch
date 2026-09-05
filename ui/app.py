"""
ui/app.py
----------
ComplianceWatch Streamlit Dashboard — main entry point.

Run with: streamlit run ui/app.py

This file configures the page, injects global CSS, and provides shared
utility functions (API client, session state helpers) used by all pages.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Add project root to sys.path so imports work correctly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Page Configuration (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ComplianceWatch — AI Risk Manager",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "ComplianceWatch: AI-powered merchant compliance monitoring for Razorpay. Razorpay AI Buildathon 2026.",
    },
)

# ---------------------------------------------------------------------------
# Global CSS Injection
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* ── Google Font ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* ── Dark Background ── */
    .stApp {
        background: linear-gradient(135deg, #080d16 0%, #0c1420 50%, #0a1228 100%);
        min-height: 100vh;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0e1a 0%, #111827 100%);
        border-right: 1px solid #1e293b;
    }
    [data-testid="stSidebar"] .stMarkdown { color: #e2e8f0; }

    /* ── Main content area ── */
    [data-testid="stAppViewContainer"] {
        padding-top: 0;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(90deg, #2563eb, #1d4ed8);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.88rem;
        padding: 0.5rem 1.2rem;
        transition: opacity 0.2s, transform 0.1s;
    }
    .stButton > button:hover {
        opacity: 0.9;
        transform: translateY(-1px);
    }

    /* ── Danger button ── */
    .danger-btn > button {
        background: linear-gradient(90deg, #dc2626, #b91c1c) !important;
    }

    /* ── Success button ── */
    .success-btn > button {
        background: linear-gradient(90deg, #059669, #047857) !important;
    }

    /* ── Metrics ── */
    [data-testid="stMetric"] {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid #1e293b;
        border-radius: 10px;
        padding: 0.75rem 1rem;
    }
    [data-testid="stMetricLabel"] { color: #94a3b8 !important; font-size: 0.75rem !important; }
    [data-testid="stMetricValue"] { color: #e2e8f0 !important; font-size: 1.5rem !important; }

    /* ── Expander ── */
    .streamlit-expanderHeader {
        background: rgba(30, 41, 59, 0.4);
        border-radius: 8px;
        color: #e2e8f0 !important;
    }

    /* ── Code blocks ── */
    .stCodeBlock { border-radius: 8px; }

    /* ── Hide default Streamlit header decoration ── */
    #MainMenu, footer { visibility: hidden; }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #0f172a; }
    ::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }

    /* ── Alert boxes ── */
    .stAlert { border-radius: 8px; }

    /* ── Divider ── */
    hr { border-color: #1e293b; margin: 1rem 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar: App Identity + Navigation Hints
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown(
        """
        <div style="padding:0.5rem 0 1.5rem;">
            <div style="font-size:1.6rem;font-weight:900;color:#e2e8f0;letter-spacing:-0.5px;">
                🛡️ ComplianceWatch
            </div>
            <div style="font-size:0.75rem;color:#64748b;margin-top:0.25rem;">
                AI Risk Manager · Razorpay Buildathon
            </div>
        </div>
        <hr style="border-color:#1e293b;margin:0 0 1rem;">
        """,
        unsafe_allow_html=True,
    )

    st.markdown("**Navigation**")
    st.markdown("Use the pages above ↑ to switch between:")
    st.markdown("- **01 Dashboard** — Risk overview grid")
    st.markdown("- **02 Merchant Detail** — Deep-dive analysis")

    st.markdown("---")
    st.markdown("**System Status**")

    # Quick API health check
    try:
        import requests
        from config.settings import get_settings
        settings = get_settings()
        resp = requests.get(f"{settings.api_base_url}/health", timeout=2)
        if resp.status_code == 200:
            data = resp.json()
            st.markdown(
                f"""
                <div style="background:rgba(16,185,129,0.1);border:1px solid #10b981;
                border-radius:8px;padding:0.5rem 0.75rem;font-size:0.78rem;color:#10b981;">
                ✅ API Online · Scheduler: {data.get('scheduler','?')}
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.error("API returned non-200 response.")
    except Exception:
        st.markdown(
            """
            <div style="background:rgba(239,68,68,0.1);border:1px solid #ef4444;
            border-radius:8px;padding:0.5rem 0.75rem;font-size:0.78rem;color:#ef4444;">
            ❌ API Offline — start it with run.py
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown(
        """
        <div style="font-size:0.7rem;color:#475569;line-height:1.6;">
        <strong style="color:#64748b;">Tech Stack</strong><br>
        🤖 Gemini 1.5 Flash (Vision)<br>
        📐 text-embedding-004 (Embeddings)<br>
        🗄️ ChromaDB (Vector Store)<br>
        🐍 FastAPI + SQLite<br>
        🎨 Streamlit Dashboard
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Landing Page Content (shown when no sub-page is selected)
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div style="
        background:linear-gradient(135deg,#0f172a,#1e293b);
        border:1px solid #1e293b;
        border-radius:20px;
        padding:3rem;
        text-align:center;
        margin:2rem auto;
        max-width:700px;
    ">
        <div style="font-size:4rem;margin-bottom:1rem;">🛡️</div>
        <h1 style="color:#e2e8f0;font-size:2.2rem;font-weight:900;margin:0 0 0.5rem;">
            ComplianceWatch
        </h1>
        <p style="color:#94a3b8;font-size:1rem;line-height:1.7;max-width:500px;margin:0 auto 2rem;">
            AI-powered merchant compliance monitoring for Razorpay.<br>
            Detects <strong style="color:#f59e0b;">Transaction Laundering</strong> via
            multimodal semantic drift analysis.
        </p>
        <div style="display:flex;gap:1rem;justify-content:center;flex-wrap:wrap;">
            <div style="background:rgba(16,185,129,0.1);border:1px solid #10b981;border-radius:8px;
            padding:0.5rem 1rem;font-size:0.82rem;color:#10b981;">
                🟢 Text Embeddings → Semantic Diff
            </div>
            <div style="background:rgba(245,158,11,0.1);border:1px solid #f59e0b;border-radius:8px;
            padding:0.5rem 1rem;font-size:0.82rem;color:#f59e0b;">
                👁 Vision AI → Policy Check
            </div>
            <div style="background:rgba(239,68,68,0.1);border:1px solid #ef4444;border-radius:8px;
            padding:0.5rem 1rem;font-size:0.82rem;color:#ef4444;">
                🚨 Alert → Account Suspend
            </div>
        </div>
    </div>
    <div style="text-align:center;color:#475569;font-size:0.85rem;margin-top:1rem;">
        👈 Select <strong>01 Dashboard</strong> from the sidebar to get started.
    </div>
    """,
    unsafe_allow_html=True,
)
