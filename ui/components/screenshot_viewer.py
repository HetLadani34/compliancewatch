"""
ui/components/screenshot_viewer.py
------------------------------------
Side-by-side screenshot comparison component for the merchant detail page.

Loads screenshots from disk (absolute paths stored in DB) and renders them
as a labelled, side-by-side comparison using Streamlit columns.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st
from PIL import Image


def render_screenshot_comparison(
    baseline_path: str | None,
    current_path: str | None,
    merchant_name: str = "Merchant",
) -> None:
    """
    Render a side-by-side baseline vs current screenshot comparison.

    Parameters
    ----------
    baseline_path : str | None
        Absolute path to the baseline (State A) PNG screenshot.
    current_path : str | None
        Absolute path to the current (State B) PNG screenshot.
    merchant_name : str
        Used in the section header for context.
    """
    st.markdown(
        """
        <div style="
            background:linear-gradient(135deg,#0f172a,#1e293b);
            border:1px solid #1e293b;
            border-radius:14px;
            padding:1rem 1.4rem 0.5rem;
            margin-bottom:1rem;
        ">
            <div style="font-size:0.8rem;color:#94a3b8;text-transform:uppercase;letter-spacing:1px;margin-bottom:0.75rem;">
                📸 Visual Evidence — Baseline vs Current
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_baseline, col_divider, col_current = st.columns([10, 1, 10])

    with col_baseline:
        _render_screenshot_panel(
            path=baseline_path,
            label="✅ BASELINE (Approved State)",
            label_colour="#10b981",
            caption="Website at time of Razorpay onboarding",
        )

    with col_divider:
        st.markdown(
            """<div style="display:flex;align-items:center;justify-content:center;
            height:300px;color:#475569;font-size:1.5rem;">⚡</div>""",
            unsafe_allow_html=True,
        )

    with col_current:
        _render_screenshot_panel(
            path=current_path,
            label="🚨 CURRENT (Live State)",
            label_colour="#ef4444",
            caption="Website as detected during latest scan",
        )


def _render_screenshot_panel(
    path: str | None,
    label: str,
    label_colour: str,
    caption: str,
) -> None:
    """Render a single screenshot panel with a label and caption."""
    st.markdown(
        f"""<div style="
            color:{label_colour};
            font-size:0.78rem;
            font-weight:700;
            text-transform:uppercase;
            letter-spacing:0.5px;
            margin-bottom:0.4rem;
        ">{label}</div>""",
        unsafe_allow_html=True,
    )

    if path is None:
        _render_placeholder("No screenshot path recorded.")
        return

    img_path = Path(path)
    if not img_path.exists():
        _render_placeholder(f"Screenshot not found:\n{img_path.name}")
        return

    try:
        img = Image.open(img_path)
        st.image(img, use_container_width=True, caption=caption)
    except Exception as e:
        _render_placeholder(f"Error loading image:\n{e}")


def _render_placeholder(message: str) -> None:
    """Render a grey placeholder box when no image is available."""
    st.markdown(
        f"""
        <div style="
            background:#1e293b;
            border:1px dashed #334155;
            border-radius:10px;
            height:200px;
            display:flex;
            align-items:center;
            justify-content:center;
            color:#475569;
            font-size:0.82rem;
            text-align:center;
            padding:1rem;
        ">📷 {message}</div>
        """,
        unsafe_allow_html=True,
    )
