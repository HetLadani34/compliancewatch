"""
ui/pages/02_merchant_detail.py
-------------------------------
Deep-dive merchant analysis page.

Shows:
- Full merchant profile header with risk badge
- Cosine drift gauge
- Side-by-side baseline vs current screenshots
- Gemini's structured JSON verdict (formatted, not raw JSON)
- Visual evidence list from Gemini
- Full scan history table
- Account action buttons (Scan Now, Suspend Account)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import requests
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import get_settings
from ui.components.risk_card import RISK_COLOURS, render_drift_gauge
from ui.components.screenshot_viewer import render_screenshot_comparison

settings = get_settings()
API_BASE = settings.api_base_url

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _api_get(path: str, timeout: int = 10) -> dict | list | None:
    try:
        r = requests.get(f"{API_BASE}{path}", timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        err_msg = str(e)
        if hasattr(e, "response") and e.response is not None and e.response.text:
            err_msg = f"{e.response.status_code} - {e.response.text}"
        st.error(f"API error ({path}): {err_msg}")
        return None


def _api_post(path: str, json_data: dict | None = None, timeout: int = 180) -> dict | None:
    try:
        r = requests.post(f"{API_BASE}{path}", json=json_data or {}, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        err_msg = str(e)
        if hasattr(e, "response") and e.response is not None and e.response.text:
            err_msg = f"{e.response.status_code} - {e.response.text}"
        st.error(f"API error ({path}): {err_msg}")
        return None


# ---------------------------------------------------------------------------
# Resolve merchant selection
# ---------------------------------------------------------------------------

merchant_id = st.session_state.get("selected_merchant_id")
merchant_name_hint = st.session_state.get("selected_merchant_name", "Merchant Detail")

# Allow direct URL-style access via query param (for future use)
# query_params = st.query_params  # Uncomment if using URL routing

if not merchant_id:
    st.markdown(
        """
        <div style="text-align:center;padding:3rem;color:#64748b;">
            <div style="font-size:3rem;">👈</div>
            <div style="font-size:1.1rem;font-weight:600;margin-top:0.5rem;">
                Select a merchant from the Dashboard first.
            </div>
            <div style="font-size:0.85rem;margin-top:0.25rem;">
                Go to <strong>01 Dashboard</strong> and click a merchant's detail button.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

# ---------------------------------------------------------------------------
# Load merchant data
# ---------------------------------------------------------------------------

merchant = _api_get(f"/api/merchants/{merchant_id}")
if not merchant:
    st.error(f"Could not load merchant `{merchant_id}`. Check API connectivity.")
    st.stop()

risk_level = merchant.get("risk_level") or "UNSCANNED"
account_status = merchant.get("status", "ACTIVE")
colours = RISK_COLOURS.get(
    "SUSPENDED" if account_status == "SUSPENDED" else risk_level,
    RISK_COLOURS["UNSCANNED"],
)

# ---------------------------------------------------------------------------
# Page Header
# ---------------------------------------------------------------------------

col_back, col_title = st.columns([1, 8])
with col_back:
    if st.button("← Back"):
        st.switch_page("pages/01_dashboard.py")

st.html(
    f'<div style="background:linear-gradient(135deg,{colours["bg"]},{colours["bg"]});border:1.5px solid {colours["border"]};border-radius:16px;padding:1.5rem 2rem;margin-bottom:1.5rem;display:flex;align-items:center;gap:1.5rem;box-shadow:0 0 30px {colours["glow"]};">'
    f'<div style="font-size:3.5rem;">{colours["emoji"]}</div>'
    f'<div>'
    f'<div style="font-size:1.8rem;font-weight:900;color:{colours["text"]};">{merchant["name"]}</div>'
    f'<div style="display:flex;gap:1rem;margin-top:0.3rem;flex-wrap:wrap;">'
    f'<span style="background:{colours["badge_bg"]};color:{colours["badge_text"]};border-radius:6px;padding:2px 10px;font-size:0.75rem;font-weight:700;">{colours["label"]}</span>'
    f'<span style="color:#64748b;font-size:0.82rem;">📂 {merchant.get("business_category","—")}</span>'
    f'<span style="color:#64748b;font-size:0.82rem;">🔑 <code>{merchant.get("mock_site_key","—")}</code></span>'
    f'<span style="color:#64748b;font-size:0.82rem;">🏦 Account: <strong>{account_status}</strong></span>'
    f'</div>'
    f'</div>'
    f'</div>'
)

# ---------------------------------------------------------------------------
# Action Buttons Row
# ---------------------------------------------------------------------------

col_scan, col_sus, col_spacer = st.columns([2, 2, 5])

with col_scan:
    if st.button("▶ Scan This Merchant Now", use_container_width=True, type="primary"):
        with st.spinner(f"Scanning {merchant['name']}..."):
            result = _api_post(f"/api/scan/merchant/{merchant_id}")
        if result:
            st.success(
                f"✅ Scan complete — Risk Level: **{result.get('risk_level', '?')}** | "
                f"Drift: {result.get('drift_report', {}).get('variance_pct', 0):.1f}%"
            )
            st.rerun()

with col_sus:
    if account_status != "SUSPENDED":
        if st.button("🚫 Suspend Account", use_container_width=True):
            confirm = st.session_state.get(f"confirm_suspend_{merchant_id}", False)
            if confirm:
                result = _api_post(f"/api/merchants/{merchant_id}/suspend")
                if result:
                    st.error(f"🚫 {result.get('message')}")
                    st.session_state[f"confirm_suspend_{merchant_id}"] = False
                    st.rerun()
            else:
                st.session_state[f"confirm_suspend_{merchant_id}"] = True
                st.warning("⚠️ Click again to confirm suspension.")
    else:
        st.markdown(
            """<div style="background:rgba(139,92,246,0.1);border:1px solid #8b5cf6;
            border-radius:8px;padding:0.5rem 0.75rem;font-size:0.82rem;color:#8b5cf6;
            text-align:center;">🚫 Account Suspended</div>""",
            unsafe_allow_html=True,
        )

st.markdown("---")

# ---------------------------------------------------------------------------
# Latest Scan Analysis (main analysis section)
# ---------------------------------------------------------------------------

scan_history = merchant.get("scan_history", [])
latest_scan = scan_history[0] if scan_history else None

if latest_scan:
    st.markdown("## 🔬 Latest Scan Analysis")

    # --- Metric Row ---
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        drift_val = latest_scan.get("cosine_variance_pct", 0.0) or 0.0
        st.metric("Semantic Drift", f"{drift_val:.1f}%")
    with m2:
        st.metric("Cosine Distance", f"{latest_scan.get('cosine_distance', 0.0):.4f}")
    with m3:
        st.metric("Drift Detected", "YES ⚠️" if latest_scan.get("drift_detected") else "NO ✅")
    with m4:
        st.metric("Vision AI Triggered", "YES 👁" if latest_scan.get("vision_triggered") else "NO")

    # --- Drift Gauge ---
    st.markdown(
        render_drift_gauge(
            variance_pct=latest_scan.get("cosine_variance_pct", 0.0) or 0.0,
            threshold_pct=settings.cosine_drift_threshold * 100,
        ),
        unsafe_allow_html=True,
    )

    # --- Screenshot Comparison ---
    render_screenshot_comparison(
        baseline_path=merchant.get("baseline_screenshot_path"),
        current_path=latest_scan.get("current_screenshot_path"),
        merchant_name=merchant["name"],
    )

    # --- Gemini Vision Analysis Results ---
    vision_result = latest_scan.get("vision_result")
    if vision_result:
        st.markdown("---")
        st.markdown("## 👁 Gemini AI Compliance Verdict")

        v_col1, v_col2 = st.columns([1, 1])

        with v_col1:
            is_violation = vision_result.get("is_policy_violation", False)
            confidence = vision_result.get("confidence_score", 0)
            category = vision_result.get("detected_banned_category") or "None"

            verdict_colour = "#ef4444" if is_violation else "#10b981"
            verdict_text = "POLICY VIOLATION DETECTED" if is_violation else "NO VIOLATION FOUND"
            verdict_emoji = "🚨" if is_violation else "✅"

            st.markdown(
                f"""
                <div style="
                    background:{verdict_colour}18;
                    border:2px solid {verdict_colour};
                    border-radius:14px;
                    padding:1.5rem;
                    text-align:center;
                ">
                    <div style="font-size:2.5rem;">{verdict_emoji}</div>
                    <div style="font-size:1rem;font-weight:800;color:{verdict_colour};margin:0.5rem 0;">
                        {verdict_text}
                    </div>
                    <div style="font-size:0.82rem;color:#94a3b8;">
                        Confidence: <strong style="color:{verdict_colour};">{confidence}%</strong>
                    </div>
                    <div style="font-size:0.82rem;color:#94a3b8;margin-top:0.25rem;">
                        Category: <strong style="color:#e2e8f0;">{category}</strong>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with v_col2:
            st.markdown(
                """
                <div style="color:#94a3b8;font-size:0.75rem;text-transform:uppercase;
                letter-spacing:0.5px;margin-bottom:0.5rem;">🤖 Gemini Reasoning</div>
                """,
                unsafe_allow_html=True,
            )
            reasoning = vision_result.get("reasoning_summary", "No reasoning provided.")
            st.markdown(
                f"""
                <div style="
                    background:#0f172a;border:1px solid #1e293b;border-radius:10px;
                    padding:1rem;font-size:0.88rem;color:#cbd5e1;line-height:1.6;
                ">
                    {reasoning}
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Visual Evidence
            evidence_list = vision_result.get("visual_evidence", [])
            if evidence_list:
                st.markdown(
                    """<div style="color:#94a3b8;font-size:0.75rem;text-transform:uppercase;
                    letter-spacing:0.5px;margin:0.75rem 0 0.4rem;">📍 Visual Evidence Detected</div>""",
                    unsafe_allow_html=True,
                )
                for item in evidence_list:
                    st.markdown(
                        f"""
                        <div style="background:#1e293b;border-left:3px solid #ef4444;
                        border-radius:4px;padding:0.35rem 0.75rem;margin-bottom:0.3rem;
                        font-size:0.82rem;color:#fca5a5;">
                            🔍 {item}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        # --- Raw JSON Expander ---
        with st.expander("📋 View Raw Gemini JSON Response"):
            st.json(vision_result)

    elif latest_scan.get("drift_detected"):
        st.info("ℹ️ Drift was detected but vision analysis result is unavailable (may have failed).")
    else:
        st.success(
            f"✅ No semantic drift detected (variance: {latest_scan.get('cosine_variance_pct', 0):.1f}%). "
            "Vision analysis was not triggered."
        )

else:
    st.info(
        "ℹ️ This merchant has no scan results yet. "
        "Click **▶ Scan This Merchant Now** above to run the first scan."
    )

# ---------------------------------------------------------------------------
# Scan History Table
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown("## 📊 Scan History")

if scan_history:
    # Build display table
    table_rows = []
    for scan in scan_history:
        risk = scan.get("risk_level", "?")
        risk_colour = {"GREEN": "🟢", "YELLOW": "🟡", "RED": "🔴"}.get(risk, "⚪")
        table_rows.append({
            "Timestamp": scan.get("scanned_at", "?")[:19].replace("T", " "),
            "Risk": f"{risk_colour} {risk}",
            "Drift": f"{scan.get('cosine_variance_pct', 0):.1f}%",
            "Distance": f"{scan.get('cosine_distance', 0):.4f}",
            "Drift Flag": "⚠️ Yes" if scan.get("drift_detected") else "✅ No",
            "Vision AI": "👁 Yes" if scan.get("vision_triggered") else "—",
        })
    st.dataframe(
        table_rows,
        use_container_width=True,
        hide_index=True,
    )
else:
    st.markdown(
        """<div style="color:#64748b;text-align:center;padding:1rem;">
        No scan history available.</div>""",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Merchant Metadata
# ---------------------------------------------------------------------------
st.markdown("---")
with st.expander("🔎 Merchant Metadata"):
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"**Merchant ID:** `{merchant['merchant_id']}`")
        st.markdown(f"**Business Category:** {merchant.get('business_category', '—')}")
        st.markdown(f"**Mock Site Key:** `{merchant.get('mock_site_key', '—')}`")
        st.markdown(f"**Registered URL:** {merchant.get('registered_url', '—')}")
    with col_b:
        st.markdown(f"**Onboarded At:** {merchant.get('onboarded_at', '—')[:19].replace('T', ' ')}")
        st.markdown(f"**Baseline Ingested:** {(merchant.get('baseline_ingested_at') or '—')[:19].replace('T', ' ')}")
        st.markdown(f"**Account Status:** {account_status}")
        if merchant.get("baseline_text_snippet"):
            st.markdown("**Baseline Text Preview:**")
            st.caption(merchant["baseline_text_snippet"][:300])
