"""
ui/pages/01_dashboard.py
-------------------------
Risk Overview Dashboard — the main grid showing all monitored merchants.

Features
--------
* Summary stats bar (total / green / yellow / red counts)
* Per-merchant risk cards with cosine drift % and last scan time
* "Onboard Demo Merchants" button (first-time setup)
* "🔴 Simulate Fraud" button — toggles mock sites to fraud state
* "🔵 Reset to Clean" button — resets mock sites back to clean state
* "▶ Run Full Scan Now" button — triggers immediate compliance scan
* Auto-refresh every 30 seconds (configurable)
* Clicking a merchant card navigates to the detail page via session state
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import requests
import streamlit as st

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import get_settings
from ui.components.risk_card import render_merchant_card, render_summary_bar

settings = get_settings()
API_BASE = settings.api_base_url
MOCK_BASE = settings.mock_server_base_url

# ---------------------------------------------------------------------------
# Helper: API calls
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


def _api_post(path: str, json: dict | None = None, timeout: int = 120) -> dict | None:
    try:
        r = requests.post(f"{API_BASE}{path}", json=json or {}, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        err_msg = str(e)
        if hasattr(e, "response") and e.response is not None and e.response.text:
            err_msg = f"{e.response.status_code} - {e.response.text}"
        st.error(f"API error ({path}): {err_msg}")
        return None


def _toggle_all_merchants(state: str) -> None:
    """Toggle all mock merchants to 'clean' or 'fraud' state."""
    keys = ["diya_store", "organic_tea", "handloom_crafts"]
    success_count = 0
    for key in keys:
        result = _api_post(
            "/api/webhooks/mock/set-state",
            json={"merchant_key": key, "state": state},
        )
        if result:
            success_count += 1
    if success_count == len(keys):
        st.success(f"✅ All merchants switched to **{state.upper()}** state.")
    else:
        st.warning(f"⚠️ {success_count}/{len(keys)} merchants updated. Check mock server.")


# ---------------------------------------------------------------------------
# Page Header
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div style="
        background:linear-gradient(90deg,#1e3a5f,#0f2544);
        border-bottom:1px solid #1e293b;
        padding:1.2rem 1.5rem;
        margin:-1rem -1rem 1.5rem -1rem;
    ">
        <div style="font-size:1.5rem;font-weight:800;color:#e2e8f0;">
            🛡️ ComplianceWatch — Risk Dashboard
        </div>
        <div style="font-size:0.82rem;color:#64748b;margin-top:0.2rem;">
            Real-time merchant compliance monitoring · Powered by Gemini 1.5 Flash
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Control Panel
# ---------------------------------------------------------------------------

col_ob, col_custom, col_fraud, col_clean, col_scan, col_refresh = st.columns([2, 2, 2, 2, 2, 1])

with col_ob:
    if st.button("🚀 Demo Merchants (3)", use_container_width=True, help="Onboard 3 pre-configured demo stores"):
        with st.spinner("Onboarding 3 demo merchants — scraping baselines & generating embeddings..."):
            result = _api_post("/api/webhooks/demo/onboard-all", timeout=180)
        if result:
            st.success(f"✅ Onboarded: {', '.join(result.get('onboarded', []))}")
            if result.get("failed"):
                st.error(f"Failed: {result['failed']}")
            st.rerun()

with col_custom:
    show_add_form = st.toggle("➕ Add Merchant", value=False, help="Onboard a custom live merchant website")

with col_fraud:
    if st.button("🔴 Simulate Fraud (All)", use_container_width=True):
        _toggle_all_merchants("fraud")
        st.info("ℹ️ Sites switched to fraud state. Run a scan to detect violations.")

with col_clean:
    if st.button("🟢 Reset to Clean (All)", use_container_width=True):
        _toggle_all_merchants("clean")
        st.info("ℹ️ Sites reset to clean state.")

with col_scan:
    if st.button("▶ Run Full Scan Now", use_container_width=True, type="primary"):
        with st.spinner("Running compliance scan across all merchants — this may take 30–90 seconds..."):
            result = _api_post("/api/scan/trigger", timeout=300)
        if result:
            st.success(
                f"✅ Scan complete in {result.get('duration_seconds', 0):.1f}s — "
                f"🟢 {result.get('green_count', 0)} · "
                f"🟡 {result.get('yellow_count', 0)} · "
                f"🔴 {result.get('red_count', 0)}"
            )
            st.rerun()

with col_refresh:
    auto_refresh = st.toggle("Auto", value=False, help="Auto-refresh every 30 seconds")

# ---------------------------------------------------------------------------
# Add Custom Merchant Form
# ---------------------------------------------------------------------------
if show_add_form:
    with st.expander("➕ **Onboard New Merchant Website**", expanded=True):
        st.markdown(
            """
            <div style="font-size:0.88rem;color:#94a3b8;margin-bottom:0.8rem;">
                Enter any live merchant website URL. Playwright will capture its initial approved baseline screenshot and compute its semantic embedding for continuous compliance monitoring.
            </div>
            """,
            unsafe_allow_html=True,
        )
        c1, c2 = st.columns([2, 1])
        with c1:
            custom_name = st.text_input("Merchant / Business Name*", placeholder="e.g. Silk & Stone Jewellers")
        with c2:
            custom_cat = st.selectbox(
                "Business Category*",
                [
                    "Apparel & Fashion",
                    "Food & Beverages",
                    "Handicrafts & Decor",
                    "Electronics & Gadgets",
                    "Health & Wellness",
                    "Home & Living",
                    "Other E-Commerce",
                ],
            )
        custom_url = st.text_input(
            "Merchant Website URL*",
            placeholder="e.g. https://example.com or http://localhost:8100/merchant/diya_store",
            help="Live URL to scrape and ingest baseline",
        )

        if st.button("🚀 Onboard & Capture Baseline", type="primary"):
            if not custom_name.strip() or not custom_url.strip():
                st.warning("⚠️ Please provide both Merchant Name and Website URL.")
            else:
                with st.spinner(f"Scraping '{custom_url}' and generating baseline vector..."):
                    import re, uuid
                    clean_key = re.sub(r"[^a-z0-9_]+", "_", custom_name.strip().lower()).strip("_")[:30] or "custom_store"
                    mock_key = f"{clean_key}_{uuid.uuid4().hex[:6]}"
                    payload = {
                        "name": custom_name.strip(),
                        "business_category": custom_cat,
                        "registered_url": custom_url.strip(),
                        "mock_site_key": mock_key,
                    }
                    res = _api_post("/api/merchants/onboard", json=payload, timeout=120)
                if res and res.get("merchant_id"):
                    st.success(f"✅ Merchant '{custom_name}' onboarded successfully! Baseline captured.")
                    time.sleep(1)
                    st.rerun()

st.markdown("---")

# ---------------------------------------------------------------------------
# Load merchant data
# ---------------------------------------------------------------------------

merchants_data = _api_get("/api/merchants") or []

if not merchants_data:
    st.markdown(
        """
        <div style="
            text-align:center;padding:3rem;background:rgba(30,41,59,0.4);
            border:1px dashed #334155;border-radius:14px;color:#64748b;
        ">
            <div style="font-size:3rem;margin-bottom:1rem;">📭</div>
            <div style="font-size:1.1rem;font-weight:600;">No merchants onboarded yet.</div>
            <div style="font-size:0.85rem;margin-top:0.5rem;">
                Click <strong>🚀 Onboard Demo Merchants</strong> above to get started.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    # ---------------------------------------------------------------------------
    # Summary Stats Bar
    # ---------------------------------------------------------------------------
    green_c = sum(1 for m in merchants_data if m.get("risk_level") == "GREEN")
    yellow_c = sum(1 for m in merchants_data if m.get("risk_level") == "YELLOW")
    red_c = sum(1 for m in merchants_data if m.get("risk_level") == "RED")
    total_c = len(merchants_data)

    st.html(render_summary_bar(green_c, yellow_c, red_c, total_c))

    # ---------------------------------------------------------------------------
    # Sort: RED first, then YELLOW, then GREEN, then UNSCANNED
    # ---------------------------------------------------------------------------
    _order = {"RED": 0, "YELLOW": 1, "GREEN": 2, None: 3}
    merchants_sorted = sorted(
        merchants_data,
        key=lambda m: _order.get(m.get("risk_level"), 3),
    )

    # ---------------------------------------------------------------------------
    # Merchant Grid (3 columns)
    # ---------------------------------------------------------------------------
    cols = st.columns(3)
    for idx, merchant in enumerate(merchants_sorted):
        with cols[idx % 3]:
            card_html = render_merchant_card(
                merchant_name=merchant.get("name", "Unknown"),
                mock_site_key=merchant.get("mock_site_key", ""),
                risk_level=merchant.get("risk_level"),
                account_status=merchant.get("status", "ACTIVE"),
                cosine_variance_pct=merchant.get("cosine_variance_pct"),
                last_scanned_at=merchant.get("last_scanned_at"),
                vision_triggered=merchant.get("vision_triggered", False)
                    if merchant.get("last_scanned_at") else False,
            )
            st.html(card_html)

            # Detail view button
            risk = merchant.get("risk_level", "UNSCANNED")
            btn_label = (
                "🔍 Investigate →" if risk in ("RED", "YELLOW")
                else "📋 View Details →"
            )
            if st.button(btn_label, key=f"detail_{merchant['merchant_id']}", use_container_width=True):
                st.session_state["selected_merchant_id"] = merchant["merchant_id"]
                st.session_state["selected_merchant_name"] = merchant["name"]
                st.switch_page("pages/02_merchant_detail.py")

    # ---------------------------------------------------------------------------
    # Flagged Merchants Alert Banner
    # ---------------------------------------------------------------------------
    flagged = [m for m in merchants_data if m.get("risk_level") in ("RED", "YELLOW")]
    if flagged:
        st.markdown("---")
        st.markdown(
            f"""
            <div style="
                background:rgba(239,68,68,0.1);border:1px solid #ef4444;
                border-radius:12px;padding:1rem 1.4rem;
            ">
                <div style="color:#ef4444;font-weight:700;font-size:1rem;margin-bottom:0.4rem;">
                    ⚠️ {len(flagged)} Merchant(s) Require Immediate Attention
                </div>
                <div style="color:#fca5a5;font-size:0.85rem;">
                    {', '.join(m['name'] for m in flagged)} — click their cards above to investigate.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------------
# Mock Server State Panel
# ---------------------------------------------------------------------------
st.markdown("---")
with st.expander("🔧 Mock Server Controls — Toggle Merchant States for Demo"):
    registry = _api_get("/api/webhooks/mock/registry")
    if registry:
        st.markdown("**Current mock site states** (controls what the scanner sees):")
        for entry in registry:
            state = entry.get("active_state", "?")
            state_badge = "🟢 CLEAN" if state == "clean" else "🔴 FRAUD"
            c1, c2, c3 = st.columns([3, 2, 3])
            with c1:
                st.markdown(f"**{entry['name']}**  \n`{entry['key']}`")
            with c2:
                st.markdown(f"**{state_badge}**")
            with c3:
                new_state = "fraud" if state == "clean" else "clean"
                btn_lbl = f"Switch to {'FRAUD 🔴' if new_state == 'fraud' else 'CLEAN 🟢'}"
                if st.button(btn_lbl, key=f"toggle_{entry['key']}"):
                    result = _api_post(
                        "/api/webhooks/mock/set-state",
                        json={"merchant_key": entry["key"], "state": new_state},
                    )
                    if result:
                        st.success(f"Switched {entry['key']} → {new_state}")
                        st.rerun()
            st.divider()
    else:
        st.warning("Mock server not reachable. Ensure it's running on port 8100.")

# ---------------------------------------------------------------------------
# Auto-refresh
# ---------------------------------------------------------------------------
if auto_refresh:
    st.markdown(
        "<div style='text-align:right;font-size:0.72rem;color:#475569;'>⟳ Auto-refresh in 30s</div>",
        unsafe_allow_html=True,
    )
    time.sleep(30)
    st.rerun()
