"""
ui/components/risk_card.py
---------------------------
Reusable Streamlit risk indicator components.

Provides styled HTML cards and metric displays for the dashboard grid,
rendered via st.markdown with unsafe_allow_html=True for rich CSS control.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Risk colour palette
# ---------------------------------------------------------------------------

RISK_COLOURS: dict[str, dict[str, str]] = {
    "GREEN": {
        "bg": "rgba(16, 185, 129, 0.12)",
        "border": "#10b981",
        "text": "#065f46",
        "badge_bg": "#10b981",
        "badge_text": "#ffffff",
        "glow": "rgba(16, 185, 129, 0.3)",
        "emoji": "🟢",
        "label": "COMPLIANT",
    },
    "YELLOW": {
        "bg": "rgba(245, 158, 11, 0.12)",
        "border": "#f59e0b",
        "text": "#78350f",
        "badge_bg": "#f59e0b",
        "badge_text": "#000000",
        "glow": "rgba(245, 158, 11, 0.3)",
        "emoji": "🟡",
        "label": "REVIEW NEEDED",
    },
    "RED": {
        "bg": "rgba(239, 68, 68, 0.12)",
        "border": "#ef4444",
        "text": "#7f1d1d",
        "badge_bg": "#ef4444",
        "badge_text": "#ffffff",
        "glow": "rgba(239, 68, 68, 0.4)",
        "emoji": "🔴",
        "label": "VIOLATION DETECTED",
    },
    "UNSCANNED": {
        "bg": "rgba(100, 116, 139, 0.10)",
        "border": "#64748b",
        "text": "#334155",
        "badge_bg": "#64748b",
        "badge_text": "#ffffff",
        "glow": "rgba(100, 116, 139, 0.2)",
        "emoji": "⚪",
        "label": "NOT SCANNED",
    },
    "SUSPENDED": {
        "bg": "rgba(139, 92, 246, 0.12)",
        "border": "#8b5cf6",
        "text": "#4c1d95",
        "badge_bg": "#8b5cf6",
        "badge_text": "#ffffff",
        "glow": "rgba(139, 92, 246, 0.3)",
        "emoji": "🚫",
        "label": "SUSPENDED",
    },
}


def _get_colours(risk_level: str | None, account_status: str = "ACTIVE") -> dict[str, str]:
    if account_status == "SUSPENDED":
        return RISK_COLOURS["SUSPENDED"]
    return RISK_COLOURS.get(risk_level or "UNSCANNED", RISK_COLOURS["UNSCANNED"])


# ---------------------------------------------------------------------------
# Merchant Risk Card
# ---------------------------------------------------------------------------

def render_merchant_card(
    merchant_name: str,
    mock_site_key: str,
    risk_level: str | None,
    account_status: str,
    cosine_variance_pct: float | None,
    last_scanned_at: str | None,
    vision_triggered: bool = False,
    on_click_key: str = "",
) -> str:
    """
    Return a styled HTML merchant card as a string.

    The card shows:
    - Merchant name and category tag
    - Risk badge with pulsing animation for RED alerts
    - Cosine variance metric
    - Last scan timestamp
    - Whether vision analysis was triggered

    Parameters
    ----------
    on_click_key : str
        A unique CSS class to add for targeting with JS (future use).
    """
    colours = _get_colours(risk_level, account_status)
    rl = risk_level or "UNSCANNED"

    # Pulse animation only for RED
    pulse_style = (
        f"animation: pulse-border 2s infinite;"
        if rl == "RED" else ""
    )

    variance_display = (
        f"{cosine_variance_pct:.1f}%" if cosine_variance_pct is not None else "N/A"
    )
    scan_display = last_scanned_at[:19].replace("T", " ") if last_scanned_at else "Never"
    vision_badge = (
        '<span style="background:#7c3aed;color:#fff;border-radius:4px;'
        'padding:1px 6px;font-size:0.65rem;margin-left:6px;">👁 AI Analysed</span>'
        if vision_triggered else ""
    )

    return f"""
<div class="merchant-card {on_click_key}" style="
    background: {colours['bg']};
    border: 1.5px solid {colours['border']};
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.75rem;
    box-shadow: 0 0 20px {colours['glow']};
    {pulse_style}
    transition: transform 0.2s;
">
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:0.6rem;">
        <div>
            <div style="font-size:1rem;font-weight:700;color:{colours['text']};">{merchant_name}</div>
            <div style="font-size:0.72rem;color:#64748b;margin-top:1px;">
                🔑 <code style="font-size:0.7rem;">{mock_site_key}</code>
                {vision_badge}
            </div>
        </div>
        <div style="
            background:{colours['badge_bg']};
            color:{colours['badge_text']};
            border-radius:8px;
            padding:4px 10px;
            font-size:0.72rem;
            font-weight:800;
            letter-spacing:0.5px;
            text-align:center;
        ">
            {colours['emoji']}<br>{colours['label']}
        </div>
    </div>
    <div style="display:flex;gap:1.5rem;margin-top:0.5rem;">
        <div>
            <div style="font-size:0.65rem;color:#94a3b8;text-transform:uppercase;letter-spacing:0.5px;">Semantic Drift</div>
            <div style="font-size:1.1rem;font-weight:800;color:{colours['border']};">{variance_display}</div>
        </div>
        <div>
            <div style="font-size:0.65rem;color:#94a3b8;text-transform:uppercase;letter-spacing:0.5px;">Last Scanned</div>
            <div style="font-size:0.8rem;font-weight:600;color:{colours['text']};">{scan_display}</div>
        </div>
    </div>
</div>
<style>
@keyframes pulse-border {{
    0%   {{ box-shadow: 0 0 8px rgba(239,68,68,0.4); }}
    50%  {{ box-shadow: 0 0 24px rgba(239,68,68,0.8); }}
    100% {{ box-shadow: 0 0 8px rgba(239,68,68,0.4); }}
}}
</style>
"""


# ---------------------------------------------------------------------------
# Summary Stats Bar
# ---------------------------------------------------------------------------

def render_summary_bar(green: int, yellow: int, red: int, total: int) -> str:
    """Render a horizontal stats bar with green/yellow/red counts."""
    return f"""
<div style="
    display:flex;
    gap:1rem;
    background:linear-gradient(135deg,#0f172a,#1e293b);
    border:1px solid #1e293b;
    border-radius:14px;
    padding:1.2rem 1.6rem;
    margin-bottom:1.5rem;
    align-items:center;
    flex-wrap:wrap;
">
    <div style="flex:1;text-align:center;">
        <div style="font-size:2rem;font-weight:900;color:#e2e8f0;">{total}</div>
        <div style="font-size:0.72rem;color:#64748b;text-transform:uppercase;">Total Merchants</div>
    </div>
    <div style="width:1px;height:40px;background:#1e293b;"></div>
    <div style="flex:1;text-align:center;">
        <div style="font-size:2rem;font-weight:900;color:#10b981;">{green}</div>
        <div style="font-size:0.72rem;color:#64748b;text-transform:uppercase;">🟢 Compliant</div>
    </div>
    <div style="width:1px;height:40px;background:#1e293b;"></div>
    <div style="flex:1;text-align:center;">
        <div style="font-size:2rem;font-weight:900;color:#f59e0b;">{yellow}</div>
        <div style="font-size:0.72rem;color:#64748b;text-transform:uppercase;">🟡 Review</div>
    </div>
    <div style="width:1px;height:40px;background:#1e293b;"></div>
    <div style="flex:1;text-align:center;">
        <div style="font-size:2rem;font-weight:900;color:#ef4444;">{red}</div>
        <div style="font-size:0.72rem;color:#64748b;text-transform:uppercase;">🔴 Violation</div>
    </div>
</div>
"""


# ---------------------------------------------------------------------------
# Cosine Drift Gauge (text-based)
# ---------------------------------------------------------------------------

def render_drift_gauge(variance_pct: float, threshold_pct: float = 30.0) -> str:
    """Render a visual drift gauge bar."""
    clamped = min(variance_pct, 100.0)
    colour = "#10b981" if clamped < threshold_pct else ("#f59e0b" if clamped < 60 else "#ef4444")
    return f"""
<div style="margin:0.5rem 0 1rem;">
    <div style="display:flex;justify-content:space-between;font-size:0.75rem;color:#94a3b8;margin-bottom:4px;">
        <span>Semantic Drift</span>
        <span style="font-weight:700;color:{colour};">{variance_pct:.1f}%</span>
    </div>
    <div style="background:#1e293b;border-radius:8px;height:12px;overflow:hidden;">
        <div style="
            width:{clamped}%;
            height:100%;
            background:linear-gradient(90deg,{colour},{colour}88);
            border-radius:8px;
            transition:width 0.5s;
        "></div>
    </div>
    <div style="font-size:0.65rem;color:#64748b;margin-top:3px;">
        Alert threshold: {threshold_pct:.0f}%
    </div>
</div>
"""
