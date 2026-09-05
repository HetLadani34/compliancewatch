"""
api/routes/webhooks.py
-----------------------
Simulated Razorpay onboarding webhook endpoint.

In the real Razorpay platform, when a merchant completes their KYC and is
approved, Razorpay sends a signed POST webhook to registered compliance tools.
This endpoint simulates that flow locally for demo purposes.

Also includes an endpoint to toggle mock merchant states (for demo workflow).

Routes
------
POST /api/webhooks/razorpay/onboarding     — Simulated onboarding webhook
POST /api/webhooks/mock/set-state          — Toggle a mock merchant's site state
GET  /api/webhooks/mock/registry           — Inspect the mock server registry
POST /api/webhooks/demo/onboard-all        — One-click onboard all demo merchants
"""

from __future__ import annotations

from typing import Annotated, Any

import httpx
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel

from config.settings import get_settings
from core_agent.compliance_agent import ComplianceAgent
from core_agent.schemas import MerchantOnboardRequest
from utils.logging_config import get_logger

logger = get_logger(__name__)
router = APIRouter()

settings = get_settings()

# ---------------------------------------------------------------------------
# Demo merchant definitions — mirrors MERCHANT_REGISTRY in mock_sites/server.py
# ---------------------------------------------------------------------------

DEMO_MERCHANTS: list[dict[str, str]] = [
    {
        "name": "Diya & Co. Handcrafts",
        "business_category": "Home Decor & Handicrafts",
        "mock_site_key": "diya_store",
    },
    {
        "name": "PureLeaf Organic Tea",
        "business_category": "Food & Beverages",
        "mock_site_key": "organic_tea",
    },
    {
        "name": "Weave & Wonder Handlooms",
        "business_category": "Apparel & Textiles",
        "mock_site_key": "handloom_crafts",
    },
]


# ---------------------------------------------------------------------------
# Dependency
# ---------------------------------------------------------------------------

def _get_agent() -> ComplianceAgent:
    from api.main import get_compliance_agent
    return get_compliance_agent()

AgentDep = Annotated[ComplianceAgent, Depends(_get_agent)]


# ---------------------------------------------------------------------------
# Request/Response Models
# ---------------------------------------------------------------------------

class RazorpayOnboardingWebhook(BaseModel):
    """Simulated Razorpay onboarding webhook payload."""
    event: str = "merchant.activated"
    merchant_name: str
    business_category: str
    mock_site_key: str


class MockStateToggle(BaseModel):
    merchant_key: str
    state: str  # "clean" or "fraud"


class DemoOnboardResult(BaseModel):
    onboarded: list[str]
    failed: list[str]
    total: int


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post(
    "/razorpay/onboarding",
    summary="Simulated Razorpay onboarding webhook",
    status_code=202,
)
async def razorpay_onboarding_webhook(
    payload: RazorpayOnboardingWebhook,
    background_tasks: BackgroundTasks,
    agent: AgentDep,
) -> dict[str, str]:
    """
    Receives a simulated Razorpay merchant.activated event and triggers
    baseline ingestion for the new merchant.

    In production, this webhook would be verified via HMAC-SHA256 signature.
    For this demo, we trust the payload directly.
    """
    logger.info(
        "Razorpay onboarding webhook received",
        extra={"merchant": payload.merchant_name, "event": payload.event},
    )

    if payload.event != "merchant.activated":
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported event type: '{payload.event}'. Only 'merchant.activated' is handled.",
        )

    request = MerchantOnboardRequest(
        name=payload.merchant_name,
        business_category=payload.business_category,
        mock_site_key=payload.mock_site_key,
    )

    # Run onboarding in the background so the webhook returns 202 immediately
    background_tasks.add_task(agent.onboard_merchant, request)

    return {
        "status": "accepted",
        "message": f"Baseline ingestion for '{payload.merchant_name}' scheduled.",
        "mock_site_key": payload.mock_site_key,
    }


@router.post(
    "/mock/set-state",
    summary="Toggle a mock merchant site between clean and fraud state",
)
async def set_mock_merchant_state(payload: MockStateToggle) -> dict[str, Any]:
    """
    Calls the mock server's set-state endpoint to switch a merchant's website
    between its clean (State A) and fraud (State B) version.

    This is the key control surface for the demo — call this before running
    a scan to simulate a merchant switching to banned content.
    """
    if payload.state not in ("clean", "fraud"):
        raise HTTPException(status_code=400, detail="State must be 'clean' or 'fraud'.")

    mock_url = f"{settings.mock_server_base_url}/merchant/{payload.merchant_key}/set-state"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(mock_url, json={"state": payload.state})
            response.raise_for_status()
            result = response.json()
        logger.info(
            "Mock merchant state toggled",
            extra={"key": payload.merchant_key, "state": payload.state},
        )
        return result
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=exc.response.status_code,
            detail=f"Mock server error: {exc.response.text}",
        )
    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail=f"Mock server at {settings.mock_server_base_url} is not reachable. Is it running?",
        )


@router.get(
    "/mock/registry",
    summary="Inspect the mock server's merchant registry and current states",
)
async def get_mock_registry() -> list[dict[str, Any]]:
    """Proxy to the mock server's /registry endpoint for dashboard display."""
    registry_url = f"{settings.mock_server_base_url}/registry"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(registry_url)
            response.raise_for_status()
            return response.json()
    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail="Mock server is not reachable. Start it with: python -m uvicorn mock_sites.server:app --port 8100",
        )


@router.post(
    "/demo/onboard-all",
    response_model=DemoOnboardResult,
    summary="One-click onboard all three demo merchants",
)
async def onboard_all_demo_merchants(agent: AgentDep) -> DemoOnboardResult:
    """
    Onboards all three pre-configured demo merchants simultaneously.
    This is the 'Onboard Demo Merchants' button in the Streamlit dashboard.

    Each merchant's baseline is ingested sequentially (respects Gemini rate limits).
    """
    logger.info("Demo onboard-all triggered.")
    onboarded: list[str] = []
    failed: list[str] = []

    for merchant_data in DEMO_MERCHANTS:
        try:
            request = MerchantOnboardRequest(**merchant_data)
            result = await agent.onboard_merchant(request)
            onboarded.append(result.merchant_name)
            logger.info("Demo merchant onboarded.", extra={"name": result.merchant_name})
        except Exception as exc:
            failed.append(f"{merchant_data['name']}: {str(exc)}")
            logger.error(
                "Demo merchant onboarding failed.",
                exc_info=True,
                extra={"name": merchant_data["name"]},
            )

    return DemoOnboardResult(
        onboarded=onboarded,
        failed=failed,
        total=len(DEMO_MERCHANTS),
    )
