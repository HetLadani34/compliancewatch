"""
api/routes/merchants.py
------------------------
CRUD endpoints for merchant management.

Routes
------
POST   /api/merchants/onboard        — Onboard a new merchant (triggers baseline ingestion)
GET    /api/merchants                — List all merchants with latest risk status
GET    /api/merchants/{id}           — Get detailed merchant info + scan history
POST   /api/merchants/{id}/suspend   — Mock-suspend a merchant's account
GET    /api/merchants/{id}/status    — Quick status check
"""

from __future__ import annotations

import json
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Path, status
from pydantic import BaseModel

from core_agent.compliance_agent import ComplianceAgent
from core_agent.schemas import MerchantOnboardRequest, OnboardingResult
from database.engine import get_async_session
from database.models import MerchantStatus, RiskLevel
from database.repository import MerchantRepository, ScanResultRepository
from utils.exceptions import MerchantNotFoundError, VectorStoreError
from utils.logging_config import get_logger

logger = get_logger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# Dependency: ComplianceAgent
# ---------------------------------------------------------------------------

def _get_agent() -> ComplianceAgent:
    from api.main import get_compliance_agent
    return get_compliance_agent()


AgentDep = Annotated[ComplianceAgent, Depends(_get_agent)]


# ---------------------------------------------------------------------------
# Response Models
# ---------------------------------------------------------------------------

class MerchantSummary(BaseModel):
    merchant_id: str
    name: str
    business_category: str
    mock_site_key: str
    status: str
    risk_level: str | None
    cosine_distance: float | None
    cosine_variance_pct: float | None
    last_scanned_at: str | None
    baseline_screenshot_path: str | None
    onboarded_at: str


class MerchantDetail(MerchantSummary):
    registered_url: str
    baseline_text_snippet: str | None
    baseline_ingested_at: str | None
    scan_history: list[dict[str, Any]]


class SuspendResponse(BaseModel):
    merchant_id: str
    new_status: str
    message: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post(
    "/onboard",
    response_model=OnboardingResult,
    status_code=status.HTTP_201_CREATED,
    summary="Onboard a new merchant and capture baseline",
)
async def onboard_merchant(
    request: MerchantOnboardRequest,
    agent: AgentDep,
) -> OnboardingResult:
    """
    Onboard a merchant by scraping their clean (State A) website and
    storing the baseline embedding in ChromaDB.

    This endpoint is idempotent — calling it again for an existing merchant
    will refresh their baseline.
    """
    logger.info(
        "Onboarding request received",
        extra={"mock_site_key": request.mock_site_key, "merchant_name": request.name},
    )
    try:
        result = await agent.onboard_merchant(request)
        return result
    except VectorStoreError as exc:
        raise HTTPException(status_code=503, detail=f"Vector store error: {exc.message}")
    except Exception as exc:
        logger.error("Onboarding failed", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Onboarding failed: {str(exc)}")


@router.get(
    "",
    response_model=list[MerchantSummary],
    summary="List all merchants with their latest risk status",
)
async def list_merchants() -> list[MerchantSummary]:
    """
    Return all merchants with their most recent scan result for the
    dashboard overview grid.
    """
    async with get_async_session() as session:
        merchant_repo = MerchantRepository(session)
        scan_repo = ScanResultRepository(session)
        merchants = await merchant_repo.list_all(include_suspended=True)

        summaries: list[MerchantSummary] = []
        for m in merchants:
            latest = await scan_repo.get_latest_for_merchant(m.id)
            summaries.append(MerchantSummary(
                merchant_id=m.id,
                name=m.name,
                business_category=m.business_category,
                mock_site_key=m.mock_site_key,
                status=m.status.value,
                risk_level=latest.risk_level.value if latest else None,
                cosine_distance=latest.cosine_distance if latest else None,
                cosine_variance_pct=latest.cosine_variance_pct if latest else None,
                last_scanned_at=latest.scanned_at.isoformat() if latest else None,
                baseline_screenshot_path=m.baseline_screenshot_path,
                onboarded_at=m.onboarded_at.isoformat(),
            ))

    return summaries


@router.get(
    "/{merchant_id}",
    response_model=MerchantDetail,
    summary="Get detailed merchant info and scan history",
)
async def get_merchant(
    merchant_id: Annotated[str, Path(description="Merchant UUID")],
) -> MerchantDetail:
    """Return full merchant details with up to 20 most recent scan results."""
    try:
        async with get_async_session() as session:
            merchant_repo = MerchantRepository(session)
            scan_repo = ScanResultRepository(session)

            merchant = await merchant_repo.get_by_id(merchant_id, load_relations=False)
            latest = await scan_repo.get_latest_for_merchant(merchant_id)
            history = await scan_repo.get_history(merchant_id, limit=20)

            scan_history_dicts = []
            for scan in history:
                vision_data = None
                if scan.vision_result_json:
                    try:
                        vision_data = json.loads(scan.vision_result_json)
                    except json.JSONDecodeError:
                        vision_data = {"error": "Could not parse vision result"}
                scan_history_dicts.append({
                    "scan_id": scan.id,
                    "scanned_at": scan.scanned_at.isoformat(),
                    "risk_level": scan.risk_level.value,
                    "cosine_distance": scan.cosine_distance,
                    "cosine_variance_pct": scan.cosine_variance_pct,
                    "drift_detected": scan.drift_detected,
                    "vision_triggered": scan.vision_triggered,
                    "current_screenshot_path": scan.current_screenshot_path,
                    "vision_result": vision_data,
                })

            return MerchantDetail(
                merchant_id=merchant.id,
                name=merchant.name,
                business_category=merchant.business_category,
                mock_site_key=merchant.mock_site_key,
                registered_url=merchant.registered_url,
                status=merchant.status.value,
                risk_level=latest.risk_level.value if latest else None,
                cosine_distance=latest.cosine_distance if latest else None,
                cosine_variance_pct=latest.cosine_variance_pct if latest else None,
                last_scanned_at=latest.scanned_at.isoformat() if latest else None,
                baseline_screenshot_path=merchant.baseline_screenshot_path,
                baseline_text_snippet=merchant.baseline_text_snippet,
                baseline_ingested_at=merchant.baseline_ingested_at.isoformat() if merchant.baseline_ingested_at else None,
                onboarded_at=merchant.onboarded_at.isoformat(),
                scan_history=scan_history_dicts,
            )
    except MerchantNotFoundError:
        raise HTTPException(status_code=404, detail=f"Merchant '{merchant_id}' not found.")


@router.post(
    "/{merchant_id}/suspend",
    response_model=SuspendResponse,
    summary="Suspend a merchant's Razorpay account",
)
async def suspend_merchant(
    merchant_id: Annotated[str, Path(description="Merchant UUID")],
    agent: AgentDep,
) -> SuspendResponse:
    """
    Mock-suspend a merchant, resolving all their open alerts.
    Simulates the real Razorpay action of freezing payment gateway access.
    """
    try:
        result = await agent.suspend_merchant(merchant_id)
        return SuspendResponse(
            merchant_id=result["merchant_id"],
            new_status=result["new_status"],
            message=result["message"],
        )
    except MerchantNotFoundError:
        raise HTTPException(status_code=404, detail=f"Merchant '{merchant_id}' not found.")
    except Exception as exc:
        logger.error("Suspension failed", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get(
    "/{merchant_id}/status",
    summary="Quick risk status check for a merchant",
)
async def get_merchant_status(
    merchant_id: Annotated[str, Path(description="Merchant UUID")],
) -> dict[str, Any]:
    """Lightweight endpoint returning only the current risk level and account status."""
    try:
        async with get_async_session() as session:
            merchant_repo = MerchantRepository(session)
            scan_repo = ScanResultRepository(session)
            merchant = await merchant_repo.get_by_id(merchant_id)
            latest = await scan_repo.get_latest_for_merchant(merchant_id)
            return {
                "merchant_id": merchant_id,
                "account_status": merchant.status.value,
                "risk_level": latest.risk_level.value if latest else "UNSCANNED",
                "last_scanned_at": latest.scanned_at.isoformat() if latest else None,
            }
    except MerchantNotFoundError:
        raise HTTPException(status_code=404, detail=f"Merchant '{merchant_id}' not found.")
