"""
api/routes/scanning.py
-----------------------
Endpoints to trigger compliance scans — both manual and per-merchant.

Routes
------
POST /api/scan/trigger              — Trigger a full scan of ALL active merchants
POST /api/scan/merchant/{id}        — Scan a single specific merchant
GET  /api/scan/results/{merchant_id} — Retrieve scan history for a merchant
GET  /api/scan/latest               — Get latest scan result for every merchant
"""

from __future__ import annotations

import json
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Path, status
from pydantic import BaseModel

from core_agent.compliance_agent import ComplianceAgent
from core_agent.schemas import FullScanSummary, ScanReport
from database.engine import get_async_session
from database.repository import ScanResultRepository
from utils.exceptions import MerchantNotFoundError, VectorStoreError
from utils.logging_config import get_logger

logger = get_logger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# Dependency
# ---------------------------------------------------------------------------

def _get_agent() -> ComplianceAgent:
    from api.main import get_compliance_agent
    return get_compliance_agent()

AgentDep = Annotated[ComplianceAgent, Depends(_get_agent)]


# ---------------------------------------------------------------------------
# Response Models
# ---------------------------------------------------------------------------

class ScanTriggerResponse(BaseModel):
    scan_run_id: str
    total_scanned: int
    green_count: int
    yellow_count: int
    red_count: int
    error_count: int
    duration_seconds: float
    flagged_merchants: list[str]


class ScanResultResponse(BaseModel):
    scan_id: str
    merchant_id: str
    scanned_at: str
    risk_level: str
    cosine_distance: float | None
    cosine_variance_pct: float | None
    drift_detected: bool
    vision_triggered: bool
    vision_result: dict[str, Any] | None
    current_screenshot_path: str | None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post(
    "/trigger",
    response_model=ScanTriggerResponse,
    summary="Trigger a full compliance scan of all active merchants",
)
async def trigger_full_scan(agent: AgentDep) -> ScanTriggerResponse:
    """
    Manually trigger a full compliance scan across all active merchants.

    This is the same logic that the APScheduler background job runs on its
    configured interval. Use this button in the dashboard for on-demand scans.
    """
    logger.info("Manual full scan triggered via API.")
    try:
        summary: FullScanSummary = await agent.run_full_scan()
        return ScanTriggerResponse(
            scan_run_id=summary.scan_run_id,
            total_scanned=summary.total_merchants_scanned,
            green_count=summary.green_count,
            yellow_count=summary.yellow_count,
            red_count=summary.red_count,
            error_count=len(summary.errors),
            duration_seconds=summary.duration_seconds,
            flagged_merchants=[r.merchant_id for r in summary.flagged_merchants],
        )
    except Exception as exc:
        logger.error("Full scan trigger failed.", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Scan failed: {str(exc)}")


@router.post(
    "/merchant/{merchant_id}",
    summary="Scan a single merchant",
)
async def scan_single_merchant(
    merchant_id: Annotated[str, Path(description="Merchant UUID to scan")],
    agent: AgentDep,
) -> dict[str, Any]:
    """
    Run a compliance scan on one specific merchant and return the full ScanReport.
    """
    logger.info("Single merchant scan triggered via API.", extra={"merchant_id": merchant_id})
    try:
        report: ScanReport = await agent.scan_merchant(merchant_id)
        # Serialise via model_dump for clean JSON (handles nested Pydantic models)
        report_dict = report.model_dump(mode="json")
        return report_dict
    except MerchantNotFoundError:
        raise HTTPException(status_code=404, detail=f"Merchant '{merchant_id}' not found.")
    except VectorStoreError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Vector store error (has merchant been onboarded?): {exc.message}",
        )
    except Exception as exc:
        logger.error("Single merchant scan failed.", exc_info=True, extra={"merchant_id": merchant_id})
        raise HTTPException(status_code=500, detail=str(exc))


@router.get(
    "/results/{merchant_id}",
    response_model=list[ScanResultResponse],
    summary="Get scan history for a merchant",
)
async def get_scan_results(
    merchant_id: Annotated[str, Path(description="Merchant UUID")],
    limit: int = 20,
) -> list[ScanResultResponse]:
    """Return the last N scan results for a merchant, newest first."""
    async with get_async_session() as session:
        scan_repo = ScanResultRepository(session)
        results = await scan_repo.get_history(merchant_id, limit=limit)

    output: list[ScanResultResponse] = []
    for r in results:
        vision_data: dict[str, Any] | None = None
        if r.vision_result_json:
            try:
                vision_data = json.loads(r.vision_result_json)
            except json.JSONDecodeError:
                vision_data = {"error": "Unparseable vision result"}

        output.append(ScanResultResponse(
            scan_id=r.id,
            merchant_id=r.merchant_id,
            scanned_at=r.scanned_at.isoformat(),
            risk_level=r.risk_level.value,
            cosine_distance=r.cosine_distance,
            cosine_variance_pct=r.cosine_variance_pct,
            drift_detected=r.drift_detected,
            vision_triggered=r.vision_triggered,
            vision_result=vision_data,
            current_screenshot_path=r.current_screenshot_path,
        ))
    return output


@router.get(
    "/latest",
    summary="Get latest scan result for every merchant",
)
async def get_all_latest_scans() -> list[dict[str, Any]]:
    """
    Lightweight endpoint returning just the latest scan result for every
    merchant. Used by the Streamlit dashboard for its auto-refresh loop.
    """
    async with get_async_session() as session:
        from database.models import Merchant
        from sqlalchemy import select
        result = await session.execute(select(Merchant))
        merchants = result.scalars().all()
        scan_repo = ScanResultRepository(session)

        output = []
        for m in merchants:
            latest = await scan_repo.get_latest_for_merchant(m.id)
            output.append({
                "merchant_id": m.id,
                "name": m.name,
                "mock_site_key": m.mock_site_key,
                "account_status": m.status.value,
                "risk_level": latest.risk_level.value if latest else "UNSCANNED",
                "cosine_distance": latest.cosine_distance if latest else None,
                "cosine_variance_pct": latest.cosine_variance_pct if latest else None,
                "last_scanned_at": latest.scanned_at.isoformat() if latest else None,
                "vision_triggered": latest.vision_triggered if latest else False,
                "drift_detected": latest.drift_detected if latest else False,
            })
    return output
