"""
database/repository.py
-----------------------
Repository pattern for all database CRUD operations.

The Repository pattern decouples the application's business logic from the
underlying SQLAlchemy session management. Each repository class is focused
on a single model and accepts an `AsyncSession` injected at construction
time, making it trivially testable with mocked sessions.

Why not use SQLAlchemy directly in route handlers?
--------------------------------------------------
Mixing session management, query logic, and HTTP response logic in the same
function makes the code hard to test and reason about. Repositories give us
a clean, typed interface that can be swapped for a test double with minimal
friction.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Sequence

from sqlalchemy import desc, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from database.models import Alert, AlertSeverity, Merchant, MerchantStatus, RiskLevel, ScanResult
from utils.exceptions import DatabaseError, MerchantNotFoundError
from utils.logging_config import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# MerchantRepository
# ---------------------------------------------------------------------------


class MerchantRepository:
    """Data access layer for the `merchants` table."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(
        self,
        *,
        name: str,
        business_category: str,
        registered_url: str,
        mock_site_key: str,
    ) -> Merchant:
        """
        Persist a new merchant record and return the populated ORM object.

        Raises
        ------
        DatabaseError
            If the insert fails (e.g., duplicate mock_site_key).
        """
        merchant = Merchant(
            name=name,
            business_category=business_category,
            registered_url=registered_url,
            mock_site_key=mock_site_key,
            status=MerchantStatus.ACTIVE,
        )
        self._session.add(merchant)
        try:
            await self._session.flush()  # Assigns PK without committing
            await self._session.refresh(merchant)
        except Exception as exc:
            raise DatabaseError(
                message=f"Failed to create merchant '{name}': {exc}",
                operation="insert",
                table="merchants",
            ) from exc

        logger.info("Merchant created", extra={"merchant_id": merchant.id, "name": name})
        return merchant

    async def get_by_id(self, merchant_id: str, *, load_relations: bool = False) -> Merchant:
        """
        Fetch a single merchant by primary key.

        Parameters
        ----------
        merchant_id : str
            UUID of the merchant.
        load_relations : bool
            If True, eagerly loads `scan_results` and `alerts` in the same query.

        Raises
        ------
        MerchantNotFoundError
            If no merchant with the given ID exists.
        """
        stmt = select(Merchant).where(Merchant.id == merchant_id)
        if load_relations:
            stmt = stmt.options(
                selectinload(Merchant.scan_results),
                selectinload(Merchant.alerts),
            )

        result = await self._session.execute(stmt)
        merchant = result.scalar_one_or_none()
        if merchant is None:
            raise MerchantNotFoundError(merchant_id=merchant_id, source="database")
        return merchant

    async def get_by_mock_site_key(self, mock_site_key: str) -> Merchant | None:
        """Return the merchant matching the given mock site key, or None."""
        stmt = select(Merchant).where(Merchant.mock_site_key == mock_site_key)
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def list_all(self, *, include_suspended: bool = True) -> Sequence[Merchant]:
        """
        Return all merchants, optionally filtering out suspended accounts.

        Each merchant is loaded with its most recent scan result for the
        dashboard overview (avoids N+1 queries).
        """
        stmt = (
            select(Merchant)
            .options(selectinload(Merchant.scan_results))
            .order_by(Merchant.onboarded_at)
        )
        if not include_suspended:
            stmt = stmt.where(Merchant.status != MerchantStatus.SUSPENDED)

        result = await self._session.execute(stmt)
        return result.scalars().all()

    async def update_baseline(
        self,
        merchant_id: str,
        *,
        screenshot_path: str,
        text_snippet: str,
        ingested_at: datetime,
    ) -> None:
        """Update baseline metadata after a successful onboarding scrape."""
        stmt = (
            update(Merchant)
            .where(Merchant.id == merchant_id)
            .values(
                baseline_screenshot_path=screenshot_path,
                baseline_text_snippet=text_snippet[:500],  # Store first 500 chars
                baseline_ingested_at=ingested_at,
            )
        )
        await self._session.execute(stmt)
        logger.debug("Merchant baseline updated", extra={"merchant_id": merchant_id})

    async def update_status(
        self,
        merchant_id: str,
        status: MerchantStatus,
        *,
        action_description: str | None = None,
    ) -> Merchant:
        """
        Change a merchant's account status and return the updated record.

        Raises
        ------
        MerchantNotFoundError
            If the merchant_id does not exist.
        """
        merchant = await self.get_by_id(merchant_id)
        old_status = merchant.status
        merchant.status = status

        logger.info(
            "Merchant status updated",
            extra={
                "merchant_id": merchant_id,
                "old_status": old_status.value,
                "new_status": status.value,
                "action": action_description,
            },
        )
        return merchant

    async def count_by_status(self) -> dict[str, int]:
        """Return a dict of {status_value: count} for the dashboard summary."""
        from sqlalchemy import func
        stmt = select(Merchant.status, func.count(Merchant.id)).group_by(Merchant.status)
        result = await self._session.execute(stmt)
        return {row[0].value: row[1] for row in result.all()}


# ---------------------------------------------------------------------------
# ScanResultRepository
# ---------------------------------------------------------------------------


class ScanResultRepository:
    """Data access layer for the `scan_results` table."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(
        self,
        *,
        merchant_id: str,
        current_url_scraped: str,
        current_screenshot_path: str | None,
        cosine_distance: float | None,
        drift_detected: bool,
        drift_threshold_used: float | None,
        vision_triggered: bool,
        vision_result_json: str | None,
        risk_level: RiskLevel,
        scanned_at: datetime | None = None,
    ) -> ScanResult:
        """
        Persist a new scan result and return the ORM object.

        Parameters
        ----------
        vision_result_json : str | None
            Should be a JSON-serialised VisionAnalysisResult dict.
            Pass None when vision analysis was not triggered.
        """
        scan = ScanResult(
            merchant_id=merchant_id,
            current_url_scraped=current_url_scraped,
            current_screenshot_path=current_screenshot_path,
            cosine_distance=cosine_distance,
            drift_detected=drift_detected,
            drift_threshold_used=drift_threshold_used,
            vision_triggered=vision_triggered,
            vision_result_json=vision_result_json,
            risk_level=risk_level,
            scanned_at=scanned_at or datetime.now(tz=timezone.utc).replace(tzinfo=None),
        )
        self._session.add(scan)
        try:
            await self._session.flush()
            await self._session.refresh(scan)
        except Exception as exc:
            raise DatabaseError(
                message=f"Failed to create scan result for merchant {merchant_id}: {exc}",
                operation="insert",
                table="scan_results",
            ) from exc

        logger.info(
            "Scan result created",
            extra={
                "scan_id": scan.id,
                "merchant_id": merchant_id,
                "risk_level": risk_level.value,
                "cosine_distance": cosine_distance,
            },
        )
        return scan

    async def get_latest_for_merchant(self, merchant_id: str) -> ScanResult | None:
        """Return the most recent scan result for a given merchant, or None."""
        stmt = (
            select(ScanResult)
            .where(ScanResult.merchant_id == merchant_id)
            .order_by(desc(ScanResult.scanned_at))
            .limit(1)
        )
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_history(
        self, merchant_id: str, *, limit: int = 20
    ) -> Sequence[ScanResult]:
        """Return the last `limit` scan results for a merchant, newest first."""
        stmt = (
            select(ScanResult)
            .where(ScanResult.merchant_id == merchant_id)
            .order_by(desc(ScanResult.scanned_at))
            .limit(limit)
        )
        result = await self._session.execute(stmt)
        return result.scalars().all()

    async def get_high_risk_results(self) -> Sequence[ScanResult]:
        """Return all RED-risk scan results (newest first) for the alerts panel."""
        stmt = (
            select(ScanResult)
            .where(ScanResult.risk_level == RiskLevel.RED)
            .order_by(desc(ScanResult.scanned_at))
        )
        result = await self._session.execute(stmt)
        return result.scalars().all()


# ---------------------------------------------------------------------------
# AlertRepository
# ---------------------------------------------------------------------------


class AlertRepository:
    """Data access layer for the `alerts` table."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(
        self,
        *,
        merchant_id: str,
        scan_result_id: str,
        alert_type: str,
        severity: AlertSeverity,
        detected_category: str | None,
        confidence_score: int | None,
    ) -> Alert:
        """Create a new compliance alert linked to a scan result."""
        alert = Alert(
            merchant_id=merchant_id,
            scan_result_id=scan_result_id,
            alert_type=alert_type,
            severity=severity,
            detected_category=detected_category,
            confidence_score=confidence_score,
        )
        self._session.add(alert)
        try:
            await self._session.flush()
            await self._session.refresh(alert)
        except Exception as exc:
            raise DatabaseError(
                message=f"Failed to create alert for merchant {merchant_id}: {exc}",
                operation="insert",
                table="alerts",
            ) from exc

        logger.warning(
            "Compliance alert created",
            extra={
                "alert_id": alert.id,
                "merchant_id": merchant_id,
                "alert_type": alert_type,
                "severity": severity.value,
                "confidence": confidence_score,
            },
        )
        return alert

    async def resolve(
        self,
        alert_id: str,
        *,
        action_taken: str,
    ) -> Alert | None:
        """
        Mark an alert as resolved with the action that was taken.

        Returns the updated Alert, or None if not found.
        """
        stmt = select(Alert).where(Alert.id == alert_id)
        result = await self._session.execute(stmt)
        alert = result.scalar_one_or_none()
        if alert is None:
            return None

        alert.resolved = True
        alert.action_taken = action_taken
        alert.resolved_at = datetime.now(tz=timezone.utc).replace(tzinfo=None)

        logger.info(
            "Alert resolved",
            extra={"alert_id": alert_id, "action": action_taken},
        )
        return alert

    async def list_unresolved(self) -> Sequence[Alert]:
        """Return all unresolved alerts, newest first."""
        stmt = (
            select(Alert)
            .where(Alert.resolved == False)  # noqa: E712
            .options(selectinload(Alert.merchant))
            .order_by(desc(Alert.created_at))
        )
        result = await self._session.execute(stmt)
        return result.scalars().all()

    async def get_for_merchant(self, merchant_id: str) -> Sequence[Alert]:
        """Return all alerts for a specific merchant, newest first."""
        stmt = (
            select(Alert)
            .where(Alert.merchant_id == merchant_id)
            .order_by(desc(Alert.created_at))
        )
        result = await self._session.execute(stmt)
        return result.scalars().all()
