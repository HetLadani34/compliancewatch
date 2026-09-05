"""
core_agent/compliance_agent.py
--------------------------------
The ComplianceAgent: central orchestrator of the entire monitoring pipeline.

This is the "brain" of ComplianceWatch. It wires together every component —
scraper, embeddings, vector store, drift detector, and vision analyzer — into
two clean, high-level workflows:

  1. onboard_merchant()  — Called once per merchant at signup time.
                           Captures the baseline (State A) and stores it.

  2. scan_merchant()     — Called periodically for every active merchant.
                           Captures the current state, compares it to the
                           baseline, and conditionally triggers vision analysis.

  3. run_full_scan()     — Scans ALL active merchants and returns a summary.

Pipeline (onboard_merchant)
----------------------------
  Mock Site (State A)
    → Playwright Scraper   [get text + screenshot]
    → Gemini Embedding     [text → 768-dim vector]
    → ChromaDB             [store baseline vector]
    → SQLite               [save Merchant record + baseline metadata]

Pipeline (scan_merchant)
------------------------
  Mock Site (Current State)
    → Playwright Scraper   [get text + screenshot]
    → Gemini Embedding     [text → 768-dim vector]
    → ChromaDB             [retrieve baseline vector]
    → DriftDetector        [cosine distance + threshold verdict]
    → [if drift detected]
        → VisionAnalyzer   [Gemini Flash: screenshot + text → JSON verdict]
    → SQLite               [save ScanResult + optional Alert]
    → Return ScanReport

Error Handling Strategy
-----------------------
* Per-merchant errors during run_full_scan() are caught, logged, and added
  to the summary's `errors` list. A single failing merchant never aborts the
  entire scan run.
* All component errors use our custom exception hierarchy for structured logging.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from core_agent.drift_detector import DriftDetector
from core_agent.embedding_service import EmbeddingService
from core_agent.schemas import (
    DriftReport,
    FullScanSummary,
    MerchantOnboardRequest,
    OnboardingResult,
    ScanReport,
    VisionAnalysisResult,
)
from core_agent.scraper import WebsiteScraper, save_screenshot
from core_agent.vision_analyzer import VisionAnalyzer
from database.engine import get_async_session
from database.models import Alert, AlertSeverity, Merchant, MerchantStatus, RiskLevel, ScanResult
from database.repository import AlertRepository, MerchantRepository, ScanResultRepository
from utils.exceptions import (
    ComplianceWatchError,
    MerchantNotFoundError,
    ScrapingError,
    VectorStoreError,
)
from utils.logging_config import get_logger
from vector_store.chroma_client import get_vector_store

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Action Recommendation Templates
# ---------------------------------------------------------------------------

_ACTION_TEMPLATES: dict[str, str] = {
    "GREEN": "No action required. Merchant website content is compliant.",
    "YELLOW": (
        "Manual review recommended. Drift detected but AI confidence is low. "
        "A human reviewer should examine the screenshots."
    ),
    "RED": (
        "Immediate action required. AI has detected a high-confidence policy violation. "
        "Recommend suspending payment gateway access pending merchant response."
    ),
}


# ---------------------------------------------------------------------------
# ComplianceAgent
# ---------------------------------------------------------------------------


class ComplianceAgent:
    """
    Orchestrates the full ComplianceWatch pipeline.

    This class is designed to be instantiated once and reused across requests.
    All component dependencies (EmbeddingService, VisionAnalyzer, etc.) are
    created lazily on first use and shared across calls.

    Usage
    -----
    >>> agent = ComplianceAgent()
    >>> result = await agent.onboard_merchant(MerchantOnboardRequest(...))
    >>> report = await agent.scan_merchant(merchant_id="abc-123")
    >>> summary = await agent.run_full_scan()
    """

    def __init__(self) -> None:
        from config.settings import get_settings
        self._settings = get_settings()

        # Lazily initialised components (created on first use)
        self._embedding_service: EmbeddingService | None = None
        self._vision_analyzer: VisionAnalyzer | None = None
        self._drift_detector: DriftDetector | None = None

        logger.info("ComplianceAgent initialised.")

    # ------------------------------------------------------------------
    # Lazy Component Accessors
    # ------------------------------------------------------------------

    @property
    def embedding_service(self) -> EmbeddingService:
        if self._embedding_service is None:
            self._embedding_service = EmbeddingService()
        return self._embedding_service

    @property
    def vision_analyzer(self) -> VisionAnalyzer:
        if self._vision_analyzer is None:
            self._vision_analyzer = VisionAnalyzer()
        return self._vision_analyzer

    @property
    def drift_detector(self) -> DriftDetector:
        if self._drift_detector is None:
            self._drift_detector = DriftDetector()
        return self._drift_detector

    # ------------------------------------------------------------------
    # Public API: Onboarding
    # ------------------------------------------------------------------

    async def onboard_merchant(
        self, request: MerchantOnboardRequest
    ) -> OnboardingResult:
        """
        Onboard a new merchant by capturing and storing their baseline website state.

        Steps
        -----
        1. Check if merchant with this mock_site_key already exists (idempotent).
        2. Create DB record for the merchant.
        3. Scrape the baseline URL (State A: /merchant/{key}/clean).
        4. Generate embedding from the baseline text.
        5. Store embedding in ChromaDB.
        6. Save baseline screenshot to disk.
        7. Update DB record with baseline metadata.

        Parameters
        ----------
        request : MerchantOnboardRequest
            Contains merchant name, category, and mock_site_key.

        Returns
        -------
        OnboardingResult
            Confirmation payload with merchant_id and baseline metadata.
        """
        mock_key = request.mock_site_key
        baseline_url = f"{self._settings.mock_server_base_url}/merchant/{mock_key}/clean"

        logger.info(
            "Starting merchant onboarding",
            extra={"mock_site_key": mock_key, "name": request.name, "baseline_url": baseline_url},
        )

        async with get_async_session() as session:
            merchant_repo = MerchantRepository(session)

            # --- Idempotency check ---
            existing = await merchant_repo.get_by_mock_site_key(mock_key)
            if existing:
                logger.info(
                    "Merchant already onboarded, re-running baseline ingestion",
                    extra={"merchant_id": existing.id, "mock_site_key": mock_key},
                )
                merchant = existing
            else:
                # --- Create merchant DB record ---
                merchant = await merchant_repo.create(
                    name=request.name,
                    business_category=request.business_category,
                    registered_url=baseline_url,
                    mock_site_key=mock_key,
                )

        merchant_id = merchant.id

        # --- Scrape baseline (outside session to avoid holding it open during I/O) ---
        async with WebsiteScraper() as scraper:
            baseline_content = await scraper.scrape(baseline_url, merchant_id=merchant_id)

        # --- Generate embedding ---
        embedding_result = await self.embedding_service.embed_async(baseline_content.text)

        # --- Store in ChromaDB ---
        vector_store = get_vector_store()
        await vector_store.store_baseline_async(
            merchant_id=merchant_id,
            embedding_vector=embedding_result.vector,
            merchant_name=request.name,
            mock_site_key=mock_key,
            text_snippet=baseline_content.text[:200],
        )

        # --- Save baseline screenshot ---
        screenshot_path = save_screenshot(
            screenshot_bytes=baseline_content.screenshot_bytes,
            directory=self._settings.baselines_screenshot_dir,
            filename=f"{mock_key}_baseline",
        )

        # --- Update DB with baseline metadata ---
        ingested_at = datetime.now(tz=timezone.utc).replace(tzinfo=None)
        async with get_async_session() as session:
            merchant_repo = MerchantRepository(session)
            await merchant_repo.update_baseline(
                merchant_id=merchant_id,
                screenshot_path=str(screenshot_path),
                text_snippet=baseline_content.text[:500],
                ingested_at=ingested_at,
            )

        logger.info(
            "Merchant onboarding complete",
            extra={
                "merchant_id": merchant_id,
                "mock_site_key": mock_key,
                "baseline_screenshot": str(screenshot_path),
                "embedding_dim": embedding_result.dimension,
            },
        )

        return OnboardingResult(
            merchant_id=merchant_id,
            merchant_name=request.name,
            mock_site_key=mock_key,
            baseline_url=baseline_url,
            baseline_screenshot_path=str(screenshot_path),
            baseline_text_snippet=baseline_content.text[:300],
            onboarded_at=ingested_at,
        )

    # ------------------------------------------------------------------
    # Public API: Individual Merchant Scan
    # ------------------------------------------------------------------

    async def scan_merchant(self, merchant_id: str) -> ScanReport:
        """
        Run a full compliance scan on a single merchant.

        Steps
        -----
        1. Load merchant record from DB.
        2. Scrape the merchant's current active URL.
        3. Generate current embedding.
        4. Load baseline embedding from ChromaDB.
        5. Compute cosine distance → DriftReport.
        6. If drift detected: run VisionAnalyzer.
        7. Determine final risk_level.
        8. Persist ScanResult (and optional Alert) to DB.
        9. Return ScanReport.

        Parameters
        ----------
        merchant_id : str
            UUID of the merchant to scan.

        Returns
        -------
        ScanReport
            Complete scan report including drift analysis and optional vision result.

        Raises
        ------
        MerchantNotFoundError
            If the merchant_id doesn't exist in the database.
        VectorStoreError
            If no baseline embedding exists (merchant not onboarded).
        ScrapingError
            If the current website cannot be scraped.
        """
        # --- Load merchant ---
        async with get_async_session() as session:
            merchant_repo = MerchantRepository(session)
            merchant = await merchant_repo.get_by_id(merchant_id)
            merchant_name = merchant.name
            mock_site_key = merchant.mock_site_key
            baseline_screenshot_path = merchant.baseline_screenshot_path

        if merchant.status == MerchantStatus.SUSPENDED:
            logger.info(
                "Skipping scan for suspended merchant",
                extra={"merchant_id": merchant_id},
            )
            # Return the last known state without scanning
            return await self._build_skipped_report(merchant)

        current_url = f"{self._settings.mock_server_base_url}/merchant/{mock_site_key}"
        baseline_url = f"{self._settings.mock_server_base_url}/merchant/{mock_site_key}/clean"
        scan_id = str(uuid.uuid4())

        logger.info(
            "Starting merchant scan",
            extra={"merchant_id": merchant_id, "mock_site_key": mock_site_key, "scan_id": scan_id},
        )

        # --- Scrape current state ---
        async with WebsiteScraper() as scraper:
            current_content = await scraper.scrape(current_url, merchant_id=merchant_id)

        # --- Save current screenshot ---
        current_screenshot_path = save_screenshot(
            screenshot_bytes=current_content.screenshot_bytes,
            directory=self._settings.current_screenshot_dir,
            filename=f"{mock_site_key}_current_{scan_id[:8]}",
        )

        # --- Generate current embedding ---
        current_embedding = await self.embedding_service.embed_async(current_content.text)

        # --- Retrieve baseline embedding from ChromaDB ---
        vector_store = get_vector_store()
        baseline_vector = await vector_store.get_baseline_vector_async(merchant_id)

        # --- Store current scan embedding for trend analysis ---
        await vector_store.store_current_scan_async(
            merchant_id=merchant_id,
            scan_id=scan_id,
            embedding_vector=current_embedding.vector,
            text_snippet=current_content.text[:200],
        )

        # --- Compute drift ---
        drift_report = self.drift_detector.evaluate(
            merchant_id=merchant_id,
            baseline_vector=baseline_vector,
            current_vector=current_embedding.vector,
        )

        # --- Conditional vision analysis ---
        vision_result: VisionAnalysisResult | None = None
        if drift_report.drift_detected:
            logger.warning(
                "Drift detected — triggering vision analysis",
                extra={
                    "merchant_id": merchant_id,
                    "cosine_distance": drift_report.cosine_distance,
                    "variance_pct": drift_report.variance_pct,
                },
            )
            vision_result = await self.vision_analyzer.analyze(
                screenshot_bytes=current_content.screenshot_bytes,
                page_text=current_content.text,
                page_html=current_content.html,
                merchant_url=current_url,
                drift_pct=drift_report.variance_pct,
                drift_threshold=drift_report.threshold_used,
                merchant_id=merchant_id,
            )

        # --- Determine final risk level ---
        risk_level = _compute_risk_level(drift_report, vision_result)

        # --- Persist to database ---
        await self._persist_scan_result(
            merchant_id=merchant_id,
            scan_id=scan_id,
            current_url=current_url,
            current_screenshot_path=str(current_screenshot_path),
            drift_report=drift_report,
            vision_result=vision_result,
            risk_level=risk_level,
        )

        scan_report = ScanReport(
            merchant_id=merchant_id,
            merchant_name=merchant_name,
            mock_site_key=mock_site_key,
            baseline_url=baseline_url,
            current_url=current_url,
            baseline_screenshot_path=baseline_screenshot_path,
            current_screenshot_path=str(current_screenshot_path),
            drift_report=drift_report,
            vision_analysis=vision_result,
            risk_level=risk_level,
            action_recommended=_ACTION_TEMPLATES[risk_level],
        )

        logger.info(
            "Merchant scan completed",
            extra={
                "merchant_id": merchant_id,
                "risk_level": risk_level,
                "drift_detected": drift_report.drift_detected,
                "vision_triggered": vision_result is not None,
            },
        )
        return scan_report

    # ------------------------------------------------------------------
    # Public API: Full Scan Run
    # ------------------------------------------------------------------

    async def run_full_scan(self) -> FullScanSummary:
        """
        Scan ALL active merchants and return an aggregate summary.

        Per-merchant errors are caught and recorded in the summary's `errors`
        list — a single failing merchant never aborts the entire run.

        Returns
        -------
        FullScanSummary
            Aggregate counts (green/yellow/red) + all individual ScanReports.
        """
        scan_run_id = str(uuid.uuid4())
        started_at = datetime.now(tz=timezone.utc).replace(tzinfo=None)

        logger.info("Starting full compliance scan run", extra={"scan_run_id": scan_run_id})

        # --- Load all active merchants ---
        async with get_async_session() as session:
            merchant_repo = MerchantRepository(session)
            all_merchants = await merchant_repo.list_all(include_suspended=False)

        merchant_ids = [m.id for m in all_merchants]
        logger.info(
            "Full scan: merchants to process",
            extra={"count": len(merchant_ids), "scan_run_id": scan_run_id},
        )

        scan_reports: list[ScanReport] = []
        errors: list[str] = []

        # We scan merchants sequentially (not concurrently) to:
        # 1. Stay within Gemini free-tier rate limits
        # 2. Avoid overwhelming the local mock server
        for merchant_id in merchant_ids:
            try:
                report = await self.scan_merchant(merchant_id)
                scan_reports.append(report)
            except ComplianceWatchError as exc:
                error_msg = f"merchant_id={merchant_id}: {exc.message}"
                logger.error(
                    "Merchant scan failed during full run",
                    exc_info=True,
                    extra={"merchant_id": merchant_id, "error": exc.message},
                )
                errors.append(error_msg)
            except Exception as exc:
                error_msg = f"merchant_id={merchant_id}: Unexpected error — {exc}"
                logger.error(
                    "Unexpected error during merchant scan",
                    exc_info=True,
                    extra={"merchant_id": merchant_id},
                )
                errors.append(error_msg)

        completed_at = datetime.now(tz=timezone.utc).replace(tzinfo=None)

        summary = FullScanSummary(
            scan_run_id=scan_run_id,
            started_at=started_at,
            completed_at=completed_at,
            total_merchants_scanned=len(scan_reports),
            green_count=sum(1 for r in scan_reports if r.risk_level == "GREEN"),
            yellow_count=sum(1 for r in scan_reports if r.risk_level == "YELLOW"),
            red_count=sum(1 for r in scan_reports if r.risk_level == "RED"),
            scan_reports=scan_reports,
            errors=errors,
        )

        logger.info(
            "Full scan run complete",
            extra={
                "scan_run_id": scan_run_id,
                "duration_seconds": summary.duration_seconds,
                "total": summary.total_merchants_scanned,
                "green": summary.green_count,
                "yellow": summary.yellow_count,
                "red": summary.red_count,
                "errors": len(errors),
            },
        )
        return summary

    # ------------------------------------------------------------------
    # Account Action: Suspend Merchant
    # ------------------------------------------------------------------

    async def suspend_merchant(self, merchant_id: str) -> dict[str, str]:
        """
        Mark a merchant as SUSPENDED in the database and resolve their alerts.

        This simulates the real Razorpay action of freezing a merchant's
        payment gateway access pending investigation.
        """
        async with get_async_session() as session:
            merchant_repo = MerchantRepository(session)
            alert_repo = AlertRepository(session)

            merchant = await merchant_repo.update_status(
                merchant_id=merchant_id,
                status=MerchantStatus.SUSPENDED,
                action_description="ACCOUNT_SUSPENDED_COMPLIANCE_VIOLATION",
            )

            # Resolve all open alerts for this merchant
            unresolved = await alert_repo.list_unresolved()
            resolved_count = 0
            for alert in unresolved:
                if alert.merchant_id == merchant_id:
                    await alert_repo.resolve(
                        alert.id,
                        action_taken="ACCOUNT_SUSPENDED",
                    )
                    resolved_count += 1

        logger.warning(
            "Merchant account suspended",
            extra={
                "merchant_id": merchant_id,
                "merchant_name": merchant.name,
                "alerts_resolved": resolved_count,
            },
        )
        return {
            "merchant_id": merchant_id,
            "new_status": "SUSPENDED",
            "alerts_resolved": str(resolved_count),
            "message": f"Account for '{merchant.name}' has been suspended. {resolved_count} alert(s) resolved.",
        }

    # ------------------------------------------------------------------
    # Private Helpers
    # ------------------------------------------------------------------

    async def _persist_scan_result(
        self,
        merchant_id: str,
        scan_id: str,
        current_url: str,
        current_screenshot_path: str,
        drift_report: DriftReport,
        vision_result: VisionAnalysisResult | None,
        risk_level: str,
    ) -> None:
        """Persist ScanResult and optionally an Alert to the database."""
        import json as json_module

        vision_json: str | None = None
        if vision_result is not None:
            vision_json = vision_result.model_dump_json()

        async with get_async_session() as session:
            scan_repo = ScanResultRepository(session)
            alert_repo = AlertRepository(session)

            scan_result = await scan_repo.create(
                merchant_id=merchant_id,
                current_url_scraped=current_url,
                current_screenshot_path=current_screenshot_path,
                cosine_distance=drift_report.cosine_distance,
                drift_detected=drift_report.drift_detected,
                drift_threshold_used=drift_report.threshold_used,
                vision_triggered=vision_result is not None,
                vision_result_json=vision_json,
                risk_level=RiskLevel(risk_level),
            )

            # Create an alert if this is a RED or YELLOW result with a violation
            if vision_result is not None and vision_result.is_policy_violation:
                severity = (
                    AlertSeverity.CRITICAL
                    if risk_level == "RED"
                    else AlertSeverity.MEDIUM
                )
                await alert_repo.create(
                    merchant_id=merchant_id,
                    scan_result_id=scan_result.id,
                    alert_type=vision_result.detected_banned_category or "UNKNOWN_VIOLATION",
                    severity=severity,
                    detected_category=vision_result.detected_banned_category,
                    confidence_score=vision_result.confidence_score,
                )

    async def _build_skipped_report(self, merchant: Merchant) -> ScanReport:
        """Build a placeholder ScanReport for suspended merchants (no actual scan)."""
        from core_agent.schemas import DriftReport
        mock_drift = DriftReport(
            merchant_id=merchant.id,
            cosine_distance=0.0,
            drift_detected=False,
            threshold_used=self._settings.cosine_drift_threshold,
        )
        return ScanReport(
            merchant_id=merchant.id,
            merchant_name=merchant.name,
            mock_site_key=merchant.mock_site_key,
            baseline_url=merchant.registered_url,
            current_url=merchant.registered_url,
            baseline_screenshot_path=merchant.baseline_screenshot_path,
            drift_report=mock_drift,
            vision_analysis=None,
            risk_level="RED",  # Suspended merchants stay RED
            action_recommended="Account is currently SUSPENDED. No scan performed.",
        )


# ---------------------------------------------------------------------------
# Risk Level Computation Logic
# ---------------------------------------------------------------------------


def _compute_risk_level(
    drift_report: DriftReport,
    vision_result: VisionAnalysisResult | None,
) -> str:
    """
    Determine the final risk level for a scan using both drift and vision data.

    Decision Table
    --------------
    No drift detected                       → GREEN
    Drift detected, vision not triggered    → YELLOW  (shouldn't happen, but safe default)
    Drift detected, no violation found      → GREEN   (false positive drift)
    Drift detected, violation, conf < 60   → YELLOW  (uncertain, needs review)
    Drift detected, violation, conf >= 60  → RED     (confirmed violation)
    """
    if not drift_report.drift_detected:
        return "GREEN"

    if vision_result is None:
        # Drift detected but vision wasn't triggered (shouldn't occur in normal flow)
        return "YELLOW"

    if not vision_result.is_policy_violation:
        # Drift was detected but Gemini cleared the merchant
        return "GREEN"

    # Vision confirmed a violation
    return vision_result.risk_label
