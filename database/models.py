"""
database/models.py
-------------------
SQLAlchemy ORM models (table definitions) for ComplianceWatch.

Tables
------
merchants       — Registered merchants and their baseline metadata
scan_results    — Outcome of each monitoring scan per merchant
alerts          — Active policy violation alerts and their resolution status

Design notes
------------
* UUIDs are used as primary keys (stored as strings in SQLite) to avoid
  exposing sequential integer IDs in API responses.
* Timestamps are stored as UTC-naive datetimes (the application layer is
  responsible for ensuring UTC inputs).
* Enum columns use Python enum types for type safety; their string values
  are stored in the DB for human readability.
* `__repr__` methods are provided on every model to aid debugging.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


# ---------------------------------------------------------------------------
# Shared Base & Helper
# ---------------------------------------------------------------------------


class Base(DeclarativeBase):
    """Shared declarative base for all ORM models."""


def _new_uuid() -> str:
    """Generate a new UUID4 string. Used as the default for PK columns."""
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class MerchantStatus(str, enum.Enum):
    """Lifecycle status of a merchant's Razorpay account."""
    ACTIVE = "ACTIVE"
    UNDER_REVIEW = "UNDER_REVIEW"
    SUSPENDED = "SUSPENDED"


class RiskLevel(str, enum.Enum):
    """Risk classification of a completed scan."""
    GREEN = "GREEN"       # No semantic drift detected
    YELLOW = "YELLOW"     # Drift detected, vision analysis inconclusive
    RED = "RED"           # Vision analysis confirmed policy violation


class AlertSeverity(str, enum.Enum):
    """Severity tier for a generated alert."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


# ---------------------------------------------------------------------------
# Merchants Table
# ---------------------------------------------------------------------------


class Merchant(Base):
    """
    Represents a merchant onboarded to the Razorpay platform.

    Each merchant has a single "baseline" — the approved website state
    captured at the time of onboarding. All subsequent scans are compared
    against this baseline.
    """

    __tablename__ = "merchants"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=_new_uuid, index=True
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    business_category: Mapped[str] = mapped_column(String(100), nullable=False)
    registered_url: Mapped[str] = mapped_column(String(2048), nullable=False)

    # The key used to address this merchant on the mock server
    # e.g. "diya_store", "organic_tea", "handloom_crafts"
    mock_site_key: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)

    # Account lifecycle
    status: Mapped[MerchantStatus] = mapped_column(
        Enum(MerchantStatus), nullable=False, default=MerchantStatus.ACTIVE
    )

    # Baseline metadata (populated during onboarding)
    baseline_screenshot_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    baseline_text_snippet: Mapped[str | None] = mapped_column(
        Text, nullable=True,
        comment="First 500 chars of baseline text, stored for display purposes."
    )
    baseline_ingested_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    # Timestamps
    onboarded_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    scan_results: Mapped[list[ScanResult]] = relationship(
        "ScanResult", back_populates="merchant", cascade="all, delete-orphan",
        order_by="desc(ScanResult.scanned_at)"
    )
    alerts: Mapped[list[Alert]] = relationship(
        "Alert", back_populates="merchant", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return (
            f"Merchant(id={self.id!r}, name={self.name!r}, "
            f"status={self.status.value!r}, url={self.registered_url!r})"
        )

    @property
    def latest_risk_level(self) -> RiskLevel | None:
        """Return the risk level of the most recent scan, or None if unscanned."""
        if self.scan_results:
            return self.scan_results[0].risk_level
        return None


# ---------------------------------------------------------------------------
# Scan Results Table
# ---------------------------------------------------------------------------


class ScanResult(Base):
    """
    Records the outcome of a single monitoring scan for one merchant.

    The pipeline produces one ScanResult per merchant per scan run.
    When drift is detected and vision analysis is triggered, the Gemini
    response is stored verbatim as a JSON string in `vision_result_json`.
    """

    __tablename__ = "scan_results"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=_new_uuid, index=True
    )
    merchant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("merchants.id", ondelete="CASCADE"), nullable=False, index=True
    )

    scanned_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now(), index=True
    )

    # Snapshot metadata
    current_screenshot_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    current_url_scraped: Mapped[str] = mapped_column(String(2048), nullable=False)

    # Drift analysis
    cosine_distance: Mapped[float | None] = mapped_column(Float, nullable=True)
    drift_detected: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    drift_threshold_used: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Vision analysis (only populated if drift_detected == True)
    vision_triggered: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    vision_result_json: Mapped[str | None] = mapped_column(
        Text, nullable=True,
        comment="Raw JSON string of the VisionAnalysisResult Pydantic model."
    )

    # Final verdict
    risk_level: Mapped[RiskLevel] = mapped_column(
        Enum(RiskLevel), nullable=False, default=RiskLevel.GREEN
    )

    # Relationship
    merchant: Mapped[Merchant] = relationship("Merchant", back_populates="scan_results")
    alert: Mapped[Alert | None] = relationship(
        "Alert", back_populates="scan_result", uselist=False, cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return (
            f"ScanResult(id={self.id!r}, merchant_id={self.merchant_id!r}, "
            f"risk={self.risk_level.value!r}, distance={self.cosine_distance})"
        )

    @property
    def cosine_variance_pct(self) -> float | None:
        """Cosine distance expressed as a percentage (0–100) for display."""
        if self.cosine_distance is not None:
            return round(self.cosine_distance * 100, 2)
        return None


# ---------------------------------------------------------------------------
# Alerts Table
# ---------------------------------------------------------------------------


class Alert(Base):
    """
    An active policy violation alert generated from a scan result.

    One ScanResult produces at most one Alert (enforced by the unique
    constraint on scan_result_id). Alerts are resolved when a merchant is
    suspended or cleared after manual review.
    """

    __tablename__ = "alerts"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=_new_uuid, index=True
    )
    merchant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("merchants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    scan_result_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("scan_results.id", ondelete="CASCADE"), nullable=False, unique=True
    )

    alert_type: Mapped[str] = mapped_column(
        String(100), nullable=False,
        comment="Category of violation, e.g. 'VAPE_CONTENT', 'CRYPTO_GAMBLING'."
    )
    severity: Mapped[AlertSeverity] = mapped_column(
        Enum(AlertSeverity), nullable=False, default=AlertSeverity.HIGH
    )
    detected_category: Mapped[str | None] = mapped_column(String(200), nullable=True)
    confidence_score: Mapped[int | None] = mapped_column(
        Integer, nullable=True,
        comment="Gemini vision confidence score 0–100."
    )

    # Resolution
    resolved: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    action_taken: Mapped[str | None] = mapped_column(
        String(200), nullable=True,
        comment="Human-readable description of the action taken, e.g. 'ACCOUNT_SUSPENDED'."
    )
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )

    # Relationships
    merchant: Mapped[Merchant] = relationship("Merchant", back_populates="alerts")
    scan_result: Mapped[ScanResult] = relationship("ScanResult", back_populates="alert")

    def __repr__(self) -> str:
        return (
            f"Alert(id={self.id!r}, merchant_id={self.merchant_id!r}, "
            f"type={self.alert_type!r}, severity={self.severity.value!r}, "
            f"resolved={self.resolved})"
        )
