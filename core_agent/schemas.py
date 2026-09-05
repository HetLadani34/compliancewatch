"""
core_agent/schemas.py
----------------------
Pydantic models that act as the strict type-gate for ALL data flowing
through the ComplianceWatch pipeline.

Design Philosophy
-----------------
Every LLM output, every inter-module data transfer, and every API response
is represented as a Pydantic model. This means:
  1. Malformed LLM outputs are caught at the boundary (before they corrupt DB state).
  2. IDE auto-complete and static type-checkers work across the entire codebase.
  3. FastAPI can directly use these as response_model=... with zero extra work.

Model Map
---------
Scraping Layer
  ScrapedContent          — Raw output of one Playwright scrape

Embedding Layer
  EmbeddingResult         — Embedding vector + metadata

Drift Detection Layer
  DriftReport             — Cosine distance + threshold verdict

Vision Analysis Layer  ← Gemini output is FORCED into this schema
  VisionAnalysisResult    — Structured JSON output from Gemini 1.5 Flash

Orchestration Layer
  ScanReport              — Full report combining drift + vision results
  MerchantOnboardRequest  — Input to onboard a new merchant
  OnboardingResult        — Output after successful onboarding
  FullScanSummary         — Aggregate stats after a full scan run
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Scraping Layer
# ---------------------------------------------------------------------------


class ScrapedContent(BaseModel):
    """
    The complete output of a single Playwright scrape of one URL.

    Attributes
    ----------
    url : str
        The URL that was scraped.
    text : str
        The full visible text content extracted from the page (innerText).
        This is what gets embedded for semantic comparison.
    html : str
        The raw HTML source. Passed to Gemini alongside the screenshot for
        richer context during vision analysis.
    screenshot_bytes : bytes
        A PNG screenshot of the full page, used for vision analysis.
    scraped_at : datetime
        UTC timestamp of when the scrape was performed.
    text_char_count : int
        Character length of the extracted text (auto-computed).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    url: str = Field(..., description="The URL that was scraped.")
    text: str = Field(..., description="Visible text extracted from the page via innerText.")
    html: str = Field(..., description="Raw HTML source of the page.")
    screenshot_bytes: bytes = Field(..., description="PNG screenshot of the full rendered page.")
    scraped_at: datetime = Field(default_factory=datetime.utcnow)
    text_char_count: int = Field(default=0, description="Auto-computed character count of text.")

    @model_validator(mode="after")
    def compute_char_count(self) -> "ScrapedContent":
        self.text_char_count = len(self.text)
        return self

    @property
    def text_preview(self) -> str:
        """Return the first 300 characters for logging/display."""
        return self.text[:300].replace("\n", " ").strip()


# ---------------------------------------------------------------------------
# Embedding Layer
# ---------------------------------------------------------------------------


class EmbeddingResult(BaseModel):
    """
    Wraps a text embedding vector with provenance metadata.

    Attributes
    ----------
    vector : list[float]
        The raw embedding vector. Dimensionality depends on the model
        (text-embedding-004 produces 768-dimensional vectors).
    model_used : str
        Name of the embedding model that produced this vector.
    text_char_count : int
        Length of the input text that was embedded (for debugging).
    truncated : bool
        True if the input text was truncated before embedding due to
        the model's token limit.
    """

    vector: list[float] = Field(..., description="The raw embedding vector.")
    model_used: str = Field(..., description="Embedding model identifier.")
    text_char_count: int = Field(default=0)
    truncated: bool = Field(
        default=False,
        description="True if the input text was truncated to fit the model's token limit.",
    )

    @property
    def dimension(self) -> int:
        """Return the dimensionality of the embedding vector."""
        return len(self.vector)


# ---------------------------------------------------------------------------
# Drift Detection Layer
# ---------------------------------------------------------------------------


class DriftReport(BaseModel):
    """
    The outcome of comparing a merchant's current embedding against their baseline.

    Attributes
    ----------
    merchant_id : str
        UUID of the merchant being evaluated.
    cosine_distance : float
        Cosine distance between baseline and current embeddings. Range: 0.0–2.0.
        (0.0 = identical, 1.0 = orthogonal, 2.0 = opposite — in practice values
        above 0.5 are extremely rare for web pages.)
    cosine_similarity : float
        1 - cosine_distance. Auto-computed.
    drift_detected : bool
        True when cosine_distance exceeds the configured threshold.
    threshold_used : float
        The threshold value that was applied for this comparison.
    variance_pct : float
        cosine_distance expressed as a percentage (0–100) for dashboard display.
    """

    merchant_id: str
    cosine_distance: Annotated[float, Field(ge=0.0, le=2.0)]
    cosine_similarity: float = Field(default=0.0)
    drift_detected: bool
    threshold_used: float
    variance_pct: float = Field(default=0.0)

    @model_validator(mode="after")
    def compute_derived(self) -> "DriftReport":
        self.cosine_similarity = round(1.0 - self.cosine_distance, 6)
        self.variance_pct = round(self.cosine_distance * 100, 2)
        return self


# ---------------------------------------------------------------------------
# Vision Analysis Layer — STRICT LLM OUTPUT GATE
# ---------------------------------------------------------------------------


class VisionAnalysisResult(BaseModel):
    """
    Structured output from Gemini 1.5 Flash multimodal analysis.

    CRITICAL: This model is used as the RESPONSE SCHEMA passed directly to
    the Gemini API via `response_schema`. The API is instructed to return JSON
    that validates against this exact structure. Any response that fails
    validation is treated as an error and retried.

    Attributes
    ----------
    is_policy_violation : bool
        True if the page content violates Razorpay's Acceptable Use Policy.
    confidence_score : int
        Gemini's confidence in its verdict. Range: 0–100.
        0–39  = Low confidence (uncertain)
        40–69 = Medium confidence
        70–89 = High confidence
        90–100 = Very high confidence
    detected_banned_category : str | None
        The specific policy category violated, or None if no violation.
        Expected values: "VAPE_ECIG", "CRYPTO_TRADING", "ONLINE_GAMBLING",
        "ADULT_CONTENT", "WEAPONS", "NARCOTICS", "NONE"
    reasoning_summary : str
        Gemini's explanation of its verdict. Capped at 500 characters.
        Must contain at most 3 sentences.
    visual_evidence : list[str]
        List of specific visual or textual elements Gemini identified as
        evidence for its verdict (e.g., ["Vape product listings", "18+ age gate"]).
    """

    is_policy_violation: bool = Field(
        ...,
        description="True if the merchant website violates Razorpay's Acceptable Use Policy.",
    )
    confidence_score: Annotated[int, Field(ge=0, le=100)] = Field(
        ...,
        description="Gemini's confidence in its verdict, from 0 (uncertain) to 100 (certain).",
    )
    detected_banned_category: str | None = Field(
        default=None,
        description=(
            "The policy category violated. One of: VAPE_ECIG, CRYPTO_TRADING, "
            "ONLINE_GAMBLING, ADULT_CONTENT, WEAPONS, NARCOTICS, NONE"
        ),
    )
    reasoning_summary: Annotated[str, Field(max_length=500)] = Field(
        ...,
        description="A 1-3 sentence explanation of the verdict.",
    )
    visual_evidence: list[str] = Field(
        default_factory=list,
        description="Specific textual or visual elements that support the verdict.",
    )

    @field_validator("detected_banned_category")
    @classmethod
    def normalise_category(cls, v: str | None) -> str | None:
        """Normalise 'NONE' string to Python None, and upper-case the category."""
        if v is None or v.upper() in ("NONE", "NULL", "N/A", ""):
            return None
        return v.upper()

    @field_validator("reasoning_summary")
    @classmethod
    def trim_reasoning(cls, v: str) -> str:
        """Truncate to 500 chars with an ellipsis marker if needed."""
        return v[:497] + "..." if len(v) > 500 else v

    @property
    def risk_label(self) -> Literal["GREEN", "YELLOW", "RED"]:
        """
        Convert the raw analysis into a three-tier risk label.

        RED    — violation confirmed with confidence >= 60
        YELLOW — violation suspected but confidence < 60
        GREEN  — no violation detected
        """
        if not self.is_policy_violation:
            return "GREEN"
        return "RED" if self.confidence_score >= 60 else "YELLOW"


# ---------------------------------------------------------------------------
# Orchestration Layer
# ---------------------------------------------------------------------------


class ScanReport(BaseModel):
    """
    The complete output of the compliance agent pipeline for one merchant,
    for one scan cycle. Stored as JSON in the `scan_results` table and
    returned directly to the API / dashboard.
    """

    merchant_id: str
    merchant_name: str
    mock_site_key: str
    scan_timestamp: datetime = Field(default_factory=datetime.utcnow)

    # URLs involved
    baseline_url: str
    current_url: str

    # Screenshot paths on disk (relative to project root)
    baseline_screenshot_path: str | None = None
    current_screenshot_path: str | None = None

    # Drift analysis results
    drift_report: DriftReport

    # Vision analysis (None when drift was not detected)
    vision_analysis: VisionAnalysisResult | None = None

    # Final verdicts
    risk_level: Literal["GREEN", "YELLOW", "RED"]
    action_recommended: str

    @field_validator("action_recommended", mode="before")
    @classmethod
    def default_action(cls, v: str | None) -> str:
        return v or "No action required."

    @property
    def is_flagged(self) -> bool:
        return self.risk_level in ("YELLOW", "RED")

    @property
    def cosine_variance_pct(self) -> float:
        return self.drift_report.variance_pct


class MerchantOnboardRequest(BaseModel):
    """
    Input payload for onboarding a new merchant.

    Sent by the API when a simulated Razorpay onboarding webhook fires.
    """

    name: str = Field(..., min_length=2, max_length=255)
    business_category: str = Field(..., min_length=2, max_length=100)
    mock_site_key: str = Field(
        ...,
        description="Key in the mock server's MERCHANT_REGISTRY. e.g. 'diya_store'.",
        pattern=r"^[a-z][a-z0-9_]{1,49}$",
    )


class OnboardingResult(BaseModel):
    """
    Result returned after a merchant is successfully onboarded.
    """

    merchant_id: str
    merchant_name: str
    mock_site_key: str
    baseline_url: str
    baseline_screenshot_path: str | None
    baseline_text_snippet: str
    onboarded_at: datetime
    status: str = "ACTIVE"
    message: str = "Merchant onboarded successfully. Baseline captured."


class FullScanSummary(BaseModel):
    """
    Aggregate statistics returned after a full monitoring scan run across
    all active merchants.
    """

    scan_run_id: str = Field(..., description="A UUID identifying this scan run.")
    started_at: datetime
    completed_at: datetime
    total_merchants_scanned: int
    green_count: int
    yellow_count: int
    red_count: int
    scan_reports: list[ScanReport]
    errors: list[str] = Field(
        default_factory=list,
        description="Merchant IDs or error messages for scans that failed.",
    )

    @property
    def duration_seconds(self) -> float:
        return (self.completed_at - self.started_at).total_seconds()

    @property
    def flagged_merchants(self) -> list[ScanReport]:
        return [r for r in self.scan_reports if r.is_flagged]
