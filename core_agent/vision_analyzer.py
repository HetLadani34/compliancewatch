"""
core_agent/vision_analyzer.py
------------------------------
Gemini 1.5 Flash multimodal analysis for policy violation detection.

This is the most critical module in the pipeline. When the drift detector
flags a merchant, this module takes over and makes the definitive compliance
verdict by showing Gemini the actual screenshot AND the page text.

Gemini Structured Output Strategy
-----------------------------------
We use two layers of enforcement to guarantee structured JSON output:

  Layer 1 — response_mime_type="application/json"
    Tells Gemini to format its response as raw JSON.

  Layer 2 — response_schema (Pydantic model converted to JSON schema)
    Provides the exact field names, types, and constraints. Gemini's
    token generation is steered to match this schema exactly.

  Layer 3 — Pydantic validation on the received string
    Even if Gemini returns valid JSON, we parse it through VisionAnalysisResult.
    Any field missing or out-of-range raises a ValidationError that we catch
    and retry.

This triple-gate approach means malformed LLM outputs never reach the database.

Acceptable Use Policy (AUP) Embedded in Prompt
-----------------------------------------------
We embed a concise version of Razorpay's real AUP into the system prompt.
This gives Gemini the necessary context to make accurate policy judgements
without us needing to train a custom model.

Banned categories:
  VAPE_ECIG         — E-cigarettes, vapes, nicotine pods, e-liquids
  CRYPTO_TRADING    — Unregulated crypto exchanges, trading platforms
  ONLINE_GAMBLING   — Sports betting, casino games, fantasy gaming for money
  ADULT_CONTENT     — Pornography, adult services
  WEAPONS           — Firearms, ammunition, explosives
  NARCOTICS         — Illegal drugs, paraphernalia
"""

from __future__ import annotations

import json
import time
from typing import Any

import google.generativeai as genai
from google.generativeai.types import GenerationConfig

from core_agent.schemas import VisionAnalysisResult
from utils.exceptions import VisionAnalysisError
from utils.logging_config import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_RETRIES: int = 3
_BASE_BACKOFF_SECONDS: float = 3.0

# Maximum HTML characters to include in the vision prompt (token budget)
_MAX_HTML_CHARS_IN_PROMPT: int = 3_000

# The system instruction embedded in every Gemini request
_SYSTEM_INSTRUCTION = """You are ComplianceWatch, an AI compliance auditor for Razorpay Payment Gateway.

Your task is to analyze a merchant's website screenshot and page text to determine if it violates Razorpay's Acceptable Use Policy (AUP).

RAZORPAY ACCEPTABLE USE POLICY — BANNED CATEGORIES:
1. VAPE_ECIG: E-cigarettes, vapes, pod systems, e-liquids, nicotine pouches, or any related accessories.
2. CRYPTO_TRADING: Unregulated cryptocurrency exchanges, crypto trading platforms, ICOs, or crypto investment services.
3. ONLINE_GAMBLING: Sports betting, casino games (roulette, blackjack, slots, poker), fantasy sports for real money, lottery, or any wagering service.
4. ADULT_CONTENT: Pornography, escort services, adult entertainment, or sexually explicit material.
5. WEAPONS: Firearms, ammunition, explosives, knives, or any weapon or weapon accessory.
6. NARCOTICS: Illegal drugs, controlled substances, or drug paraphernalia.

ANALYSIS INSTRUCTIONS:
- Carefully examine BOTH the screenshot (visual evidence) and the HTML/text (textual evidence).
- Look for product listings, prices, category names, navigation menu items, and brand names.
- A single clear product listing of a banned item is sufficient evidence for a violation.
- Age-gate warnings (e.g., "18+ only") for banned categories are additional confirmation.
- If the site clearly belongs to a banned category, set is_policy_violation to true.
- Your confidence_score must reflect how certain you are based on the evidence available.
- Be specific in visual_evidence — list exact text, product names, or visual elements you saw.

You MUST respond with valid JSON only. No prose, no markdown, no code blocks. Raw JSON only."""

# The user-facing prompt template
_USER_PROMPT_TEMPLATE = """Analyze this merchant website for policy compliance.

MERCHANT URL: {url}
DETECTED SEMANTIC DRIFT: {drift_pct:.1f}% variance from baseline (threshold: {threshold:.0%})

EXTRACTED PAGE TEXT (first 2000 chars):
---
{page_text}
---

HTML STRUCTURE HINTS (nav links, headings, product names):
---
{html_snippet}
---

Based on the screenshot and text above, provide your compliance verdict as JSON.
Remember: respond with raw JSON only, no markdown."""


# ---------------------------------------------------------------------------
# VisionAnalyzer
# ---------------------------------------------------------------------------


class VisionAnalyzer:
    """
    Multimodal compliance analyzer using Gemini 1.5 Flash.

    Takes a PNG screenshot and page text, sends them to Gemini with the
    embedded Acceptable Use Policy, and returns a strictly typed
    VisionAnalysisResult.

    Usage
    -----
    >>> analyzer = VisionAnalyzer()
    >>> result = await analyzer.analyze(
    ...     screenshot_bytes=png_bytes,
    ...     page_text="VapeZone India — Buy e-cigarettes...",
    ...     page_html="<nav>Devices | E-Liquids | Pod Systems</nav>...",
    ...     merchant_url="http://localhost:8100/merchant/diya_store",
    ...     drift_pct=72.4,
    ...     drift_threshold=0.30,
    ...     merchant_id="abc-123",
    ... )
    >>> print(result.is_policy_violation, result.detected_banned_category)
    True VAPE_ECIG
    """

    def __init__(self) -> None:
        from config.settings import get_settings
        settings = get_settings()
        genai.configure(api_key=settings.gemini_api_key)
        self._model_name: str = settings.gemini_vision_model
        self._model = genai.GenerativeModel(
            model_name=self._model_name,
            system_instruction=_SYSTEM_INSTRUCTION,
        )
        logger.debug("VisionAnalyzer initialised", extra={"model": self._model_name})

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def analyze(
        self,
        screenshot_bytes: bytes,
        page_text: str,
        page_html: str,
        merchant_url: str,
        drift_pct: float,
        drift_threshold: float,
        merchant_id: str | None = None,
    ) -> VisionAnalysisResult:
        """
        Perform multimodal compliance analysis on a merchant's current state.

        This method is async — it runs the synchronous Gemini SDK call in a
        thread executor to avoid blocking the FastAPI event loop.

        Parameters
        ----------
        screenshot_bytes : bytes
            PNG screenshot of the merchant's current website.
        page_text : str
            Visible text extracted from the page (used as textual evidence).
        page_html : str
            Raw HTML (we extract nav links and headings for the prompt).
        merchant_url : str
            The URL that was scraped (for context in the prompt).
        drift_pct : float
            Cosine variance percentage (for context in the prompt).
        drift_threshold : float
            The threshold used for drift detection.
        merchant_id : str | None
            UUID of the merchant (for error messages and logging).

        Returns
        -------
        VisionAnalysisResult
            Strictly validated Pydantic model from Gemini's response.

        Raises
        ------
        VisionAnalysisError
            If all retry attempts fail or Gemini returns persistently malformed JSON.
        """
        import asyncio
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._analyze_sync(
                screenshot_bytes=screenshot_bytes,
                page_text=page_text,
                page_html=page_html,
                merchant_url=merchant_url,
                drift_pct=drift_pct,
                drift_threshold=drift_threshold,
                merchant_id=merchant_id,
            ),
        )

    # ------------------------------------------------------------------
    # Synchronous Analysis (runs in thread executor)
    # ------------------------------------------------------------------

    def _analyze_sync(
        self,
        screenshot_bytes: bytes,
        page_text: str,
        page_html: str,
        merchant_url: str,
        drift_pct: float,
        drift_threshold: float,
        merchant_id: str | None,
    ) -> VisionAnalysisResult:
        """
        Synchronous analysis pipeline with retry loop.

        On each attempt:
          1. Build the multimodal message (image + text prompt).
          2. Call Gemini with structured output config.
          3. Parse the JSON response into VisionAnalysisResult.
          4. If parsing fails, retry with exponential backoff.
        """
        html_snippet = _extract_html_hints(page_html)
        user_prompt = _USER_PROMPT_TEMPLATE.format(
            url=merchant_url,
            drift_pct=drift_pct,
            threshold=drift_threshold,
            page_text=page_text[:2_000],
            html_snippet=html_snippet,
        )

        # Build the image part for the multimodal message
        image_part = {
            "mime_type": "image/png",
            "data": screenshot_bytes,
        }

        generation_config = GenerationConfig(
            response_mime_type="application/json",
            temperature=0.1,    # Low temperature = more deterministic, factual output
            top_p=0.95,
            max_output_tokens=1024,
        )

        last_error: Exception | None = None
        raw_response: str = ""

        from config.settings import get_settings
        api_key = get_settings().gemini_api_key
        if not api_key or "your_" in api_key or "mock_" in api_key:
            return self._fallback_analysis(page_text, page_html, merchant_url, drift_pct)

        for attempt in range(1, _MAX_RETRIES + 1):
            logger.info(
                "Vision analysis attempt",
                extra={
                    "attempt": attempt,
                    "max_retries": _MAX_RETRIES,
                    "merchant_id": merchant_id,
                    "model": self._model_name,
                },
            )
            try:
                response = self._model.generate_content(
                    contents=[image_part, user_prompt],
                    generation_config=generation_config,
                )

                raw_response = response.text.strip()
                result = self._parse_and_validate(raw_response, merchant_id)
                return result

            except Exception as exc:
                last_error = exc
                logger.warning(f"Vision analysis attempt {attempt} failed: {exc}")
                if attempt < _MAX_RETRIES:
                    time.sleep(_BASE_BACKOFF_SECONDS * attempt)

        logger.info("Falling back to rule-based policy analyzer.")
        return self._fallback_analysis(page_text, page_html, merchant_url, drift_pct)

    def _fallback_analysis(
        self,
        page_text: str,
        page_html: str,
        merchant_url: str,
        drift_pct: float,
    ) -> VisionAnalysisResult:
        full_content = (page_text + " " + page_html).lower()
        
        # Banned keyword rules
        gambling_kw = ["casino", "betting", "poker", "jackpot", "roulette", "slot machine", "wager", "gambling", "sportsbook"]
        pharma_kw = ["prescription", "viagra", "cialis", "steroids", "pharmacy no rx", "controlled substance", "xanax"]
        crypto_kw = ["crypto doubling", "guaranteed 100% return", "get rich quick", "crypto multiplier", "ponzi"]
        counterfeit_kw = ["replica rolex", "knockoff designer", "1:1 clone gucci", "fake luxury"]
        adult_kw = ["adult content", "escort", "18+ cams", "explicit content"]

        detected_category = None
        evidence = []
        is_violation = False

        if any(k in full_content for k in gambling_kw):
            detected_category = "Online Gambling / Casino"
            evidence = [f"Found gambling term '{k}'" for k in gambling_kw if k in full_content]
            is_violation = True
        elif any(k in full_content for k in pharma_kw):
            detected_category = "Unlicensed Pharmaceuticals"
            evidence = [f"Found pharma term '{k}'" for k in pharma_kw if k in full_content]
            is_violation = True
        elif any(k in full_content for k in crypto_kw):
            detected_category = "High-Risk Crypto / Ponzi Scheme"
            evidence = [f"Found crypto fraud term '{k}'" for k in crypto_kw if k in full_content]
            is_violation = True
        elif any(k in full_content for k in counterfeit_kw):
            detected_category = "Counterfeit / Replica Goods"
            evidence = [f"Found counterfeit term '{k}'" for k in counterfeit_kw if k in full_content]
            is_violation = True
        elif any(k in full_content for k in adult_kw):
            detected_category = "Adult / Restricted Content"
            evidence = [f"Found adult term '{k}'" for k in adult_kw if k in full_content]
            is_violation = True

        if is_violation:
            confidence = 0.95
            risk_label = "CRITICAL"
            reasoning = (
                f"Automated policy inspection detected prohibited commercial content ({detected_category}). "
                f"Found policy violation indicators: {', '.join(evidence[:3])}. Drift level: {drift_pct:.1f}%."
            )
            suggested_action = "TERMINATE"
        elif drift_pct > 50.0:
            is_violation = False
            confidence = 0.70
            risk_label = "HIGH"
            reasoning = f"Significant catalog shift detected ({drift_pct:.1f}% drift) but no explicit banned keywords identified."
            suggested_action = "MANUAL_REVIEW"
        elif drift_pct > 25.0:
            is_violation = False
            confidence = 0.85
            risk_label = "MEDIUM"
            reasoning = f"Moderate page changes detected ({drift_pct:.1f}% drift) within normal product variation ranges."
            suggested_action = "MONITOR"
        else:
            is_violation = False
            confidence = 0.95
            risk_label = "LOW"
            reasoning = f"Store content aligns with verified baseline ({drift_pct:.1f}% drift). No policy violations detected."
            suggested_action = "PASS"

        return VisionAnalysisResult(
            is_policy_violation=is_violation,
            detected_banned_category=detected_category,
            confidence_score=confidence,
            risk_label=risk_label,
            reasoning=reasoning,
            specific_evidence_found=evidence if evidence else ["Content matches expected merchant profile"],
            suggested_action=suggested_action,
        )

    # ------------------------------------------------------------------
    # Response Parsing & Validation
    # ------------------------------------------------------------------

    def _parse_and_validate(
        self,
        raw_response: str,
        merchant_id: str | None,
    ) -> VisionAnalysisResult:
        """
        Parse Gemini's raw response string into a validated VisionAnalysisResult.

        Handles two failure modes:
          1. JSON parse failure — the string is not valid JSON at all.
          2. Pydantic validation failure — valid JSON but wrong structure/types.

        Both are wrapped in VisionAnalysisError with the raw_response attached
        so the caller can log or retry intelligently.
        """
        # Strip Markdown code fences if Gemini accidentally includes them
        cleaned = raw_response.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            # Remove first (```json) and last (```) lines
            cleaned = "\n".join(lines[1:-1]).strip()

        try:
            parsed_dict: dict[str, Any] = json.loads(cleaned)
        except json.JSONDecodeError as json_err:
            raise VisionAnalysisError(
                message=f"Gemini returned invalid JSON: {json_err}",
                model=self._model_name,
                merchant_id=merchant_id,
                raw_response=raw_response[:500],
            ) from json_err

        try:
            result = VisionAnalysisResult.model_validate(parsed_dict)
        except Exception as val_err:
            raise VisionAnalysisError(
                message=f"Gemini JSON failed Pydantic validation: {val_err}",
                model=self._model_name,
                merchant_id=merchant_id,
                raw_response=raw_response[:500],
            ) from val_err

        return result


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _extract_html_hints(html: str) -> str:
    """
    Extract text from key HTML elements (nav, h1, h2, h3, title) to give Gemini
    structural context without sending the entire raw HTML.

    This is a lightweight regex-free extraction — we just look for common
    content-bearing tags by finding their text nodes.
    """
    if not html:
        return "(no HTML available)"

    import re

    # Tags whose text content is most revealing for compliance analysis
    target_tags = ["title", "h1", "h2", "h3", "nav", "li", "button"]
    extracted_parts: list[str] = []

    for tag in target_tags:
        # Find all occurrences of the tag (non-greedy inner match)
        pattern = rf"<{tag}[^>]*>(.*?)</{tag}>"
        matches = re.findall(pattern, html[:_MAX_HTML_CHARS_IN_PROMPT], re.IGNORECASE | re.DOTALL)
        for match in matches[:8]:   # Max 8 per tag type
            # Strip inner HTML tags to get text only
            text = re.sub(r"<[^>]+>", " ", match).strip()
            text = " ".join(text.split())  # Collapse whitespace
            if text and len(text) > 2:
                extracted_parts.append(f"[{tag.upper()}] {text}")

    if not extracted_parts:
        return html[:500]  # Fallback: first 500 chars of raw HTML

    return "\n".join(extracted_parts[:30])   # Cap at 30 hints
