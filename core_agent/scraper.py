"""
core_agent/scraper.py
----------------------
Async Playwright-based web scraper for capturing merchant website snapshots.

Responsibilities
----------------
1. Launch a headless Chromium browser (managed as a context-manager resource).
2. Navigate to a given URL and wait for network activity to settle.
3. Extract the full visible text (innerText) from the rendered DOM.
4. Capture a full-page PNG screenshot.
5. Return the raw HTML source.

Design Notes
------------
* The scraper is implemented as an async context manager class to guarantee
  proper browser lifecycle management (launch on enter, close on exit).
  This prevents leaked Chromium processes even on error paths.
* We deliberately avoid JS-heavy interactions (clicks, form fills) — we only
  need a static snapshot of what the page presents to a visitor.
* Page load strategy: 'networkidle' wait ensures JS-rendered content is visible
  but we add a hard timeout so a slow/broken page doesn't block the pipeline.
* Screenshots are full-page PNG (not viewport-only) to capture below-the-fold
  product listings that matter most for compliance checking.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Self

from playwright.async_api import (
    Browser,
    BrowserContext,
    Error as PlaywrightError,
    Page,
    Playwright,
    async_playwright,
)

from core_agent.schemas import ScrapedContent
from utils.exceptions import ScrapingError
from utils.logging_config import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Maximum time (ms) to wait for the page to finish loading
_PAGE_LOAD_TIMEOUT_MS: int = 30_000

# After load, additional ms to wait for any JS animations to settle
_RENDER_SETTLE_WAIT_MS: int = 1_500

# Maximum characters of page text to embed. Gemini text-embedding-004 has a
# token limit; 8 000 chars ≈ ~2 000 tokens, well within the 2 048 token limit.
MAX_TEXT_CHARS_FOR_EMBEDDING: int = 8_000


class WebsiteScraper:
    """
    Async context manager that manages a single headless Chromium instance.

    Usage
    -----
    >>> async with WebsiteScraper() as scraper:
    ...     content = await scraper.scrape("http://localhost:8100/merchant/diya_store/clean")
    ...     print(content.text_preview)

    The same scraper instance can be reused for multiple scrape() calls,
    sharing the underlying browser process (faster for batch operations).
    """

    def __init__(
        self,
        headless: bool = True,
        viewport_width: int = 1440,
        viewport_height: int = 900,
    ) -> None:
        self._headless = headless
        self._viewport_width = viewport_width
        self._viewport_height = viewport_height

        self._playwright: Playwright | None = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None

    # ------------------------------------------------------------------
    # Context Manager Protocol
    # ------------------------------------------------------------------

    async def __aenter__(self) -> Self:
        await self._launch()
        return self

    async def __aexit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        await self._close()

    # ------------------------------------------------------------------
    # Internal Lifecycle
    # ------------------------------------------------------------------

    async def _launch(self) -> None:
        """Start Playwright and launch a headless Chromium browser."""
        logger.debug("Launching headless Chromium browser...")
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            headless=self._headless,
            args=[
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-dev-shm-usage",  # Prevent /dev/shm exhaustion in containers
                "--disable-gpu",
            ],
        )
        self._context = await self._browser.new_context(
            viewport={"width": self._viewport_width, "height": self._viewport_height},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/126.0.0.0 Safari/537.36"
            ),
            # Block analytics and tracking scripts to keep pages lightweight
            java_script_enabled=True,
        )
        logger.debug("Chromium browser launched successfully.")

    async def _close(self) -> None:
        """Tear down context, browser, and playwright in reverse order."""
        try:
            if self._context:
                await self._context.close()
            if self._browser:
                await self._browser.close()
            if self._playwright:
                await self._playwright.stop()
        except Exception:
            logger.warning("Error during browser teardown — may have already been closed.")
        finally:
            self._context = None
            self._browser = None
            self._playwright = None
        logger.debug("Chromium browser closed.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def scrape(self, url: str, merchant_id: str | None = None) -> ScrapedContent:
        """
        Scrape a single URL and return a populated ScrapedContent object.

        Parameters
        ----------
        url : str
            The full URL to scrape (e.g. http://localhost:8100/merchant/diya_store)
        merchant_id : str | None
            Optional merchant ID for enriched error messages and log context.

        Returns
        -------
        ScrapedContent
            Contains the raw text, HTML, and screenshot bytes.

        Raises
        ------
        ScrapingError
            On any Playwright error, timeout, or empty-page result.
        RuntimeError
            If called outside an async context manager (browser not launched).
        """
        if self._context is None:
            raise RuntimeError(
                "WebsiteScraper must be used as an async context manager. "
                "Use: async with WebsiteScraper() as scraper: ..."
            )

        log_extra = {"url": url, "merchant_id": merchant_id}
        logger.info("Starting page scrape", extra=log_extra)

        page: Page = await self._context.new_page()
        try:
            return await self._perform_scrape(page, url, merchant_id)
        finally:
            await page.close()

    async def _perform_scrape(
        self,
        page: Page,
        url: str,
        merchant_id: str | None,
    ) -> ScrapedContent:
        """
        Internal scrape logic executed within an already-opened Page.

        We navigate, wait for idle network, then capture text + screenshot in
        parallel (screenshot is a separate call, not truly parallel, but we
        keep them sequential for stability).
        """
        try:
            # Navigate with a combined load + network-idle strategy
            await page.goto(
                url,
                wait_until="networkidle",
                timeout=_PAGE_LOAD_TIMEOUT_MS,
            )
        except PlaywrightError as nav_err:
            raise ScrapingError(
                message=f"Navigation failed: {nav_err}",
                url=url,
                merchant_id=merchant_id,
                original_error=nav_err,
            ) from nav_err

        # Allow any CSS transitions / lazy-loaded images to finish rendering
        await asyncio.sleep(_RENDER_SETTLE_WAIT_MS / 1_000)

        # --- Extract visible text ---
        try:
            raw_text: str = await page.evaluate("() => document.body.innerText")
        except PlaywrightError as text_err:
            raise ScrapingError(
                message=f"Failed to extract page text: {text_err}",
                url=url,
                merchant_id=merchant_id,
                original_error=text_err,
            ) from text_err

        cleaned_text = _clean_text(raw_text)

        if not cleaned_text.strip():
            raise ScrapingError(
                message="Page returned empty text — may be a bot block or blank page.",
                url=url,
                merchant_id=merchant_id,
            )

        # --- Extract raw HTML ---
        try:
            html: str = await page.content()
        except PlaywrightError as html_err:
            logger.warning(
                "Failed to extract HTML source, continuing without it.",
                extra={"url": url, "error": str(html_err)},
            )
            html = ""

        # --- Capture full-page screenshot ---
        try:
            screenshot_bytes: bytes = await page.screenshot(
                full_page=True,
                type="png",
                animations="disabled",  # Freeze animations for a clean shot
            )
        except PlaywrightError as shot_err:
            raise ScrapingError(
                message=f"Screenshot capture failed: {shot_err}",
                url=url,
                merchant_id=merchant_id,
                original_error=shot_err,
            ) from shot_err

        content = ScrapedContent(
            url=url,
            text=cleaned_text[:MAX_TEXT_CHARS_FOR_EMBEDDING],
            html=html[:50_000],       # Cap HTML at 50KB for DB storage
            screenshot_bytes=screenshot_bytes,
            scraped_at=datetime.now(tz=timezone.utc).replace(tzinfo=None),
        )

        logger.info(
            "Scrape completed successfully",
            extra={
                "url": url,
                "merchant_id": merchant_id,
                "text_chars": content.text_char_count,
                "screenshot_bytes": len(screenshot_bytes),
                "truncated": len(raw_text) > MAX_TEXT_CHARS_FOR_EMBEDDING,
            },
        )
        return content


# ---------------------------------------------------------------------------
# Convenience Function (for single-shot scrapes without reuse)
# ---------------------------------------------------------------------------


async def scrape_once(url: str, merchant_id: str | None = None) -> ScrapedContent:
    """
    Convenience wrapper for one-off scrapes that creates, uses, and tears down
    a WebsiteScraper in a single call.

    Prefer the context manager form when scraping multiple URLs in sequence
    to avoid the browser launch overhead on every call.
    """
    async with WebsiteScraper() as scraper:
        return await scraper.scrape(url, merchant_id=merchant_id)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _clean_text(raw: str) -> str:
    """
    Normalise whitespace in extracted page text.

    Playwright's innerText can contain runs of blank lines and tab characters.
    We collapse them into single spaces / newlines to reduce token waste.
    """
    lines = (line.strip() for line in raw.splitlines())
    non_empty = (line for line in lines if line)
    return "\n".join(non_empty)


def save_screenshot(
    screenshot_bytes: bytes,
    directory: Path,
    filename: str,
) -> Path:
    """
    Persist a screenshot to disk.

    Parameters
    ----------
    screenshot_bytes : bytes
        Raw PNG bytes from Playwright.
    directory : Path
        Target directory (must already exist or will be created).
    filename : str
        Filename without extension (e.g. 'merchant_abc123_baseline').

    Returns
    -------
    Path
        The absolute path to the saved PNG file.
    """
    directory.mkdir(parents=True, exist_ok=True)
    out_path = directory / f"{filename}.png"
    out_path.write_bytes(screenshot_bytes)
    logger.debug("Screenshot saved", extra={"path": str(out_path), "bytes": len(screenshot_bytes)})
    return out_path
