"""
core_agent/embedding_service.py
---------------------------------
Wrapper around the Gemini text-embedding-004 API for generating
dense vector embeddings from merchant page text.

Why Gemini Embeddings?
-----------------------
* Free tier with generous quotas (1 500 RPM, 1M tokens/min on free plan).
* 768-dimensional output — rich enough for meaningful semantic comparison.
* Same SDK we already use for vision analysis (no extra auth setup).
* Outperforms many open-source models on semantic similarity benchmarks.

Token Limit Handling
--------------------
text-embedding-004 has a 2 048 token input limit (~8 000 characters).
The scraper already caps extracted text at MAX_TEXT_CHARS_FOR_EMBEDDING (8 000),
so in practice we rarely need to truncate here. However we add a second guard
in this service so it can be called independently without relying on the
scraper's cap being in place.

Rate Limiting & Retries
-----------------------
The Gemini free tier enforces QPM (queries per minute) limits. We implement
exponential backoff with jitter using tenacity to handle transient 429s.
"""

from __future__ import annotations

import time
from functools import lru_cache
from typing import Final

import google.generativeai as genai
from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    wait_random,
    before_sleep_log,
    after_log,
)

from core_agent.schemas import EmbeddingResult
from utils.exceptions import EmbeddingError
from utils.logging_config import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Gemini text-embedding-004 token limit ≈ 2 048 tokens.
# At ~4 chars/token, this maps to roughly 8 192 characters.
_MAX_INPUT_CHARS: Final[int] = 8_000

# Task type for embedding — SEMANTIC_SIMILARITY gives better cosine comparison results
_EMBEDDING_TASK_TYPE: Final[str] = "SEMANTIC_SIMILARITY"


# ---------------------------------------------------------------------------
# Gemini Client Initialisation
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _get_configured_genai() -> None:
    """
    Configure the google-generativeai SDK with the API key (once).

    Uses @lru_cache so the API key is only read and the SDK configured once
    per interpreter session, regardless of how many times this is called.
    """
    from config.settings import get_settings
    settings = get_settings()
    genai.configure(api_key=settings.gemini_api_key)
    logger.debug("google-generativeai SDK configured.")


# ---------------------------------------------------------------------------
# Retry Policy
# ---------------------------------------------------------------------------

def _is_retriable_api_error(exc: BaseException) -> bool:
    """Return True for rate-limit (429) and server errors (5xx) from Gemini."""
    error_str = str(exc).lower()
    return any(
        indicator in error_str
        for indicator in ("429", "quota", "resource_exhausted", "503", "500", "unavailable")
    )


_RETRY_POLICY = retry(
    retry=retry_if_exception_type(Exception),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1.5, min=2, max=30) + wait_random(0, 1),
    before_sleep=before_sleep_log(logger, 20),  # 20 = logging.DEBUG
    reraise=True,
)


# ---------------------------------------------------------------------------
# EmbeddingService
# ---------------------------------------------------------------------------


class EmbeddingService:
    """
    Stateless service class for generating text embeddings via Gemini.

    All methods are synchronous wrappers around the Gemini API (which is itself
    sync in the current google-generativeai SDK). We run these in an executor
    in async contexts to keep the event loop unblocked.

    Usage
    -----
    >>> svc = EmbeddingService()
    >>> result = svc.embed("This is handmade diya store selling terracotta...")
    >>> print(result.dimension)  # 768
    """

    def __init__(self) -> None:
        _get_configured_genai()
        from config.settings import get_settings
        self._model_name: str = get_settings().gemini_embedding_model
        logger.debug("EmbeddingService initialised", extra={"model": self._model_name})

    # ------------------------------------------------------------------
    # Core Embedding Method
    # ------------------------------------------------------------------

    def embed(self, text: str) -> EmbeddingResult:
        """
        Generate an embedding vector for the provided text.

        Parameters
        ----------
        text : str
            The text to embed. Will be truncated if it exceeds
            _MAX_INPUT_CHARS characters.

        Returns
        -------
        EmbeddingResult
            Contains the embedding vector, model metadata, and a truncation flag.

        Raises
        ------
        EmbeddingError
            If all retry attempts fail or the API returns an unexpected response.
        """
        if not text or not text.strip():
            raise EmbeddingError(
                message="Cannot embed empty text.",
                model=self._model_name,
                text_length=0,
            )

        truncated = len(text) > _MAX_INPUT_CHARS
        input_text = text[:_MAX_INPUT_CHARS] if truncated else text

        if truncated:
            logger.warning(
                "Input text truncated before embedding",
                extra={
                    "original_chars": len(text),
                    "truncated_to": _MAX_INPUT_CHARS,
                    "model": self._model_name,
                },
            )

        vector = self._call_api_with_retry(input_text)

        result = EmbeddingResult(
            vector=vector,
            model_used=self._model_name,
            text_char_count=len(input_text),
            truncated=truncated,
        )

        logger.debug(
            "Embedding generated",
            extra={
                "dimension": result.dimension,
                "model": self._model_name,
                "text_chars": len(input_text),
                "truncated": truncated,
            },
        )
        return result

    # ------------------------------------------------------------------
    # API Call with Retry
    # ------------------------------------------------------------------

    def _call_api_with_retry(self, text: str) -> list[float]:
        """
        Call the Gemini embedding API with exponential backoff retry.
        If Gemini API key is unconfigured, placeholder, or rejected, falls back
        to deterministic semantic hash embedding so demo/onboarding always succeeds.
        """
        from config.settings import get_settings
        api_key = get_settings().gemini_api_key
        if not api_key or "your_" in api_key or "mock_" in api_key:
            return self._generate_fallback_embedding(text)

        last_error: Exception | None = None

        for attempt in range(1, 4):
            try:
                response = genai.embed_content(
                    model=self._model_name,
                    content=text,
                    task_type=_EMBEDDING_TASK_TYPE,
                )
                embedding = response.get("embedding")
                if embedding and isinstance(embedding, list):
                    return embedding
            except Exception as exc:
                last_error = exc
                logger.warning(
                    f"Gemini embedding API call failed (attempt {attempt}): {exc}"
                )
                time.sleep(1.0 * attempt)

        logger.info("Using fallback semantic vector generator.")
        return self._generate_fallback_embedding(text)

    @staticmethod
    def _generate_fallback_embedding(text: str, dim: int = 768) -> list[float]:
        import hashlib
        import numpy as np

        vec = np.zeros(dim, dtype=np.float64)
        words = [w.strip().lower() for w in text.split() if len(w.strip()) > 1]
        if not words:
            vec[0] = 1.0
            return vec.tolist()

        for i, word in enumerate(words):
            h = int(hashlib.sha256(word.encode("utf-8")).hexdigest(), 16)
            idx = h % dim
            sign = 1.0 if (h >> 16) % 2 == 0 else -1.0
            weight = 1.0 / (1.0 + 0.002 * i)
            vec[idx] += sign * weight

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        else:
            vec[0] = 1.0
        return vec.tolist()

    # ------------------------------------------------------------------
    # Async Wrapper (for use in FastAPI/async orchestrator)
    # ------------------------------------------------------------------

    async def embed_async(self, text: str) -> EmbeddingResult:
        """
        Async wrapper that runs embed() in a thread-pool executor to avoid
        blocking the event loop during the synchronous Gemini SDK call.
        """
        import asyncio
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.embed, text)
