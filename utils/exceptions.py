"""
utils/exceptions.py
--------------------
Custom exception hierarchy for ComplianceWatch.

Every exception carries rich contextual attributes so that upstream
callers and log handlers have enough information to route, display,
and diagnose failures without needing to parse free-form message strings.

Hierarchy
---------
ComplianceWatchError (base)
├── ConfigurationError          — Bad or missing config values
├── ScrapingError               — Playwright / network failures
├── EmbeddingError              — Gemini embedding API failures
├── VisionAnalysisError         — Gemini vision API failures
├── DriftDetectionError         — Vector math / threshold logic failures
├── MerchantNotFoundError       — Lookup failures in DB or vector store
├── DatabaseError               — SQLAlchemy / SQLite failures
└── VectorStoreError            — ChromaDB failures
"""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Base Exception
# ---------------------------------------------------------------------------


class ComplianceWatchError(Exception):
    """
    Root exception for all ComplianceWatch errors.

    All domain-specific exceptions inherit from this class, making it
    easy to catch any application-level error with a single except clause
    while still allowing granular handling where needed.

    Attributes
    ----------
    message : str
        Human-readable description of the error.
    context : dict[str, Any]
        Arbitrary key-value pairs providing additional diagnostic context
        (e.g., merchant_id, url, model_name). Never contains secrets.
    """

    def __init__(self, message: str, **context: Any) -> None:
        super().__init__(message)
        self.message = message
        self.context: dict[str, Any] = context

    def __repr__(self) -> str:
        ctx_str = ", ".join(f"{k}={v!r}" for k, v in self.context.items())
        return f"{self.__class__.__name__}({self.message!r}, {ctx_str})"

    def __str__(self) -> str:
        if self.context:
            ctx_str = " | ".join(f"{k}={v}" for k, v in self.context.items())
            return f"{self.message} [{ctx_str}]"
        return self.message


# ---------------------------------------------------------------------------
# Specific Exceptions
# ---------------------------------------------------------------------------


class ConfigurationError(ComplianceWatchError):
    """
    Raised when required configuration values are missing or invalid.

    Example: GEMINI_API_KEY not set in .env, or a port number out of range.
    """

    def __init__(self, message: str, field_name: str | None = None, **context: Any) -> None:
        super().__init__(message, field_name=field_name, **context)
        self.field_name = field_name


class ScrapingError(ComplianceWatchError):
    """
    Raised when the Playwright scraper fails to load or parse a page.

    Attributes
    ----------
    url : str
        The URL that was being scraped when the error occurred.
    merchant_id : str | None
        The merchant associated with the failed scrape, if known.
    original_error : Exception | None
        The underlying Playwright or network exception.
    """

    def __init__(
        self,
        message: str,
        url: str,
        merchant_id: str | None = None,
        original_error: Exception | None = None,
        **context: Any,
    ) -> None:
        super().__init__(message, url=url, merchant_id=merchant_id, **context)
        self.url = url
        self.merchant_id = merchant_id
        self.original_error = original_error


class EmbeddingError(ComplianceWatchError):
    """
    Raised when the Gemini embedding API call fails.

    Attributes
    ----------
    model : str
        The embedding model that was being used.
    text_length : int | None
        Length of the input text (for debugging token-limit issues).
    """

    def __init__(
        self,
        message: str,
        model: str,
        text_length: int | None = None,
        **context: Any,
    ) -> None:
        super().__init__(message, model=model, text_length=text_length, **context)
        self.model = model
        self.text_length = text_length


class VisionAnalysisError(ComplianceWatchError):
    """
    Raised when the Gemini vision/multimodal API call fails or returns
    malformed output that cannot be parsed into VisionAnalysisResult.

    Attributes
    ----------
    model : str
        The vision model that was being used.
    merchant_id : str | None
        The merchant whose screenshot was being analysed.
    raw_response : str | None
        The raw text response from Gemini, if available (useful for debugging
        JSON parse failures).
    """

    def __init__(
        self,
        message: str,
        model: str,
        merchant_id: str | None = None,
        raw_response: str | None = None,
        **context: Any,
    ) -> None:
        super().__init__(message, model=model, merchant_id=merchant_id, **context)
        self.model = model
        self.merchant_id = merchant_id
        self.raw_response = raw_response


class DriftDetectionError(ComplianceWatchError):
    """
    Raised when cosine distance computation fails (e.g., vector dimension mismatch).

    Attributes
    ----------
    merchant_id : str
        The merchant for whom drift detection was attempted.
    baseline_dim : int | None
        Dimensionality of the baseline embedding.
    current_dim : int | None
        Dimensionality of the current embedding.
    """

    def __init__(
        self,
        message: str,
        merchant_id: str,
        baseline_dim: int | None = None,
        current_dim: int | None = None,
        **context: Any,
    ) -> None:
        super().__init__(
            message,
            merchant_id=merchant_id,
            baseline_dim=baseline_dim,
            current_dim=current_dim,
            **context,
        )
        self.merchant_id = merchant_id
        self.baseline_dim = baseline_dim
        self.current_dim = current_dim


class MerchantNotFoundError(ComplianceWatchError):
    """
    Raised when a merchant cannot be located in the database or vector store.

    Attributes
    ----------
    merchant_id : str
        The ID that was searched for.
    source : str
        Where the lookup failed ("database", "chroma", etc.).
    """

    def __init__(self, merchant_id: str, source: str = "database", **context: Any) -> None:
        super().__init__(
            f"Merchant '{merchant_id}' not found in {source}.",
            merchant_id=merchant_id,
            source=source,
            **context,
        )
        self.merchant_id = merchant_id
        self.source = source


class DatabaseError(ComplianceWatchError):
    """
    Raised when a SQLAlchemy / SQLite operation fails.

    Attributes
    ----------
    operation : str
        The DB operation that failed (e.g., "insert", "query", "commit").
    table : str | None
        The table involved, if known.
    """

    def __init__(
        self,
        message: str,
        operation: str,
        table: str | None = None,
        **context: Any,
    ) -> None:
        super().__init__(message, operation=operation, table=table, **context)
        self.operation = operation
        self.table = table


class VectorStoreError(ComplianceWatchError):
    """
    Raised when a ChromaDB operation fails.

    Attributes
    ----------
    operation : str
        The ChromaDB operation that failed (e.g., "upsert", "query", "delete").
    collection : str | None
        The collection involved, if known.
    """

    def __init__(
        self,
        message: str,
        operation: str,
        collection: str | None = None,
        **context: Any,
    ) -> None:
        super().__init__(message, operation=operation, collection=collection, **context)
        self.operation = operation
        self.collection = collection
