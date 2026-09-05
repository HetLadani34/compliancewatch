"""
vector_store/chroma_client.py
------------------------------
ChromaDB persistent client for storing and querying merchant embeddings.

Architecture
------------
We use a SINGLE ChromaDB collection ("merchant_baselines") for all merchants.
Each document in the collection is keyed by a compound ID and tagged with
metadata (merchant_id, scan_type, timestamp) enabling precise filtering.

Document ID convention
----------------------
  Baseline : "{merchant_id}_baseline"
  Current  : "{merchant_id}_current_{scan_id}"

This means there is always exactly ONE baseline document per merchant, and
current-scan documents accumulate over time (useful for drift trend analysis).

Why ChromaDB?
-------------
* Runs fully in-process — no separate server to manage.
* Persists to disk in the configured directory.
* Supports cosine similarity search natively.
* The Python client is well-maintained and has a clean API.

Threading
---------
ChromaDB's PersistentClient is thread-safe for reads. Writes are serialised
internally by ChromaDB. However, since our FastAPI app uses asyncio, we
run all ChromaDB operations in a thread executor to avoid blocking the loop.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any

import chromadb
from chromadb import Collection, PersistentClient
from chromadb.config import Settings as ChromaSettings

from utils.exceptions import VectorStoreError
from utils.logging_config import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Metadata Keys — keep as constants to prevent typo-bugs
# ---------------------------------------------------------------------------
_META_MERCHANT_ID: str = "merchant_id"
_META_SCAN_TYPE: str = "scan_type"          # "baseline" or "current"
_META_MERCHANT_NAME: str = "merchant_name"
_META_MOCK_SITE_KEY: str = "mock_site_key"
_META_STORED_AT: str = "stored_at_iso"
_META_SCAN_ID: str = "scan_id"
_META_TEXT_SNIPPET: str = "text_snippet"    # First 200 chars for debugging


# ---------------------------------------------------------------------------
# VectorStoreClient
# ---------------------------------------------------------------------------


class VectorStoreClient:
    """
    High-level ChromaDB client for ComplianceWatch.

    Provides typed methods for all vector store operations needed by the
    compliance pipeline. All public methods have async variants that run
    the sync ChromaDB calls in a thread executor.

    Usage
    -----
    >>> client = VectorStoreClient()
    >>> client.store_baseline("merchant-uuid", embedding_vector, metadata={})
    >>> baseline = client.get_baseline_vector("merchant-uuid")
    """

    def __init__(self) -> None:
        from config.settings import get_settings
        settings = get_settings()
        self._persist_dir: str = settings.chroma_persist_dir
        self._collection_name: str = settings.chroma_collection_name
        self._client: PersistentClient = self._build_client()
        self._collection: Collection = self._get_or_create_collection()
        logger.info(
            "VectorStoreClient initialised",
            extra={
                "persist_dir": self._persist_dir,
                "collection": self._collection_name,
            },
        )

    # ------------------------------------------------------------------
    # Client & Collection Setup
    # ------------------------------------------------------------------

    def _build_client(self) -> PersistentClient:
        """Create and return a ChromaDB PersistentClient."""
        try:
            return chromadb.PersistentClient(
                path=self._persist_dir,
                settings=ChromaSettings(
                    anonymized_telemetry=False,   # Disable telemetry
                    allow_reset=True,             # Allow reset in tests
                ),
            )
        except Exception as exc:
            raise VectorStoreError(
                message=f"Failed to initialise ChromaDB client: {exc}",
                operation="init",
                collection=self._collection_name,
            ) from exc

    def _get_or_create_collection(self) -> Collection:
        """
        Get the merchant baselines collection, or create it with cosine
        distance metric if it doesn't yet exist.
        """
        try:
            collection = self._client.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},  # Use cosine distance for similarity
            )
            logger.debug(
                "ChromaDB collection ready",
                extra={
                    "collection": self._collection_name,
                    "count": collection.count(),
                },
            )
            return collection
        except Exception as exc:
            raise VectorStoreError(
                message=f"Failed to get/create ChromaDB collection: {exc}",
                operation="get_or_create_collection",
                collection=self._collection_name,
            ) from exc

    # ------------------------------------------------------------------
    # Baseline Operations
    # ------------------------------------------------------------------

    def store_baseline(
        self,
        merchant_id: str,
        embedding_vector: list[float],
        merchant_name: str = "",
        mock_site_key: str = "",
        text_snippet: str = "",
    ) -> str:
        """
        Upsert the baseline embedding for a merchant.

        Uses upsert (not add) so re-running onboarding overwrites the old baseline
        cleanly without duplicates.

        Parameters
        ----------
        merchant_id : str
            UUID of the merchant.
        embedding_vector : list[float]
            The 768-dimensional embedding from Gemini text-embedding-004.
        merchant_name : str
            Human-readable name for the metadata payload.
        mock_site_key : str
            The mock site key (e.g. "diya_store") for debugging.
        text_snippet : str
            First 200 chars of the baseline text for display/debugging.

        Returns
        -------
        str
            The document ID used in ChromaDB.
        """
        doc_id = f"{merchant_id}_baseline"
        metadata: dict[str, Any] = {
            _META_MERCHANT_ID: merchant_id,
            _META_SCAN_TYPE: "baseline",
            _META_MERCHANT_NAME: merchant_name,
            _META_MOCK_SITE_KEY: mock_site_key,
            _META_STORED_AT: datetime.now(tz=timezone.utc).isoformat(),
            _META_TEXT_SNIPPET: text_snippet[:200],
        }

        try:
            self._collection.upsert(
                ids=[doc_id],
                embeddings=[embedding_vector],
                metadatas=[metadata],
                documents=[text_snippet[:200]],  # Documents field for human readability
            )
        except Exception as exc:
            raise VectorStoreError(
                message=f"Failed to store baseline embedding for merchant {merchant_id}: {exc}",
                operation="upsert_baseline",
                collection=self._collection_name,
            ) from exc

        logger.info(
            "Baseline embedding stored",
            extra={
                "doc_id": doc_id,
                "merchant_id": merchant_id,
                "vector_dim": len(embedding_vector),
            },
        )
        return doc_id

    def get_baseline_vector(self, merchant_id: str) -> list[float]:
        """
        Retrieve the baseline embedding vector for a merchant.

        Parameters
        ----------
        merchant_id : str
            UUID of the merchant.

        Returns
        -------
        list[float]
            The 768-dimensional baseline embedding vector.

        Raises
        ------
        VectorStoreError
            If no baseline exists for the merchant.
        """
        doc_id = f"{merchant_id}_baseline"
        try:
            result = self._collection.get(
                ids=[doc_id],
                include=["embeddings"],
            )
        except Exception as exc:
            raise VectorStoreError(
                message=f"ChromaDB query failed for merchant {merchant_id}: {exc}",
                operation="get_baseline",
                collection=self._collection_name,
            ) from exc

        embeddings = result.get("embeddings")
        if embeddings is None or len(embeddings) == 0 or embeddings[0] is None:
            raise VectorStoreError(
                message=f"No baseline embedding found for merchant '{merchant_id}'. "
                        "Has this merchant been onboarded?",
                operation="get_baseline",
                collection=self._collection_name,
                merchant_id=merchant_id,
            )

        vector = list(embeddings[0])
        logger.debug(
            "Baseline vector retrieved",
            extra={"merchant_id": merchant_id, "vector_dim": len(vector)},
        )
        return vector

    def has_baseline(self, merchant_id: str) -> bool:
        """Return True if a baseline embedding exists for this merchant."""
        doc_id = f"{merchant_id}_baseline"
        try:
            result = self._collection.get(ids=[doc_id], include=[])
            return len(result.get("ids", [])) > 0
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Current Scan Operations
    # ------------------------------------------------------------------

    def store_current_scan(
        self,
        merchant_id: str,
        scan_id: str,
        embedding_vector: list[float],
        text_snippet: str = "",
    ) -> str:
        """
        Store the embedding from a current monitoring scan.

        These accumulate over time (one per scan per merchant), enabling
        drift trend analysis in future versions.
        """
        doc_id = f"{merchant_id}_current_{scan_id}"
        metadata: dict[str, Any] = {
            _META_MERCHANT_ID: merchant_id,
            _META_SCAN_TYPE: "current",
            _META_SCAN_ID: scan_id,
            _META_STORED_AT: datetime.now(tz=timezone.utc).isoformat(),
            _META_TEXT_SNIPPET: text_snippet[:200],
        }

        try:
            self._collection.upsert(
                ids=[doc_id],
                embeddings=[embedding_vector],
                metadatas=[metadata],
                documents=[text_snippet[:200]],
            )
        except Exception as exc:
            raise VectorStoreError(
                message=f"Failed to store current scan embedding: {exc}",
                operation="upsert_current",
                collection=self._collection_name,
            ) from exc

        logger.debug(
            "Current scan embedding stored",
            extra={"doc_id": doc_id, "merchant_id": merchant_id},
        )
        return doc_id

    # ------------------------------------------------------------------
    # Utility / Admin Operations
    # ------------------------------------------------------------------

    def delete_merchant_embeddings(self, merchant_id: str) -> int:
        """
        Delete all embeddings (baseline + all current scans) for a merchant.

        Returns the number of documents deleted.
        Useful when a merchant is offboarded.
        """
        try:
            results = self._collection.get(
                where={_META_MERCHANT_ID: {"$eq": merchant_id}},
                include=[],
            )
            ids_to_delete = results.get("ids", [])
            if ids_to_delete:
                self._collection.delete(ids=ids_to_delete)
            logger.info(
                "Merchant embeddings deleted",
                extra={"merchant_id": merchant_id, "deleted_count": len(ids_to_delete)},
            )
            return len(ids_to_delete)
        except Exception as exc:
            raise VectorStoreError(
                message=f"Failed to delete embeddings for merchant {merchant_id}: {exc}",
                operation="delete",
                collection=self._collection_name,
            ) from exc

    def collection_stats(self) -> dict[str, int]:
        """Return basic statistics about the collection."""
        try:
            total = self._collection.count()
            return {"total_documents": total}
        except Exception:
            return {"total_documents": -1}

    # ------------------------------------------------------------------
    # Async Variants (run sync calls in thread executor)
    # ------------------------------------------------------------------

    async def store_baseline_async(
        self,
        merchant_id: str,
        embedding_vector: list[float],
        **kwargs: str,
    ) -> str:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.store_baseline(merchant_id, embedding_vector, **kwargs),
        )

    async def get_baseline_vector_async(self, merchant_id: str) -> list[float]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.get_baseline_vector, merchant_id)

    async def store_current_scan_async(
        self,
        merchant_id: str,
        scan_id: str,
        embedding_vector: list[float],
        text_snippet: str = "",
    ) -> str:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.store_current_scan(merchant_id, scan_id, embedding_vector, text_snippet),
        )

    async def has_baseline_async(self, merchant_id: str) -> bool:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.has_baseline, merchant_id)


# ---------------------------------------------------------------------------
# Module-level Singleton
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_vector_store() -> VectorStoreClient:
    """
    Return the shared VectorStoreClient singleton.

    @lru_cache ensures ChromaDB is opened only once. Call this from
    anywhere in the application to get the same client instance.
    """
    return VectorStoreClient()
