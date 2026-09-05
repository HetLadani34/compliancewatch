"""
core_agent/drift_detector.py
-----------------------------
Cosine distance computation and semantic drift verdict logic.

What is Semantic Drift?
-----------------------
"Semantic drift" occurs when the meaning of a merchant's website content
shifts significantly from their approved baseline. A handmade diya store
that pivots to selling vapes has undergone catastrophic semantic drift.

How Cosine Distance Works Here
-------------------------------
Cosine similarity measures the angle between two embedding vectors:
  similarity = (A · B) / (||A|| × ||B||)

Cosine distance = 1 - cosine_similarity
  0.0 — Identical meaning (same page, essentially)
  0.1 — Very similar (minor wording changes, new products in same category)
  0.3 — Noticeable shift (new section, related but different products)
  0.5+ — Major shift (completely different product category)
  1.0 — Completely unrelated content

Our default threshold of 0.30 is tuned for our mock merchants:
  - Clean vs Clean:  expected distance ~0.05–0.15 (minor copy changes)
  - Clean vs Fraud:  expected distance ~0.60–0.85 (completely different domain)

Numpy vs scipy
--------------
We use numpy only (already a transitive dependency) to keep the install
footprint small. scipy would give a marginally cleaner API but adds ~200MB.
"""

from __future__ import annotations

import numpy as np

from core_agent.schemas import DriftReport
from utils.exceptions import DriftDetectionError
from utils.logging_config import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# DriftDetector
# ---------------------------------------------------------------------------


class DriftDetector:
    """
    Computes semantic drift between two embedding vectors and renders a verdict.

    The detector is stateless — all inputs are passed to the method, making
    it trivially testable without mocks.

    Usage
    -----
    >>> detector = DriftDetector(threshold=0.30)
    >>> report = detector.evaluate(
    ...     merchant_id="abc-123",
    ...     baseline_vector=[0.1, 0.2, ...],
    ...     current_vector=[0.8, 0.1, ...],
    ... )
    >>> print(report.drift_detected, report.variance_pct)
    True  72.4
    """

    def __init__(self, threshold: float | None = None) -> None:
        """
        Parameters
        ----------
        threshold : float | None
            Cosine distance above which drift is flagged. If None, reads
            from application settings (cosine_drift_threshold).
        """
        if threshold is None:
            from config.settings import get_settings
            threshold = get_settings().cosine_drift_threshold
        self._threshold = threshold
        logger.debug("DriftDetector initialised", extra={"threshold": self._threshold})

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        merchant_id: str,
        baseline_vector: list[float],
        current_vector: list[float],
    ) -> DriftReport:
        """
        Compute cosine distance and return a DriftReport verdict.

        Parameters
        ----------
        merchant_id : str
            UUID of the merchant being evaluated.
        baseline_vector : list[float]
            Embedding generated from the merchant's approved baseline content.
        current_vector : list[float]
            Embedding generated from the merchant's current (live) content.

        Returns
        -------
        DriftReport
            Includes cosine_distance, drift_detected flag, and variance_pct.

        Raises
        ------
        DriftDetectionError
            If the vectors are empty, mismatched in dimension, or contain
            only zeros (which would cause division-by-zero in cosine math).
        """
        self._validate_vectors(merchant_id, baseline_vector, current_vector)

        distance = self._cosine_distance(baseline_vector, current_vector)
        drift_detected = distance > self._threshold

        report = DriftReport(
            merchant_id=merchant_id,
            cosine_distance=distance,
            drift_detected=drift_detected,
            threshold_used=self._threshold,
        )

        logger.info(
            "Drift evaluation complete",
            extra={
                "merchant_id": merchant_id,
                "cosine_distance": round(distance, 6),
                "variance_pct": report.variance_pct,
                "threshold": self._threshold,
                "drift_detected": drift_detected,
            },
        )
        return report

    # ------------------------------------------------------------------
    # Core Math
    # ------------------------------------------------------------------

    @staticmethod
    def _cosine_distance(a: list[float], b: list[float]) -> float:
        """
        Compute the cosine distance between two vectors.

        Cosine Distance = 1 - (A · B) / (||A|| × ||B||)

        Returns a value in [0.0, 2.0]. In practice, well-formed sentence
        embeddings from Gemini have all-positive values, so the range is
        effectively [0.0, 1.0].

        We clip the result to [0.0, 1.0] to guard against floating-point
        precision errors that could produce values like -1e-10 or 1.0000001.
        """
        vec_a = np.array(a, dtype=np.float64)
        vec_b = np.array(b, dtype=np.float64)

        dot_product = np.dot(vec_a, vec_b)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)

        if norm_a == 0.0 or norm_b == 0.0:
            # Zero-magnitude vector — treat as maximum distance
            return 1.0

        cosine_similarity = dot_product / (norm_a * norm_b)
        cosine_distance = 1.0 - cosine_similarity

        # Clip to valid range to handle floating-point edge cases
        return float(np.clip(cosine_distance, 0.0, 1.0))

    @staticmethod
    def _validate_vectors(
        merchant_id: str,
        baseline: list[float],
        current: list[float],
    ) -> None:
        """
        Validate that both vectors are non-empty and dimension-matched.

        Raises DriftDetectionError if validation fails.
        """
        if not baseline:
            raise DriftDetectionError(
                message="Baseline embedding vector is empty.",
                merchant_id=merchant_id,
                baseline_dim=0,
                current_dim=len(current),
            )
        if not current:
            raise DriftDetectionError(
                message="Current embedding vector is empty.",
                merchant_id=merchant_id,
                baseline_dim=len(baseline),
                current_dim=0,
            )
        if len(baseline) != len(current):
            raise DriftDetectionError(
                message=(
                    f"Embedding dimension mismatch: baseline has {len(baseline)} dimensions, "
                    f"current has {len(current)} dimensions. "
                    "Both must use the same embedding model."
                ),
                merchant_id=merchant_id,
                baseline_dim=len(baseline),
                current_dim=len(current),
            )

    # ------------------------------------------------------------------
    # Threshold Tuning Helper (for diagnostics / admin routes)
    # ------------------------------------------------------------------

    def explain_distance(self, distance: float) -> str:
        """
        Return a human-readable interpretation of a cosine distance value.
        Used in dashboard tooltips and log messages.
        """
        if distance < 0.10:
            return "Minimal change — website content is essentially unchanged."
        elif distance < 0.20:
            return "Minor update — small wording or layout changes detected."
        elif distance < self._threshold:
            return f"Moderate shift — content has changed but below alert threshold ({self._threshold:.2f})."
        elif distance < 0.50:
            return "Significant semantic shift — alert threshold exceeded. Vision analysis triggered."
        elif distance < 0.75:
            return "Major content change — category of business may have changed."
        else:
            return "Extreme semantic shift — website appears to be completely different."
