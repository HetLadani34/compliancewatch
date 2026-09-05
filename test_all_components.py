"""
test_all_components.py
----------------------
End-to-end unit and integration validation for all ComplianceWatch components.
"""
import asyncio
import os
import sys
import unittest
import uuid
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

class TestComplianceWatch(unittest.IsolatedAsyncioTestCase):

    def test_01_config_and_settings(self):
        from config.settings import get_settings
        settings = get_settings()
        self.assertIsNotNone(settings)
        self.assertEqual(settings.mock_server_port, 8100)
        self.assertEqual(settings.api_port, 8000)
        self.assertTrue(Path("data").exists())

    def test_02_drift_detector_math(self):
        from core_agent.drift_detector import DriftDetector
        detector = DriftDetector(threshold=0.30)

        # Identical vectors -> 0.0 distance
        v1 = [1.0, 0.0, 0.0, 1.0]
        v2 = [1.0, 0.0, 0.0, 1.0]
        report_clean = detector.evaluate("test-clean", v1, v2)
        self.assertAlmostEqual(report_clean.cosine_distance, 0.0, places=4)
        self.assertFalse(report_clean.drift_detected)

        # Orthogonal / different vectors -> distance > threshold
        v3 = [0.0, 1.0, 1.0, 0.0]
        report_fraud = detector.evaluate("test-fraud", v1, v3)
        self.assertGreater(report_fraud.cosine_distance, 0.5)
        self.assertTrue(report_fraud.drift_detected)

    def test_03_mock_templates(self):
        from mock_sites.server import MERCHANT_REGISTRY
        self.assertIn("diya_store", MERCHANT_REGISTRY)
        self.assertIn("organic_tea", MERCHANT_REGISTRY)
        self.assertIn("handloom_crafts", MERCHANT_REGISTRY)

        for key, config in MERCHANT_REGISTRY.items():
            clean_html = config.get_html("clean")
            self.assertIn("<html", clean_html.lower())
            fraud_html = config.get_html("fraud")
            self.assertIn("<html", fraud_html.lower())

    async def test_04_database_and_repository(self):
        from database.engine import init_db, get_async_session
        from database.repository import MerchantRepository, ScanResultRepository
        from database.models import Merchant, RiskLevel

        await init_db()

        async with get_async_session() as session:
            m_repo = MerchantRepository(session)
            s_repo = ScanResultRepository(session)

            # Create test merchant
            unique_key = f"diya_store_{uuid.uuid4().hex[:6]}"
            merchant = await m_repo.create(
                name="Test Artisan Crafts",
                business_category="Handicrafts",
                mock_site_key=unique_key,
                registered_url=f"http://localhost:8100/merchant/{unique_key}",
            )
            self.assertIsNotNone(merchant.id)

            await m_repo.update_baseline(
                merchant_id=merchant.id,
                screenshot_path="data/screenshots/baselines/test.png",
                text_snippet="Handmade terracotta diyas and festive lamps.",
                ingested_at=datetime.now(tz=timezone.utc),
            )

            # Record scan result
            scan = await s_repo.create(
                merchant_id=merchant.id,
                current_url_scraped="http://localhost:8100/merchant/diya_store",
                current_screenshot_path="data/screenshots/current/test.png",
                cosine_distance=0.04,
                drift_detected=False,
                drift_threshold_used=0.30,
                vision_triggered=False,
                vision_result_json=None,
                risk_level=RiskLevel.GREEN,
            )
            self.assertIsNotNone(scan.id)

            # Fetch latest
            latest = await s_repo.get_latest_for_merchant(merchant.id)
            self.assertIsNotNone(latest)
            self.assertEqual(latest.risk_level, RiskLevel.GREEN)

    def test_05_chroma_vector_store(self):
        from vector_store.chroma_client import get_vector_store
        store = get_vector_store()
        self.assertIsNotNone(store)

        # Store test embedding
        dummy_vector = [0.05] * 768
        doc_id = store.store_baseline(
            merchant_id="test_merchant_123",
            embedding_vector=dummy_vector,
            merchant_name="Diya Test",
            mock_site_key="diya_store",
            text_snippet="Sample handmade crafts",
        )
        self.assertEqual(doc_id, "test_merchant_123_baseline")

        retrieved = store.get_baseline_vector("test_merchant_123")
        self.assertIsNotNone(retrieved)
        self.assertEqual(len(retrieved), 768)

        # Clean up test entry
        store.delete_merchant_embeddings("test_merchant_123")

    def test_06_ui_components_render(self):
        from ui.components.risk_card import render_merchant_card, render_summary_bar, render_drift_gauge
        from ui.components.screenshot_viewer import render_screenshot_comparison

        card = render_merchant_card(
            merchant_name="Diya & Co.",
            mock_site_key="diya_store",
            risk_level="RED",
            account_status="ACTIVE",
            cosine_variance_pct=75.4,
            last_scanned_at="2026-09-05T12:00:00",
            vision_triggered=True,
        )
        self.assertIn("Diya & Co.", card)
        self.assertIn("75.4%", card)

        summary = render_summary_bar(green=5, yellow=2, red=1, total=8)
        self.assertIn("Compliant", summary)
        self.assertIn("Violation", summary)

        gauge = render_drift_gauge(variance_pct=45.0, threshold_pct=30.0)
        self.assertIn("45.0%", gauge)

    async def test_07_fastapi_endpoints(self):
        from httpx import AsyncClient, ASGITransport
        from api.main import app
        from mock_sites.server import app as mock_app

        # Test mock server health
        async with AsyncClient(transport=ASGITransport(app=mock_app), base_url="http://test") as client:
            resp = await client.get("/health")
            self.assertEqual(resp.status_code, 200)
            self.assertEqual(resp.json()["status"], "ok")

            reg_resp = await client.get("/registry")
            self.assertEqual(reg_resp.status_code, 200)
            self.assertEqual(len(reg_resp.json()), 3)

        # Test main API health & root
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            root_resp = await client.get("/")
            self.assertEqual(root_resp.status_code, 200)

            health_resp = await client.get("/health")
            self.assertEqual(health_resp.status_code, 200)

            merchants_resp = await client.get("/api/merchants")
            self.assertEqual(merchants_resp.status_code, 200)
            self.assertIsInstance(merchants_resp.json(), list)

if __name__ == "__main__":
    unittest.main()
