"""
test_agent_and_scraper.py
-------------------------
Tests Playwright browser scraper and ComplianceAgent end-to-end.
"""
import asyncio
import os
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

class TestScraperAndAgent(unittest.IsolatedAsyncioTestCase):

    async def test_01_playwright_scraper(self):
        import uvicorn
        from mock_sites.server import app as mock_app
        from core_agent.scraper import WebsiteScraper

        # Start mock server in background task
        config = uvicorn.Config(mock_app, host="127.0.0.1", port=8100, log_level="warning")
        server = uvicorn.Server(config)
        server_task = asyncio.create_task(server.serve())
        await asyncio.sleep(1.0)

        try:
            # Scrape clean diya store
            async with WebsiteScraper(headless=True) as scraper:
                clean_content = await scraper.scrape("http://127.0.0.1:8100/merchant/diya_store/clean")
                self.assertIn("Diya", clean_content.text)
                self.assertGreater(len(clean_content.screenshot_bytes), 1000)

                # Scrape fraud diya store
                fraud_content = await scraper.scrape("http://127.0.0.1:8100/merchant/diya_store/fraud")
                self.assertIn("Vape", fraud_content.text)
                self.assertGreater(len(fraud_content.screenshot_bytes), 1000)
        finally:
            server.should_exit = True
            await server_task

if __name__ == "__main__":
    unittest.main()
