"""
api/main.py
------------
FastAPI application entry point for ComplianceWatch.

Responsibilities
----------------
* Database initialisation on startup (create_all tables).
* CORS middleware so the Streamlit frontend can call the API.
* Mount all route modules under /api/...
* Background APScheduler cron that runs run_full_scan() on a configurable
  interval — this is the "automatic monitoring" half of the system.
* A shared ComplianceAgent singleton injected via FastAPI dependency.

Lifespan Pattern
----------------
We use FastAPI's asynccontextmanager lifespan (the modern replacement for
on_event("startup")) so startup/shutdown logic is co-located and async-safe.
"""

import asyncio
import logging
import sys
from contextlib import asynccontextmanager

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
from functools import lru_cache
from typing import AsyncGenerator

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routes import merchants, scanning, webhooks
from config.settings import get_settings
from core_agent.compliance_agent import ComplianceAgent
from database.engine import init_db
from utils.logging_config import configure_logging, get_logger

# ---------------------------------------------------------------------------
# Bootstrap logging as early as possible
# ---------------------------------------------------------------------------
settings = get_settings()
configure_logging(log_level=settings.log_level, log_file=settings.log_file)
logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Shared ComplianceAgent singleton
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_compliance_agent() -> ComplianceAgent:
    """Return the shared ComplianceAgent instance (created once)."""
    return ComplianceAgent()


# ---------------------------------------------------------------------------
# Background Scheduler
# ---------------------------------------------------------------------------

_scheduler: AsyncIOScheduler | None = None


async def _run_scheduled_scan() -> None:
    """
    Background task executed by APScheduler on each interval tick.
    Errors are caught here so a failing scan never kills the scheduler.
    """
    agent = get_compliance_agent()
    logger.info("Scheduled scan triggered by APScheduler.")
    try:
        summary = await agent.run_full_scan()
        logger.info(
            "Scheduled scan complete",
            extra={
                "green": summary.green_count,
                "yellow": summary.yellow_count,
                "red": summary.red_count,
                "errors": len(summary.errors),
            },
        )
    except Exception as exc:
        logger.error("Scheduled scan failed with unhandled exception.", exc_info=True)


# ---------------------------------------------------------------------------
# Lifespan (startup + shutdown)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    FastAPI lifespan context manager.

    On startup:
      1. Initialise the SQLite database (create tables).
      2. Start the APScheduler background scan job.

    On shutdown:
      1. Shut down the scheduler gracefully.
    """
    global _scheduler

    logger.info("ComplianceWatch API starting up...")

    # 1 — Initialise database
    await init_db()
    logger.info("Database initialised.")

    # 2 — Start background scheduler
    _scheduler = AsyncIOScheduler(timezone="UTC")
    _scheduler.add_job(
        func=_run_scheduled_scan,
        trigger=IntervalTrigger(seconds=settings.scan_interval_seconds),
        id="compliance_scan",
        name="Periodic Compliance Scan",
        replace_existing=True,
        misfire_grace_time=60,
    )
    _scheduler.start()
    logger.info(
        "Background scan scheduler started.",
        extra={"interval_seconds": settings.scan_interval_seconds},
    )

    yield  # --- Application runs here ---

    # Shutdown
    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=False)
        logger.info("Scheduler shut down.")
    logger.info("ComplianceWatch API shut down.")


# ---------------------------------------------------------------------------
# FastAPI Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="ComplianceWatch API",
    description=(
        "AI-powered merchant compliance monitoring system for Razorpay. "
        "Detects semantic drift and policy violations via multimodal AI analysis."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# --- CORS: allow Streamlit (port 8501) and any localhost origin ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "http://127.0.0.1:8501",
        "http://localhost:3000",
        "*",  # Permissive for local dev; tighten for production
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Mount route modules ---
app.include_router(merchants.router, prefix="/api/merchants", tags=["Merchants"])
app.include_router(scanning.router, prefix="/api/scan", tags=["Scanning"])
app.include_router(webhooks.router, prefix="/api/webhooks", tags=["Webhooks"])


# ---------------------------------------------------------------------------
# Health & Root Endpoints
# ---------------------------------------------------------------------------

@app.get("/", tags=["health"])
async def root() -> JSONResponse:
    return JSONResponse({
        "service": "ComplianceWatch API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
    })


@app.get("/health", tags=["health"])
async def health_check() -> JSONResponse:
    """Liveness probe for the API server."""
    scheduler_status = "running" if (_scheduler and _scheduler.running) else "stopped"
    return JSONResponse({
        "status": "healthy",
        "scheduler": scheduler_status,
        "scan_interval_seconds": settings.scan_interval_seconds,
    })


@app.get("/api/scheduler/status", tags=["admin"])
async def scheduler_status() -> JSONResponse:
    """Return APScheduler job status."""
    if _scheduler is None:
        return JSONResponse({"running": False, "jobs": []})
    jobs = [
        {
            "id": job.id,
            "name": job.name,
            "next_run_time": str(job.next_run_time) if job.next_run_time else None,
        }
        for job in _scheduler.get_jobs()
    ]
    return JSONResponse({"running": _scheduler.running, "jobs": jobs})
