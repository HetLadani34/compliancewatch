"""
mock_sites/server.py
---------------------
A FastAPI mini-application that simulates merchant websites.

This server exists solely to give Playwright a real HTTP endpoint to scrape,
avoiding bot-detection issues that come with targeting live websites.

Merchant Registry
-----------------
Each merchant entry defines:
  - clean_template  : The approved "State A" HTML template
  - fraud_template  : The post-fraud "State B" HTML template
  - active_state    : "clean" or "fraud" — controls what GET /merchant/{key}
                      returns without an explicit state param

Routes
------
GET  /merchant/{key}               → Returns whichever state is currently active
GET  /merchant/{key}/{state}       → Returns the explicit state ("clean" or "fraud")
POST /merchant/{key}/set-state     → Toggle state {"state": "clean"|"fraud"}
GET  /health                       → Liveness check
GET  /registry                     → Lists all merchants and their current states
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException, Path as PathParam
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

logger = logging.getLogger("compliance_watch.mock_sites")

# ---------------------------------------------------------------------------
# Template directory
# ---------------------------------------------------------------------------

TEMPLATES_DIR = Path(__file__).parent / "templates"


def _load_template(filename: str) -> str:
    """Read an HTML template file from disk. Raises on missing files."""
    path = TEMPLATES_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Mock site template not found: {path}")
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# In-memory merchant registry
# ---------------------------------------------------------------------------

MerchantState = Literal["clean", "fraud"]


class MerchantSiteConfig:
    """Holds template filenames and the currently active state for one merchant."""

    __slots__ = ("name", "clean_template", "fraud_template", "_active_state")

    def __init__(self, name: str, clean_template: str, fraud_template: str) -> None:
        self.name = name
        self.clean_template = clean_template
        self.fraud_template = fraud_template
        self._active_state: MerchantState = "clean"

    @property
    def active_state(self) -> MerchantState:
        return self._active_state

    @active_state.setter
    def active_state(self, value: MerchantState) -> None:
        if value not in ("clean", "fraud"):
            raise ValueError(f"Invalid state '{value}'. Must be 'clean' or 'fraud'.")
        self._active_state = value

    def get_html(self, state: MerchantState | None = None) -> str:
        """Return the HTML for the requested state (or the active state if None)."""
        target = state or self._active_state
        template_file = self.clean_template if target == "clean" else self.fraud_template
        return _load_template(template_file)


# Registry of all mock merchants
MERCHANT_REGISTRY: dict[str, MerchantSiteConfig] = {
    "diya_store": MerchantSiteConfig(
        name="Diya & Co. Handcrafts",
        clean_template="diya_store_clean.html",
        fraud_template="diya_store_fraud.html",
    ),
    "organic_tea": MerchantSiteConfig(
        name="PureLeaf Organic Tea",
        clean_template="organic_tea_clean.html",
        fraud_template="organic_tea_fraud.html",
    ),
    "handloom_crafts": MerchantSiteConfig(
        name="Weave & Wonder Handlooms",
        clean_template="handloom_clean.html",
        fraud_template="handloom_current.html",  # Control — stays clean
    ),
}

# ---------------------------------------------------------------------------
# FastAPI Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="ComplianceWatch — Mock Merchant Server",
    description="Serves simulated merchant websites for local scraping.",
    version="1.0.0",
    docs_url="/docs",
)


# ---------------------------------------------------------------------------
# Request / Response Models
# ---------------------------------------------------------------------------


class SetStateRequest(BaseModel):
    state: Literal["clean", "fraud"]


class MerchantRegistryEntry(BaseModel):
    key: str
    name: str
    active_state: str
    baseline_url: str
    current_url: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health", tags=["health"])
async def health_check() -> JSONResponse:
    """Simple liveness probe."""
    return JSONResponse({"status": "ok", "service": "mock-merchant-server"})


@app.get("/registry", tags=["admin"], response_model=list[MerchantRegistryEntry])
async def list_registry() -> list[MerchantRegistryEntry]:
    """
    Return the full merchant registry with each merchant's current active state.
    Used by the compliance agent to discover which merchants to onboard.
    """
    from config.settings import get_settings
    settings = get_settings()
    base = settings.mock_server_base_url

    return [
        MerchantRegistryEntry(
            key=key,
            name=config.name,
            active_state=config.active_state,
            baseline_url=f"{base}/merchant/{key}/clean",
            current_url=f"{base}/merchant/{key}",
        )
        for key, config in MERCHANT_REGISTRY.items()
    ]


@app.get(
    "/merchant/{merchant_key}",
    response_class=HTMLResponse,
    tags=["merchant-sites"],
    summary="Get current active state of a merchant site",
)
async def get_merchant_active(
    merchant_key: Annotated[str, PathParam(description="Merchant site key, e.g. 'diya_store'")],
) -> HTMLResponse:
    """Serve the merchant's currently active page (used during monitoring scans)."""
    config = MERCHANT_REGISTRY.get(merchant_key)
    if config is None:
        raise HTTPException(status_code=404, detail=f"Merchant '{merchant_key}' not found.")
    html_content = config.get_html()
    logger.debug("Serving merchant page", extra={"key": merchant_key, "state": config.active_state})
    return HTMLResponse(content=html_content)


@app.get(
    "/merchant/{merchant_key}/{state}",
    response_class=HTMLResponse,
    tags=["merchant-sites"],
    summary="Get a specific state of a merchant site",
)
async def get_merchant_by_state(
    merchant_key: Annotated[str, PathParam(description="Merchant site key")],
    state: Annotated[Literal["clean", "fraud"], PathParam(description="Which version to serve")],
) -> HTMLResponse:
    """
    Serve the explicit 'clean' or 'fraud' version of a merchant's site.
    Used during baseline ingestion to always grab the clean state.
    """
    config = MERCHANT_REGISTRY.get(merchant_key)
    if config is None:
        raise HTTPException(status_code=404, detail=f"Merchant '{merchant_key}' not found.")
    html_content = config.get_html(state=state)
    logger.debug(
        "Serving explicit merchant state",
        extra={"key": merchant_key, "requested_state": state},
    )
    return HTMLResponse(content=html_content)


@app.post(
    "/merchant/{merchant_key}/set-state",
    tags=["admin"],
    summary="Toggle a merchant's active state",
)
async def set_merchant_state(
    merchant_key: Annotated[str, PathParam(description="Merchant site key")],
    body: SetStateRequest,
) -> JSONResponse:
    """
    Switch a merchant's active state between 'clean' and 'fraud'.

    This endpoint simulates what happens in the real world when a fraudulent
    merchant secretly replaces their approved website with banned content.
    """
    config = MERCHANT_REGISTRY.get(merchant_key)
    if config is None:
        raise HTTPException(status_code=404, detail=f"Merchant '{merchant_key}' not found.")

    old_state = config.active_state
    config.active_state = body.state

    logger.info(
        "Merchant state toggled",
        extra={"key": merchant_key, "old_state": old_state, "new_state": body.state},
    )
    return JSONResponse(
        {
            "merchant_key": merchant_key,
            "old_state": old_state,
            "new_state": body.state,
            "message": f"Merchant '{merchant_key}' is now serving its '{body.state}' page.",
        }
    )
