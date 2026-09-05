"""
config/settings.py
------------------
Application-wide configuration loaded from the .env file.

Uses pydantic-settings for validated, type-safe environment variable
loading. A module-level singleton is provided via get_settings() so
that all parts of the application share the exact same configuration
instance (loaded and validated exactly once).
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent


class AppSettings(BaseSettings):
    """
    Centralised application configuration.

    All fields map 1-to-1 with entries in the .env file.
    Pydantic validates types and provides clear error messages when
    required fields are missing or malformed.
    """

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # Silently ignore unknown env vars
    )

    # ------------------------------------------------------------------
    # Gemini / AI
    # ------------------------------------------------------------------
    gemini_api_key: str = Field(
        default=os.environ.get("GEMINI_API_KEY", ""),
        description="Google Gemini API key (free tier). Get one at https://aistudio.google.com/app/apikey",
    )
    gemini_embedding_model: str = Field(
        default="models/text-embedding-004",
        description="Gemini embedding model identifier.",
    )
    gemini_vision_model: str = Field(
        default="gemini-1.5-flash",
        description="Gemini vision/chat model identifier.",
    )

    # ------------------------------------------------------------------
    # Database
    # ------------------------------------------------------------------
    database_url: str = Field(
        default="sqlite+aiosqlite:///data/compliance_watch.db",
        description="SQLAlchemy async database URL.",
    )

    # ------------------------------------------------------------------
    # ChromaDB
    # ------------------------------------------------------------------
    chroma_persist_dir: str = Field(
        default="data/chroma_db",
        description="Local directory where ChromaDB persists its data.",
    )
    chroma_collection_name: str = Field(
        default="merchant_baselines",
        description="Name of the ChromaDB collection storing baseline embeddings.",
    )

    # ------------------------------------------------------------------
    # Screenshot Storage
    # ------------------------------------------------------------------
    screenshot_dir: str = Field(
        default="data/screenshots",
        description="Root directory for saved merchant screenshots.",
    )

    # ------------------------------------------------------------------
    # Service Ports
    # ------------------------------------------------------------------
    mock_server_port: int = Field(
        default=8100,
        ge=1024,
        le=65535,
        description="Port for the local mock merchant website server.",
    )
    api_port: int = Field(
        default=8000,
        ge=1024,
        le=65535,
        description="Port for the FastAPI compliance backend.",
    )

    # ------------------------------------------------------------------
    # Agent Tuning
    # ------------------------------------------------------------------
    cosine_drift_threshold: float = Field(
        default=0.30,
        ge=0.0,
        le=1.0,
        description=(
            "Cosine distance above this value is treated as a semantic drift "
            "and triggers the vision analysis pipeline. Range: 0.0–1.0."
        ),
    )
    scan_interval_seconds: int = Field(
        default=300,
        ge=30,
        description="How often the background monitoring cron job runs (seconds).",
    )

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    log_level: str = Field(
        default="INFO",
        description="Python logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).",
    )
    log_file: str = Field(
        default="data/compliance_watch.log",
        description="Path to the application log file.",
    )

    # ------------------------------------------------------------------
    # Derived / Computed Properties
    # ------------------------------------------------------------------
    @property
    def baselines_screenshot_dir(self) -> Path:
        return Path(self.screenshot_dir) / "baselines"

    @property
    def current_screenshot_dir(self) -> Path:
        return Path(self.screenshot_dir) / "current"

    @property
    def mock_server_base_url(self) -> str:
        return f"http://127.0.0.1:{self.mock_server_port}"

    @property
    def api_base_url(self) -> str:
        return f"http://127.0.0.1:{self.api_port}"

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------
    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in allowed:
            raise ValueError(f"log_level must be one of {allowed}, got '{v}'")
        return upper

    def ensure_data_directories(self) -> None:
        """Create all required data directories if they do not already exist."""
        dirs_to_create = [
            Path(self.screenshot_dir),
            self.baselines_screenshot_dir,
            self.current_screenshot_dir,
            Path(self.chroma_persist_dir),
            Path(self.log_file).parent,
        ]
        for directory in dirs_to_create:
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """
    Return the application settings singleton.

    The @lru_cache decorator ensures .env is parsed only once per
    interpreter lifetime, making this function safe to call from
    anywhere in the codebase without performance concerns.
    """
    settings = AppSettings()  # type: ignore[call-arg]
    settings.ensure_data_directories()
    return settings
