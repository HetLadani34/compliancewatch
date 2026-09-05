"""
utils/logging_config.py
------------------------
Structured, production-grade logging configuration for ComplianceWatch.

Design decisions
----------------
* Two handlers: console (human-readable ColourFormatter) and file (JSON lines).
* JSON file logs are machine-parseable and ready for ingestion into any log
  aggregation platform (ELK, Grafana Loki, etc.) without extra processing.
* A module-level factory function `get_logger()` is the single entry-point
  that all other modules should use — no bare `logging.getLogger(__name__)`
  scattered across the codebase.
* Configuration is applied lazily on first call and is idempotent (calling
  configure_logging() multiple times is safe).
"""

from __future__ import annotations

import json
import logging
import logging.config
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# JSON Log Formatter
# ---------------------------------------------------------------------------


class JsonLineFormatter(logging.Formatter):
    """
    Formats each log record as a single JSON object on its own line.

    Output fields
    -------------
    timestamp   — ISO-8601 UTC timestamp
    level       — Log level name (INFO, WARNING, etc.)
    logger      — Logger name (typically the module's __name__)
    message     — The formatted log message
    module      — Source module name
    function    — Source function name
    line        — Source line number
    exc_info    — Exception traceback (only when an exception is attached)
    **extra     — Any extra key-value pairs passed via the `extra` argument
    """

    EXCLUDED_LOGRECORD_ATTRS: frozenset[str] = frozenset(
        {
            "args", "asctime", "created", "exc_info", "exc_text", "filename",
            "funcName", "id", "levelname", "levelno", "lineno", "module",
            "msecs", "message", "msg", "name", "pathname", "process",
            "processName", "relativeCreated", "stack_info", "thread",
            "threadName", "taskName",
        }
    )

    def format(self, record: logging.LogRecord) -> str:
        record.message = record.getMessage()

        log_dict: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.message,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Attach exception info if present
        if record.exc_info:
            log_dict["exc_info"] = self.formatException(record.exc_info)

        # Attach any extra fields the caller passed in via extra={}
        for key, value in record.__dict__.items():
            if key not in self.EXCLUDED_LOGRECORD_ATTRS:
                try:
                    json.dumps(value)  # Only include JSON-serialisable values
                    log_dict[key] = value
                except (TypeError, ValueError):
                    log_dict[key] = str(value)

        return json.dumps(log_dict, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Colour Console Formatter
# ---------------------------------------------------------------------------

_LEVEL_COLOURS: dict[str, str] = {
    "DEBUG":    "\033[36m",   # Cyan
    "INFO":     "\033[32m",   # Green
    "WARNING":  "\033[33m",   # Yellow
    "ERROR":    "\033[31m",   # Red
    "CRITICAL": "\033[35m",   # Magenta
}
_RESET = "\033[0m"


class ColourConsoleFormatter(logging.Formatter):
    """Human-readable, colour-coded formatter for the console handler."""

    FMT = "%(asctime)s | {colour}%(levelname)-8s{reset} | %(name)s:%(lineno)d | %(message)s"
    DATE_FMT = "%H:%M:%S"

    def format(self, record: logging.LogRecord) -> str:
        colour = _LEVEL_COLOURS.get(record.levelname, "")
        formatter = logging.Formatter(
            fmt=self.FMT.format(colour=colour, reset=_RESET),
            datefmt=self.DATE_FMT,
        )
        return formatter.format(record)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_LOGGING_CONFIGURED = False


def configure_logging(log_level: str = "INFO", log_file: str = "data/compliance_watch.log") -> None:
    """
    Initialise the logging system.

    This function is idempotent — subsequent calls after the first are no-ops.
    Call this once at application startup (e.g., in FastAPI's lifespan handler
    and at the top of the Streamlit entry point).

    Parameters
    ----------
    log_level : str
        Minimum log level for the console handler (e.g., "DEBUG", "INFO").
    log_file : str
        Path to the JSON log file. Parent directories are created if needed.
    """
    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED:
        return

    # Ensure log directory exists
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    config: dict[str, Any] = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "console": {
                "()": ColourConsoleFormatter,
            },
            "json_file": {
                "()": JsonLineFormatter,
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
                "formatter": "console",
                "level": log_level,
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "filename": str(log_path),
                "maxBytes": 10 * 1024 * 1024,  # 10 MB per file
                "backupCount": 5,
                "formatter": "json_file",
                "level": "DEBUG",  # File always captures everything
                "encoding": "utf-8",
            },
        },
        "root": {
            "level": "DEBUG",
            "handlers": ["console", "file"],
        },
        "loggers": {
            # Quieten noisy third-party loggers
            "uvicorn.access": {"level": "WARNING"},
            "uvicorn.error": {"level": "INFO"},
            "httpx": {"level": "WARNING"},
            "httpcore": {"level": "WARNING"},
            "chromadb": {"level": "WARNING"},
            "google.generativeai": {"level": "WARNING"},
            "playwright": {"level": "WARNING"},
        },
    }

    logging.config.dictConfig(config)
    _LOGGING_CONFIGURED = True

    # Emit a startup message so we know logging is alive
    startup_logger = logging.getLogger("compliance_watch.startup")
    startup_logger.info(
        "Logging configured",
        extra={"log_level": log_level, "log_file": str(log_path)},
    )


def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger, prefixed with 'compliance_watch.' for easy
    filtering in log aggregators.

    Usage
    -----
    >>> logger = get_logger(__name__)
    >>> logger.info("Scraping merchant", extra={"merchant_id": "abc123"})

    Parameters
    ----------
    name : str
        Typically pass `__name__` from the calling module. The 'compliance_watch.'
        prefix will be prepended automatically if not already present.
    """
    prefix = "compliance_watch"
    qualified_name = name if name.startswith(prefix) else f"{prefix}.{name}"
    return logging.getLogger(qualified_name)
