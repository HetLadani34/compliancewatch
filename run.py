"""
run.py
-------
ComplianceWatch Unified Process Launcher.

Starts all three services in parallel subprocess groups:

  Port 8100 — Mock Merchant Server (FastAPI / Uvicorn)
  Port 8000 — ComplianceWatch API   (FastAPI / Uvicorn)
  Port 8501 — Streamlit Dashboard

Handles:
  * Graceful shutdown on Ctrl+C (SIGINT) — kills all child processes cleanly.
  * Colour-coded prefixed log output per service.
  * Startup sequencing with health-check retries (mock server must be up
    before the API starts, so onboarding webhooks don't fail immediately).
  * Windows compatibility (no signal.SIGKILL; uses process group termination).

Usage
-----
  python run.py
  python run.py --no-browser      # Skip auto-opening the browser

Environment
-----------
  Requires the .env file to be present in the project root.
  Copy .env.example to .env and fill in your GEMINI_API_KEY.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
import webbrowser
from dataclasses import dataclass
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Colour codes for terminal output
# ---------------------------------------------------------------------------

RESET = "\033[0m"
BOLD = "\033[1m"

COLOURS = {
    "mock": "\033[36m",    # Cyan
    "api": "\033[34m",     # Blue
    "ui": "\033[35m",      # Magenta
    "launcher": "\033[33m", # Yellow
    "error": "\033[31m",   # Red
    "success": "\033[32m", # Green
}


def _log(service: str, message: str) -> None:
    colour = COLOURS.get(service, RESET)
    prefix = f"{colour}{BOLD}[{service.upper():^8}]{RESET}"
    print(f"{prefix} {message}", flush=True)


# ---------------------------------------------------------------------------
# Service Definitions
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable  # Use the same interpreter that's running run.py


@dataclass
class ServiceConfig:
    name: str
    command: list[str]
    port: int
    health_url: str | None
    startup_delay: float = 2.0      # Seconds to wait after launching before checking health


UI_PORT = int(os.environ.get("PORT", "8501"))

SERVICES: list[ServiceConfig] = [
    ServiceConfig(
        name="mock",
        command=[
            PYTHON, "-m", "uvicorn",
            "mock_sites.server:app",
            "--host", "0.0.0.0",
            "--port", "8100",
            "--log-level", "warning",
        ],
        port=8100,
        health_url="http://127.0.0.1:8100/registry",
        startup_delay=3.0,
    ),
    ServiceConfig(
        name="api",
        command=[
            PYTHON, "-m", "uvicorn",
            "api.main:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--log-level", "info",
        ],
        port=8000,
        health_url="http://127.0.0.1:8000/health",
        startup_delay=4.0,
    ),
    ServiceConfig(
        name="ui",
        command=[
            PYTHON, "-m", "streamlit",
            "run", "ui/app.py",
            "--server.port", str(UI_PORT),
            "--server.address", "0.0.0.0",
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false",
            "--server.enableCORS", "false",
            "--server.enableXsrfProtection", "false",
        ],
        port=UI_PORT,
        health_url=None,  # Streamlit doesn't have a simple JSON health endpoint
        startup_delay=5.0,
    ),
]


# ---------------------------------------------------------------------------
# Process Manager
# ---------------------------------------------------------------------------

class ProcessManager:
    """
    Manages child subprocesses for all three services.

    Each process's stdout/stderr is forwarded to the parent's stdout with a
    coloured service prefix, running in a dedicated daemon thread.
    """

    def __init__(self) -> None:
        self._processes: dict[str, subprocess.Popen] = {}
        self._threads: list[threading.Thread] = []

    def start(self, service: ServiceConfig) -> None:
        """Launch a service subprocess and start its log-forwarding thread."""
        _log("launcher", f"Starting {service.name} on port {service.port}...")

        proc = subprocess.Popen(
            service.command,
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            bufsize=1,
            universal_newlines=True,
        )
        self._processes[service.name] = proc

        # Daemon thread forwards subprocess output with service prefix
        thread = threading.Thread(
            target=self._forward_output,
            args=(proc, service.name),
            daemon=True,
            name=f"log-{service.name}",
        )
        thread.start()
        self._threads.append(thread)

    def _forward_output(self, proc: subprocess.Popen, service_name: str) -> None:
        """Read lines from subprocess stdout and print them with a prefix."""
        assert proc.stdout is not None
        try:
            for line in proc.stdout:
                line = line.rstrip()
                if line:
                    _log(service_name, line)
        except ValueError:
            pass  # Pipe closed — subprocess exited

    def stop_all(self) -> None:
        """Send termination signal to all child processes."""
        _log("launcher", "Shutting down all services...")
        for name, proc in self._processes.items():
            if proc.poll() is None:
                _log("launcher", f"Terminating {name} (PID {proc.pid})...")
                proc.terminate()

        # Give processes 5s to exit gracefully before force-killing
        deadline = time.time() + 5
        for name, proc in self._processes.items():
            remaining = max(0.0, deadline - time.time())
            try:
                proc.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                _log("launcher", f"Force-killing {name} (PID {proc.pid})...")
                proc.kill()

    def all_running(self) -> bool:
        """Return True if all managed processes are still alive."""
        return all(proc.poll() is None for proc in self._processes.values())

    def any_died(self) -> str | None:
        """Return the name of the first process that has exited, or None."""
        for name, proc in self._processes.items():
            if proc.poll() is not None:
                return name
        return None


# ---------------------------------------------------------------------------
# Health Check
# ---------------------------------------------------------------------------

def _wait_for_health(url: str, service_name: str, timeout: float = 30.0) -> bool:
    """Poll a health URL until it returns HTTP 200 or timeout expires."""
    import urllib.request
    import urllib.error

    deadline = time.time() + timeout
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        try:
            with urllib.request.urlopen(url, timeout=3) as resp:
                if resp.status == 200:
                    _log("launcher", f"✅ {service_name} is healthy ({url})")
                    return True
        except Exception:
            pass
        _log("launcher", f"⏳ Waiting for {service_name}... (attempt {attempt})")
        time.sleep(2)
    _log("error", f"❌ {service_name} did not become healthy within {timeout:.0f}s")
    return False


# ---------------------------------------------------------------------------
# Pre-flight Checks
# ---------------------------------------------------------------------------

def _check_env() -> bool:
    """Verify that GEMINI_API_KEY is set via environment or .env file."""
    # 1. Check direct environment variable (e.g. Docker, Hugging Face Spaces, Render)
    env_key = os.environ.get("GEMINI_API_KEY")
    if env_key and env_key.strip() and env_key.strip() != "your_gemini_api_key_here":
        return True

    # 2. Check .env file if present
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line.startswith("GEMINI_API_KEY=") and len(line) > len("GEMINI_API_KEY="):
                    val = line.split("=", 1)[1].strip().strip('"').strip("'")
                    if val and val != "your_gemini_api_key_here":
                        return True

    _log("error", "GEMINI_API_KEY is not set!")
    _log("error", "Set it in your .env file or export GEMINI_API_KEY in your environment.")
    _log("error", "Get your free key at: https://aistudio.google.com/apikey")
    return False


def _check_dependencies() -> bool:
    """Quick check that key packages are importable."""
    required = [
        ("fastapi", "fastapi"),
        ("uvicorn", "uvicorn"),
        ("streamlit", "streamlit"),
        ("playwright", "playwright"),
        ("chromadb", "chromadb"),
        ("google.generativeai", "google-generativeai"),
        ("sqlalchemy", "sqlalchemy"),
        ("pydantic", "pydantic"),
    ]
    missing = []
    for import_name, pip_name in required:
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pip_name)

    if missing:
        _log("error", f"Missing packages: {', '.join(missing)}")
        _log("error", "Run: pip install -r requirements.txt")
        return False
    return True


def _check_playwright_browsers() -> bool:
    """Verify that Playwright Chromium browser is installed."""
    try:
        result = subprocess.run(
            [PYTHON, "-m", "playwright", "install", "--dry-run", "chromium"],
            capture_output=True, text=True, timeout=10,
        )
        if "chromium" in result.stdout.lower() or result.returncode == 0:
            return True
    except Exception:
        pass

    _log("launcher", "Installing Playwright Chromium browser (one-time setup)...")
    try:
        subprocess.run(
            [PYTHON, "-m", "playwright", "install", "chromium"],
            check=True, timeout=180,
        )
        _log("success", "Playwright Chromium installed successfully.")
        return True
    except subprocess.CalledProcessError:
        _log("error", "Failed to install Playwright Chromium.")
        _log("error", "Run manually: playwright install chromium")
        return False


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> int:
    """Launch all services and block until Ctrl+C."""
    print()
    _log("launcher", "=" * 55)
    _log("launcher", "  🛡️  ComplianceWatch — AI Risk Manager")
    _log("launcher", "  Razorpay AI Buildathon 2026")
    _log("launcher", "=" * 55)
    print()

    # Pre-flight
    _log("launcher", "Running pre-flight checks...")
    if not _check_env():
        return 1
    if not _check_dependencies():
        return 1
    _log("success", "Pre-flight checks passed.")

    # Playwright browser install (non-blocking if already installed)
    _check_playwright_browsers()

    manager = ProcessManager()

    try:
        # --- Start Mock Server first ---
        mock_cfg = next(s for s in SERVICES if s.name == "mock")
        manager.start(mock_cfg)
        time.sleep(mock_cfg.startup_delay)
        if mock_cfg.health_url:
            _wait_for_health(mock_cfg.health_url, "Mock Server", timeout=20)

        # --- Start ComplianceWatch API ---
        api_cfg = next(s for s in SERVICES if s.name == "api")
        manager.start(api_cfg)
        time.sleep(api_cfg.startup_delay)
        if api_cfg.health_url:
            _wait_for_health(api_cfg.health_url, "ComplianceWatch API", timeout=30)

        # --- Start Streamlit UI ---
        ui_cfg = next(s for s in SERVICES if s.name == "ui")
        manager.start(ui_cfg)
        time.sleep(ui_cfg.startup_delay)

        # --- Print startup summary ---
        print()
        _log("success", "=" * 55)
        _log("success", "  All services are running!")
        _log("success", "")
        _log("success", f"  🌐 Dashboard : http://localhost:{ui_cfg.port}")
        _log("success", f"  📡 API       : http://localhost:{api_cfg.port}/docs")
        _log("success", f"  🏪 Mock Sites: http://localhost:{mock_cfg.port}/registry")
        _log("success", "")
        _log("success", "  Press Ctrl+C to stop all services.")
        _log("success", "=" * 55)
        print()

        # Open browser automatically
        if not args.no_browser:
            time.sleep(2)
            webbrowser.open(f"http://localhost:{ui_cfg.port}")

        # --- Monitor loop ---
        while True:
            time.sleep(3)
            dead = manager.any_died()
            if dead:
                _log("error", f"Service '{dead}' exited unexpectedly. Shutting down.")
                break

    except KeyboardInterrupt:
        print()
        _log("launcher", "Ctrl+C received.")
    finally:
        manager.stop_all()
        _log("launcher", "All services stopped. Goodbye.")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ComplianceWatch unified launcher")
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not auto-open the dashboard in a browser.",
    )
    parsed = parser.parse_args()
    sys.exit(main(parsed))
