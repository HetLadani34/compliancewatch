---
title: ComplianceWatch AI Risk Manager
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8501
pinned: false
---

# 🛡️ ComplianceWatch — AI Risk Manager

> **Razorpay AI Buildathon 2026 · Track: AI Risk Manager**

ComplianceWatch is an AI-powered merchant compliance monitoring system that detects **Transaction Laundering** — the practice of fraudsters obtaining a payment gateway account with a clean website (e.g., selling handmade diyas) and then switching the website to sell banned content (vapes, crypto trading, etc.) after approval.

---

## 📐 How It Works

```
Merchant Onboarding (One-time)
──────────────────────────────
Mock Site (State A — Clean)
  → Playwright Scraper       captures text + full-page PNG
  → Gemini text-embedding-004 → 768-dim semantic vector
  → ChromaDB                 stores baseline vector
  → SQLite                   stores merchant + baseline metadata

Periodic Monitoring (Every N minutes)
──────────────────────────────────────
Mock Site (Live State)
  → Playwright Scraper       captures current text + PNG
  → Gemini text-embedding-004 → 768-dim current vector
  → ChromaDB (retrieve baseline)
  → DriftDetector            cosine_distance > threshold? 
      ↓ YES (Drift Detected)
      → Gemini 1.5 Flash     multimodal vision analysis
      → VisionAnalysisResult (Pydantic-gated JSON)
      → SQLite               persist ScanResult + Alert
      → Dashboard            RED/YELLOW/GREEN badge
```

---

## 🧱 Architecture

```
compliancewatch/
├── config/
│   └── settings.py          # Pydantic BaseSettings singleton
├── utils/
│   ├── exceptions.py        # 8-class custom exception hierarchy
│   └── logging_config.py    # Structured JSON + colour console logging
├── database/
│   ├── engine.py            # Async SQLAlchemy engine + session factory
│   ├── models.py            # Merchant / ScanResult / Alert ORM models
│   └── repository.py        # Repository pattern CRUD layer
├── core_agent/
│   ├── schemas.py           # Pydantic gate for all inter-module data
│   ├── scraper.py           # Async Playwright scraper
│   ├── embedding_service.py # Gemini text-embedding-004 wrapper
│   ├── drift_detector.py    # NumPy cosine distance + threshold logic
│   ├── vision_analyzer.py   # Gemini 1.5 Flash multimodal AUP checker
│   └── compliance_agent.py  # Main orchestrator (onboard + scan + suspend)
├── vector_store/
│   └── chroma_client.py     # ChromaDB persistent client
├── mock_sites/
│   ├── server.py            # FastAPI mock merchant server (port 8100)
│   └── templates/           # 6 HTML templates (clean + fraud × 3)
├── api/
│   ├── main.py              # FastAPI app + APScheduler background scanner
│   └── routes/
│       ├── merchants.py     # CRUD + onboard + suspend routes
│       ├── scanning.py      # Manual scan trigger routes
│       └── webhooks.py      # Razorpay webhook simulation + demo controls
├── ui/
│   ├── app.py               # Streamlit entry point + global CSS
│   ├── components/
│   │   ├── risk_card.py     # Merchant risk cards + summary stats bar
│   │   └── screenshot_viewer.py  # Side-by-side screenshot comparison
│   └── pages/
│       ├── 01_dashboard.py  # Risk overview grid + scan controls
│       └── 02_merchant_detail.py # Deep-dive analysis + Gemini verdict
├── data/                    # Auto-created: DB, screenshots, ChromaDB, logs
├── run.py                   # 🚀 Unified process launcher
├── requirements.txt
└── .env.example
```

---

## ⚙️ Setup Guide

### 1. Prerequisites

- Python 3.11 or 3.12
- pip (or pip3)
- A free Google Gemini API key (see below)

### 2. Clone & Create Virtual Environment

```bash
# Clone the repository
git clone <your-repo-url>
cd compliancewatch

# Create and activate a virtual environment
python -m venv .venv

# On Windows:
.venv\Scripts\activate

# On macOS/Linux:
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Get Your FREE Gemini API Key

ComplianceWatch uses Google's **Gemini API** for both vision analysis and text embeddings. The free tier is more than sufficient for this project.

**Steps to get your free key:**

1. Go to **[https://aistudio.google.com/apikey](https://aistudio.google.com/apikey)**
2. Sign in with your Google account (no credit card required).
3. Click **"Create API Key"** → Select a Google Cloud project (or create a new one).
4. Copy the generated API key.

> **Free tier limits:**  
> • `gemini-1.5-flash` — 1,500 RPM, 1M tokens/min  
> • `text-embedding-004` — 1,500 RPM, 1M tokens/min  
> Both limits are far above what this demo requires.

### 5. Configure the `.env` File

```bash
# Copy the example configuration file
copy .env.example .env      # Windows
cp .env.example .env        # macOS/Linux
```

Open `.env` in any text editor and fill in your API key:

```env
# ── REQUIRED: Your Gemini API key ──────────────────────────────────
GEMINI_API_KEY=AIzaSy...your_key_here...

# ── Optional: Tune these if needed ─────────────────────────────────
COSINE_DRIFT_THRESHOLD=0.30   # Sensitivity (lower = more sensitive)
SCAN_INTERVAL_SECONDS=300     # Background scan interval (5 min default)
LOG_LEVEL=INFO
```

> **Security Note:** `.env` is listed in `.gitignore` and will never be committed to version control.

### 6. Install Playwright Browser (one-time)

```bash
playwright install chromium
```

> The unified launcher (`run.py`) will also attempt this automatically if Chromium is not detected.

---

## 🚀 Running the Project

### Option A: Unified Launcher (Recommended)

The `run.py` launcher starts all three services with a single command and handles sequenced startup, health checks, and graceful shutdown.

```bash
python run.py
```

This starts:
| Service | Port | URL |
|---|---|---|
| 🏪 Mock Merchant Server | 8100 | http://localhost:8100/registry |
| 📡 ComplianceWatch API | 8000 | http://localhost:8000/docs |
| 🎨 Streamlit Dashboard | 8501 | http://localhost:8501 |

The browser opens automatically. Press **Ctrl+C** to stop all services.

```bash
# Optional: suppress auto browser open
python run.py --no-browser
```

### Option B: Manual Start (3 terminals)

```bash
# Terminal 1 — Mock Sites
uvicorn mock_sites.server:app --port 8100 --reload

# Terminal 2 — ComplianceWatch API
uvicorn api.main:app --port 8000 --reload

# Terminal 3 — Streamlit Dashboard
streamlit run ui/app.py
```

---

## 🎬 Demo Walkthrough

Once all services are running, follow these steps in the **Streamlit Dashboard**:

### Step 1: Onboard Demo Merchants
Click **🚀 Onboard Demo Merchants** on the dashboard.

This scrapes the three clean mock websites and stores their baseline embeddings. It will take ~30–60 seconds (Playwright browser startup + 3 Gemini embedding calls).

You should see all 3 merchants appear with **🟢 COMPLIANT** badges.

### Step 2: Simulate Fraud
Click **🔴 Simulate Fraud (All)** to switch all mock websites to their "fraud" versions:
- **Diya & Co.** → VapeZone India (e-cigarettes, pod systems)
- **PureLeaf Organic Tea** → CryptoPulse Exchange (crypto trading)
- **Weave & Wonder** → Stays clean (control merchant)

### Step 3: Run a Compliance Scan
Click **▶ Run Full Scan Now**.

The AI pipeline runs for each merchant:
1. Playwright scrapes the live (now fraudulent) pages
2. Gemini embeddings are compared to baselines
3. Cosine distance > 0.30 → Vision analysis triggered
4. Gemini 1.5 Flash examines the screenshot + text
5. Structured JSON verdict returned

After ~60–120 seconds you'll see:
- 🔴 **Diya & Co.** — RED · VAPE_ECIG detected
- 🔴 **PureLeaf Organic Tea** — RED · CRYPTO_TRADING detected  
- 🟢 **Weave & Wonder** — GREEN · No violation

### Step 4: Investigate a Flagged Merchant
Click **🔍 Investigate →** on a RED merchant to see:
- Side-by-side baseline vs current screenshots
- Gemini's reasoning and visual evidence list
- Cosine drift gauge
- Full scan history table

### Step 5: Suspend Account
On the detail page, click **🚫 Suspend Account** to simulate freezing the merchant's Razorpay access.

---

## 🔧 Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `GEMINI_API_KEY` | **Required** | Gemini API key from Google AI Studio |
| `GEMINI_EMBEDDING_MODEL` | `models/text-embedding-004` | Embedding model |
| `GEMINI_VISION_MODEL` | `gemini-1.5-flash` | Vision analysis model |
| `DATABASE_URL` | `sqlite+aiosqlite:///data/compliance_watch.db` | Database path |
| `CHROMA_PERSIST_DIR` | `data/chroma_db` | ChromaDB storage directory |
| `COSINE_DRIFT_THRESHOLD` | `0.30` | Drift sensitivity (0.0–1.0) |
| `SCAN_INTERVAL_SECONDS` | `300` | Background scan frequency |
| `MOCK_SERVER_PORT` | `8100` | Mock site server port |
| `API_PORT` | `8000` | FastAPI backend port |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| Vision AI | Gemini 1.5 Flash (multimodal) |
| Embeddings | Gemini text-embedding-004 (768-dim) |
| Vector Store | ChromaDB (persistent, local) |
| Database | SQLite + SQLAlchemy (async) |
| Backend API | FastAPI + APScheduler |
| Web Scraping | Playwright (headless Chromium) |
| Dashboard | Streamlit |
| Config | Pydantic-Settings |
| Data Validation | Pydantic v2 |

---

## 📁 Data Directory

All runtime data is stored in `data/` (gitignored):

```
data/
├── compliance_watch.db        # SQLite database
├── compliance_watch.log       # Structured application logs
├── chroma_db/                 # ChromaDB vector store
└── screenshots/
    ├── baselines/             # Baseline (State A) screenshots
    └── current/               # Latest scan screenshots
```

---

## 🐛 Troubleshooting

**"GEMINI_API_KEY is not set"**  
→ Ensure `.env` exists and has a valid API key (not the placeholder value).

**"Mock server is not reachable"**  
→ Make sure port 8100 is free. Run `run.py` or start mock server manually.

**"No baseline embedding found"**  
→ Click "Onboard Demo Merchants" first before running scans.

**Playwright browser not found**  
→ Run `playwright install chromium` manually.

**Gemini 429 Rate Limit**  
→ The free tier allows 1,500 RPM. If you hit limits during a full scan, increase `SCAN_INTERVAL_SECONDS` in `.env`.

---

## 📜 License

MIT License. Built for the Razorpay AI Buildathon 2026.
