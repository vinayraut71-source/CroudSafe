# CrowdSafe AI — Intelligent Crowd Safety & Stampede Prevention System

> An enterprise-grade, real-time computer vision and multimodal AI platform for crowd density monitoring, movement telemetry, stampede prevention, automated video analysis, and conversational crowd intelligence.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Prerequisites & Requirements](#prerequisites--requirements)
- [Installation & Setup](#installation--setup)
- [Configuration](#configuration)
- [Running the Application](#running-the-application)
- [Web Dashboard & Interfaces](#web-dashboard--interfaces)
- [Video Upload & Automated Analysis](#video-upload--automated-analysis)
- [Gemini Multimodal Intelligence](#gemini-multimodal-intelligence)
- [API Reference](#api-reference)
- [Project Directory Structure](#project-directory-structure)
- [Automated Testing](#automated-testing)
- [Tech Stack](#tech-stack)

---

## Overview

**CrowdSafe AI** is an advanced crowd intelligence platform designed to monitor physical spaces in real time, identify dangerous crowd surges, forecast bottleneck hazards, and empower operators with automated incident reporting and conversational AI.

The platform integrates deep-learning computer vision (YOLOv8 and Kalman-filter multi-object tracking) with Google Gemini multimodal intelligence, delivering deterministic telemetry alongside contextual reasoning, multilingual support, and automated PDF dossier generation.

---

## Key Features

### 1. Computer Vision & Real-Time Telemetry
- **YOLOv8 Person Detection:** High-accuracy real-time person detection with bounding box localization and privacy-preserving face/head blurring.
- **Multi-Object Tracking:** SortTracker algorithm with persistent tracking IDs, trajectory paths, velocity vectors, and direction entropy calculation.
- **Dynamic Density & Flow Analysis:** Grid-based spatial density calculation, cumulative movement heatmaps, and bi-directional entrance/exit flow counting.
- **Stampede & Anomaly Prevention:** Automated detection of sudden speed spikes, compression waves, pressure surges, cluster formations, and restricted-zone intrusions.
- **Deterministic Alert Engine:** Multi-tier threshold evaluation (`SAFE`, `WARNING`, `DANGER`, `CRITICAL`) with cooldown controls and voice alerts.
- **LSTM Risk Predictor:** Forward-looking risk forecasting over 30s, 60s, and 120s horizons.

### 2. Gemini Multimodal Crowd Intelligence
- **Natural Language Assistant:** Ask questions about live occupancy, zone congestion, flow rates, risk levels, and historical incidents.
- **Database & Telemetry Grounding:** Uses structured function calling to query live metrics and SQLite records; never fabricates sensor values or timestamps.
- **Conversational Memory:** Context-aware multi-turn conversations with session isolation.
- **Multilingual Support:** Native interaction in **English**, **Hindi (हिन्दी)**, and **Marathi (मराठी)**.
- **Gemini Vision Analyst:** On-demand multimodal frame analysis identifying spatial clustering, blocked pathways, fallen persons, and physical obstacles.
- **AI Incident Explanation:** Generates human-readable cause-and-effect breakdowns for safety alerts.

### 3. Video Upload & Automated Analysis
- **Recorded Video Processing:** Upload pre-recorded `.mp4`, `.avi`, `.mov`, or `.mkv` footage.
- **Complete CV Pipeline:** Executes frame-by-frame YOLO detection, tracking, density, and flow analysis in a non-blocking background thread.
- **Annotated Video Generation:** Produces an annotated MP4 video with real-time HUD overlays, bounding boxes, and thermal trails.
- **Visual Analytics:** Generates dual-axis headcount & risk score timeline charts and spatial heatmaps.
- **Publication-Ready PDF Dossier:** Automatically compiles a comprehensive A4 PDF report with metadata, analytics tables, embedded graphs, Gemini AI safety assessments, incident logs, and security verification blocks.

### 4. Incident Management & Semantic Search
- **Semantic Incident Search:** Natural-language search across historical incidents powered by TF-IDF vector similarity.
- **Automated Incident Reports:** Converts structured telemetry into formal incident reports exportable to PDF.
- **Automated Clip Recording:** Records rolling MP4 incident clips upon `DANGER` or `CRITICAL` trigger events.

---

## System Architecture

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        VIDEO INPUT SOURCES                             │
│     Webcam (DirectShow)   │   RTSP Stream   │   Video File Upload     │
└────────────────────────────────────┬───────────────────────────────────┘
                                     │
                                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   REAL-TIME CV & TRACKING PIPELINE                     │
│  - YOLOv8 Person Detector (models/yolov8n.pt)                          │
│  - MultiTracker (Kalman Filter + Hungarian Algorithm)                  │
│  - Density Analyzer & Spatial Heatmaps                                 │
│  - Movement & Direction Entropy Analyzer                               │
│  - Safety & Alert Engine (SAFE, WARNING, DANGER, CRITICAL)             │
│  - LSTM Risk Forecaster & Acoustic Anomaly Monitor                     │
└────────────────────────────────────┬───────────────────────────────────┘
                                     │
                                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│                     CORE DATA & SERVER RUNTIME                         │
│  - FastAPI Web Server & MJPEG Stream (/stream)                         │
│  - Real-Time WebSocket Telemetry (/ws)                                 │
│  - SQLite Database (crowd_safety.db) & JSONL Audit Logs                │
│  - Background Video Job Manager (Thread-pool execution)                │
└──────────────┬──────────────────────────────────────────┬──────────────┘
               │                                          │
               ▼                                          ▼
┌──────────────────────────────┐        ┌────────────────────────────────┐
│   GEMINI MULTIMODAL SUITE    │        │      OPERATOR DASHBOARD        │
│ - NLP Crowd Assistant        │        │ - Live HUD & Camera Feed       │
│ - Multimodal Vision Analyst  │◄──────►│ - Video Upload & Progress UI   │
│ - AI Incident Explainer      │        │ - Multilingual Chat Console    │
│ - ReportLab PDF Dossier Gen  │        │ - Incident Timelines & Reports │
└──────────────────────────────┘        └────────────────────────────────┘
```

---

## Prerequisites & Requirements

- **Operating System:** Windows 10/11, Linux, or macOS
- **Python:** Python 3.9 to Python 3.14
- **Hardware:**
  - Standard USB/integrated webcam (for live camera mode)
  - Minimum 4 GB RAM (8 GB recommended)
  - CPU or CUDA-compatible GPU for accelerated YOLO inference
- **Optional:** Google Gemini API Key (for conversational AI and vision analysis features)

---

## Installation & Setup

### 1. Extract / Navigate to Project Directory

Open your command prompt or terminal in the project directory:

```bash
cd crowd-detection-main
```

### 2. Set Up a Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

Install all core packages:

```bash
pip install -r requirements.txt
```

*Note: The YOLO model weights (`models/yolov8n.pt`) will automatically load or download on first run.*

---

## Configuration

Copy the example environment file to configure your credentials:

```bash
# Windows (PowerShell / Command Prompt)
copy .env.example .env

# Linux / macOS
cp .env.example .env
```

Open `.env` and set your configuration options:

```env
# Google Gemini API Key (for NLP Assistant, Vision Analyst, and AI Dossier Summaries)
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-1.5-flash

# Optional: Outbound Emergency Alerts (leave blank to simulate in console)
TWILIO_ACCOUNT_SID=
TWILIO_AUTH_TOKEN=
TWILIO_FROM=
ALERT_PHONES=

SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=
SMTP_PASS=
ALERT_EMAILS=
WEBHOOK_URL=
```

> **Note:** If no Gemini API key is provided, the platform automatically switches to its offline rule-based NLP fallback engine without crashing.

---

## Running the Application

### Option A: Standard Quick Launch (Windows)

```bash
python launch.py
```

### Option B: Command Line Options

```bash
# 1. Live Webcam (uses primary webcam device index 0)
python main.py --source 0 --no-window

# 2. Simulation Mode (runs synthetic crowd scenarios without requiring a camera)
python main.py --sim

# 3. Analyze a Specific Video File
python main.py --source "path/to/crowd_video.mp4"

# 4. Custom Port
python main.py --port 8080
```

### Command-Line Arguments:

| Argument | Description | Default |
|---|---|---|
| `--source VALUE` | Camera index (`0`, `1`, etc.), video file path, or RTSP stream URL | `0` |
| `--sim` | Run in simulation mode with synthetic crowd behavior | `False` |
| `--no-window` | Run headless (disables OpenCV desktop window; web dashboard only) | `False` |
| `--port PORT` | Web server port | `8000` |

---

## Web Dashboard & Interfaces

Once started, open your web browser to:

👉 **[http://localhost:8000](http://localhost:8000)**

*(A mobile-optimized view is also available at `http://localhost:8000/mobile`)*

### Dashboard Modules:
- **Main Viewport:** High-resolution video stream with real-time tracking boxes, speed indicators, and HUD telemetry.
- **Metrics Bar:** Instant readouts for People Count, Density %, Average Speed, Risk Score %, Direction Entropy, and Acoustic SPL.
- **Alert Feed:** Chronological real-time alerts with color-coded severity (`SAFE`, `WARNING`, `DANGER`, `CRITICAL`).
- **Video Feed Source:** One-click toggle between **Live Webcam (Camera 1)**, **Camera 2**, and **Drone / Simulation Mode**.
- **Intelligence & Incident Tools:**
  - 🤖 **Ask Gemini Crowd Assistant**
  - 👁️ **Gemini Vision Analyst**
  - 🎥 **Video Upload & Analysis**
  - 🔍 **Semantic Incident Search**
  - 📑 **Incident Reports & PDF**

---

## Video Upload & Automated Analysis

The platform allows operators to upload pre-recorded security footage for forensic investigation:

1. Click **🎥 Video Upload & Analysis** in the dashboard.
2. Drag and drop or browse any `.mp4`, `.avi`, `.mov`, or `.mkv` video file.
3. Click **Upload & Run Crowd Intelligence**.
4. The system executes background computer vision processing with real-time progress, FPS, and ETA tracking.
5. Review the results:
   - **Annotated Video:** View processed video with tracked IDs and HUD telemetry.
   - **Crowd Timeline Chart:** Dual-axis visualization of headcount and risk score progression.
   - **Spatial Density Heatmap:** Cumulative thermal map of crowd movement and high-density bottlenecks.
   - **Interactive Incident Timeline:** Clickable timestamp tags to jump to critical events.
   - **Gemini AI Assessment:** Automated executive summary and marshal deployment recommendations.
   - **📄 Download PDF Dossier:** Export a formal, multi-page PDF report.

---

## Gemini Multimodal Intelligence

### Conversational Crowd Assistant
Operators can query live telemetry using natural language in English, Hindi, or Marathi:
- *"How many people are currently detected?"*
- *"Which zone has the highest crowd density?"*
- *"What are the latest critical alerts?"*
- *"Explain the current crowd risk level."*
- *"सध्या किती लोक उपस्थित आहेत?"* (Marathi)
- *"वर्तमान भीड़ की स्थिति क्या है?"* (Hindi)

### Multimodal Vision Analyst
- Captures live snapshots from camera feeds.
- Inspects physical scene properties: pathway blockages, obstacle hazards, irregular clustering, and visible fallen-person indicators.
- Distinguishes observable visual facts from interpretations to maintain safety integrity.

---

## API Reference

The FastAPI backend exposes the following primary endpoints:

### Core Streaming & Controls
| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Web dashboard interface |
| `GET` | `/stream` | Real-time MJPEG annotated video stream |
| `GET` | `/api/status` | Current metrics, risk score, alert level, and predictions |
| `GET` | `/api/alerts` | Active alert feed |
| `POST` | `/api/control/{action}` | Pipeline controls (`start`, `stop`, `live_cam`, `sim_normal`, `sim_dense`, `sim_panic`) |
| `POST` | `/api/camera/{cam_id}` | Switch camera source (`CAM-1`, `CAM-2`, `DRONE-1`) |
| `WS` | `/ws` | Real-time bidirectional WebSocket telemetry feed |

### Gemini AI & NLP Intelligence
| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/ai/chat` | Conversational NLP assistant query |
| `POST` | `/api/ai/analyze-frame` | Gemini multimodal visual scene analysis |
| `POST` | `/api/ai/explain-incident` | AI-generated root cause incident explanation |
| `POST` | `/api/nlp/search` | Semantic incident retrieval |
| `POST` | `/api/nlp/reports/generate` | Generate structured incident report |
| `GET` | `/api/nlp/reports/{report_id}/pdf` | Download formal incident PDF report |

### Video Upload & Analysis
| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/video/upload` | Upload video file for background processing |
| `GET` | `/api/video/status/{job_id}` | Processing status, progress %, and result payload |
| `GET` | `/api/video/jobs` | List recent video analysis jobs |
| `GET` | `/api/video/report/{job_id}/pdf` | Download complete video analysis PDF dossier |

---

## Project Directory Structure

```text
crowd-detection-main/
├── config/
│   └── settings.py              # Centralized configuration, thresholds, and defaults
├── src/
│   ├── detector.py              # YOLOv8 person detection and privacy blurring
│   ├── tracker.py               # Kalman-filter MultiTracker (SORT)
│   ├── density.py               # Density grid, flow counting, and spatial heatmaps
│   ├── movement.py              # Speed, velocity vectors, and direction entropy
│   ├── alerts.py                # Alert evaluation engine with cooldowns
│   ├── predictor.py             # LSTM risk forecasting model
│   ├── logger.py                # SQLite and JSONL audit logging
│   ├── hud.py                   # Real-time HUD visual overlay renderer
│   ├── notifications.py         # SMS, email, and webhook notifications
│   ├── recorder.py              # Rolling MP4 clip recorder
│   ├── simulator.py             # Synthetic crowd simulation engine
│   ├── gemini_service/          # Google Gemini multimodal integration
│   │   ├── client.py            # Gemini API client & rate limiting
│   │   ├── agent.py             # Function-calling conversational agent
│   │   ├── vision.py            # Multimodal frame scene analyst
│   │   └── explainer.py         # Incident explanation service
│   ├── nlp_assistant/           # NLP Assistant, semantic search, and reports
│   │   ├── assistant.py         # Conversational assistant coordinator
│   │   ├── semantic_search.py   # Vector similarity incident retrieval
│   │   ├── report_generator.py  # ReportLab incident PDF generator
│   │   └── translator.py        # Multilingual alert translator
│   └── video_analysis/          # Automated video file analysis pipeline
│       ├── analyzer.py          # VideoFileAnalyzer & VideoJobManager
│       └── report.py            # Video analysis PDF dossier generator
├── web/
│   ├── server.py                # FastAPI web server and API endpoints
│   └── templates/               # Dashboard and mobile UI templates
│       ├── dashboard.html       # Full operations dashboard
│       └── mobile.html          # Mobile-optimized interface
├── models/                      # YOLO model weights (yolov8n.pt)
├── tests/                       # Comprehensive automated test suites (112 tests)
│   ├── test_nlp_assistant.py    # NLP assistant & semantic search tests
│   ├── test_gemini_multimodal.py# Gemini API & mock multimodal tests
│   └── test_video_analysis.py   # Video upload, processing & PDF tests
├── main.py                      # Application entry point and orchestrator
├── launch.py                    # Windows launcher script
├── requirements.txt             # Project dependencies
├── .env.example                 # Environment configuration template
└── README.md                    # Project documentation
```

---

## Automated Testing

CrowdSafe AI includes an extensive automated test suite covering all modules, computer vision processing, API endpoints, mock Gemini integrations, and PDF generators:

```bash
# Run all test suites
pytest -v

# Run specific module tests
pytest tests/test_video_analysis.py -v
pytest tests/test_gemini_multimodal.py -v
pytest tests/test_nlp_assistant.py -v
```

**Status:** `112 passed, 0 failed` across all suites.

---

## Tech Stack

- **Core & Server:** Python 3.9+, FastAPI, Uvicorn, WebSockets, Jinja2
- **Computer Vision & Tracking:** OpenCV, Ultralytics YOLOv8, NumPy, SciPy, FilterPy
- **Multimodal AI:** Google Gemini API (`google-generativeai`), Scikit-Learn (TF-IDF vector search)
- **Data & Visualization:** Matplotlib, SQLite3, ReportLab (PDF generation)
- **Audio & Alerts:** Pygame, pyttsx3, PyAudio, Twilio SDK
