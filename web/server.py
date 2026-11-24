"""
FastAPI-based web dashboard backend.
Serves the HTML dashboard, video stream via MJPEG, and a WebSocket for live data.
"""

import asyncio
import base64
import json
import time
import datetime
import os
import threading
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Depends, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import API_HOST, API_PORT, RECORDING_DIR
from src.nlp_assistant import (
    CrowdAssistant,
    AlertTranslator,
    IncidentReportManager,
    SemanticIncidentSearcher,
    require_rate_limit,
)
from src.nlp_assistant.schemas import (
    ChatQueryRequest,
    IncidentReportCreate,
    VerificationUpdate,
    SemanticSearchRequest,
    TranslateAlertRequest,
    AIChatRequest,
    AIChatResponse,
    ClearHistoryRequest,
    ConversationHistoryResponse,
    VisionAnalysisRequest,
    VisionAnalysisResponse,
    IncidentExplanationRequest,
    IncidentExplanationResponse,
)
from src.gemini_service import (
    GeminiClient,
    GeminiCrowdAgent,
    GeminiVisionAnalyst,
    IncidentExplainer,
)
from src.video_analysis import video_job_manager

# NLP & Gemini AI singletons
nlp_assistant = CrowdAssistant()
alert_translator = AlertTranslator()
report_manager = IncidentReportManager()
semantic_searcher = SemanticIncidentSearcher()
gemini_client = GeminiClient()
gemini_agent = GeminiCrowdAgent(gemini_client)
gemini_vision = GeminiVisionAnalyst(gemini_client)
incident_explainer = IncidentExplainer(gemini_client)

# Shared state object injected from main pipeline
_state: dict = {
    "frame": None,
    "metrics": {},
    "alert_feed": [],
    "alert_level": "SAFE",
    "risk_score": 0.0,
    "predictions": {},
    "running": False,
    "sim_mode": False,
}

app = FastAPI(title="Crowd Safety Dashboard", version="1.0")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(BASE_DIR, "web", "static")
TEMPLATE_DIR = os.path.join(BASE_DIR, "web", "templates")

if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATE_DIR)

os.makedirs(RECORDING_DIR, exist_ok=True)
app.mount("/recordings", StaticFiles(directory=RECORDING_DIR), name="recordings")


def get_state() -> dict:
    return _state


def update_state(**kwargs):
    _state.update(kwargs)


# ── MJPEG stream ────────────────────────────────────────────────────────────

def _frame_generator():
    last_seq = -1
    while True:
        jpeg_frame = _state.get("jpeg_frame")
        seq = _state.get("frame_seq", 0)
        if jpeg_frame is not None and seq != last_seq:
            last_seq = seq
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg_frame + b"\r\n")
            time.sleep(0.015)
        else:
            frame = _state.get("frame")
            if frame is not None and jpeg_frame is None:
                try:
                    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
                except Exception:
                    pass
            time.sleep(0.01)


@app.get("/stream")
def video_stream():
    return StreamingResponse(
        _frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        }
    )


# ── WebSocket live data ──────────────────────────────────────────────────────

class ConnectionManager:
    def __init__(self):
        self.active: list = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast(self, data: dict):
        msg = json.dumps(data)
        dead = []
        for ws in self.active:
            try:
                await ws.send_text(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


manager = ConnectionManager()


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)
    try:
        while True:
            await ws.receive_text()   # keep alive pings
    except WebSocketDisconnect:
        manager.disconnect(ws)


async def _broadcast_loop():
    while True:
        await asyncio.sleep(0.5)
        try:
            data = {
                "metrics": _state.get("metrics", {}),
                "alert_level": _state.get("alert_level", "SAFE"),
                "risk_score": _state.get("risk_score", 0),
                "predictions": _state.get("predictions", {}),
                "alert_feed": _state.get("alert_feed", [])[:10],
                "running": _state.get("running", False),
            }
            await manager.broadcast(data)
        except Exception:
            pass


# ── REST endpoints ────────────────────────────────────────────────────────────

@app.get("/api/status")
def api_status():
    m = _state.get("metrics", {})
    return JSONResponse({
        "alert_level": _state.get("alert_level", "SAFE"),
        "risk_score": _state.get("risk_score", 0),
        "count": m.get("count", 0),
        "density": m.get("density", 0),
        "running": _state.get("running", False),
        "predictions": _state.get("predictions", {}),
    })


@app.get("/api/alerts")
def api_alerts():
    return JSONResponse({"alerts": _state.get("alert_feed", [])})


@app.post("/api/control/{action}")
async def api_control(action: str):
    if action == "start":
        _state["running"] = True
    elif action == "stop":
        _state["running"] = False
    elif action in ("live_cam", "camera"):
        _state["sim_cmd"] = "live_cam"
        _state["current_camera"] = "CAM-1"
    elif action == "sim_normal":
        _state["sim_cmd"] = "normal"
    elif action == "sim_building":
        _state["sim_cmd"] = "building"
    elif action == "sim_dense":
        _state["sim_cmd"] = "dense"
    elif action == "sim_panic":
        _state["sim_cmd"] = "panic"
    return JSONResponse({"status": "ok", "action": action})


@app.get("/api/timeline")
def api_timeline():
    timeline = _state.get("timeline", "")
    if not timeline:
        timeline = "No recent incident recorded."
    return JSONResponse({"timeline": timeline})

@app.get("/api/recordings")
def api_list_recordings():
    """Returns a list of all recorded mp4 incident clips."""
    recs = []
    if os.path.exists(RECORDING_DIR):
        for f in reversed(sorted(os.listdir(RECORDING_DIR))):
            if f.endswith(".mp4"):
                recs.append(f)
    return JSONResponse({"recordings": recs})

@app.post("/api/camera/{cam_id}")
async def api_switch_camera(cam_id: str):
    """Multi-camera switching endpoint API."""
    _state["current_camera"] = cam_id
    if cam_id in ("CAM-1", "0", "WEBCAM", "LIVE"):
        _state["sim_cmd"] = "live_cam"
    elif cam_id in ("CAM-2", "DRONE-1", "SIM"):
        _state["sim_cmd"] = "normal"
    print(f"[Dashboard] Switched to Camera: {cam_id}")
    return JSONResponse({"status": "ok", "camera": cam_id})


# ── NLP-Based Crowd Intelligence Assistant APIs ──────────────────────────────

@app.post("/api/nlp/chat")
async def api_nlp_chat(req: ChatQueryRequest, request: Request):
    """Natural Language Crowd Assistant: answer operator questions using live & DB telemetry."""
    require_rate_limit(request, max_requests=40, window_seconds=60)
    res = nlp_assistant.ask(req.query, session_id=req.session_id)
    return JSONResponse(res.dict())


@app.post("/api/nlp/search")
async def api_nlp_search(req: SemanticSearchRequest, request: Request):
    """Semantic Incident Search: retrieve historical incidents and alerts using semantic similarity."""
    require_rate_limit(request, max_requests=60, window_seconds=60)
    res = semantic_searcher.search(query=req.query, limit=req.limit, min_similarity=req.min_similarity)
    return JSONResponse(res.dict())


@app.post("/api/nlp/reports/generate")
async def api_nlp_generate_report(req: IncidentReportCreate, request: Request):
    """Automatic Incident Report Generation: convert structured alerts/telemetry into formal report."""
    require_rate_limit(request, max_requests=30, window_seconds=60)
    live_metrics = _state.get("metrics", {})
    count = req.count if req.count is not None else live_metrics.get("count", 0)
    density = req.density if req.density is not None else live_metrics.get("density", 0.0)
    occupancy_pct = req.occupancy_pct if req.occupancy_pct is not None else ((live_metrics.get("occupancy_pct", 0.0) or 0.0) * 100)
    risk_score = float(_state.get("risk_score", 0.0))

    report = report_manager.create_report(
        event_type=req.event_type,
        severity=req.severity,
        camera_id=req.camera_id,
        zone=req.zone,
        count=count,
        density=density,
        occupancy_pct=occupancy_pct,
        risk_score=risk_score,
        evidence_url=req.evidence_url,
        operator_notes=req.operator_notes,
        data=live_metrics
    )
    return JSONResponse(report.dict())


@app.get("/api/nlp/reports")
def api_nlp_list_reports(limit: int = 50):
    """List historical incident reports."""
    reports = report_manager.list_reports(limit=limit)
    return JSONResponse([r.dict() for r in reports])


@app.get("/api/nlp/reports/{report_id}")
def api_nlp_get_report(report_id: str):
    """Get single incident report details."""
    report = report_manager.get_report(report_id)
    if not report:
        raise HTTPException(status_code=404, detail="Incident report not found")
    return JSONResponse(report.dict())


@app.get("/api/nlp/reports/{report_id}/pdf")
def api_nlp_export_pdf(report_id: str):
    """Export incident report as a styled A4 PDF."""
    report = report_manager.get_report(report_id)
    if not report:
        raise HTTPException(status_code=404, detail="Incident report not found")
    pdf_path = report_manager.export_pdf(report_id)
    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename=f"{report_id}.pdf"
    )


@app.post("/api/nlp/reports/{report_id}/verify")
async def api_nlp_verify_report(report_id: str, req: VerificationUpdate, request: Request):
    """Update operator verification status on an incident report."""
    require_rate_limit(request, max_requests=40, window_seconds=60)
    updated = report_manager.update_verification(
        report_id=report_id,
        status=req.status,
        operator_notes=req.operator_notes
    )
    if not updated:
        raise HTTPException(status_code=404, detail="Incident report not found")
    return JSONResponse(updated.dict())


@app.get("/api/nlp/alerts")
def api_nlp_alerts(lang: str = "en"):
    """Return live alert feed translated into requested language (en, hi, mr)."""
    feed = _state.get("alert_feed", [])
    translated = alert_translator.translate_feed(feed, target_lang=lang)
    return JSONResponse({"lang": lang, "alerts": translated})


@app.post("/api/nlp/translate")
async def api_nlp_translate(req: TranslateAlertRequest, request: Request):
    """Translate arbitrary list of alert payloads while preserving numbers, timestamps, and levels."""
    require_rate_limit(request, max_requests=60, window_seconds=60)
    translated = alert_translator.translate_feed(req.alerts, target_lang=req.target_lang)
    return JSONResponse({"target_lang": req.target_lang, "alerts": translated})


# ── Gemini Multimodal AI Endpoints ──────────────────────────────────────────

@app.post("/api/ai/chat")
async def api_ai_chat(req: AIChatRequest, request: Request):
    """
    Module 1: Gemini Conversational Assistant.
    Provides multi-turn conversational Q&A with function calling, session memory,
    and automatic multilingual responses (EN, HI, MR).
    """
    require_rate_limit(request, max_requests=40, window_seconds=60)
    res = gemini_agent.ask(
        query=req.query,
        session_id=req.session_id,
        lang=req.lang,
        clear_history=req.clear_history
    )
    return JSONResponse(res.dict())


@app.post("/api/ai/clear-history")
async def api_ai_clear_history(req: ClearHistoryRequest, request: Request):
    """Clear conversation history for a given session."""
    gemini_agent.clear_session(req.session_id)
    return JSONResponse({"status": "ok", "session_id": req.session_id, "cleared": True})


@app.get("/api/ai/conversation-history")
def api_ai_conversation_history(session_id: str = "default_session"):
    """Retrieve message history for a given conversation session."""
    history = gemini_agent.get_session_history(session_id)
    return JSONResponse({
        "session_id": session_id,
        "messages": [m.dict() for m in history]
    })


@app.get("/api/ai/snapshot")
def api_ai_snapshot(camera_id: str = "CAM-1"):
    """
    Capture the latest high-resolution frame from the live camera stream.
    Returns relative image URL and base64 string for immediate preview.
    """
    frame_bytes, filename, rel_url = gemini_vision.capture_live_frame(camera_id)
    if not frame_bytes:
        raise HTTPException(status_code=503, detail="Camera frame unavailable or capture failed")

    b64_str = base64.b64encode(frame_bytes).decode("utf-8")
    return JSONResponse({
        "status": "ok",
        "camera_id": camera_id,
        "snapshot_url": rel_url,
        "image_base64": f"data:image/jpeg;base64,{b64_str}",
        "timestamp": datetime.datetime.now().strftime("%H:%M:%S")
    })


@app.post("/api/ai/analyze-frame")
async def api_ai_analyze_frame(req: VisionAnalysisRequest, request: Request):
    """
    Module 2: Gemini Vision-Based Crowd Scene Analyst.
    Performs multimodal scene analysis on the live camera frame or an uploaded snapshot.
    """
    require_rate_limit(request, max_requests=30, window_seconds=60)
    res = gemini_vision.analyze_frame(
        image_base64=req.image_base64,
        camera_id=req.camera_id,
        lang=req.lang,
        use_live_frame=req.use_live_frame
    )
    return JSONResponse(res.dict())


@app.post("/api/ai/explain-incident")
async def api_ai_explain_incident(req: IncidentExplanationRequest, request: Request):
    """
    Module 3: AI Incident Explanation.
    Explains the causal indicators behind structured alerts and provides practical operator actions.
    """
    require_rate_limit(request, max_requests=40, window_seconds=60)
    res = incident_explainer.explain_incident(
        incident_id=req.incident_id,
        alert_payload=req.alert_payload,
        lang=req.lang,
        include_snapshot=req.include_snapshot
    )
    return JSONResponse(res.dict())


@app.post("/api/ai/generate-report")
async def api_ai_generate_report(req: IncidentReportCreate, request: Request):
    """
    Module 4: Intelligent Incident Report Generator.
    Enriches incident reports with Gemini causal analysis, visual observations, and recommendations.
    """
    require_rate_limit(request, max_requests=30, window_seconds=60)
    live_metrics = _state.get("metrics", {})
    count = req.count if req.count is not None else live_metrics.get("count", 0)
    density = req.density if req.density is not None else live_metrics.get("density", 0.0)
    occupancy_pct = req.occupancy_pct if req.occupancy_pct is not None else ((live_metrics.get("occupancy_pct", 0.0) or 0.0) * 100)
    risk_score = float(_state.get("risk_score", 0.0))

    report = report_manager.create_report(
        event_type=req.event_type,
        severity=req.severity,
        camera_id=req.camera_id,
        zone=req.zone,
        count=count,
        density=density,
        occupancy_pct=occupancy_pct,
        risk_score=risk_score,
        evidence_url=req.evidence_url,
        operator_notes=req.operator_notes,
        ai_explanation=req.ai_explanation,
        visual_observations=req.visual_observations,
        recommended_actions=req.recommended_actions,
        data=live_metrics
    )
    return JSONResponse(report.dict())


# ── Video Upload & Offline Analysis APIs ────────────────────────────────────

UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
ALLOWED_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}

@app.post("/api/video/upload")
async def api_video_upload(request: Request, file: UploadFile = File(...)):
    """
    Upload a recorded video (MP4, AVI, MOV) for automated processing.
    Runs YOLO detection, SortTracker, density, HUD rendering, Gemini AI summary, and PDF generation.
    """
    require_rate_limit(request, max_requests=20, window_seconds=60)

    filename = file.filename or "uploaded_video.mp4"
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_VIDEO_EXTS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported video format '{ext}'. Supported formats: {', '.join(sorted(ALLOWED_VIDEO_EXTS))}"
        )

    # Save to disk
    save_filename = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{filename}"
    file_path = os.path.join(UPLOAD_DIR, save_filename)

    try:
        if hasattr(file, "file") and hasattr(file.file, "read"):
            contents = file.file.read()
        else:
            contents = await file.read()
        with open(file_path, "wb") as f:
            f.write(contents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded video: {e}")

    # Create & dispatch async background job
    job = video_job_manager.create_job(file_path=file_path, filename=filename)

    return JSONResponse({
        "status": "queued",
        "job_id": job.job_id,
        "filename": filename,
        "message": "Video queued for automated CV and Gemini multimodal intelligence processing."
    })


@app.get("/api/video/status/{job_id}")
def api_video_status(job_id: str):
    """Check processing status, frame progress, and results for a video analysis job."""
    job = video_job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Video analysis job '{job_id}' not found")
    return JSONResponse(job.to_dict())


@app.get("/api/video/jobs")
def api_video_jobs(limit: int = 20):
    """Retrieve list of recent video analysis jobs and their statuses."""
    return JSONResponse(video_job_manager.list_jobs(limit=limit))


@app.get("/api/video/report/{job_id}/pdf")
def api_video_report_pdf(job_id: str):
    """Download publication-quality PDF dossier for completed video analysis."""
    job = video_job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Video job '{job_id}' not found")

    if job.status != "completed":
        raise HTTPException(status_code=400, detail=f"Job '{job_id}' is not yet completed (current: {job.status})")

    pdf_rel = job.result.get("pdf_report_url", "")
    if not pdf_rel:
        raise HTTPException(status_code=404, detail="PDF report not available for this job")

    clean_rel = pdf_rel.replace("/recordings/", "")
    full_path = os.path.join(RECORDING_DIR, clean_rel)
    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail="PDF report file not found on disk")

    return FileResponse(
        full_path,
        media_type="application/pdf",
        filename=f"CrowdSafe_Video_Report_{job_id}.pdf"
    )



@app.get("/")
def index(request: Request):
    """Main dashboard interface."""
    resp = templates.TemplateResponse("dashboard.html", {"request": request})
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp

@app.get("/mobile")
def mobile_index(request: Request):
    """Mobile Command Center interface."""
    resp = templates.TemplateResponse("mobile.html", {"request": request})
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp


@app.on_event("startup")
async def startup():
    asyncio.create_task(_broadcast_loop())


def start_server(host: str = API_HOST, port: int = API_PORT):
    import uvicorn
    uvicorn.run(app, host=host, port=port, log_level="warning")
