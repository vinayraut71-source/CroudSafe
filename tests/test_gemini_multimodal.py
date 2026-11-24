"""
Test suite for Gemini-Powered NLP and Visual Intelligence for CrowdSafe AI.
Tests GeminiClient, GeminiCrowdAgent, GeminiVisionAnalyst, IncidentExplainer,
and Report Generator across connectivity, function calling, conversational memory,
multilingual output, vision analysis, error fallbacks, and PDF rendering.
All external LLM calls are mocked to enable 100% offline testing without consuming credits.
"""

import os
import sys
import json
import time
import sqlite3
import pytest
import numpy as np
import cv2
from unittest.mock import MagicMock, patch

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.gemini_service.client import GeminiClient
from src.gemini_service.agent import (
    GeminiCrowdAgent,
    get_live_crowd_telemetry,
    get_zone_density_breakdown,
    get_flow_counts,
    get_latest_alerts,
    compare_crowd_last_hour,
    get_camera_occupancy_overview,
)
from src.gemini_service.vision_analyst import GeminiVisionAnalyst
from src.gemini_service.explainer import IncidentExplainer
from src.nlp_assistant.report_generator import IncidentReportManager
from src.nlp_assistant.schemas import (
    AIChatRequest,
    AIChatResponse,
    ClearHistoryRequest,
    VisionAnalysisRequest,
    VisionAnalysisResponse,
    IncidentExplanationRequest,
    IncidentExplanationResponse,
)


@pytest.fixture
def mock_live_state(monkeypatch):
    """Provide realistic web server state."""
    state = {
        "frame": np.zeros((480, 640, 3), dtype=np.uint8),
        "metrics": {
            "count": 18,
            "density": 0.45,
            "occupancy_pct": 0.18,
            "entry_count": 25,
            "exit_count": 7,
            "avg_speed": 2.1,
            "speed_multiplier": 1.4,
            "direction_entropy": 0.82,
            "acoustic_db": 55.0,
            "acoustic_panic": False,
            "zone_grid": [[0.1, 0.2], [0.45, 0.3]],
            "running": True
        },
        "alert_feed": [
            {"level": "WARNING", "message": "Elevated density in Zone-R2C1", "time_str": "14:30:00"}
        ],
        "alert_level": "WARNING",
        "risk_score": 45.0,
        "current_camera": "CAM-1"
    }
    monkeypatch.setattr("web.server.get_state", lambda: state)
    return state


@pytest.fixture
def temp_db(tmp_path):
    """Create a temporary SQLite database initialized with tables."""
    db_file = str(tmp_path / "test_crowd_safety.db")
    with sqlite3.connect(db_file) as conn:
        conn.execute("""
            CREATE TABLE events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                timestamp REAL,
                event_type TEXT,
                level TEXT,
                message TEXT,
                data TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                timestamp REAL,
                count INTEGER,
                density REAL,
                avg_speed REAL,
                risk_score REAL,
                alert_level TEXT
            )
        """)
        now = time.time()
        conn.execute("INSERT INTO events (timestamp, level, message) VALUES (?, 'CRITICAL', 'CRITICAL: Rapid stampede surge detected')", (now - 60,))
        conn.execute("INSERT INTO events (timestamp, level, message) VALUES (?, 'WARNING', 'WARNING: Density reached 40%')", (now - 120,))
        for i in range(10):
            conn.execute("INSERT INTO metrics (timestamp, count, density, avg_speed, risk_score, alert_level) VALUES (?, ?, ?, ?, ?, ?)",
                         (now - (i * 60), 10 + i, 0.3 + (i * 0.02), 2.0, 30.0 + (i * 2), "WARNING"))
        conn.commit()
    return db_file


# ── Test Suite 1: Client & Connectivity ──────────────────────────────────────

class TestGeminiClient:
    def test_client_missing_key_fallback(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "")
        monkeypatch.setenv("GOOGLE_API_KEY", "")
        client = GeminiClient(api_key="")
        assert client.is_available() is False
        res = client.generate_text("Hello")
        assert res["success"] is False
        assert res["fallback_needed"] is True

    def test_client_with_key_initializes(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "test_mock_key_12345")
        with patch("google.genai.Client") as mock_genai:
            client = GeminiClient()
            assert client.is_available() is True
            mock_genai.assert_called_once()

    def test_client_handles_quota_exhausted_gracefully(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "test_mock_key_12345")
        with patch("google.genai.Client") as mock_genai:
            mock_instance = MagicMock()
            mock_instance.models.generate_content.side_effect = Exception("429 RESOURCE_EXHAUSTED: Rate limit exceeded")
            mock_genai.return_value = mock_instance

            client = GeminiClient()
            res = client.generate_text("Status update")
            assert res["success"] is False
            assert res["fallback_needed"] is True
            assert "Quota Exhausted" in res["error"]


# ── Test Suite 2: Grounding Function Tools ───────────────────────────────────

class TestGroundingTools:
    def test_live_crowd_telemetry(self, mock_live_state):
        t = get_live_crowd_telemetry()
        assert t["headcount"] == 18
        assert t["density_pct"] == 45.0
        assert t["occupancy_pct"] == 18.0
        assert t["alert_level"] == "WARNING"
        assert t["current_camera"] == "CAM-1"

    def test_zone_density_breakdown(self, mock_live_state):
        z = get_zone_density_breakdown()
        assert z["highest_density_zone"] == "Zone-R2C1"
        assert z["max_density_pct"] == 45.0

    def test_flow_counts(self, mock_live_state):
        f = get_flow_counts()
        assert f["entry_count"] == 25
        assert f["exit_count"] == 7
        assert f["net_flow"] == 18

    def test_compare_crowd_last_hour(self, temp_db, monkeypatch):
        monkeypatch.setattr("src.gemini_service.agent.DB_PATH", temp_db)
        comp = compare_crowd_last_hour()
        assert "avg_headcount" in comp
        assert comp["peak_headcount"] >= 18

    def test_camera_occupancy_overview(self, mock_live_state):
        c = get_camera_occupancy_overview()
        assert c["active_camera"] == "CAM-1"
        assert len(c["cameras"]) >= 2


# ── Test Suite 3: Conversational NLP Assistant (Module 1) ────────────────────

class TestGeminiCrowdAgent:
    def test_offline_fallback_qa(self, mock_live_state, temp_db, monkeypatch):
        monkeypatch.setattr("src.nlp_assistant.assistant.DB_PATH", temp_db)
        agent = GeminiCrowdAgent(client=GeminiClient(api_key=""))
        res = agent.ask("How many people are currently detected?", session_id="s1")
        assert isinstance(res, AIChatResponse)
        assert res.fallback_used is True
        assert "18" in res.answer or "headcount" in res.answer.lower()

    def test_conversational_memory_followup(self, mock_live_state):
        agent = GeminiCrowdAgent(client=GeminiClient(api_key=""))
        agent.ask("How many people are there?", session_id="mem_test")
        agent.ask("Are there any danger alerts?", session_id="mem_test")
        history = agent.get_session_history("mem_test")
        assert len(history) == 4  # 2 user + 2 assistant turns
        assert history[0].role == "user"
        assert history[1].role == "assistant"

    def test_clear_session_history(self, mock_live_state):
        agent = GeminiCrowdAgent(client=GeminiClient(api_key=""))
        agent.ask("Test message 1", session_id="clear_me")
        assert len(agent.get_session_history("clear_me")) == 2
        agent.clear_session("clear_me")
        assert len(agent.get_session_history("clear_me")) == 0

    def test_multilingual_responses(self, mock_live_state):
        agent = GeminiCrowdAgent(client=GeminiClient(api_key=""))
        res_hi = agent.ask("What is the density?", lang="hi")
        assert res_hi.lang == "hi"
        res_mr = agent.ask("What is the density?", lang="mr")
        assert res_mr.lang == "mr"

    def test_mocked_gemini_api_success(self, mock_live_state, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "mock_key")
        with patch("google.genai.Client") as mock_genai:
            mock_inst = MagicMock()
            mock_resp = MagicMock()
            mock_resp.text = "There are currently 18 people detected on CAM-1 with 45.0% density."
            mock_fc = MagicMock()
            mock_fc.name = "get_live_crowd_telemetry"
            mock_resp.function_calls = [mock_fc]
            mock_inst.models.generate_content.return_value = mock_resp
            mock_genai.return_value = mock_inst

            agent = GeminiCrowdAgent()
            res = agent.ask("How many people are currently detected?")
            assert res.fallback_used is False
            assert "18 people" in res.answer


# ── Test Suite 4: Vision Analyst (Module 2) ──────────────────────────────────

class TestGeminiVisionAnalyst:
    def test_analyze_with_image_bytes(self, mock_live_state):
        analyst = GeminiVisionAnalyst(client=GeminiClient(api_key=""))
        dummy_img = np.zeros((240, 320, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", dummy_img)
        res = analyst.analyze_frame(image_bytes=buf.tobytes(), camera_id="CAM-1", lang="en")
        assert isinstance(res, VisionAnalysisResponse)
        assert res.camera_id == "CAM-1"
        assert len(res.observed_conditions) > 0
        assert len(res.recommended_operator_actions) > 0
        assert res.fallback_used is True

    def test_missing_frame_handling(self, monkeypatch):
        monkeypatch.setattr("web.server.get_state", lambda: {"frame": None})
        analyst = GeminiVisionAnalyst(client=GeminiClient(api_key=""))
        res = analyst.analyze_frame(use_live_frame=True)
        assert "unavailable" in res.scene_summary.lower()

    def test_mocked_gemini_vision_json_output(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "mock_key")
        with patch("google.genai.Client") as mock_genai:
            mock_inst = MagicMock()
            mock_resp = MagicMock()
            mock_resp.text = json.dumps({
                "scene_summary": "Dispersed pedestrian group near central doorway.",
                "observed_conditions": ["Clear exit pathways", "Nominal walking pace"],
                "potential_safety_concerns": ["None currently observed"],
                "supporting_visual_evidence": ["Central walkway unobstructed"],
                "confidence_or_uncertainty": "High confidence",
                "recommended_operator_actions": ["Maintain standard observation"]
            })
            mock_inst.models.generate_content.return_value = mock_resp
            mock_genai.return_value = mock_inst

            analyst = GeminiVisionAnalyst()
            dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
            _, buf = cv2.imencode(".jpg", dummy_img)
            res = analyst.analyze_frame(image_bytes=buf.tobytes())
            assert res.fallback_used is False
            assert "Dispersed pedestrian group" in res.scene_summary


# ── Test Suite 5: AI Incident Explanation (Module 3) ─────────────────────────

class TestIncidentExplainer:
    def test_heuristic_explanation_generation(self):
        explainer = IncidentExplainer(client=GeminiClient(api_key=""))
        payload = {
            "level": "DANGER",
            "risk_score": 72.0,
            "metrics": {"count": 40, "density": 0.65, "speed_multiplier": 2.2, "compressed": True}
        }
        res = explainer.explain_incident(alert_payload=payload, include_snapshot=False)
        assert isinstance(res, IncidentExplanationResponse)
        assert res.alert_level == "DANGER"
        assert res.official_risk_score == 72.0
        assert len(res.causal_indicators) >= 2
        assert len(res.recommended_actions) >= 2


# ── Test Suite 6: Incident Reports & PDF Export (Module 4) ────────────────────

class TestIncidentReportGeneratorWithAI:
    def test_report_generation_and_pdf(self, temp_db, tmp_path, monkeypatch):
        monkeypatch.setattr("src.nlp_assistant.report_generator.REPORT_DIR", str(tmp_path / "reports"))
        mgr = IncidentReportManager(db_path=temp_db)
        rep = mgr.create_report(
            event_type="STAMPEDE_RISK_SURGE",
            severity="CRITICAL",
            camera_id="CAM-1",
            count=85,
            density=0.78,
            risk_score=91.0,
            ai_explanation="Severe directional convergence with speed acceleration to 2.4x baseline.",
            recommended_actions=["Deploy marshals immediately", "Open secondary exit gates"]
        )
        assert rep.id.startswith("INC-")
        assert "Severe directional convergence" in rep.ai_explanation
        pdf_path = mgr.export_pdf(rep.id)
        assert os.path.exists(pdf_path)
        with open(pdf_path, "rb") as f:
            header = f.read(5)
            assert header == b"%PDF-"


# ── Test Suite 7: Schema Validation ──────────────────────────────────────────

class TestSchemas:
    def test_ai_chat_request_validation(self):
        req = AIChatRequest(query="Status check", lang="hi")
        assert req.query == "Status check"
        assert req.lang == "hi"

        with pytest.raises(Exception):
            AIChatRequest(query="")  # min_length=1

        with pytest.raises(Exception):
            AIChatRequest(query="Valid", lang="french")  # pattern rejection
