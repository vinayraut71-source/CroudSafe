"""
Comprehensive tests for the NLP-Based Crowd Intelligence Assistant module.
Covers: CrowdAssistant, AlertTranslator, IncidentReportManager,
        SemanticIncidentSearcher, RateLimiter, and Pydantic schemas.

All external dependencies (web.server, LLM APIs) are mocked.
Tests are runnable without network access.
"""

import os
import sys
import re
import time
import json
import sqlite3
import tempfile
import datetime
import threading
from unittest.mock import patch, MagicMock

import pytest

# ── Ensure project root is on sys.path ──────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ── Imports under test ──────────────────────────────────────────────────────
from src.nlp_assistant.schemas import (
    ChatQueryRequest,
    ChatQueryResponse,
    ChatSource,
    IncidentReport,
    IncidentReportCreate,
    VerificationUpdate,
    SemanticSearchRequest,
    SemanticSearchItem,
    SemanticSearchResponse,
    TranslateAlertRequest,
    TranslateAlertResponse,
)
from src.nlp_assistant.translator import AlertTranslator
from src.nlp_assistant.rate_limiter import RateLimiter, default_limiter, require_rate_limit
from src.nlp_assistant.assistant import CrowdAssistant
from config.settings import MAX_CAPACITY


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def tmp_db(tmp_path):
    """Create an isolated temporary SQLite database with standard schema."""
    db_file = str(tmp_path / "test_crowd.db")
    with sqlite3.connect(db_file) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                event_type TEXT,
                level TEXT,
                message TEXT,
                data TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                count INTEGER,
                density REAL,
                risk_score REAL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS incidents (
                id TEXT PRIMARY KEY,
                session_id TEXT,
                timestamp REAL,
                time_str TEXT,
                event_type TEXT,
                severity TEXT,
                camera_id TEXT,
                zone TEXT,
                count INTEGER,
                density REAL,
                occupancy_pct REAL,
                risk_score REAL,
                evidence_url TEXT,
                verification_status TEXT,
                operator_notes TEXT,
                verified_at REAL,
                data TEXT
            )
        """)
        conn.commit()
    return db_file


@pytest.fixture
def seeded_db(tmp_db):
    """tmp_db pre-populated with sample events and metrics."""
    now = time.time()
    with sqlite3.connect(tmp_db) as conn:
        # Historical events
        conn.execute(
            "INSERT INTO events (timestamp, event_type, level, message, data) VALUES (?,?,?,?,?)",
            (now - 300, "DENSITY_SURGE", "CRITICAL", "Density exceeded 80% in Zone-A", "{}"),
        )
        conn.execute(
            "INSERT INTO events (timestamp, event_type, level, message, data) VALUES (?,?,?,?,?)",
            (now - 200, "SPEED_SPIKE", "DANGER", "Speed spike detected: 4.2 px/f", "{}"),
        )
        conn.execute(
            "INSERT INTO events (timestamp, event_type, level, message, data) VALUES (?,?,?,?,?)",
            (now - 100, "DIRECTION_SURGE", "WARNING", "Direction surge (85% aligned)", "{}"),
        )
        conn.execute(
            "INSERT INTO events (timestamp, event_type, level, message, data) VALUES (?,?,?,?,?)",
            (now - 50, "ACOUSTIC_PANIC", "DANGER", "Acoustic panic detected: 92.3 dB", "{}"),
        )
        # Metrics
        conn.execute(
            "INSERT INTO metrics (timestamp, count, density, risk_score) VALUES (?,?,?,?)",
            (now - 300, 85, 0.72, 78.5),
        )
        conn.execute(
            "INSERT INTO metrics (timestamp, count, density, risk_score) VALUES (?,?,?,?)",
            (now - 100, 45, 0.35, 32.0),
        )
        conn.commit()
    return tmp_db


def _make_live_state(
    count=42,
    density=0.35,
    occupancy_pct=0.42,
    entry_count=100,
    exit_count=58,
    avg_speed=2.15,
    speed_multiplier=1.3,
    direction_entropy=0.82,
    acoustic_db=55.7,
    acoustic_panic=False,
    compressed=False,
    alert_level="WARNING",
    risk_score=38.5,
    predictions=None,
    zone_grid=None,
    running=True,
    current_camera="CAM-1",
):
    """Build a mock return value for web.server.get_state()."""
    if predictions is None:
        predictions = {30: 42.1, 60: 48.3, 120: 55.0}
    if zone_grid is None:
        zone_grid = [
            [{"count": 10, "density": 0.25, "risk": "SAFE"}, {"count": 18, "density": 0.45, "risk": "WARNING"}],
            [{"count": 5, "density": 0.12, "risk": "SAFE"}, {"count": 9, "density": 0.22, "risk": "SAFE"}],
        ]
    return {
        "metrics": {
            "count": count,
            "density": density,
            "occupancy_pct": occupancy_pct,
            "entry_count": entry_count,
            "exit_count": exit_count,
            "avg_speed": avg_speed,
            "speed_multiplier": speed_multiplier,
            "direction_entropy": direction_entropy,
            "acoustic_db": acoustic_db,
            "acoustic_panic": acoustic_panic,
            "compressed": compressed,
            "zone_grid": zone_grid,
        },
        "alert_level": alert_level,
        "risk_score": risk_score,
        "predictions": predictions,
        "running": running,
        "current_camera": current_camera,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 1. NUMERICAL ACCURACY TESTS — CrowdAssistant
# ═══════════════════════════════════════════════════════════════════════════

class TestNumericalAccuracy:
    """Verify the assistant returns exact counts/densities from mock live state
    and never fabricates data."""

    def _ask(self, query, live_kwargs=None, db_path=None):
        """Helper: ask with mocked live state and no LLM keys."""
        live = _make_live_state(**(live_kwargs or {}))
        mock_get_state = MagicMock(return_value=live)
        db = db_path or ":memory:"
        assistant = CrowdAssistant(db_path=db)
        # Remove API keys so _augment_with_llm returns grounded answer as-is
        assistant.gemini_key = None
        assistant.openai_key = None
        with patch.dict("sys.modules", {"web": MagicMock(), "web.server": MagicMock(get_state=mock_get_state)}):
            return assistant.ask(query)

    def test_occupancy_count_exact(self):
        resp = self._ask("How many people are there?", {"count": 73, "occupancy_pct": 0.73})
        assert "73" in resp.answer
        assert resp.intent == "OCCUPANCY_COUNT"
        assert resp.data["count"] == 73

    def test_density_pct_matches_live(self):
        resp = self._ask("What is the zone density?", {"density": 0.456})
        assert resp.intent == "ZONE_DENSITY"
        # density_pct should be 45.6
        assert "45.6%" in resp.answer

    def test_flow_counts_exact(self):
        resp = self._ask("How many entered and exited?", {"entry_count": 210, "exit_count": 145})
        assert resp.intent == "FLOW_COUNTS"
        assert resp.data["entry_count"] == 210
        assert resp.data["exit_count"] == 145
        assert resp.data["net_inside"] == 65
        assert "210" in resp.answer
        assert "145" in resp.answer

    def test_risk_forecast_exact(self):
        preds = {30: 44.12, 60: 51.99, 120: 63.77}
        resp = self._ask("What is the risk forecast?", {"risk_score": 41.3, "predictions": preds})
        assert resp.intent == "RISK_FORECAST"
        assert resp.data["current_risk_score"] == 41.3
        assert resp.data["forecast_30s"] == 44.1  # rounded to 1 decimal
        assert resp.data["forecast_60s"] == 52.0
        assert resp.data["forecast_120s"] == 63.8

    def test_acoustic_status_exact(self):
        resp = self._ask("What is the noise level?", {"acoustic_db": 88.4, "acoustic_panic": True})
        assert resp.intent == "ACOUSTIC_STATUS"
        assert resp.data["acoustic_db"] == 88.4
        assert resp.data["acoustic_panic"] is True
        assert "88.4" in resp.answer

    def test_movement_speed_exact(self):
        resp = self._ask(
            "What is the crowd speed?",
            {"avg_speed": 3.14, "speed_multiplier": 2.05, "direction_entropy": 0.33, "compressed": True},
        )
        assert resp.intent == "MOVEMENT_SPEED"
        assert resp.data["avg_speed"] == 3.14
        assert resp.data["speed_multiplier"] == 2.05
        assert resp.data["direction_entropy"] == 0.33
        assert resp.data["compressed"] is True

    def test_general_status_exact(self):
        resp = self._ask("Give me a status update", {"count": 55, "density": 0.44, "alert_level": "DANGER", "risk_score": 62.3})
        assert resp.intent == "GENERAL_STATUS"
        assert resp.data["count"] == 55
        assert resp.data["alert_level"] == "DANGER"

    def test_incidents_from_db(self, seeded_db):
        """Counts returned match exactly what was inserted into the DB."""
        resp = self._ask("Show me past incidents", db_path=seeded_db)
        assert resp.intent == "INCIDENTS_ALERTS"
        assert resp.data["critical_incidents"] == 1
        assert resp.data["danger_incidents"] == 2
        assert resp.data["warning_incidents"] == 1
        assert resp.data["total_logged_events"] == 4

    def test_no_fabricated_incidents_empty_db(self, tmp_db):
        """With an empty DB, no incidents should be reported."""
        resp = self._ask("Any alerts logged?", db_path=tmp_db)
        assert resp.data["critical_incidents"] == 0
        assert resp.data["danger_incidents"] == 0
        assert resp.data["total_logged_events"] == 0

    def test_peak_count_from_db(self, seeded_db):
        """Peak count from DB matches the max inserted metric."""
        live_kw = {"count": 30}
        resp = self._ask("How many people are here?", live_kwargs=live_kw, db_path=seeded_db)
        # DB peak is 85, live count is 30 → peak_observed should be max(85, 30)=85
        assert resp.data["peak_observed"] == 85


# ═══════════════════════════════════════════════════════════════════════════
# 2. INTENT CLASSIFICATION TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestIntentClassification:
    """Ensure _determine_intent maps queries to the correct intent."""

    @pytest.fixture(autouse=True)
    def _assistant(self):
        self.asst = CrowdAssistant.__new__(CrowdAssistant)

    @pytest.mark.parametrize(
        "query, expected",
        [
            ("How many people in the venue?", "OCCUPANCY_COUNT"),
            ("What is the headcount?", "OCCUPANCY_COUNT"),
            ("crowd capacity percentage", "OCCUPANCY_COUNT"),
            ("zone density breakdown", "ZONE_DENSITY"),
            ("sector A status", "ZONE_DENSITY"),
            ("entry and exit count", "FLOW_COUNTS"),
            ("how many people exited?", "FLOW_COUNTS"),
            ("past incidents log", "INCIDENTS_ALERTS"),
            ("any critical alerts?", "INCIDENTS_ALERTS"),
            ("risk prediction next 2 minutes", "RISK_FORECAST"),
            ("forecast trend", "RISK_FORECAST"),
            ("noise level", "ACOUSTIC_STATUS"),
            ("decibel reading", "ACOUSTIC_STATUS"),
            ("movement speed", "MOVEMENT_SPEED"),
            ("is there a rush?", "MOVEMENT_SPEED"),
            ("system overview", "GENERAL_STATUS"),
        ],
    )
    def test_intent(self, query, expected):
        assert self.asst._determine_intent(query) == expected


# ═══════════════════════════════════════════════════════════════════════════
# 3. TRANSLATION PRESERVATION TESTS — AlertTranslator
# ═══════════════════════════════════════════════════════════════════════════

class TestTranslationPreservation:
    """Verify numbers, timestamps, percentages, and classifications
    survive EN→HI and EN→MR translation unchanged."""

    @pytest.fixture(autouse=True)
    def _translator(self):
        self.tr = AlertTranslator()
        # Neutralise any LLM keys
        self.tr.gemini_key = None
        self.tr.openai_key = None

    # ── exact-match dictionary translations ──

    def test_safe_hi(self):
        assert self.tr.translate_message("SAFE", "hi") == "सुरक्षित"

    def test_critical_mr(self):
        assert self.tr.translate_message("CRITICAL", "mr") == "अतिगंभीर"

    def test_english_passthrough(self):
        msg = "All metrics within normal parameters."
        assert self.tr.translate_message(msg, "en") == msg

    def test_empty_passthrough(self):
        assert self.tr.translate_message("", "hi") == ""

    # ── dynamic-pattern: numbers/percentages/units preserved ──

    def test_direction_surge_numbers_preserved_hi(self):
        msg = "Direction surge (85% aligned)"
        result = self.tr.translate_message(msg, "hi")
        assert "85%" in result or "85" in result

    def test_direction_surge_numbers_preserved_mr(self):
        msg = "Direction surge (92% aligned)"
        result = self.tr.translate_message(msg, "mr")
        assert "92" in result

    def test_speed_spike_units_preserved(self):
        msg = "Speed spike detected: 4.5 px/f"
        result_hi = self.tr.translate_message(msg, "hi")
        assert "4.5" in result_hi
        assert "px/f" in result_hi

    def test_density_percentage_preserved(self):
        msg = "High crowd density: 78.3%"
        result_mr = self.tr.translate_message(msg, "mr")
        assert "78.3%" in result_mr

    def test_acoustic_db_preserved(self):
        msg = "Acoustic panic detected: 91.2 dB"
        result_hi = self.tr.translate_message(msg, "hi")
        assert "91.2" in result_hi
        assert "dB" in result_hi

    def test_zone_intrusion_numbers_preserved(self):
        msg = "Zone A1 intrusion: 12 people"
        result_hi = self.tr.translate_message(msg, "hi")
        assert "A1" in result_hi
        assert "12" in result_hi

    def test_lost_child_count_preserved(self):
        msg = "Possible lost child(ren) detected: 3"
        result_mr = self.tr.translate_message(msg, "mr")
        assert "3" in result_mr

    # ── translate_alert preserves metadata fields ──

    def test_translate_alert_preserves_level(self):
        alert = {"message": "SAFE", "level": "WARNING", "time_str": "14:30:05"}
        out = self.tr.translate_alert(alert, "hi")
        assert out["level"] == "WARNING"
        assert out["original_message"] == "SAFE"

    def test_translate_feed_batch(self):
        alerts = [
            {"message": "SAFE", "level": "SAFE"},
            {"message": "High crowd density: 60.0%", "level": "DANGER"},
        ]
        out = self.tr.translate_feed(alerts, "mr")
        assert len(out) == 2
        assert out[0]["level"] == "SAFE"
        assert "60.0%" in out[1]["message"]

    # ── Timestamps embedded in sentences survive ──

    def test_timestamp_preserved_in_keyword_fallback(self):
        msg = "Alert driven by: density 10:45:22 spike"
        result = self.tr.translate_message(msg, "hi")
        assert "10:45:22" in result


# ═══════════════════════════════════════════════════════════════════════════
# 4. RETRIEVAL RELEVANCE TESTS — SemanticIncidentSearcher
# ═══════════════════════════════════════════════════════════════════════════

class TestRetrievalRelevance:
    """Verify semantic search returns relevant results for domain queries."""

    @pytest.fixture
    def searcher(self, seeded_db):
        """Searcher backed by the seeded test DB."""
        from src.nlp_assistant.semantic_search import SemanticIncidentSearcher
        s = SemanticIncidentSearcher(db_path=seeded_db)
        return s

    def test_density_query_returns_density_event(self, searcher):
        resp = searcher.search("density surge")
        assert resp.count > 0
        types = [r.event_type for r in resp.results]
        assert "DENSITY_SURGE" in types

    def test_speed_query_returns_speed_event(self, searcher):
        resp = searcher.search("speed spike running")
        assert resp.count > 0
        types = [r.event_type for r in resp.results]
        assert "SPEED_SPIKE" in types

    def test_acoustic_query_returns_acoustic_event(self, searcher):
        resp = searcher.search("screaming noise panic loud")
        assert resp.count > 0
        types = [r.event_type for r in resp.results]
        assert "ACOUSTIC_PANIC" in types

    def test_direction_query(self, searcher):
        resp = searcher.search("direction surge aligned")
        assert resp.count > 0
        types = [r.event_type for r in resp.results]
        assert "DIRECTION_SURGE" in types

    def test_empty_db_returns_zero(self, tmp_db):
        from src.nlp_assistant.semantic_search import SemanticIncidentSearcher
        s = SemanticIncidentSearcher(db_path=tmp_db)
        resp = s.search("anything")
        assert resp.count == 0
        assert resp.results == []

    def test_similarity_scores_bounded(self, searcher):
        resp = searcher.search("density")
        for item in resp.results:
            assert 0.0 <= item.similarity <= 1.0

    def test_limit_respected(self, searcher):
        resp = searcher.search("surge", limit=2)
        assert len(resp.results) <= 2

    def test_query_string_echoed(self, searcher):
        resp = searcher.search("hello world")
        assert resp.query == "hello world"


# ═══════════════════════════════════════════════════════════════════════════
# 5. PDF GENERATION TEST — IncidentReportManager
# ═══════════════════════════════════════════════════════════════════════════

class TestPDFGeneration:
    """Verify PDF file is created and is a valid PDF."""

    def test_export_pdf_creates_valid_file(self, tmp_db, tmp_path):
        """Create a report, export it, and verify the file starts with %PDF."""
        with patch("src.nlp_assistant.report_generator.REPORT_DIR", str(tmp_path)), \
             patch("src.nlp_assistant.report_generator.RECORDING_DIR", str(tmp_path)):
            from src.nlp_assistant.report_generator import IncidentReportManager
            mgr = IncidentReportManager(db_path=tmp_db)
            mgr.reports_dir = str(tmp_path / "incidents")
            os.makedirs(mgr.reports_dir, exist_ok=True)

            report = mgr.create_report(
                event_type="DENSITY_SURGE",
                severity="CRITICAL",
                camera_id="CAM-2",
                zone="Zone-B",
                count=88,
                density=0.72,
                occupancy_pct=88.0,
                risk_score=76.5,
                data={"avg_speed": 3.2, "acoustic_db": 80.1, "direction_entropy": 0.35,
                      "speed_multiplier": 1.8, "acoustic_panic": True},
            )
            pdf_path = mgr.export_pdf(report.id)

            assert os.path.isfile(pdf_path)
            assert pdf_path.endswith(".pdf")
            with open(pdf_path, "rb") as f:
                header = f.read(5)
            assert header == b"%PDF-"

    def test_export_pdf_nonexistent_report_raises(self, tmp_db, tmp_path):
        with patch("src.nlp_assistant.report_generator.REPORT_DIR", str(tmp_path)), \
             patch("src.nlp_assistant.report_generator.RECORDING_DIR", str(tmp_path)):
            from src.nlp_assistant.report_generator import IncidentReportManager
            mgr = IncidentReportManager(db_path=tmp_db)
            with pytest.raises(ValueError, match="not found"):
                mgr.export_pdf("INC-FAKE-0000")


# ═══════════════════════════════════════════════════════════════════════════
# 6. RATE LIMITER TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestRateLimiter:
    """Verify rate limit enforcement."""

    def test_allows_under_limit(self):
        rl = RateLimiter(max_requests=5, window_seconds=60)
        for _ in range(5):
            assert rl.check("user1") is True

    def test_blocks_over_limit(self):
        rl = RateLimiter(max_requests=3, window_seconds=60)
        for _ in range(3):
            assert rl.check("user1") is True
        assert rl.check("user1") is False

    def test_separate_keys_independent(self):
        rl = RateLimiter(max_requests=2, window_seconds=60)
        assert rl.check("a") is True
        assert rl.check("a") is True
        assert rl.check("a") is False  # a is exhausted
        assert rl.check("b") is True   # b is independent

    def test_window_expiry(self):
        """After the window elapses, requests are allowed again."""
        rl = RateLimiter(max_requests=2, window_seconds=1)
        assert rl.check("u") is True
        assert rl.check("u") is True
        assert rl.check("u") is False
        time.sleep(1.1)
        assert rl.check("u") is True  # window expired

    def test_custom_override_params(self):
        rl = RateLimiter(max_requests=100, window_seconds=100)
        # Override to very tight limit
        for _ in range(2):
            assert rl.check("c", max_requests=2, window_seconds=60) is True
        assert rl.check("c", max_requests=2, window_seconds=60) is False

    def test_thread_safety(self):
        """Rate limiter should be safe under concurrent access."""
        rl = RateLimiter(max_requests=50, window_seconds=60)
        results = []

        def do_check():
            for _ in range(10):
                results.append(rl.check("shared_key"))

        threads = [threading.Thread(target=do_check) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 50
        assert results.count(True) == 50  # all should pass under limit

    def test_require_rate_limit_raises_429(self):
        """The FastAPI dependency raises HTTPException 429 when limit is exceeded."""
        from fastapi import HTTPException

        mock_request = MagicMock()
        mock_request.client.host = "10.0.0.1"
        mock_request.url.path = "/api/test"

        # Drain the limiter for this key
        key = "10.0.0.1:/api/test"
        for _ in range(40):
            default_limiter.check(key)

        with pytest.raises(HTTPException) as exc_info:
            require_rate_limit(mock_request, max_requests=40, window_seconds=60)
        assert exc_info.value.status_code == 429


# ═══════════════════════════════════════════════════════════════════════════
# 7. SCHEMA VALIDATION TESTS — Pydantic v2 Models
# ═══════════════════════════════════════════════════════════════════════════

class TestSchemaValidation:
    """Verify Pydantic models accept valid input and reject invalid input."""

    # ── ChatQueryRequest ──

    def test_chat_query_valid(self):
        r = ChatQueryRequest(query="How many people?")
        assert r.query == "How many people?"

    def test_chat_query_with_session(self):
        r = ChatQueryRequest(query="status", session_id="abc")
        assert r.session_id == "abc"

    def test_chat_query_empty_rejected(self):
        with pytest.raises(Exception):
            ChatQueryRequest(query="")

    def test_chat_query_too_long_rejected(self):
        with pytest.raises(Exception):
            ChatQueryRequest(query="x" * 501)

    # ── IncidentReportCreate ──

    def test_incident_create_valid(self):
        r = IncidentReportCreate(event_type="STAMPEDE_RISK", severity="CRITICAL")
        assert r.camera_id == "CAM-1"  # default
        assert r.zone == "Zone-1"      # default

    def test_incident_create_density_bounds(self):
        r = IncidentReportCreate(event_type="X", severity="Y", density=0.5)
        assert r.density == 0.5
        with pytest.raises(Exception):
            IncidentReportCreate(event_type="X", severity="Y", density=1.5)
        with pytest.raises(Exception):
            IncidentReportCreate(event_type="X", severity="Y", density=-0.1)

    def test_incident_create_count_non_negative(self):
        r = IncidentReportCreate(event_type="X", severity="Y", count=0)
        assert r.count == 0
        with pytest.raises(Exception):
            IncidentReportCreate(event_type="X", severity="Y", count=-1)

    # ── VerificationUpdate ──

    def test_verification_valid_statuses(self):
        for s in ("UNVERIFIED", "VERIFIED_FALSE_POSITIVE", "CONFIRMED_HAZARD", "RESOLVED"):
            v = VerificationUpdate(status=s)
            assert v.status == s

    def test_verification_invalid_status_rejected(self):
        with pytest.raises(Exception):
            VerificationUpdate(status="INVALID")

    # ── SemanticSearchRequest ──

    def test_search_request_valid(self):
        r = SemanticSearchRequest(query="stampede near gate")
        assert r.limit == 10  # default
        assert r.min_similarity == 0.05  # default

    def test_search_request_empty_rejected(self):
        with pytest.raises(Exception):
            SemanticSearchRequest(query="")

    def test_search_request_limit_bounds(self):
        with pytest.raises(Exception):
            SemanticSearchRequest(query="x", limit=0)
        with pytest.raises(Exception):
            SemanticSearchRequest(query="x", limit=51)

    # ── TranslateAlertRequest ──

    def test_translate_request_valid(self):
        r = TranslateAlertRequest(alerts=[{"message": "hi"}], target_lang="hi")
        assert r.target_lang == "hi"

    def test_translate_request_invalid_lang(self):
        with pytest.raises(Exception):
            TranslateAlertRequest(alerts=[], target_lang="fr")

    # ── ChatQueryResponse round-trip ──

    def test_chat_response_round_trip(self):
        r = ChatQueryResponse(
            answer="42 people",
            intent="OCCUPANCY_COUNT",
            confidence=0.98,
            sources=[ChatSource(source_type="Live", timestamp="10:00:00", details="cam")],
            data={"count": 42},
            timestamp="10:00:00",
        )
        d = r.model_dump()
        assert d["data"]["count"] == 42
        r2 = ChatQueryResponse.model_validate(d)
        assert r2.answer == "42 people"

    # ── IncidentReport ──

    def test_incident_report_defaults(self):
        r = IncidentReport(
            id="INC-1",
            timestamp=time.time(),
            time_str="2026-01-01 00:00:00",
            event_type="TEST",
            severity="SAFE",
            camera_id="CAM-1",
            zone="Zone-1",
            count=0,
            density=0.0,
            occupancy_pct=0.0,
            risk_score=0.0,
        )
        assert r.verification_status == "UNVERIFIED"
        assert r.data == {}
        assert r.evidence_url is None


# ═══════════════════════════════════════════════════════════════════════════
# 8. REPORT MANAGER CRUD TESTS — IncidentReportManager
# ═══════════════════════════════════════════════════════════════════════════

class TestReportManagerCRUD:
    """Verify CRUD operations for incident reports."""

    @pytest.fixture
    def mgr(self, tmp_db, tmp_path):
        with patch("src.nlp_assistant.report_generator.REPORT_DIR", str(tmp_path)), \
             patch("src.nlp_assistant.report_generator.RECORDING_DIR", str(tmp_path)):
            from src.nlp_assistant.report_generator import IncidentReportManager
            m = IncidentReportManager(db_path=tmp_db)
            yield m

    def test_create_report(self, mgr):
        r = mgr.create_report(event_type="STAMPEDE_RISK", severity="CRITICAL", count=50, density=0.65)
        assert r.id.startswith("INC-")
        assert r.event_type == "STAMPEDE_RISK"
        assert r.severity == "CRITICAL"
        assert r.count == 50
        assert r.density == 0.65
        assert r.verification_status == "UNVERIFIED"

    def test_get_report(self, mgr):
        r = mgr.create_report(event_type="DENSITY_SURGE", severity="DANGER")
        fetched = mgr.get_report(r.id)
        assert fetched is not None
        assert fetched.id == r.id
        assert fetched.event_type == "DENSITY_SURGE"

    def test_get_nonexistent_returns_none(self, mgr):
        assert mgr.get_report("INC-DOES-NOT-EXIST") is None

    def test_list_reports_empty(self, mgr):
        assert mgr.list_reports() == []

    def test_list_reports_order(self, mgr):
        r1 = mgr.create_report(event_type="A", severity="SAFE")
        time.sleep(0.05)
        r2 = mgr.create_report(event_type="B", severity="WARNING")
        reports = mgr.list_reports()
        assert len(reports) == 2
        # Most recent first
        assert reports[0].id == r2.id
        assert reports[1].id == r1.id

    def test_list_reports_limit(self, mgr):
        for i in range(5):
            mgr.create_report(event_type=f"T{i}", severity="SAFE")
        assert len(mgr.list_reports(limit=3)) == 3

    def test_update_verification(self, mgr):
        r = mgr.create_report(event_type="TEST", severity="DANGER")
        updated = mgr.update_verification(r.id, "CONFIRMED_HAZARD", operator_notes="Confirmed by control room")
        assert updated is not None
        assert updated.verification_status == "CONFIRMED_HAZARD"
        assert updated.operator_notes == "Confirmed by control room"
        assert updated.verified_at is not None

    def test_update_verification_nonexistent(self, mgr):
        result = mgr.update_verification("INC-NOPE", "RESOLVED")
        assert result is None

    def test_update_verification_preserves_notes_when_none(self, mgr):
        r = mgr.create_report(event_type="T", severity="SAFE", operator_notes="original note")
        updated = mgr.update_verification(r.id, "RESOLVED")
        assert updated.operator_notes == "original note"

    def test_create_report_stores_data(self, mgr):
        data = {"speed": 3.5, "zones": [1, 2]}
        r = mgr.create_report(event_type="T", severity="WARNING", data=data)
        fetched = mgr.get_report(r.id)
        assert fetched.data == data

    def test_report_id_uniqueness(self, mgr):
        ids = set()
        for _ in range(10):
            r = mgr.create_report(event_type="T", severity="SAFE")
            ids.add(r.id)
        assert len(ids) == 10
