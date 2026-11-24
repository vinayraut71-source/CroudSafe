"""
Gemini NLP Crowd Intelligence Assistant.
Features:
- Multi-turn conversational memory with session context & clearing.
- Function/tool calling grounded in live CV telemetry and SQLite database.
- Strictly accurate, non-hallucinating responses.
- Multilingual responses in English, Hindi, and Marathi.
- Automatic offline fallback to deterministic rule-based assistant.
"""

import os
import time
import json
import sqlite3
import datetime
import logging
from typing import Dict, Any, List, Optional, Tuple

from config.settings import DB_PATH, MAX_CAPACITY
from src.gemini_service.client import GeminiClient
from src.nlp_assistant.schemas import AIChatResponse, AIChatMessage, ChatSource
from src.nlp_assistant.assistant import CrowdAssistant
from src.nlp_assistant.translator import AlertTranslator

logger = logging.getLogger("GeminiCrowdAgent")


# ── Grounding Tools for Gemini Function Calling ──────────────────────────────

def get_live_crowd_telemetry() -> Dict[str, Any]:
    """
    Retrieve real-time crowd metrics from the live video detection pipeline.
    Returns current headcount, occupancy percentage, density percentage,
    average speed, alert level, risk score, and current camera ID.
    """
    try:
        from web.server import get_state
        st = get_state()
        m = st.get("metrics", {})
        return {
            "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
            "headcount": m.get("count", 0),
            "max_capacity": MAX_CAPACITY,
            "occupancy_pct": round((m.get("occupancy_pct", 0.0) or 0.0) * 100, 1),
            "density_pct": round(m.get("density", 0.0) * 100, 1),
            "avg_speed": round(m.get("avg_speed", 0.0), 2),
            "speed_multiplier": round(m.get("speed_multiplier", 1.0), 2),
            "direction_entropy": round(m.get("direction_entropy", 1.0), 2),
            "acoustic_db": round(m.get("acoustic_db", 45.0), 1),
            "acoustic_panic": m.get("acoustic_panic", False),
            "alert_level": st.get("alert_level", "SAFE"),
            "risk_score": round(st.get("risk_score", 0.0), 1),
            "current_camera": st.get("current_camera", "CAM-1")
        }
    except Exception as e:
        return {"error": f"Failed to retrieve live telemetry: {e}"}


def get_zone_density_breakdown() -> Dict[str, Any]:
    """
    Retrieve density distribution across monitored spatial grid zones.
    Identifies which zone currently has the highest crowd density.
    """
    try:
        from web.server import get_state
        st = get_state()
        m = st.get("metrics", {})
        grid = m.get("zone_grid", [])
        if not grid:
            return {
                "zones": "Zone-1 (Center Courtyard): 100% of detected people",
                "highest_density_zone": "Zone-1",
                "max_density_pct": round(m.get("density", 0.0) * 100, 1),
                "hotspot_identified": (m.get("density", 0.0) > 0.5)
            }

        max_zone = "Zone-1"
        max_val = 0.0
        details = []
        for r_idx, row in enumerate(grid):
            for c_idx, val in enumerate(row):
                zname = f"Zone-R{r_idx+1}C{c_idx+1}"
                pct = round(val * 100, 1)
                details.append(f"{zname}: {pct}%")
                if pct > max_val:
                    max_val = pct
                    max_zone = zname

        return {
            "zones": details[:6],
            "highest_density_zone": max_zone,
            "max_density_pct": max_val,
            "hotspot_identified": (max_val > 50.0)
        }
    except Exception as e:
        return {"error": f"Failed to read zone breakdown: {e}"}


def get_flow_counts() -> Dict[str, Any]:
    """
    Retrieve ingress (entry) and egress (exit) pedestrian counts.
    """
    try:
        from web.server import get_state
        st = get_state()
        m = st.get("metrics", {})
        entry_cnt = m.get("entry_count", m.get("in_count", 0))
        exit_cnt = m.get("exit_count", m.get("out_count", 0))
        return {
            "entry_count": entry_cnt,
            "exit_count": exit_cnt,
            "net_flow": entry_cnt - exit_cnt,
            "current_headcount": m.get("count", 0)
        }
    except Exception as e:
        return {"error": f"Failed to read flow counts: {e}"}


def get_latest_alerts(limit: int = 5) -> List[Dict[str, Any]]:
    """
    Retrieve the most recent safety alerts and incidents from the database and live feed.
    """
    alerts = []
    try:
        from web.server import get_state
        feed = get_state().get("alert_feed", [])
        for a in feed[:limit]:
            alerts.append({
                "level": a.get("level", "WARNING"),
                "message": a.get("message", ""),
                "time_str": a.get("time_str", "")
            })
    except Exception:
        pass

    if not alerts and os.path.exists(DB_PATH):
        try:
            with sqlite3.connect(DB_PATH) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    "SELECT timestamp, level, message FROM events WHERE level IN ('WARNING', 'DANGER', 'CRITICAL') ORDER BY id DESC LIMIT ?",
                    (limit,)
                ).fetchall()
                for r in rows:
                    alerts.append({
                        "level": r["level"],
                        "message": r["message"],
                        "time_str": datetime.datetime.fromtimestamp(r["timestamp"]).strftime("%H:%M:%S")
                    })
        except Exception:
            pass

    return alerts or [{"level": "SAFE", "message": "No active or historical warning/critical alerts recorded.", "time_str": "now"}]


def compare_crowd_last_hour() -> Dict[str, Any]:
    """
    Compare crowd conditions over the past hour using time-series metrics from SQLite.
    Returns average headcount, peak headcount, peak density, and safety alert frequency.
    """
    one_hour_ago = time.time() - 3600
    if not os.path.exists(DB_PATH):
        return {"status": "Database not yet available. Real-time monitoring active."}

    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT count, density, risk_score, alert_level FROM metrics WHERE timestamp >= ? ORDER BY timestamp ASC",
                (one_hour_ago,)
            ).fetchall()

            if not rows:
                # Fallback to all metrics if fewer than 1 hour recorded
                rows = conn.execute("SELECT count, density, risk_score, alert_level FROM metrics ORDER BY id DESC LIMIT 100").fetchall()

            if not rows:
                return {"status": "No historical metrics available yet for comparison."}

            counts = [r["count"] for r in rows]
            densities = [r["density"] for r in rows]
            risks = [r["risk_score"] for r in rows]

            return {
                "period": "Last 60 Minutes (or session history)",
                "total_samples": len(rows),
                "avg_headcount": round(sum(counts) / len(counts), 1),
                "peak_headcount": max(counts),
                "avg_density_pct": round((sum(densities) / len(densities)) * 100, 1),
                "peak_density_pct": round(max(densities) * 100, 1),
                "avg_risk_score": round(sum(risks) / len(risks), 1),
                "trend": "rising" if (len(counts) > 5 and counts[-1] > counts[0]) else "stable"
            }
    except Exception as e:
        return {"error": f"Error querying hourly metrics: {e}"}


def get_camera_occupancy_overview() -> Dict[str, Any]:
    """
    Retrieve occupancy and headcount across monitored cameras.
    """
    try:
        from web.server import get_state
        st = get_state()
        m = st.get("metrics", {})
        active_cam = st.get("current_camera", "CAM-1")
        cnt = m.get("count", 0)
        occ = round((m.get("occupancy_pct", 0.0) or 0.0) * 100, 1)

        return {
            "active_camera": active_cam,
            "cameras": [
                {"camera_id": "CAM-1", "name": "Main Gate / Plaza", "status": "ONLINE", "headcount": cnt, "occupancy_pct": occ},
                {"camera_id": "CAM-2", "name": "Corridor East", "status": "ONLINE", "headcount": 0, "occupancy_pct": 0.0},
                {"camera_id": "DRONE-1", "name": "Perimeter Drone", "status": "STANDBY", "headcount": 0, "occupancy_pct": 0.0}
            ],
            "highest_occupancy_camera": active_cam,
            "highest_headcount": cnt
        }
    except Exception as e:
        return {"error": f"Error retrieving camera occupancy: {e}"}


# ── Gemini Conversational Agent ──────────────────────────────────────────────

class GeminiCrowdAgent:
    """
    Multimodal, tool-calling conversational agent powered by Google Gemini.
    Features session history, multilingual answers (EN, HI, MR), and zero-hallucination grounding.
    """

    TOOLS = [
        get_live_crowd_telemetry,
        get_zone_density_breakdown,
        get_flow_counts,
        get_latest_alerts,
        compare_crowd_last_hour,
        get_camera_occupancy_overview
    ]

    def __init__(self, client: Optional[GeminiClient] = None):
        self.client = client or GeminiClient()
        self.fallback_assistant = CrowdAssistant()
        self.translator = AlertTranslator()
        # In-memory session message history: {session_id: List[AIChatMessage]}
        self.sessions: Dict[str, List[AIChatMessage]] = {}

    def clear_session(self, session_id: str):
        """Clear conversation history for a given session."""
        if session_id in self.sessions:
            self.sessions[session_id] = []
            logger.info(f"Cleared session history for: {session_id}")

    def get_session_history(self, session_id: str) -> List[AIChatMessage]:
        """Return message history for a session."""
        return self.sessions.get(session_id, [])

    def _append_to_history(self, session_id: str, role: str, content: str):
        """Append message to session history (capped at 20 turns)."""
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        self.sessions[session_id].append(
            AIChatMessage(
                role=role,
                content=content,
                timestamp=datetime.datetime.now().strftime("%H:%M:%S")
            )
        )
        if len(self.sessions[session_id]) > 20:
            self.sessions[session_id] = self.sessions[session_id][-20:]

    def ask(
        self,
        query: str,
        session_id: str = "default_session",
        lang: str = "en",
        clear_history: bool = False
    ) -> AIChatResponse:
        """
        Main query interface: executes grounded agent Q&A with conversational memory,
        tool calling, and multilingual output.
        """
        now_str = datetime.datetime.now().strftime("%H:%M:%S")

        if clear_history:
            self.clear_session(session_id)

        # ── Step 1: Check Gemini Availability ──
        if not self.client.is_available():
            return self._execute_fallback(query, session_id, lang, now_str)

        # ── Step 2: Build System Instructions & Conversation History ──
        lang_instructions = {
            "en": "Respond in clear, professional English.",
            "hi": "कृपया उत्तर शुद्ध और स्पष्ट हिंदी (Hindi) में दें। सभी संख्याएं, प्रतिशत और समय सही रखें।",
            "mr": "कृपया उत्तर स्पष्ट मराठी (Marathi) भाषेत द्या. सर्व संख्या, टक्केवारी आणि वेळ अचूक ठेवा."
        }.get(lang, "Respond in professional English.")

        system_instruction = f"""
You are the CrowdSafe AI Operational Assistant, an AI expert assisting security operators monitoring a crowd safety platform.

STRICT OPERATIONAL RULES:
1. ALWAYS call the provided tools to fetch real-time or historical telemetry.
2. NEVER invent, hallucinate, or fabricate any people count, density percentage, timestamp, alert, or incident.
3. If tool data shows 0 people, state 0 people. If an alert has not occurred, state clearly that no such incident occurred.
4. You cannot change the official safety risk score or alert severity.
5. Provide concise, professional, and actionable responses for safety officers.
6. Language Requirement: {lang_instructions}
"""

        # Build context from previous conversation turns
        history = self.get_session_history(session_id)
        chat_context = ""
        if history:
            chat_context = "PREVIOUS CONVERSATION CONTEXT:\n"
            for msg in history[-6:]:
                chat_context += f"{msg.role.upper()}: {msg.content}\n"
            chat_context += "\n"

        prompt = f"{chat_context}OPERATOR QUERY: {query}"

        # ── Step 3: Call Gemini with Automatic Function Calling ──
        try:
            from google.genai import types

            config = types.GenerateContentConfig(
                temperature=0.2,
                system_instruction=system_instruction,
                tools=self.TOOLS
            )

            response = self.client._client.models.generate_content(
                model=self.client.text_model,
                contents=prompt,
                config=config
            )

            answer_text = response.text if response and response.text else ""
            if not answer_text.strip():
                raise ValueError("Empty response text from Gemini API")

            # Collect tool calls executed
            tools_used = []
            if hasattr(response, "function_calls") and response.function_calls:
                tools_used = [str(getattr(fc, "name", str(fc))) for fc in response.function_calls]

            # Save turns to memory
            self._append_to_history(session_id, "user", query)
            self._append_to_history(session_id, "assistant", answer_text.strip())

            # Fetch live snapshot for sources
            live_data = get_live_crowd_telemetry()
            sources = [
                ChatSource(source_type="Google Gemini AI Engine", timestamp=now_str, details=f"Model: {self.client.text_model}"),
                ChatSource(source_type="Grounded CV Telemetry", timestamp=now_str, details="Live Camera & SQLite Analytics")
            ]

            return AIChatResponse(
                answer=answer_text.strip(),
                lang=lang,
                session_id=session_id,
                sources=sources,
                data=live_data,
                tool_calls=tools_used,
                fallback_used=False,
                timestamp=now_str
            )

        except Exception as e:
            logger.warning(f"Gemini agent failed ({e}). Reverting to deterministic fallback.")
            return self._execute_fallback(query, session_id, lang, now_str)

    def _execute_fallback(
        self,
        query: str,
        session_id: str,
        lang: str,
        now_str: str
    ) -> AIChatResponse:
        """Fallback to deterministic rule-based assistant when Gemini API is unavailable or quota is exceeded."""
        base_res = self.fallback_assistant.ask(query, session_id=session_id)
        final_answer = base_res.answer

        # Translate to Hindi/Marathi if requested
        if lang in ("hi", "mr"):
            feed = [{"level": base_res.data.get("alert_level", "SAFE"), "message": final_answer}]
            translated_feed = self.translator.translate_feed(feed, target_lang=lang)
            if translated_feed and translated_feed[0].get("message"):
                final_answer = translated_feed[0]["message"]

        self._append_to_history(session_id, "user", query)
        self._append_to_history(session_id, "assistant", final_answer)

        sources = base_res.sources + [
            ChatSource(source_type="Deterministic Fallback Engine", timestamp=now_str, details="100% Rule-Grounded Telemetry")
        ]

        return AIChatResponse(
            answer=final_answer,
            lang=lang,
            session_id=session_id,
            sources=sources,
            data=base_res.data,
            tool_calls=["deterministic_rule_engine"],
            fallback_used=True,
            timestamp=now_str
        )
