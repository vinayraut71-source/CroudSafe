"""
AI Incident Explanation Engine.
Synthesizes structured CV analytics, acoustic indicators, and camera snapshots
to generate clear, human-readable causal explanations and practical operator actions.
Strictly preserves official alert levels and deterministic risk scores.
"""

import os
import json
import time
import sqlite3
import datetime
import logging
from typing import Optional, Dict, Any, List

from config.settings import DB_PATH
from src.gemini_service.client import GeminiClient
from src.gemini_service.vision_analyst import GeminiVisionAnalyst
from src.nlp_assistant.schemas import IncidentExplanationResponse

logger = logging.getLogger("IncidentExplainer")


class IncidentExplainer:
    """
    Explains crowd safety alerts and incidents using Gemini reasoning.
    Distinguishes sensor-measured triggers from AI narrative recommendations.
    """

    def __init__(self, client: Optional[GeminiClient] = None):
        self.client = client or GeminiClient()
        self.vision_analyst = GeminiVisionAnalyst(self.client)

    def explain_incident(
        self,
        incident_id: Optional[str] = None,
        alert_payload: Optional[Dict[str, Any]] = None,
        lang: str = "en",
        include_snapshot: bool = True
    ) -> IncidentExplanationResponse:
        """
        Produce a causal explanation and actionable recommendations for an alert or incident.
        """
        now_str = datetime.datetime.now().strftime("%H:%M:%S")

        # ── Step 1: Gather Incident Telemetry ──
        metrics = {}
        alert_level = "WARNING"
        official_risk_score = 0.0
        snapshot_url = None

        if incident_id and os.path.exists(DB_PATH):
            try:
                with sqlite3.connect(DB_PATH) as conn:
                    conn.row_factory = sqlite3.Row
                    row = conn.execute("SELECT * FROM incidents WHERE id = ?", (incident_id,)).fetchone()
                    if row:
                        alert_level = row["severity"]
                        official_risk_score = float(row["risk_score"] or 0.0)
                        snapshot_url = row["evidence_url"]
                        metrics = json.loads(row["data"]) if row["data"] else {}
                        metrics["count"] = row["count"]
                        metrics["density"] = row["density"]
                        metrics["occupancy_pct"] = row["occupancy_pct"]
            except Exception as e:
                logger.warning(f"Error fetching incident {incident_id}: {e}")

        if not metrics:
            # Fallback to provided payload or live state
            if alert_payload:
                metrics = alert_payload.get("metrics", alert_payload)
                alert_level = alert_payload.get("level", alert_payload.get("alert_level", "WARNING"))
                official_risk_score = float(alert_payload.get("risk_score", metrics.get("risk_score", 0.0)))
            else:
                try:
                    from web.server import get_state
                    st = get_state()
                    metrics = st.get("metrics", {})
                    alert_level = st.get("alert_level", "WARNING")
                    official_risk_score = float(st.get("risk_score", 0.0))
                except Exception:
                    pass

        # ── Step 2: Determine Causal Indicators ──
        causal_indicators = self._identify_causal_factors(metrics, alert_level)

        # ── Step 3: Capture Camera Snapshot if requested & available ──
        frame_bytes = None
        if include_snapshot and not snapshot_url:
            frame_bytes, _, snap_url = self.vision_analyst.capture_live_frame()
            if snap_url:
                snapshot_url = snap_url

        # ── Step 4: Generate Gemini Explanation or Heuristic Fallback ──
        if self.client.is_available():
            try:
                explanation, recommendations = self._generate_gemini_explanation(
                    alert_level=alert_level,
                    official_risk_score=official_risk_score,
                    metrics=metrics,
                    causal_indicators=causal_indicators,
                    frame_bytes=frame_bytes,
                    lang=lang,
                    now_str=now_str
                )
                fallback_used = False
            except Exception as e:
                logger.warning(f"Gemini explanation error ({e}). Reverting to deterministic explanation.")
                explanation, recommendations = self._generate_heuristic_explanation(
                    alert_level, official_risk_score, metrics, causal_indicators, lang
                )
                fallback_used = True
        else:
            explanation, recommendations = self._generate_heuristic_explanation(
                alert_level, official_risk_score, metrics, causal_indicators, lang
            )
            fallback_used = True

        # ── Step 5: Persist AI Explanation to DB if incident_id exists ──
        if incident_id and os.path.exists(DB_PATH):
            try:
                with sqlite3.connect(DB_PATH) as conn:
                    conn.execute("""
                        UPDATE incidents 
                        SET ai_explanation = ?, visual_observations = ?, evidence_url = COALESCE(evidence_url, ?)
                        WHERE id = ?
                    """, (explanation, "\n".join(recommendations), snapshot_url, incident_id))
                    conn.commit()
            except Exception as e:
                logger.warning(f"Failed to update incident {incident_id} with explanation: {e}")

        return IncidentExplanationResponse(
            incident_id=incident_id,
            alert_level=alert_level,
            official_risk_score=official_risk_score,
            causal_indicators=causal_indicators,
            gemini_explanation=explanation,
            recommended_actions=recommendations,
            visual_evidence=snapshot_url,
            timestamp=now_str,
            fallback_used=fallback_used
        )

    def _identify_causal_factors(self, metrics: Dict[str, Any], alert_level: str) -> List[str]:
        """Isolate the specific sensor thresholds that contributed to the alert."""
        factors = []
        density = metrics.get("density", 0.0)
        density_pct = round(density * 100, 1) if density <= 1.0 else round(density, 1)
        speed_mult = round(metrics.get("speed_multiplier", 1.0), 2)
        entropy = round(metrics.get("direction_entropy", 1.0), 2)
        acoustic_panic = metrics.get("acoustic_panic", False)
        acoustic_db = metrics.get("acoustic_db", 0.0)
        compressed = metrics.get("compressed", False)

        if density_pct >= 55.0:
            factors.append(f"High Area Density ({density_pct}% coverage exceeding 55% safe threshold)")
        elif density_pct >= 30.0:
            factors.append(f"Elevated Density ({density_pct}% coverage entering warning band)")

        if speed_mult >= 2.0:
            factors.append(f"Rapid Acceleration / Surge Velocity ({speed_mult}x normal baseline)")
        elif speed_mult >= 1.5:
            factors.append(f"Elevated Flow Velocity ({speed_mult}x baseline)")

        if entropy < 0.4:
            factors.append(f"Directional Alignment Surge (Entropy {entropy} indicates unidirectional rushing)")

        if acoustic_panic or acoustic_db >= 80.0:
            factors.append(f"Acoustic Alarm ({round(acoustic_db, 1)} dB frequency spike indicating screaming or panic)")

        if compressed:
            factors.append("Spatial Compression Detected (Inter-person proximity below critical 50px buffer)")

        if not factors:
            factors.append(f"Composite risk evaluation flagged safety state ({alert_level})")

        return factors

    def _generate_gemini_explanation(
        self,
        alert_level: str,
        official_risk_score: float,
        metrics: Dict[str, Any],
        causal_indicators: List[str],
        frame_bytes: Optional[bytes],
        lang: str,
        now_str: str
    ) -> Tuple[str, List[str]]:
        """Call Gemini to synthesize a concise causal explanation and recommendations."""
        lang_prompt = {
            "en": "Respond in concise, professional English.",
            "hi": "कृपया उत्तर शुद्ध और स्पष्ट हिंदी (Hindi) में दें।",
            "mr": "कृपया उत्तर स्पष्ट मराठी (Marathi) भाषेत द्या."
        }.get(lang, "Respond in English.")

        prompt = f"""
You are the Incident Intelligence Explainer for the CrowdSafe AI surveillance platform.
Timestamp: {now_str}
OFFICIAL ALERT SEVERITY: {alert_level} (Deterministic)
OFFICIAL RISK SCORE: {official_risk_score}% (Deterministic)

MEASURED TELEMETRY INDICATORS:
- Causal Factors Flagged by Safety Engine: {json.dumps(causal_indicators)}
- Full Metrics: {json.dumps(metrics)}

TASK:
1. Explain in 2-3 clear, professional sentences what physical conditions triggered this {alert_level} alert.
2. Directly reference the specific contributing measurements (e.g. density, speed surge, acoustic level).
3. Do NOT contradict or change the official alert severity or risk score.
4. Provide 3 specific, practical, and actionable instructions for the security/facilities operator.
5. Language: {lang_prompt}

OUTPUT FORMAT (Respond STRICTLY as JSON with these keys):
{{
  "explanation": "2-3 sentence narrative explaining the causal factors",
  "recommendations": ["Operator Action 1", "Operator Action 2", "Operator Action 3"]
}}
"""

        system_instruction = (
            "You are a crowd safety expert analyst. Explain incidents based strictly on provided sensor metrics. "
            "Never minimize hazards and never invent metrics not in the data. Always output JSON."
        )

        if frame_bytes:
            res = self.client.generate_multimodal(
                prompt=prompt,
                image=frame_bytes,
                system_instruction=system_instruction,
                temperature=0.2
            )
        else:
            res = self.client.generate_text(
                prompt=prompt,
                system_instruction=system_instruction,
                temperature=0.2
            )

        if not res["success"] or not res["text"]:
            raise RuntimeError(res.get("error") or "Failed to receive Gemini explanation")

        raw_text = res["text"].strip()
        if "```json" in raw_text:
            raw_text = raw_text.split("```json")[1].split("```")[0].strip()
        elif "```" in raw_text:
            raw_text = raw_text.split("```")[1].split("```")[0].strip()

        data = json.loads(raw_text)
        return data.get("explanation", ""), data.get("recommendations", [])

    def _generate_heuristic_explanation(
        self,
        alert_level: str,
        official_risk_score: float,
        metrics: Dict[str, Any],
        causal_indicators: List[str],
        lang: str
    ) -> Tuple[str, List[str]]:
        """Deterministic rule-based explanation when Gemini API is unavailable."""
        reasons_str = "; ".join(causal_indicators)
        cnt = metrics.get("count", 0)

        if alert_level == "CRITICAL":
            narrative = (
                f"A CRITICAL safety alert was triggered due to severe crowd convergence. "
                f"Primary contributing indicators: {reasons_str}. "
                f"Monitored headcount is {cnt} with official deterministic risk assessed at {official_risk_score}%."
            )
            recs = [
                "Immediately deploy on-site safety stewards to the affected corridor.",
                "Broadcast automated PA dispersal instructions to redirect pedestrian flow toward secondary exits.",
                "Halt additional ingress at the primary entrance gates."
            ]
        elif alert_level == "DANGER":
            narrative = (
                f"A DANGER alert was triggered due to elevated physical metrics approaching safe containment capacity. "
                f"Contributing factors: {reasons_str}. Current measured risk score is {official_risk_score}%."
            )
            recs = [
                "Verify live camera video and monitor crowd headway closely.",
                "Ensure designated egress pathways and exit gates remain fully unlatched.",
                "Prepare venue marshals for localized crowd metering if density continues to rise."
            ]
        else:
            narrative = (
                f"A WARNING alert was triggered by early boundary deviations. "
                f"Key factors: {reasons_str}. Baseline risk score is {official_risk_score}%."
            )
            recs = [
                "Acknowledge the alert on the operator console.",
                "Inspect adjacent camera sectors to identify incoming pedestrian volume.",
                "Maintain continuous sensor telemetry monitoring."
            ]

        if lang == "hi":
            narrative = f"सुरक्षा चेतावनी ({alert_level}): मुख्य कारण — {reasons_str}। वर्तमान जोखिम स्कोर: {official_risk_score}%।"
        elif lang == "mr":
            narrative = f"सुरक्षा इशारा ({alert_level}): प्रमुख कारणे — {reasons_str}. सध्याचा धोका निर्देशांक: {official_risk_score}%."

        return narrative, recs

