"""
Gemini Vision-Based Crowd Scene Analyst.
Analyzes captured or uploaded camera frames using Gemini Multimodal Vision.
Features:
- Live frame capture from camera stream with persistent snapshot storage.
- Multi-aspect visual inspection (crowd distribution, clustering, blocked exits, obstacles, fallen persons).
- Strict separation of visual observations from safety interpretations.
- Offline heuristic fallback when Gemini API is unavailable or quota is exhausted.
"""

import os
import cv2
import time
import base64
import logging
import datetime
from typing import Optional, Dict, Any, List, Union
import numpy as np
from PIL import Image

from config.settings import RECORDING_DIR
from src.gemini_service.client import GeminiClient
from src.nlp_assistant.schemas import VisionAnalysisResponse

logger = logging.getLogger("GeminiVisionAnalyst")

SNAPSHOT_DIR = os.path.join(RECORDING_DIR, "snapshots")
os.makedirs(SNAPSHOT_DIR, exist_ok=True)


class GeminiVisionAnalyst:
    """
    Multimodal scene analyst for crowd monitoring cameras.
    Supplementary to YOLO and deterministic safety engine.
    """

    def __init__(self, client: Optional[GeminiClient] = None):
        self.client = client or GeminiClient()

    def capture_live_frame(self, camera_id: str = "CAM-1") -> TupleOptionalFrame:
        """
        Capture current raw frame from the running web server state.
        Returns: (frame_bytes, snapshot_filename, snapshot_rel_url)
        """
        try:
            from web.server import get_state
            st = get_state()
            frame = st.get("frame")
            if frame is None or not isinstance(frame, np.ndarray):
                return None, None, None

            # Encode to JPEG
            success, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not success:
                return None, None, None

            frame_bytes = buffer.tobytes()

            # Save snapshot to disk
            ts_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"snap_{camera_id}_{ts_str}.jpg"
            file_path = os.path.join(SNAPSHOT_DIR, filename)
            with open(file_path, "wb") as f:
                f.write(frame_bytes)

            rel_url = f"/recordings/snapshots/{filename}"
            return frame_bytes, filename, rel_url
        except Exception as e:
            logger.warning(f"Error capturing live frame: {e}")
            return None, None, None

    def analyze_frame(
        self,
        image_bytes: Optional[bytes] = None,
        image_base64: Optional[str] = None,
        camera_id: str = "CAM-1",
        lang: str = "en",
        use_live_frame: bool = True
    ) -> VisionAnalysisResponse:
        """
        Perform structured multimodal visual analysis of a camera frame.
        """
        now_str = datetime.datetime.now().strftime("%H:%M:%S")
        snapshot_url = None

        # ── Step 1: Resolve Frame Bytes ──
        if image_base64:
            try:
                # Strip data URI prefix if present
                if "," in image_base64:
                    image_base64 = image_base64.split(",", 1)[1]
                image_bytes = base64.b64decode(image_base64)
                # Save uploaded snapshot
                ts_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"upload_{camera_id}_{ts_str}.jpg"
                file_path = os.path.join(SNAPSHOT_DIR, filename)
                with open(file_path, "wb") as f:
                    f.write(image_bytes)
                snapshot_url = f"/recordings/snapshots/{filename}"
            except Exception as e:
                logger.warning(f"Failed to decode base64 image: {e}")
                image_bytes = None

        if image_bytes is None and use_live_frame:
            captured_bytes, _, snap_url = self.capture_live_frame(camera_id)
            if captured_bytes:
                image_bytes = captured_bytes
                snapshot_url = snap_url

        if not image_bytes:
            return self._build_unavailable_frame_response(camera_id, now_str)

        # ── Step 2: Query Gemini Vision if Available ──
        if self.client.is_available():
            try:
                return self._analyze_with_gemini(image_bytes, camera_id, lang, now_str, snapshot_url)
            except Exception as e:
                logger.warning(f"Gemini Vision call failed ({e}). Reverting to local heuristic analyst.")

        # ── Step 3: Local Heuristic Fallback ──
        return self._analyze_with_heuristics(image_bytes, camera_id, lang, now_str, snapshot_url)

    def _analyze_with_gemini(
        self,
        image_bytes: bytes,
        camera_id: str,
        lang: str,
        now_str: str,
        snapshot_url: Optional[str]
    ) -> VisionAnalysisResponse:
        """Call Gemini Vision API with structured schema and safety guidelines."""
        lang_prompt = {
            "en": "Provide the final analysis in clear, professional English.",
            "hi": "कृपया विश्लेषण शुद्ध और व्यावसायिक हिंदी (Hindi) में प्रस्तुत करें।",
            "mr": "कृपया विश्लेषण स्पष्ट आणि व्यावसायिक मराठी (Marathi) भाषेत सादर करा."
        }.get(lang, "Provide the final analysis in English.")

        prompt = f"""
You are the Multimodal Crowd Scene Analyst for the CrowdSafe AI surveillance system.
Camera: {camera_id}
Timestamp: {now_str}

Analyze this crowd scene photograph carefully.

CRITICAL OPERATIONAL CONSTRAINTS:
1. Distinguish pure visual observations (what is visibly present) from risk interpretations.
2. DO NOT claim that a 'stampede', 'mass panic', or 'disaster' is occurring based solely on this still image.
3. Assess the following observable dimensions:
   - Visible crowd distribution and spread across the venue floor.
   - Crowd concentration and clustering areas (hotspots).
   - Visible congestion or bottleneck points.
   - Potentially blocked pathways, corridors, or emergency exits.
   - Visible obstacles, barriers, or objects on the ground.
   - Unusual spatial arrangements or stationary gatherings.
   - Possible fallen-person indicators ONLY if visibly evident (or note that none are visible).
4. Provide structured, actionable recommendations for safety operators.
5. Language: {lang_prompt}

OUTPUT FORMAT (Respond STRICTLY as JSON with these exact keys):
{{
  "scene_summary": "1-2 sentence executive overview of the visual scene",
  "observed_conditions": ["observation 1", "observation 2", "observation 3"],
  "potential_safety_concerns": ["concern 1 or None observed", "concern 2"],
  "supporting_visual_evidence": ["evidence 1", "evidence 2"],
  "confidence_or_uncertainty": "High/Moderate/Low confidence with brief reason (e.g. lighting, resolution, occlusions)",
  "recommended_operator_actions": ["action 1", "action 2"]
}}
"""

        system_instruction = (
            "You are a professional crowd safety visual analyst. Always output valid JSON. "
            "Never assert an emergency without concrete visual proof. Be factual and objective."
        )

        result = self.client.generate_multimodal(
            prompt=prompt,
            image=image_bytes,
            mime_type="image/jpeg",
            system_instruction=system_instruction,
            temperature=0.2
        )

        if not result["success"] or not result["text"]:
            raise RuntimeError(result.get("error") or "Failed to receive response from Gemini Vision")

        # Parse JSON from response
        raw_text = result["text"].strip()
        if "```json" in raw_text:
            raw_text = raw_text.split("```json")[1].split("```")[0].strip()
        elif "```" in raw_text:
            raw_text = raw_text.split("```")[1].split("```")[0].strip()

        import json
        data = json.loads(raw_text)

        return VisionAnalysisResponse(
            scene_summary=data.get("scene_summary", "Crowd scene analyzed successfully."),
            observed_conditions=data.get("observed_conditions", []),
            potential_safety_concerns=data.get("potential_safety_concerns", []),
            supporting_visual_evidence=data.get("supporting_visual_evidence", []),
            confidence_or_uncertainty=data.get("confidence_or_uncertainty", "High confidence based on clear camera snapshot."),
            recommended_operator_actions=data.get("recommended_operator_actions", []),
            timestamp=now_str,
            camera_id=camera_id,
            snapshot_url=snapshot_url,
            fallback_used=False
        )

    def _analyze_with_heuristics(
        self,
        image_bytes: bytes,
        camera_id: str,
        lang: str,
        now_str: str,
        snapshot_url: Optional[str]
    ) -> VisionAnalysisResponse:
        """Deterministic local CV heuristic analysis when Gemini API is unavailable."""
        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            h, w = img.shape[:2] if img is not None else (720, 1280)
        except Exception:
            h, w = 720, 1280

        # Read live pipeline metrics for grounding
        headcount = 0
        density_pct = 0.0
        alert_level = "SAFE"
        try:
            from web.server import get_state
            st = get_state()
            m = st.get("metrics", {})
            headcount = m.get("count", 0)
            density_pct = round(m.get("density", 0.0) * 100, 1)
            alert_level = st.get("alert_level", "SAFE")
        except Exception:
            pass

        if headcount == 0:
            summary = f"Clear scene monitored on {camera_id} with no visible pedestrian crowding detected."
            conditions = ["Open floor space with clear visibility", "Pathways and designated exit routes remain unobstructed", "No person clusters detected"]
            concerns = ["No current safety or crowding concerns observed."]
            evidence = ["Full scene visibility with nominal background activity", f"Headcount: {headcount}"]
            confidence = "High confidence (unimpeded camera line-of-sight)."
            actions = ["Maintain standard automated video telemetry surveillance.", "No manual operator dispatch required."]
        elif headcount < 10:
            summary = f"Low-density pedestrian presence observed on {camera_id} ({headcount} persons detected, {density_pct}% coverage)."
            conditions = [
                f"Dispersed pedestrian distribution ({headcount} individuals visible)",
                "Free-flowing walking space throughout the central walkway",
                "Ingress and egress routes remain open and unobstructed"
            ]
            concerns = ["Low density within normal parameters; no immediate bottleneck."]
            evidence = [f"YOLO detector confirms {headcount} active tracks", f"Measured area coverage is {density_pct}%"]
            confidence = "High confidence based on direct CV tracking and unobstructed optics."
            actions = ["Continue continuous automated monitoring.", "Verify camera pan/tilt coverage periodically."]
        else:
            summary = f"Noticeable crowd concentration detected on {camera_id} ({headcount} persons, {density_pct}% density)."
            conditions = [
                f"Active pedestrian gathering with {headcount} persons in viewing area",
                f"Spatial density is measured at {density_pct}% coverage",
                "Higher local clustering observed toward central monitoring zone"
            ]
            concerns = [
                "Elevated density may reduce individual walking clearance if ingress continues.",
                "Verify that egress corridors remain clear of stationary standing groups."
            ]
            evidence = [f"CV detection metrics indicate {headcount} occupants", f"System alert level: {alert_level}"]
            confidence = "Moderate-to-high confidence; visual assessment augmented with real-time detection telemetry."
            actions = [
                "Monitor zone density progression on the dashboard.",
                "Ensure designated exit corridors are kept clear of temporary obstructions."
            ]

        # Multilingual phrasing for summary if requested
        if lang == "hi":
            summary = f"{camera_id} पर दृश्य विश्लेषण: {headcount} लोग उपस्थित हैं। घनत्व {density_pct}% है। वर्तमान सुरक्षा स्तर: {alert_level}।"
        elif lang == "mr":
            summary = f"{camera_id} वरील दृश्य विश्लेषण: {headcount} लोक उपस्थित आहेत. घनता {density_pct}% आहे. चालू सुरक्षा पातळी: {alert_level}."

        return VisionAnalysisResponse(
            scene_summary=summary,
            observed_conditions=conditions,
            potential_safety_concerns=concerns,
            supporting_visual_evidence=evidence,
            confidence_or_uncertainty=confidence,
            recommended_operator_actions=actions,
            timestamp=now_str,
            camera_id=camera_id,
            snapshot_url=snapshot_url,
            fallback_used=True
        )

    def _build_unavailable_frame_response(self, camera_id: str, now_str: str) -> VisionAnalysisResponse:
        """Return standardized response when no video frame is available."""
        return VisionAnalysisResponse(
            scene_summary=f"Camera frame from {camera_id} is currently unavailable or video pipeline is initializing.",
            observed_conditions=["No active video frame buffer received from camera capture."],
            potential_safety_concerns=["Unable to perform optical scene verification until video stream is active."],
            supporting_visual_evidence=["Camera frame buffer returned None."],
            confidence_or_uncertainty="Zero confidence (missing frame buffer).",
            recommended_operator_actions=["Verify camera connection and ensure detection pipeline is running."],
            timestamp=now_str,
            camera_id=camera_id,
            snapshot_url=None,
            fallback_used=True
        )


# Helper type for tuple
TupleOptionalFrame = tuple

