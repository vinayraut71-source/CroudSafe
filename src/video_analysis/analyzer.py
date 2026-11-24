"""
Video Analysis Processing Pipeline for CrowdSafe AI.
Processes uploaded video files (MP4, AVI, MOV) using the existing YOLO detection,
SortTracker, spatial density, and HUD visualizer.
Generates:
1. Full annotated video with bounding boxes, tracking IDs, heatmaps, and HUD metrics.
2. Temporal crowd-count & risk timeline charts.
3. Spatial density heatmap snapshot.
4. AI-powered executive summary and congestion insights via Gemini.
5. Downloadable formal PDF intelligence dossier.
"""

import os
import cv2
import time
import uuid
import json
import logging
import datetime
import threading
import tempfile
from typing import Dict, Any, List, Optional
import numpy as np

# Set headless Matplotlib cache
os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "matplotlib_crowdsafe")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config.settings import (
    RECORDING_DIR, MAX_CAPACITY, YOLO_CONFIDENCE_OFFLINE, YOLO_IMGSZ_OFFLINE
)
from src.detector import PersonDetector
from src.tracker import MultiTracker
from src.density import DensityAnalyzer
from src.movement import MovementAnalyzer
from src.alerts import AlertEngine
from src.hud import HUDRenderer
from src.fight_detector import FightDetector
from src.gemini_service.client import GeminiClient
from src.video_analysis.report import generate_video_analysis_pdf

logger = logging.getLogger("VideoFileAnalyzer")

ANALYZED_DIR = os.path.join(RECORDING_DIR, "analyzed_videos")
os.makedirs(ANALYZED_DIR, exist_ok=True)


class VideoJob:
    """Represents the real-time state of an active or completed video analysis task."""

    def __init__(self, job_id: str, filename: str, source_path: str):
        self.job_id = job_id
        self.filename = filename
        self.source_path = source_path
        self.status = "queued"          # queued | processing | completed | failed
        self.progress_pct = 0.0
        self.current_frame = 0
        self.total_frames = 0
        self.fps = 0.0
        self.eta_seconds = 0.0
        self.start_time = time.time()
        self.completed_time = None
        self.result = {}
        self.error = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "filename": self.filename,
            "status": self.status,
            "progress_pct": round(self.progress_pct, 1),
            "current_frame": self.current_frame,
            "total_frames": self.total_frames,
            "fps": round(self.fps, 1),
            "eta_seconds": round(self.eta_seconds, 1),
            "elapsed_seconds": round(time.time() - self.start_time, 1),
            "result": self.result,
            "error": self.error
        }


class VideoJobManager:
    """Singleton manager coordinating video analysis worker jobs."""

    def __init__(self):
        self.jobs: Dict[str, VideoJob] = {}
        self.detector = None    # Lazy init to save memory until requested

    def create_job(self, file_path: str, filename: str) -> VideoJob:
        job_id = f"vid_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        job = VideoJob(job_id=job_id, filename=filename, source_path=file_path)
        self.jobs[job_id] = job

        # Spawn asynchronous processing worker
        thread = threading.Thread(target=self._run_job, args=(job,), daemon=True)
        thread.start()
        return job

    def get_job(self, job_id: str) -> Optional[VideoJob]:
        return self.jobs.get(job_id)

    def list_jobs(self, limit: int = 20) -> List[Dict[str, Any]]:
        sorted_jobs = sorted(self.jobs.values(), key=lambda j: j.start_time, reverse=True)
        return [j.to_dict() for j in sorted_jobs[:limit]]

    def _run_job(self, job: VideoJob):
        """Worker thread executing end-to-end video processing."""
        job.status = "processing"
        logger.info(f"Starting video processing job {job.job_id} for file {job.filename}")

        try:
            analyzer = VideoFileAnalyzer(job)
            result = analyzer.process()
            job.result = result
            job.status = "completed"
            job.completed_time = time.time()
            job.progress_pct = 100.0
            logger.info(f"Video job {job.job_id} successfully completed.")
        except Exception as e:
            job.status = "failed"
            job.error = str(e)
            logger.error(f"Video job {job.job_id} failed: {e}", exc_info=True)


class VideoFileAnalyzer:
    """Executes frame-by-frame CV pipeline over a video file."""

    def __init__(self, job: VideoJob):
        self.job = job

    def process(self) -> Dict[str, Any]:
        cap = cv2.VideoCapture(self.job.source_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {self.job.source_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        orig_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_sec = total_frames / max(orig_fps, 1.0)

        self.job.total_frames = total_frames

        # Normalise resolution for performance and HUD layout
        W, H = 1280, 720

        # Initialize high-recall detection and analytics modules
        detector = PersonDetector(conf=YOLO_CONFIDENCE_OFFLINE, imgsz=YOLO_IMGSZ_OFFLINE)
        tracker = MultiTracker()
        density_eng = DensityAnalyzer(W, H)
        movement_eng = MovementAnalyzer()
        alert_eng = AlertEngine()
        fight_eng = FightDetector(proximity_px=95.0)
        hud = HUDRenderer(W, H)

        # Output annotated video setup
        out_filename = f"annotated_{self.job.job_id}.mp4"
        out_path = os.path.join(ANALYZED_DIR, out_filename)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, orig_fps, (W, H))

        prev_centers = {}
        time_series_counts = []
        time_series_densities = []
        time_series_risks = []
        time_series_times = []
        incident_events = []

        peak_count = 0
        peak_count_sec = 0.0
        peak_density = 0.0
        max_risk = 0.0
        overall_level = "SAFE"

        frame_idx = 0
        t_start = time.time()
        fps_calc_ts = t_start
        raw_detections = []

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            time.sleep(0.005)   # Generously yield CPU to live webcam pipeline & streaming
            frame = cv2.resize(frame, (W, H))
            sec_elapsed = frame_idx / max(orig_fps, 1.0)
            time_formatted = f"{int(sec_elapsed//60):02d}:{int(sec_elapsed%60):02d}"

            # 1. Detection & Tracking (adaptive stride: every 2nd frame if >= 15 FPS)
            if frame_idx % 2 == 0 or not raw_detections:
                raw_detections = detector.detect(frame)
            tracked = tracker.update(raw_detections)
            count = len(tracked)

            # 2. Density & Flow
            density = density_eng.compute_global_density(tracked)
            density_level = density_eng.get_density_level(density)
            zone_grid = density_eng.compute_zone_density(tracked)
            occupancy_pct = density_eng.get_occupancy_pct(count)
            density_eng.update_counts(count, density)
            heatmap = density_eng.update_heatmap(tracked)
            density_eng.update_flow_counts(tracked, prev_centers)
            prev_centers = {d.track_id: d.center for d in tracked}

            # 3. Movement
            speeds = movement_eng.update(tracked)
            avg_speed = movement_eng.get_avg_speed(speeds)
            speed_mult = movement_eng.get_speed_multiplier(avg_speed)
            speed_level = movement_eng.detect_speed_spike(avg_speed)
            direction_entropy = movement_eng.compute_direction_entropy(tracked)
            compressed, _, _ = movement_eng.detect_compression(tracked)

            # 4. Fight & Physical Altercation Detection
            fight_res = fight_eng.detect(tracked, speeds=speeds, frame=frame)
            is_fight = fight_res["is_fight"]
            fight_score = fight_res["fight_score"]
            combatants = fight_res["combatants"]

            # 5. Alert & Risk Engine
            metrics_dict = {
                "count": count,
                "entry_count": density_eng.in_count,
                "exit_count": density_eng.out_count,
                "density": density,
                "density_level": density_level,
                "speed_level": speed_level,
                "speed_multiplier": speed_mult,
                "avg_speed": avg_speed,
                "compressed": compressed,
                "direction_entropy": direction_entropy,
                "occupancy_pct": occupancy_pct,
                "fight_detected": is_fight,
                "fight_score": fight_score,
                "combatants": combatants,
                "acoustic_panic": False,
                "acoustic_db": 50.0,
                "weather_risk_mult": 1.0,
                "zone_grid": [[{"count": z["count"], "density": z["density"], "risk": z["risk"]} for z in row] for row in zone_grid]
            }

            alert_level = alert_eng.evaluate(metrics_dict)
            risk_score = alert_eng.risk_score

            # Severity escalation
            severity_order = {"SAFE": 0, "WARNING": 1, "DANGER": 2, "CRITICAL": 3}
            if severity_order.get(alert_level, 0) > severity_order.get(overall_level, 0):
                overall_level = alert_level

            # Peak tracking
            if count > peak_count:
                peak_count = count
                peak_count_sec = sec_elapsed
            if density > peak_density:
                peak_density = density
            if risk_score > max_risk:
                max_risk = risk_score

            # Record significant incident event (including violent altercations)
            if is_fight and (not incident_events or sec_elapsed - incident_events[-1]["time_sec"] >= 3.0):
                f_lvl = "CRITICAL" if fight_score >= 68.0 else "DANGER"
                incident_events.append({
                    "time_sec": round(sec_elapsed, 1),
                    "time_formatted": time_formatted,
                    "level": f_lvl,
                    "count": count,
                    "density_pct": round(density * 100, 1),
                    "risk_score": round(max(risk_score, fight_score), 1),
                    "message": f"🔴 VIOLENT ALTERCATION / FIGHT (Combat Score: {fight_score:.0f}%, Combatants: {combatants})"
                })
            elif alert_level in ("WARNING", "DANGER", "CRITICAL") and (not incident_events or sec_elapsed - incident_events[-1]["time_sec"] >= 4.0):
                incident_events.append({
                    "time_sec": round(sec_elapsed, 1),
                    "time_formatted": time_formatted,
                    "level": alert_level,
                    "count": count,
                    "density_pct": round(density * 100, 1),
                    "risk_score": round(risk_score, 1),
                    "message": f"{alert_level}: Headcount {count} | Density {round(density*100, 1)}% | Speed {round(speed_mult, 2)}x"
                })

            # Time series sample every ~0.5s or frame
            if frame_idx % max(int(orig_fps / 2), 1) == 0:
                time_series_times.append(round(sec_elapsed, 1))
                time_series_counts.append(count)
                time_series_densities.append(round(density * 100, 1))
                time_series_risks.append(round(risk_score, 1))

            # 6. Render HUD & Overlays
            display = frame.copy()
            display = density_eng.render_heatmap_overlay(display, heatmap, alpha=0.35)
            display = density_eng.draw_zone_grid(display, zone_grid)
            display = hud.draw_trajectories(display, tracker.get_trajectories())
            display = detector.draw_detections(display, tracked)

            # Highlight fight combatants with prominent red boxes & labels
            if is_fight and combatants:
                comb_set = set(combatants)
                for det in tracked:
                    if det.track_id in comb_set:
                        cv2.rectangle(display, (det.x1, det.y1), (det.x2, det.y2), (0, 0, 255), 3)
                        cv2.putText(display, f"FIGHT #{det.track_id}", (det.x1, max(20, det.y1 - 8)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

            alert_msg = (alert_eng.alert_feed[0].message if alert_eng.alert_feed else "")
            display = hud.draw_status_bar(display, alert_level, count, density, risk_score, orig_fps)
            display = hud.draw_side_panel(display, metrics_dict, alert_eng.get_feed(), {})
            display = hud.draw_risk_gauge(display, risk_score)
            display = hud.draw_alert_banner(display, alert_level, alert_msg)

            # Write annotated frame
            writer.write(display)

            frame_idx += 1
            self.job.current_frame = frame_idx
            self.job.progress_pct = min((frame_idx / max(total_frames, 1)) * 100, 99.0)

            # Calculate processing rate
            t_now = time.time()
            if t_now - fps_calc_ts >= 1.0:
                cur_fps = frame_idx / (t_now - t_start)
                self.job.fps = cur_fps
                remaining_frames = total_frames - frame_idx
                self.job.eta_seconds = remaining_frames / max(cur_fps, 0.1)
                fps_calc_ts = t_now

        cap.release()
        writer.release()

        # ── Step 6: Generate Timeline Chart ──
        chart_filename = f"chart_{self.job.job_id}.png"
        chart_path = os.path.join(ANALYZED_DIR, chart_filename)
        self._generate_chart(time_series_times, time_series_counts, time_series_risks, chart_path)

        # ── Step 7: Save Final Heatmap Snapshot ──
        heatmap_filename = f"heatmap_{self.job.job_id}.png"
        heatmap_path = os.path.join(ANALYZED_DIR, heatmap_filename)
        norm_hm = cv2.normalize(density_eng.heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        color_hm = cv2.applyColorMap(norm_hm, cv2.COLORMAP_JET)
        cv2.imwrite(heatmap_path, color_hm)

        # ── Step 8: Gemini AI Safety Assessment ──
        ai_assessment = self._generate_ai_summary(
            filename=self.job.filename,
            duration_sec=duration_sec,
            peak_count=peak_count,
            peak_count_time=f"{int(peak_count_sec//60):02d}:{int(peak_count_sec%60):02d}",
            avg_count=round(np.mean(time_series_counts) if time_series_counts else 0, 1),
            peak_density=peak_density,
            total_entry=density_eng.in_count,
            total_exit=density_eng.out_count,
            overall_level=overall_level,
            max_risk=max_risk,
            incident_events=incident_events
        )

        # ── Step 9: Generate Formal PDF Dossier ──
        pdf_filename = f"report_{self.job.job_id}.pdf"
        pdf_path = os.path.join(ANALYZED_DIR, pdf_filename)
        video_metadata = {
            "filename": self.job.filename,
            "duration_formatted": f"{int(duration_sec//60):02d}:{int(duration_sec%60):02d}",
            "total_frames": total_frames,
            "width": orig_w,
            "height": orig_h,
            "fps": orig_fps
        }
        analytics_summary = {
            "overall_risk_level": overall_level,
            "peak_count": peak_count,
            "peak_count_time": f"{int(peak_count_sec//60):02d}:{int(peak_count_sec%60):02d}",
            "avg_count": round(np.mean(time_series_counts) if time_series_counts else 0, 1),
            "peak_density_pct": round(peak_density * 100, 1),
            "total_entry": density_eng.in_count,
            "total_exit": density_eng.out_count,
            "max_risk_score": round(max_risk, 1)
        }

        generate_video_analysis_pdf(
            output_pdf_path=pdf_path,
            video_metadata=video_metadata,
            analytics_summary=analytics_summary,
            chart_image_path=chart_path,
            heatmap_image_path=heatmap_path,
            incident_events=incident_events,
            ai_assessment=ai_assessment
        )

        return {
            "annotated_video_url": f"/recordings/analyzed_videos/{out_filename}",
            "chart_url": f"/recordings/analyzed_videos/{chart_filename}",
            "heatmap_url": f"/recordings/analyzed_videos/{heatmap_filename}",
            "pdf_report_url": f"/recordings/analyzed_videos/{pdf_filename}",
            "metadata": video_metadata,
            "summary": analytics_summary,
            "time_series": {
                "times": time_series_times,
                "counts": time_series_counts,
                "risks": time_series_risks
            },
            "incident_events": incident_events,
            "ai_assessment": ai_assessment
        }

    def _generate_chart(self, times: List[float], counts: List[int], risks: List[float], out_path: str):
        """Draw dual-axis time-series chart of headcount and safety risk."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5.2), dpi=130, facecolor="#0d1117")

        # Headcount Subplot
        ax1.set_facecolor("#090d14")
        ax1.plot(times, counts, color="#00aaff", linewidth=2.0, label="Crowd Headcount")
        ax1.fill_between(times, counts, color="#00aaff", alpha=0.15)
        ax1.set_ylabel("People Detected", color="#e8eaf0", fontsize=9, fontweight="bold")
        ax1.tick_params(colors="#7a8499", labelsize=8)
        ax1.grid(True, linestyle="--", alpha=0.2, color="#1e2535")
        ax1.legend(loc="upper right", facecolor="#111722", edgecolor="#1e2535", labelcolor="#e8eaf0", fontsize=8)

        # Risk Score Subplot
        ax2.set_facecolor("#090d14")
        ax2.plot(times, risks, color="#ff6d00", linewidth=2.0, label="Composite Risk Score (%)")
        ax2.axhline(50, color="#ffc107", linestyle=":", alpha=0.6, label="Warning (50%)")
        ax2.axhline(75, color="#e53935", linestyle=":", alpha=0.6, label="Critical (75%)")
        ax2.set_xlabel("Video Timeline (Seconds)", color="#e8eaf0", fontsize=9, fontweight="bold")
        ax2.set_ylabel("Risk Score (%)", color="#e8eaf0", fontsize=9, fontweight="bold")
        ax2.set_ylim(0, 100)
        ax2.tick_params(colors="#7a8499", labelsize=8)
        ax2.grid(True, linestyle="--", alpha=0.2, color="#1e2535")
        ax2.legend(loc="upper right", facecolor="#111722", edgecolor="#1e2535", labelcolor="#e8eaf0", fontsize=8)

        plt.tight_layout()
        plt.savefig(out_path, facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close(fig)

    def _generate_ai_summary(
        self,
        filename: str,
        duration_sec: float,
        peak_count: int,
        peak_count_time: str,
        avg_count: float,
        peak_density: float,
        total_entry: int,
        total_exit: int,
        overall_level: str,
        max_risk: float,
        incident_events: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Synthesize Gemini Multimodal executive assessment or fallback."""
        client = GeminiClient()

        prompt = f"""
You are the Chief Safety Auditor for the CrowdSafe AI surveillance platform.
Analyze this video crowd analysis audit log:
- Video File: {filename}
- Duration: {duration_sec:.1f} seconds
- Peak Crowd Count: {peak_count} persons at {peak_count_time}
- Average Crowd Count: {avg_count:.1f} persons
- Peak Measured Area Density: {peak_density*100:.1f}%
- Ingress/Egress: {total_entry} entries, {total_exit} exits (Net flow: {total_entry - total_exit})
- Highest Risk Score: {max_risk:.1f}% (Overall Level: {overall_level})
- Significant Events Logged: {json.dumps(incident_events[:5])}

TASK:
1. Provide a professional 2-sentence executive summary.
2. Provide a 2-sentence congestion and bottleneck analysis based on entry/exit pacing and peak density.
3. Provide 3 specific operational recommendations for venue safety marshals.

OUTPUT STRICTLY AS JSON:
{{
  "executive_summary": "...",
  "congestion_analysis": "...",
  "recommendations": ["Recommendation 1", "Recommendation 2", "Recommendation 3"]
}}
"""
        if client.is_available():
            try:
                res = client.generate_text(prompt, temperature=0.2)
                if res["success"] and res["text"]:
                    raw = res["text"].strip()
                    if "```json" in raw:
                        raw = raw.split("```json")[1].split("```")[0].strip()
                    elif "```" in raw:
                        raw = raw.split("```")[1].split("```")[0].strip()
                    return json.loads(raw)
            except Exception as e:
                logger.warning(f"Gemini video summary failed ({e}). Using deterministic summary.")

        # Deterministic Heuristic Assessment
        return {
            "executive_summary": (
                f"Video analysis of '{filename}' recorded an average headcount of {avg_count:.1f} persons "
                f"with a maximum peak of {peak_count} persons at {peak_count_time}. "
                f"The highest safety status reached was {overall_level} with a peak risk score of {max_risk:.1f}%."
            ),
            "congestion_analysis": (
                f"Ingress totaled {total_entry} individuals versus {total_exit} exits, producing a net accumulation "
                f"of {total_entry - total_exit} pedestrians. Maximum spatial density reached {peak_density*100:.1f}%."
            ),
            "recommendations": [
                f"Monitor entry gates if inflow continues to exceed outflow ({total_entry} in vs {total_exit} out).",
                f"Keep central thoroughfares unobstructed during peak occupancy periods like {peak_count_time}.",
                "Maintain continuous sensor surveillance and verify exit latch functionality."
            ]
        }


# Singleton job manager instance
video_job_manager = VideoJobManager()

