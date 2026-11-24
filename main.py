"""
Main Pipeline Orchestrator.
Ties together: video capture → detection → tracking → density →
movement → alerts → prediction → HUD → web server.
"""

import cv2
import numpy as np
import threading
import time
import argparse
import os
import sys
import tempfile

os.environ["YOLO_CONFIG_DIR"] = os.path.join(tempfile.gettempdir(), "ultralytics")

# Check for required dependencies
def check_dependencies():
    missing = []
    try:
        import cv2
    except ImportError:
        missing.append('opencv-python')
    try:
        import numpy
    except ImportError:
        missing.append('numpy')
    try:
        import scipy
    except ImportError:
        missing.append('scipy')
    try:
        import fastapi
    except ImportError:
        missing.append('fastapi')
    
    if missing:
        print("\n" + "="*60)
        print("  ERROR: Missing Required Dependencies")
        print("="*60)
        print(f"\nThe following packages are not installed: {', '.join(missing)}")
        print("\nPlease run the setup script first:")
        print("  ./setup.sh")
        print("\nOr install manually:")
        print(f"  pip install {' '.join(missing)}")
        print("="*60 + "\n")
        sys.exit(1)

check_dependencies()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config.settings import (
    DEFAULT_SOURCE, FRAME_WIDTH, FRAME_HEIGHT, TARGET_FPS,
    SIMULATION_MODE, API_HOST, API_PORT, YOLO_CONFIDENCE, YOLO_IMGSZ_LIVE
)
from src.detector import PersonDetector
from src.tracker import MultiTracker
from src.density import DensityAnalyzer
from src.movement import MovementAnalyzer
from src.alerts import AlertEngine
from src.predictor import LSTMRiskPredictor
from src.lost_child import LostChildDetector
from src.social_distance import SocialDistanceMonitor
from src.simulator import CrowdSimulator
from src.logger import EventLogger
from src.notifications import NotificationPipeline
from src.hud import HUDRenderer
from src.acoustic import AcousticDetector
from src.voice_alert import VoiceAlertSystem
from src.explainability import ExplainabilityEngine
from src.digital_twin import DigitalTwin
from src.weather import WeatherMonitor
from src.zone_manager import ZoneManager
from src.behavior_classifier import BehaviorClassifier
from src.recorder import IncidentRecorder
from src.restricted_zones import RestrictedZoneEnforcer
from src.cluster_analysis import ClusterAnalyzer
from web.server import update_state, get_state, start_server


class CameraStream:
    """
    High-performance, threaded zero-latency camera capture reader.
    Continuously drains the driver buffer in a dedicated background worker,
    eliminating all frame queuing so read() always returns the immediate live frame.
    """
    def __init__(self, source=0, width=640, height=480):
        self.source = source
        self.width = width
        self.height = height
        self.cap = None
        self.ret = False
        self.frame = None
        self.running = False
        self.lock = threading.Lock()
        self.thread = None
        self.open()

    def open(self):
        self.release()
        try:
            if isinstance(self.source, int) and os.name == "nt":
                self.cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
                if not self.cap.isOpened():
                    self.cap = cv2.VideoCapture(self.source)
            else:
                self.cap = cv2.VideoCapture(self.source)

            if self.cap and self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                self.running = True
                self.thread = threading.Thread(target=self._worker, daemon=True)
                self.thread.start()
                return True
        except Exception as e:
            print(f"[CameraStream] Error opening camera {self.source}: {e}")
        return False

    def _worker(self):
        while self.running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.ret = ret
                    self.frame = frame
            else:
                time.sleep(0.005)

    def read(self):
        with self.lock:
            if self.frame is not None:
                return self.ret, self.frame.copy()
            return False, None

    def isOpened(self):
        return self.cap is not None and self.cap.isOpened() and self.running

    def release(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=0.3)
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        with self.lock:
            self.frame = None
            self.ret = False


class AsyncDetector:
    """
    Non-blocking background detector worker.
    Executes YOLO model inference in an isolated background thread,
    allowing the main pipeline and MJPEG video stream to run at full 30 FPS
    without waiting for CPU inference cycles.
    """
    def __init__(self, detector):
        self.detector = detector
        self.lock = threading.Lock()
        self.pending_frame = None
        self.latest_detections = []
        self.has_new = False
        self.busy = False
        self.running = True
        self.worker = threading.Thread(target=self._run, daemon=True)
        self.worker.start()

    def submit(self, frame):
        with self.lock:
            if not self.busy and self.pending_frame is None:
                self.pending_frame = frame.copy()

    def get_detections(self):
        with self.lock:
            new = self.has_new
            self.has_new = False
            return new, list(self.latest_detections)

    def _run(self):
        while self.running:
            frame_to_proc = None
            with self.lock:
                if self.pending_frame is not None:
                    frame_to_proc = self.pending_frame
                    self.pending_frame = None
                    self.busy = True
            if frame_to_proc is not None:
                dets = self.detector.detect(frame_to_proc)
                with self.lock:
                    self.latest_detections = dets
                    self.has_new = True
                    self.busy = False
            else:
                time.sleep(0.005)

    def stop(self):
        self.running = False


class CrowdSafetyPipeline:
    def __init__(self, source=None, sim_mode: bool = False,
                 show_window: bool = True, headless: bool = False):
        self.source = source if source is not None else DEFAULT_SOURCE
        self.sim_mode = sim_mode or SIMULATION_MODE
        self.show_window = show_window and not headless
        self.headless = headless
        self.running = False
        self.paused = False

        # Core modules
        self.detector    = PersonDetector(conf=YOLO_CONFIDENCE, imgsz=YOLO_IMGSZ_LIVE)
        self.async_detector = AsyncDetector(self.detector)
        self.tracker     = MultiTracker()
        self.density     = None      # init after frame size known
        self.movement    = MovementAnalyzer()
        self.alert_eng   = AlertEngine()
        self.predictor   = LSTMRiskPredictor()
        self.lost_child  = LostChildDetector()
        self.social_dist = SocialDistanceMonitor()
        self.logger      = EventLogger()
        self.notifier    = NotificationPipeline()
        self.simulator   = None
        self.hud         = None

        # Advanced Modules
        self.acoustic    = AcousticDetector()
        self.voice       = VoiceAlertSystem()
        self.explainer   = ExplainabilityEngine()
        self.digital_twin = DigitalTwin()
        self.weather     = WeatherMonitor()
        self.behavior_cls = BehaviorClassifier()
        self.recorder    = IncidentRecorder(fps=TARGET_FPS)
        self.cluster_analyzer = ClusterAnalyzer(cluster_radius=40)
        self.zone_mgr    = ZoneManager(FRAME_WIDTH, FRAME_HEIGHT)
        self.zone_enforcer = RestrictedZoneEnforcer(self.zone_mgr)
        
        # We start acoustic on run() starts

        # Register notification callback
        self.alert_eng.register_callback(self._on_alert)
        self.alert_eng.register_callback(self.notifier.send_all)
        self.alert_eng.register_callback(lambda e: self.voice.speak(e.level))
        self.alert_eng.register_callback(lambda e: self.recorder.trigger(e.level))

        # FPS tracking
        self._frame_count = 0
        self._fps = 0.0
        self._fps_ts = time.time()

        # Smooth low-latency detection & stream state
        self._detect_interval = 2
        self._last_raw_detections = []
        self._frame_seq = 0

        # Previous centers for motion heatmap
        self._prev_centers = {}

        print("=" * 60)
        print("  CrowdSafe AI — Intelligent Crowd Safety System")
        print(f"  Mode: {'SIMULATION' if self.sim_mode else 'LIVE'}")
        print(f"  Source: {self.source}")
        print(f"  Dashboard: http://{API_HOST}:{API_PORT}")
        print("=" * 60)

    # ──────────────────────────────────────────────────────────────────────
    # Alert callback
    # ──────────────────────────────────────────────────────────────────────

    def _on_alert(self, event):
        self.logger.log(
            event_type="ALERT",
            level=event.level,
            message=event.message,
            data=event.metrics
        )
        print(f"\n[ALERT] {event.level}: {event.message}\n")
        if event.level in ("WARNING", "DANGER", "CRITICAL"):
            try:
                from src.nlp_assistant.report_generator import IncidentReportManager
                mgr = IncidentReportManager()
                ev_type = event.message.split(":")[0].replace("⚠", "").replace("🔶", "").replace("🔴", "").strip() or "HAZARD_ALERT"
                mgr.create_report(
                    event_type=ev_type,
                    severity=event.level,
                    camera_id=getattr(self, "current_camera", "CAM-1"),
                    zone="Zone-1",
                    count=event.metrics.get("count", 0),
                    density=event.metrics.get("density", 0.0),
                    occupancy_pct=(event.metrics.get("occupancy_pct", 0.0) or 0.0) * 100,
                    risk_score=float(event.metrics.get("risk_score", 0.0) or 0.0),
                    data=event.metrics,
                    session_id=self.logger.session_id
                )
            except Exception:
                pass

    # ──────────────────────────────────────────────────────────────────────
    # Main loop
    # ──────────────────────────────────────────────────────────────────────

    def run(self):
        self.running = True
        self.acoustic.start()
        update_state(running=True)

        stream = None
        if not self.sim_mode:
            stream = CameraStream(self.source, FRAME_WIDTH, FRAME_HEIGHT)
            if not stream or not stream.isOpened():
                print(f"[Pipeline] Cannot open source: {self.source}. Switching to simulation.")
                self.sim_mode = True

        if self.sim_mode:
            self.simulator = CrowdSimulator(FRAME_WIDTH, FRAME_HEIGHT)

        W, H = FRAME_WIDTH, FRAME_HEIGHT
        self.density = DensityAnalyzer(W, H)
        self.hud = HUDRenderer(W, H)

        frame_interval = 1.0 / TARGET_FPS

        try:
            while self.running:
                t_start = time.time()

                # ── Handle web controls & camera feed switching ──────────
                state = get_state()
                cam_req = state.get("current_camera", None)
                sim_cmd = state.pop("sim_cmd", None)

                if sim_cmd == "live_cam" or cam_req == "CAM-1":
                    if self.sim_mode:
                        if stream is None or not stream.isOpened():
                            stream = CameraStream(self.source, FRAME_WIDTH, FRAME_HEIGHT)
                        if stream and stream.isOpened():
                            self.sim_mode = False
                            print("[Pipeline] Activated live camera feed (CAM-1).")
                elif sim_cmd in ("normal", "building", "dense", "panic") or cam_req in ("SIM", "DRONE-1", "CAM-2"):
                    if not self.sim_mode:
                        self.sim_mode = True
                        if self.simulator is None:
                            self.simulator = CrowdSimulator(FRAME_WIDTH, FRAME_HEIGHT)
                        print(f"[Pipeline] Switched to simulation mode ({sim_cmd or cam_req}).")
                    if sim_cmd in ("normal", "building", "dense", "panic") and self.simulator:
                        self.simulator.set_mode(sim_cmd)

                # ── Get frame ──────────────────────────────────
                if self.sim_mode:
                    if self.simulator is None:
                        self.simulator = CrowdSimulator(FRAME_WIDTH, FRAME_HEIGHT)
                    frame = self.simulator.render_blank_frame()
                    raw_detections = self.simulator.step()
                else:
                    ret, frame = stream.read() if stream else (False, None)
                    if not ret or frame is None or frame.size == 0:
                        # Resilient fallback: reuse last good frame during background processing / CPU spikes
                        if hasattr(self, "_last_good_frame") and self._last_good_frame is not None:
                            frame = self._last_good_frame.copy()
                        else:
                            time.sleep(0.02)
                            continue
                    else:
                        self._last_good_frame = frame.copy()

                    frame = cv2.resize(frame, (W, H))
                    self._frame_seq += 1

                    # Submit frame to non-blocking background detector
                    self.async_detector.submit(frame)

                    # Retrieve latest detections without blocking the stream
                    has_new, dets = self.async_detector.get_detections()
                    if has_new or not self._last_raw_detections:
                        raw_detections = dets
                        self._last_raw_detections = dets
                    else:
                        raw_detections = self._last_raw_detections

                # ── Track ──────────────────────────────────────
                tracked = self.tracker.update(raw_detections)

                # ── Density ────────────────────────────────────
                density = self.density.compute_global_density(tracked)
                density_level = self.density.get_density_level(density)
                zone_grid = self.density.compute_zone_density(tracked)
                count = len(tracked)
                occupancy_pct = self.density.get_occupancy_pct(count)
                self.density.update_counts(count, density)

                # ── Heatmap ────────────────────────────────────
                heatmap = self.density.update_heatmap(tracked)
                curr_centers = {d.track_id: d.center for d in tracked}
                self.density.update_flow_counts(tracked, self._prev_centers)
                motion_hm = self.density.update_motion_heatmap(self._prev_centers, curr_centers)
                self._prev_centers = curr_centers

                # ── Movement ───────────────────────────────────
                speeds = self.movement.update(tracked)
                avg_speed = self.movement.get_avg_speed(speeds)
                self.movement.update_baseline(avg_speed)
                speed_mult = self.movement.get_speed_multiplier(avg_speed)
                speed_level = self.movement.detect_speed_spike(avg_speed)
                direction_entropy = self.movement.compute_direction_entropy(tracked)
                compressed, avg_dist, pressure_wave = self.movement.detect_compression(tracked)
                
                if pressure_wave:
                    self.logger.log("PRESSURE_WAVE", "CRITICAL", "A dangerous crowd compression wave was detected propagating through the tracked area.")
                    
                # Compute flow vectors once with scale=4 for both classifier and rendering
                flow_vectors = self.movement.compute_flow_vectors(tracked, scale=4)
                anomalies = self.behavior_cls.assess_anomalies(tracked, speeds, flow_vectors, frame=frame)
                combatants = anomalies.get("combatants", [])
                behaviors = self.movement.detect_behaviors(tracked, avg_speed, combatants=combatants)
                
                if anomalies.get("fight_detected"):
                    self.logger.log("FIGHT_DETECTED", "CRITICAL",
                                    f"Physical altercation / fight detected (Combat Score: {anomalies.get('fight_score', 0):.0f}%, Combatants: {combatants})",
                                    {"score": anomalies.get("fight_score", 0), "combatants": combatants})

                dt_future = self.digital_twin.update(tracked, speeds, flow_vectors)

                # ── Social distance ────────────────────────────
                sd_violations = self.social_dist.compute_violations(tracked)

                # ── Lost child ─────────────────────────────────
                lost_children = self.lost_child.update(tracked)
                if lost_children:
                    self.logger.log("LOST_CHILD", "WARNING",
                                    f"Possible lost child(ren) detected: {len(lost_children)}",
                                    {"count": len(lost_children)})

                # ── Restricted Zones ───────────────────────────
                zone_intrusions = self.zone_enforcer.assess_intrusions(tracked)
                if zone_intrusions:
                    for z_name, ids in zone_intrusions.items():
                        self.logger.log("ZONE_VIOLATION", "WARNING", 
                                        f"Restricted zone intrusion in {z_name}", 
                                        {"zone": z_name, "count": len(ids)})

                # ── Group & Cluster Analysis ───────────────────
                cluster_stats = self.cluster_analyzer.analyze(tracked)
                if cluster_stats["dense_clusters"] > 0:
                    self.logger.log("DENSE_CLUSTER", "WARNING", f"Dangerous cluster compression developing: {cluster_stats['dense_clusters']} dense clusters detected")

                # ── Alert evaluation ───────────────────────────
                weather_data = self.weather.get_weather_data()
                ac_status = self.acoustic.get_status()
                
                metrics_dict = {
                    "count": count,
                    "entry_count": self.density.in_count,
                    "exit_count": self.density.out_count,
                    "density": density,
                    "density_level": density_level,
                    "speed_level": speed_level,
                    "speed_multiplier": speed_mult,
                    "avg_speed": avg_speed,
                    "compressed": compressed,
                    "pressure_wave": pressure_wave,
                    "avg_dist": avg_dist,
                    "direction_entropy": direction_entropy,
                    "occupancy_pct": occupancy_pct,
                    "lost_child": len(lost_children) > 0,
                    "social_violations": self.social_dist.current_violations,
                    "anomaly_score": anomalies["score"],
                    "acoustic_panic": ac_status["is_panic"],
                    "acoustic_db": ac_status["db_level"],
                    "weather_risk_mult": weather_data["risk_mult"],
                    "dense_clusters": cluster_stats["dense_clusters"],
                    "fight_detected": anomalies.get("fight_detected", False),
                    "fight_score": anomalies.get("fight_score", 0.0),
                    "combatants": combatants,
                    "zone_grid": [[{"count":z["count"],"density":z["density"],"risk":z["risk"]}
                                   for z in row] for row in zone_grid],
                }
                alert_level = self.alert_eng.evaluate(metrics_dict)
                risk_score = min(self.alert_eng.risk_score * weather_data["risk_mult"], 100.0)
                
                explanation = self.explainer.explain(metrics_dict, risk_score, alert_level)
                metrics_dict["explanation"] = explanation

                # ── Prediction ─────────────────────────────────
                self.predictor.update(density, avg_speed, risk_score, count, direction_entropy)
                predictions = self.predictor.predict()
                trend = self.predictor.get_trend()
                metrics_dict["trend"] = trend

                # ── Logging ────────────────────────────────────
                self.logger.log_metric(count, density, avg_speed, risk_score, alert_level)

                # ── FPS ────────────────────────────────────────
                self._frame_count += 1
                now = time.time()
                if now - self._fps_ts >= 1.0:
                    self._fps = self._frame_count / (now - self._fps_ts)
                    self._frame_count = 0
                    self._fps_ts = now

                # ── Render ─────────────────────────────────────
                display = frame.copy()

                # Heatmap overlay
                display = self.density.render_heatmap_overlay(display, heatmap, alpha=0.35)

                # Zone grid
                display = self.density.draw_zone_grid(display, zone_grid)

                # Flow arrows (reuse computed flow_vectors)
                display = self.movement.draw_flow_arrows(display, flow_vectors)

                # Social distance lines
                display = self.social_dist.draw_violations(display, sd_violations)

                # Trajectories
                display = self.hud.draw_trajectories(display, self.tracker.get_trajectories())

                # Bounding boxes
                display = self.detector.draw_detections(display, tracked)

                # Behavior labels & Fight combatant highlight
                comb_set = set(combatants) if combatants else set()
                for det in tracked:
                    blist = behaviors.get(det.track_id, [])
                    if det.track_id in comb_set:
                        cv2.rectangle(display, (det.x1, det.y1), (det.x2, det.y2), (0, 0, 255), 3)
                        cv2.putText(display, f"FIGHT #{det.track_id}", (det.x1, max(18, det.y1 - 6)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    elif blist:
                        label = ",".join(blist[:2])
                        cv2.putText(display, label, (det.x1, det.y2 + 14),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 0), 1)

                # Lost child overlay
                display = self.hud.draw_lost_child_alerts(display, lost_children)

                # HUD panels
                alert_msg = (self.alert_eng.alert_feed[0].message
                             if self.alert_eng.alert_feed else "")
                display = self.hud.draw_status_bar(
                    display, alert_level, count, density, risk_score, self._fps)
                display = self.hud.draw_side_panel(
                    display, metrics_dict, self.alert_eng.get_feed(), predictions)
                display = self.hud.draw_risk_gauge(display, risk_score)
                display = self.hud.draw_alert_banner(display, alert_level, alert_msg)
                display = self.hud.draw_system_health(display, self._fps)

                self.recorder.update(frame.copy())
                
                # ── Fast JPEG encode for zero-latency MJPEG stream ─────
                _, buf = cv2.imencode(".jpg", display, [cv2.IMWRITE_JPEG_QUALITY, 70])
                jpeg_bytes = buf.tobytes()

                # ── Push to web server ──────────────────────────
                update_state(
                    frame=display,
                    jpeg_frame=jpeg_bytes,
                    frame_seq=self._frame_seq,
                    metrics=metrics_dict,
                    alert_level=alert_level,
                    risk_score=risk_score,
                    predictions=predictions,
                    alert_feed=self.alert_eng.get_feed(),
                    timeline=self.logger.generate_incident_timeline(),
                    running=True,
                )

                # ── Show local window ───────────────────────────
                if self.show_window:
                    cv2.imshow("CrowdSafe AI", display)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('h'):
                        self.show_window = False   # hide window
                    elif key == ord('s'):
                        self.sim_mode = not self.sim_mode
                    elif key == ord('1') and self.sim_mode:
                        self.simulator.set_mode("normal")
                    elif key == ord('2') and self.sim_mode:
                        self.simulator.set_mode("building")
                    elif key == ord('3') and self.sim_mode:
                        self.simulator.set_mode("dense")
                    elif key == ord('4') and self.sim_mode:
                        self.simulator.set_mode("panic")

                # ── Frame rate cap ──────────────────────────────
                elapsed = time.time() - t_start
                sleep_t = max(0, frame_interval - elapsed)
                if sleep_t > 0:
                    time.sleep(sleep_t)

        except KeyboardInterrupt:
            print("\n[Pipeline] Interrupted by user.")
        finally:
            self.running = False
            self.acoustic.stop()
            self.voice.stop()
            self.recorder.close()
            update_state(running=False)
            if stream:
                stream.release()
            self.async_detector.stop()
            if self.show_window:
                cv2.destroyAllWindows()
            # Generate final report
            timeline = self.logger.generate_incident_timeline()
            print("\n" + timeline)
            self.logger.close()
            print("[Pipeline] Shutdown complete.")


# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="CrowdSafe AI — Intelligent Crowd Safety & Stampede Prevention"
    )
    parser.add_argument("--source", default=None,
                        help="Camera index (0,1,...), video file path, or RTSP URL")
    parser.add_argument("--sim", action="store_true",
                        help="Run in simulation mode (no camera required)")
    parser.add_argument("--no-window", action="store_true",
                        help="Run headless (no OpenCV window — web dashboard only)")
    parser.add_argument("--port", type=int, default=API_PORT,
                        help=f"Web dashboard port (default {API_PORT})")
    args = parser.parse_args()

    source = args.source
    if source is not None:
        try:
            source = int(source)
        except ValueError:
            pass   # keep as string (file path or RTSP URL)

    pipeline = CrowdSafetyPipeline(
        source=source,
        sim_mode=args.sim,
        show_window=not args.no_window,
        headless=args.no_window
    )

    # Start web server in background thread
    server_thread = threading.Thread(
        target=start_server,
        kwargs={"host": API_HOST, "port": args.port},
        daemon=True
    )
    server_thread.start()
    print(f"[Server] Dashboard live at http://localhost:{args.port}")
    time.sleep(1.0)   # let server start

    # Run pipeline on main thread
    pipeline.run()


if __name__ == "__main__":
    main()
