import sys, os, tempfile
sys.path.insert(0, os.path.abspath("."))
os.environ['YOLO_CONFIG_DIR'] = tempfile.gettempdir()
import cv2, math
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict, deque
import numpy as np
from ultralytics import YOLO
from src.tracker import MultiTracker
from src.detector import Detection

class CalibratedFightDetector:
    def __init__(self, proximity_px: float = 75.0, min_combat_streak: int = 2,
                 min_combat_frames: Optional[int] = None):
        self.proximity_px = proximity_px
        self.min_combat_streak = min_combat_frames if min_combat_frames is not None else min_combat_streak
        self.min_combat_frames = self.min_combat_streak
        
        self.pair_histories = defaultdict(lambda: deque(maxlen=20))
        self.pair_streaks = defaultdict(int)
        self._combatant_ids = set()
        self.latch_frames = 0

    def reset(self):
        self.pair_histories.clear()
        self.pair_streaks.clear()
        self._combatant_ids.clear()
        self.latch_frames = 0

    @staticmethod
    def _compute_iou(b1, b2):
        x1 = max(b1[0], b2[0])
        y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2])
        y2 = min(b1[3], b2[3])
        inter_w = max(0, x2 - x1)
        inter_h = max(0, y2 - y1)
        inter_area = inter_w * inter_h
        if inter_area == 0:
            return 0.0
        area1 = max(1, (b1[2] - b1[0]) * (b1[3] - b1[1]))
        area2 = max(1, (b2[2] - b2[0]) * (b2[3] - b2[1]))
        union_area = area1 + area2 - inter_area
        return float(inter_area) / float(max(1, union_area))

    def detect(self, detections: List[Detection], speeds: Optional[Dict[int, float]] = None,
               frame: Optional[np.ndarray] = None) -> Dict[str, Any]:
        n = len(detections)
        confirmed_combatants = set()
        fight_candidates = []
        max_fight_score = 0.0
        active_pair_keys = set()

        for i in range(n):
            d1 = detections[i]
            id1 = d1.track_id
            if id1 < 0:
                continue

            for j in range(i + 1, n):
                d2 = detections[j]
                id2 = d2.track_id
                if id2 < 0 or id1 == id2:
                    continue

                pair_key = (min(id1, id2), max(id1, id2))
                active_pair_keys.add(pair_key)

                avg_h = (d1.height + d2.height) / 2.0
                dy2 = abs(d1.y2 - d2.y2)
                iou = self._compute_iou(d1.bbox, d2.bbox)
                dx = d1.cx - d2.cx
                dy = d1.cy - d2.cy
                dist = math.hypot(dx, dy)

                # Aspect ratio indicates knockdown or low tackle
                fallen = (d1.width / max(1, d1.height) > 1.3) or (d2.width / max(1, d2.height) > 1.3)

                # ── 1. Ground Plane Perspective Separation Filter ──
                y_thresh = 0.75 * avg_h if (fallen or iou > 0.15) else 0.55 * avg_h
                if dy2 > y_thresh:
                    self.pair_streaks[pair_key] = max(0, self.pair_streaks[pair_key] - 1)
                    continue

                # ── 2. Real Physical Proximity Filter ──
                if dist > self.proximity_px:
                    self.pair_streaks[pair_key] = max(0, self.pair_streaks[pair_key] - 1)
                    continue

                # Record history
                self.pair_histories[pair_key].append({
                    "p1": (d1.cx, d1.cy),
                    "p2": (d2.cx, d2.cy),
                    "dist": dist,
                    "iou": iou,
                })

                hist = self.pair_histories[pair_key]
                if len(hist) < 4:
                    continue

                # Calculate velocity vectors
                v_span = min(len(hist), 5)
                vx1 = (hist[-1]["p1"][0] - hist[-v_span]["p1"][0]) / float(v_span - 1)
                vy1 = (hist[-1]["p1"][1] - hist[-v_span]["p1"][1]) / float(v_span - 1)
                vx2 = (hist[-1]["p2"][0] - hist[-v_span]["p2"][0]) / float(v_span - 1)
                vy2 = (hist[-1]["p2"][1] - hist[-v_span]["p2"][1]) / float(v_span - 1)
                spd1 = math.hypot(vx1, vy1)
                spd2 = math.hypot(vx2, vy2)

                # Relative speed calculation
                rel_speeds = []
                hist_recent = list(hist)[-5:]
                for k in range(1, len(hist_recent)):
                    ddx = (hist_recent[k]["p1"][0] - hist_recent[k]["p2"][0]) - (hist_recent[k-1]["p1"][0] - hist_recent[k-1]["p2"][0])
                    ddy = (hist_recent[k]["p1"][1] - hist_recent[k]["p2"][1]) - (hist_recent[k-1]["p1"][1] - hist_recent[k-1]["p2"][1])
                    rel_speeds.append(math.hypot(ddx, ddy))
                avg_rel_speed = float(np.mean(rel_speeds)) if rel_speeds else 0.0

                # Kinematic push-pull reversals
                rel_xs = [h["p1"][0] - h["p2"][0] for h in list(hist)[-6:]]
                flips = 0
                for k in range(2, len(rel_xs)):
                    s1 = rel_xs[k-1] - rel_xs[k-2]
                    s2 = rel_xs[k] - rel_xs[k-1]
                    if (s1 * s2) < -3.5:
                        flips += 1

                # ── 3. Companion Co-Movement Filter ──
                # Walking companions move in same direction with low relative velocity and zero fight flips
                if spd1 > 0.8 and spd2 > 0.8:
                    dot = vx1 * vx2 + vy1 * vy2
                    cos_sim = dot / (spd1 * spd2)
                    if cos_sim > 0.60 and avg_rel_speed < 2.5 and flips == 0 and not fallen:
                        self.pair_streaks[pair_key] = 0
                        continue

                # ── 4. Crossing Path / Separation Filter ──
                # Pedestrians crossing paths in opposite or angled directions and separating steadily
                dists = [h["dist"] for h in list(hist)[-4:]]
                if dists[-1] > dists[0] + 8.0 and dists[-1] > 40.0 and flips == 0 and not fallen:
                    self.pair_streaks[pair_key] = 0
                    continue

                # ── 5. Combat Criteria ──
                is_frame_combat = (flips >= 2 and avg_rel_speed > 3.0) or \
                                  (flips >= 1 and avg_rel_speed > 4.5) or \
                                  (avg_rel_speed > 5.5 and (iou > 0.18 or dist < 42.0)) or \
                                  (fallen and avg_rel_speed > 2.0)

                if is_frame_combat:
                    self.pair_streaks[pair_key] += 1
                else:
                    self.pair_streaks[pair_key] = max(0, self.pair_streaks[pair_key] - 1)

                if self.pair_streaks[pair_key] >= self.min_combat_streak:
                    score = min(98.0, 65.0 + (flips * 8.0) + (avg_rel_speed * 2.5) + (20.0 if fallen else 0.0))
                    max_fight_score = max(max_fight_score, score)
                    confirmed_combatants.add(id1)
                    confirmed_combatants.add(id2)
                    fight_candidates.append({
                        "combatants": [id1, id2],
                        "center": ((d1.cx + d2.cx) // 2, (d1.cy + d2.cy) // 2),
                        "score": round(score, 1)
                    })

        # Streak decay
        for pk in list(self.pair_streaks.keys()):
            if pk not in active_pair_keys:
                self.pair_streaks[pk] = max(0, self.pair_streaks[pk] - 1)
                if self.pair_streaks[pk] == 0:
                    del self.pair_streaks[pk]

        if len(confirmed_combatants) > 0:
            self._combatant_ids = confirmed_combatants
            self.latch_frames = 12
            is_fight = True
        elif self.latch_frames > 0:
            self.latch_frames -= 1
            is_fight = True
            max_fight_score = 75.0
        else:
            self._combatant_ids.clear()
            is_fight = False

        return {
            "is_fight": is_fight,
            "fight_score": float(max_fight_score if is_fight else 0.0),
            "combatants": list(confirmed_combatants),
            "events": fight_candidates
        }

print("Running Calibrated Evaluation...")
model = YOLO('models/yolov8n.pt')

# 1. Mall Test (Normal walking)
cap_mall = cv2.VideoCapture(r'C:/Users/Vinay/.gemini/antigravity/brain/c72e5230-35e1-4b01-8b9d-9d38f2548731/.user_uploaded/media_1794940060262_def08e8b.mp4')
tot_mall = int(cap_mall.get(cv2.CAP_PROP_FRAME_COUNT))
det_mall = CalibratedFightDetector()
tracker_mall = MultiTracker()
mall_fights = 0

for f in range(tot_mall):
    ret, frame = cap_mall.read()
    if not ret: break
    res = model(frame, classes=[0], conf=0.22, imgsz=640, verbose=False)[0]
    dets = [Detection(int(b.xyxy[0][0]), int(b.xyxy[0][1]), int(b.xyxy[0][2]), int(b.xyxy[0][3]), float(b.conf[0])) for b in res.boxes]
    tracked = tracker_mall.update(dets)
    res_f = det_mall.detect(tracked, frame=frame)
    if res_f["is_fight"]:
        mall_fights += 1
cap_mall.release()
print(f"Mall Video: {mall_fights} / {tot_mall} false fight frames.")

# 2. Fight Test (Actual physical altercation)
cap_fight = cv2.VideoCapture(r'C:/Users/Vinay/.gemini/antigravity/brain/c72e5230-35e1-4b01-8b9d-9d38f2548731/.user_uploaded/media_1794938649770_7d23391e.mp4')
tot_fight = int(cap_fight.get(cv2.CAP_PROP_FRAME_COUNT))
det_fight = CalibratedFightDetector()
tracker_fight = MultiTracker()
fight_frames = 0
fight_log = []

for f in range(tot_fight):
    ret, frame = cap_fight.read()
    if not ret: break
    res = model(frame, classes=[0], conf=0.22, imgsz=640, verbose=False)[0]
    dets = [Detection(int(b.xyxy[0][0]), int(b.xyxy[0][1]), int(b.xyxy[0][2]), int(b.xyxy[0][3]), float(b.conf[0])) for b in res.boxes]
    tracked = tracker_fight.update(dets)
    res_f = det_fight.detect(tracked, frame=frame)
    if res_f["is_fight"]:
        fight_frames += 1
        fight_log.append((f, res_f["fight_score"], res_f["combatants"]))
cap_fight.release()
print(f"Fight Video: {fight_frames} / {tot_fight} fight frames detected.")
if fight_log:
    print(f"Fight detected from frame {fight_log[0][0]} to {fight_log[-1][0]}.")

