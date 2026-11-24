"""
Fight & Violent Altercation Detector
Detects aggressive physical altercations, brawls, grappling, and violent combat dynamics.
Filters out companions walking together, pedestrian path crossings, and perspective depth overlaps.
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict, deque
import math

from src.detector import Detection


class FightDetector:
    """
    High-precision physical altercation detector.
    Identifies real fights while strictly rejecting walking pairs, companions, and passing pedestrians.
    """

    def __init__(self, proximity_px: float = 65.0, min_combat_streak: int = 3,
                 min_combat_frames: Optional[int] = None):
        self.proximity_px = proximity_px
        self.min_combat_streak = min_combat_frames if min_combat_frames is not None else min_combat_streak
        self.min_combat_frames = self.min_combat_streak
        
        # Historical interaction state
        self.pair_histories: Dict[Tuple[int, int], deque] = defaultdict(lambda: deque(maxlen=20))
        self.pair_streaks: Dict[Tuple[int, int], int] = defaultdict(int)
        self.prev_gray: Optional[np.ndarray] = None
        self._combatant_ids: set = set()

    def reset(self):
        """Reset historical tracking states."""
        self.pair_histories.clear()
        self.pair_streaks.clear()
        self.prev_gray = None
        self._combatant_ids.clear()

    @staticmethod
    def _compute_iou(b1: Tuple[int, int, int, int], b2: Tuple[int, int, int, int]) -> float:
        """Compute Intersection over Union between two bounding boxes (x1, y1, x2, y2)."""
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
        """
        Evaluate frame for real violent altercations.
        Strictly ignores people walking side-by-side or crossing paths.
        """
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

                # ── 1. Ground Plane Perspective Separation Filter ──
                # In 2D perspective cameras, one person 10ft behind another can have overlapping boxes vertically.
                # Combatants in a fight must stand on the same floor plane (similar feet y2 coordinate).
                avg_h = (d1.height + d2.height) / 2.0
                if abs(d1.y2 - d2.y2) > (0.35 * avg_h):
                    self.pair_streaks[pair_key] = max(0, self.pair_streaks[pair_key] - 1)
                    continue

                # ── 2. Real Physical Proximity & Contact Filter ──
                dx = d1.cx - d2.cx
                dy = d1.cy - d2.cy
                dist = math.hypot(dx, dy)
                if dist > self.proximity_px:
                    self.pair_streaks[pair_key] = max(0, self.pair_streaks[pair_key] - 1)
                    continue

                iou = self._compute_iou(d1.bbox, d2.bbox)
                if iou < 0.12 and dist > 45.0:
                    self.pair_streaks[pair_key] = max(0, self.pair_streaks[pair_key] - 1)
                    continue

                # Record trajectory state
                self.pair_histories[pair_key].append({
                    "p1": (d1.cx, d1.cy),
                    "p2": (d2.cx, d2.cy),
                    "dist": dist,
                    "iou": iou,
                    "w1": d1.width,
                    "h1": d1.height,
                    "w2": d2.width,
                    "h2": d2.height
                })

                hist = self.pair_histories[pair_key]
                if len(hist) < 6:
                    continue

                # ── 3. Companion Co-Movement Filter (Walking Together in Mall) ──
                # Couples, friends, and families walk in the same direction at similar speeds.
                vx1 = (hist[-1]["p1"][0] - hist[-5]["p1"][0]) / 4.0
                vy1 = (hist[-1]["p1"][1] - hist[-5]["p1"][1]) / 4.0
                vx2 = (hist[-1]["p2"][0] - hist[-5]["p2"][0]) / 4.0
                vy2 = (hist[-1]["p2"][1] - hist[-5]["p2"][1]) / 4.0
                spd1 = math.hypot(vx1, vy1)
                spd2 = math.hypot(vx2, vy2)

                if spd1 > 1.0 and spd2 > 1.0:
                    dot = vx1 * vx2 + vy1 * vy2
                    cos_sim = dot / (spd1 * spd2)
                    if cos_sim > 0.55:
                        # Walking companions moving in same direction! Not a fight.
                        self.pair_streaks[pair_key] = 0
                        continue

                # ── 4. Crossing Path / Separation Filter ──
                # Pedestrians crossing paths get close momentarily then steadily walk away.
                dists = [h["dist"] for h in list(hist)[-5:]]
                if dists[-1] > dists[0] + 10.0 and dists[-1] > 45.0:
                    self.pair_streaks[pair_key] = 0
                    continue

                # ── 5. Violent Kinematics (Push-Pull Reversals & Acceleration) ──
                # In a fight, the relative offset (p1 - p2) oscillates rapidly back and forth.
                rel_xs = [h["p1"][0] - h["p2"][0] for h in list(hist)[-6:]]
                flips = 0
                for k in range(2, len(rel_xs)):
                    s1 = rel_xs[k-1] - rel_xs[k-2]
                    s2 = rel_xs[k] - rel_xs[k-1]
                    if (s1 * s2) < -4.0: # clear push-pull reversal of at least 4px
                        flips += 1

                # Check for knockdown / tackle (aspect ratio flips horizontal)
                fallen = (d1.width / max(1, d1.height) > 1.4) or (d2.width / max(1, d2.height) > 1.4)

                # Relative speed calculation
                rel_speeds = []
                hist_recent = list(hist)[-5:]
                for k in range(1, len(hist_recent)):
                    ddx = (hist_recent[k]["p1"][0] - hist_recent[k]["p2"][0]) - (hist_recent[k-1]["p1"][0] - hist_recent[k-1]["p2"][0])
                    ddy = (hist_recent[k]["p1"][1] - hist_recent[k]["p2"][1]) - (hist_recent[k-1]["p1"][1] - hist_recent[k-1]["p2"][1])
                    rel_speeds.append(math.hypot(ddx, ddy))

                avg_rel_speed = float(np.mean(rel_speeds)) if rel_speeds else 0.0

                # ── 6. Combat Evaluation ──
                # Must have push-pull reversals OR high-speed violent struggle OR violent knockdown
                is_frame_combat = (flips >= 2 and avg_rel_speed > 4.0) or \
                                  (fallen and avg_rel_speed > 3.0) or \
                                  (flips >= 1 and avg_rel_speed > 6.5 and iou > 0.20)

                if is_frame_combat:
                    self.pair_streaks[pair_key] += 1
                else:
                    self.pair_streaks[pair_key] = max(0, self.pair_streaks[pair_key] - 1)

                # Require multi-frame confirmation streak to eliminate transient noise
                if self.pair_streaks[pair_key] >= self.min_combat_streak:
                    score = min(98.0, 60.0 + (flips * 8.0) + (avg_rel_speed * 2.5) + (20.0 if fallen else 0.0))
                    max_fight_score = max(max_fight_score, score)
                    confirmed_combatants.add(id1)
                    confirmed_combatants.add(id2)
                    fight_candidates.append({
                        "combatants": [id1, id2],
                        "center": ((d1.cx + d2.cx) // 2, (d1.cy + d2.cy) // 2),
                        "score": round(score, 1),
                        "flips": flips,
                        "rel_speed": round(avg_rel_speed, 1),
                        "iou": round(iou, 2)
                    })

        # Decay streaks for inactive pairs
        for pk in list(self.pair_streaks.keys()):
            if pk not in active_pair_keys:
                self.pair_streaks[pk] = max(0, self.pair_streaks[pk] - 1)
                if self.pair_streaks[pk] == 0:
                    del self.pair_streaks[pk]

        self._combatant_ids = confirmed_combatants
        is_fight = len(confirmed_combatants) > 0

        return {
            "is_fight": is_fight,
            "fight_score": float(max_fight_score if is_fight else 0.0),
            "combatants": list(confirmed_combatants),
            "events": fight_candidates
        }

    def is_combatant(self, track_id: int) -> bool:
        """Check if track ID is actively flagged in a confirmed fight."""
        return track_id in self._combatant_ids
