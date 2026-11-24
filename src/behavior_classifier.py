"""
Graph Neural Network Crowd Flow Prediction & Behavior Classifier.
Models the crowd as a connected system.
Integrates real-time fight and physical altercation detection.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any

from src.detector import Detection
from src.fight_detector import FightDetector


class BehaviorClassifier:
    """Classifies movement anomalies, fighting, and aggressive physical altercations."""
    def __init__(self):
        self.enabled = True
        self.fight_detector = FightDetector()
        
    def assess_anomalies(self, detections: List[Detection], speeds: Dict[int, float],
                         vectors: List[tuple], frame: Optional[np.ndarray] = None) -> dict:
        """
        Identify anomalous behavior clusters and physical altercations visually and logically.
        Returns anomaly_score (0-100), fight detection status, and list of anomalous track IDs.
        """
        anomalous_ids = []
        
        # 1. Spatial heuristic check: moving fast against the grain
        avg_speed = np.mean(list(speeds.values())) if speeds else 0.0
        
        for det in detections:
            tid = det.track_id
            spd = speeds.get(tid, 0.0)
            
            # fast outlier
            if avg_speed > 0 and spd > avg_speed * 3.0:
                anomalous_ids.append((tid, "RUNNING_FAST"))
                continue

        # 2. Fight & Physical Altercation Detection
        fight_result = self.fight_detector.detect(detections, speeds=speeds, frame=frame)
        if fight_result["is_fight"]:
            for cid in fight_result["combatants"]:
                anomalous_ids.append((cid, "FIGHTING"))
                
        # Calculate anomalous density and composite score
        score = 0.0
        if detections:
            score = (len(anomalous_ids) / len(detections)) * 100.0

        # Factor in fight score if a violent altercation is detected
        composite_score = min(score * 3.0, 100.0)
        if fight_result["is_fight"]:
            composite_score = max(composite_score, fight_result["fight_score"])
            
        return {
            "score": composite_score,
            "events": anomalous_ids,
            "fight_detected": fight_result["is_fight"],
            "fight_score": fight_result["fight_score"],
            "combatants": fight_result["combatants"],
            "fight_events": fight_result["events"]
        }
