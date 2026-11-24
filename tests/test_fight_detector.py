"""
Unit tests for Fight & Altercation Detection Engine.
Validates fight dynamics, pairwise kinematics, behavior classification, and alert triggers.
"""

import pytest
import numpy as np

from src.detector import Detection, PersonDetector
from src.fight_detector import FightDetector
from src.behavior_classifier import BehaviorClassifier
from src.movement import MovementAnalyzer
from src.alerts import AlertEngine, ALERT_CRITICAL, ALERT_DANGER, ALERT_SAFE


def test_fight_detector_initialization():
    fd = FightDetector(proximity_px=80.0)
    assert fd.proximity_px == 80.0
    assert fd.min_combat_frames >= 1
    assert len(fd.pair_histories) == 0


def test_no_fight_normal_walking():
    """Two individuals walking peacefully side-by-side should NOT trigger a fight alert."""
    fd = FightDetector(proximity_px=90.0)
    
    # 15 frames of steady parallel walking
    for frame_i in range(15):
        d1 = Detection(100 + frame_i * 4, 200, 140 + frame_i * 4, 300, 0.85, track_id=1)
        d2 = Detection(145 + frame_i * 4, 200, 185 + frame_i * 4, 300, 0.82, track_id=2)
        res = fd.detect([d1, d2], speeds={1: 4.0, 2: 4.0})
    
    assert res["is_fight"] is False
    assert res["fight_score"] < 40.0
    assert len(res["combatants"]) == 0


def test_fight_detection_overlapping_oscillating():
    """Two individuals in high-overlap reciprocating struggle should trigger a fight alert."""
    fd = FightDetector(proximity_px=90.0, min_combat_streak=2)
    
    # 12 frames of violent push-and-pull oscillations on the same ground plane
    base_positions = [
        ((200, 200, 250, 320), (225, 200, 275, 320)),
        ((220, 200, 275, 320), (210, 200, 265, 320)),
        ((195, 205, 255, 320), (235, 195, 290, 320)),
        ((225, 200, 280, 320), (215, 205, 265, 320)),
        ((190, 200, 255, 320), (230, 200, 285, 320)),
        ((225, 205, 285, 320), (210, 200, 265, 320)),
        ((195, 200, 255, 320), (235, 200, 285, 320)),
        ((225, 200, 280, 320), (210, 205, 265, 320)),
        ((190, 200, 255, 320), (230, 200, 285, 320)),
        ((225, 205, 285, 320), (210, 200, 265, 320)),
    ]
    
    res = None
    for b1, b2 in base_positions:
        d1 = Detection(b1[0], b1[1], b1[2], b1[3], 0.80, track_id=10)
        d2 = Detection(b2[0], b2[1], b2[2], b2[3], 0.75, track_id=11)
        res = fd.detect([d1, d2], speeds={10: 8.0, 11: 7.5})

    assert res is not None
    assert res["is_fight"] is True
    assert res["fight_score"] >= 50.0
    assert 10 in res["combatants"]
    assert 11 in res["combatants"]
    assert fd.is_combatant(10) is True


def test_fight_detection_with_falling():
    """An aggressive interaction followed by a fall/knockdown should trigger a fight alert."""
    fd = FightDetector(proximity_px=90.0, min_combat_streak=2)
    
    # Frames of grapple then fall
    steps = [
        ((300, 200, 350, 320), (325, 200, 375, 320)),
        ((320, 200, 375, 320), (310, 200, 360, 320)),
        ((295, 205, 355, 320), (335, 200, 385, 320)),
        ((325, 200, 380, 320), (315, 205, 365, 320)),
        ((290, 200, 350, 320), (330, 200, 380, 320)),
        ((325, 205, 380, 320), (310, 200, 360, 320)),
        # Fall: Person 21 goes horizontal (w=90, h=40 -> w/h = 2.25)
        ((310, 200, 360, 320), (300, 280, 390, 320)),
        ((310, 200, 360, 320), (300, 280, 390, 320)),
    ]
    
    res = None
    for b1, b2 in steps:
        d1 = Detection(b1[0], b1[1], b1[2], b1[3], 0.70, track_id=20)
        d2 = Detection(b2[0], b2[1], b2[2], b2[3], 0.65, track_id=21)
        res = fd.detect([d1, d2], speeds={20: 6.0, 21: 9.0})
        
    assert res is not None
    assert res["is_fight"] is True
    assert res["fight_score"] >= 55.0


def test_behavior_classifier_integration():
    """BehaviorClassifier should correctly assess fights and mark combatants."""
    bc = BehaviorClassifier()
    bc.fight_detector.min_combat_streak = 2
    
    steps = [
        ((150, 150, 195, 250), (170, 150, 215, 250)),
        ((165, 150, 215, 250), (155, 150, 195, 250)),
        ((145, 155, 195, 250), (180, 150, 225, 250)),
        ((165, 150, 215, 250), (150, 150, 195, 250)),
        ((145, 155, 195, 250), (180, 150, 225, 250)),
        ((165, 150, 215, 250), (150, 150, 195, 250)),
        ((145, 155, 195, 250), (180, 150, 225, 250)),
        ((165, 150, 215, 250), (150, 150, 195, 250)),
    ]
    
    res = None
    for b1, b2 in steps:
        d1 = Detection(b1[0], b1[1], b1[2], b1[3], 0.8, track_id=5)
        d2 = Detection(b2[0], b2[1], b2[2], b2[3], 0.8, track_id=6)
        res = bc.assess_anomalies([d1, d2], speeds={5: 7.0, 6: 7.0}, vectors=[])

    assert res is not None
    assert res["fight_detected"] is True
    assert 5 in res["combatants"]
    assert 6 in res["combatants"]
    assert any(ev[1] == "FIGHTING" for ev in res["events"])


def test_movement_analyzer_behavior_tagging():
    """MovementAnalyzer detect_behaviors should add FIGHTING for combatants."""
    ma = MovementAnalyzer()
    d1 = Detection(100, 100, 150, 220, 0.9, track_id=30)
    d2 = Detection(140, 100, 190, 220, 0.9, track_id=31)
    
    behaviors = ma.detect_behaviors([d1, d2], avg_speed=2.0, combatants=[30, 31])
    assert "FIGHTING" in behaviors[30]
    assert "FIGHTING" in behaviors[31]


def test_alert_engine_fight_elevation():
    """AlertEngine should elevate alert to CRITICAL when fight is detected."""
    ae = AlertEngine()
    
    metrics = {
        "density_level": ALERT_SAFE,
        "speed_level": ALERT_SAFE,
        "fight_detected": True,
        "fight_score": 75.0,
        "combatants": [10, 11]
    }
    
    lvl = ae.evaluate(metrics)
    assert lvl == ALERT_CRITICAL
    assert ae.risk_score >= 80.0
    
    # Check alert message format
    feed = ae.get_feed()
    assert len(feed) > 0
    assert "VIOLENT ALTERCATION" in feed[0]["message"]


def test_person_detector_custom_conf_imgsz():
    """PersonDetector should allow custom conf and imgsz parameters."""
    det = PersonDetector(conf=0.22, imgsz=640)
    assert det.conf == 0.22
    assert det.imgsz == 640
