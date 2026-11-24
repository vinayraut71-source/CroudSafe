import sys, os, tempfile
sys.path.insert(0, os.path.abspath("."))
os.environ['YOLO_CONFIG_DIR'] = tempfile.gettempdir()
import cv2, math
import numpy as np
from ultralytics import YOLO
from src.tracker import MultiTracker
from src.fight_detector import FightDetector
from src.detector import Detection

model = YOLO('models/yolov8n.pt')
tracker = MultiTracker()
fight_det = FightDetector()

cap = cv2.VideoCapture(r'C:/Users/Vinay/.gemini/antigravity/brain/c72e5230-35e1-4b01-8b9d-9d38f2548731/.user_uploaded/media_1794938649770_7d23391e.mp4')

for f_idx in range(198):
    ret, frame = cap.read()
    if not ret:
        break
    res = model(frame, classes=[0], conf=0.25, imgsz=640, verbose=False)[0]
    dets = [Detection(int(b.xyxy[0][0]), int(b.xyxy[0][1]), int(b.xyxy[0][2]), int(b.xyxy[0][3]), float(b.conf[0])) for b in res.boxes]
    tracked = tracker.update(dets)
    
    t_map = {d.track_id: d for d in tracked}
    if f_idx in (193, 194, 195):
        d1 = t_map.get(15)
        d2 = t_map.get(75)
        if d1 and d2:
            hist = fight_det.pair_histories[(15, 75)]
            if len(hist) >= 5:
                vx1 = (hist[-1]["p1"][0] - hist[-5]["p1"][0]) / 4.0
                vy1 = (hist[-1]["p1"][1] - hist[-5]["p1"][1]) / 4.0
                vx2 = (hist[-1]["p2"][0] - hist[-5]["p2"][0]) / 4.0
                vy2 = (hist[-1]["p2"][1] - hist[-5]["p2"][1]) / 4.0
                spd1 = math.hypot(vx1, vy1)
                spd2 = math.hypot(vx2, vy2)
                dot = vx1 * vx2 + vy1 * vy2
                cos_sim = dot / (spd1 * spd2) if (spd1 * spd2) > 0 else 0
                print(f"F{f_idx}: spd1={spd1:.2f}, spd2={spd2:.2f}, cos_sim={cos_sim:.2f}")

    f_res = fight_det.detect(tracked, frame=frame)
    if f_idx in (193, 194, 195):
        print(f"  Fight result F{f_idx}: {f_res['is_fight']}, streak={fight_det.pair_streaks.get((15, 75), 0)}")

cap.release()

