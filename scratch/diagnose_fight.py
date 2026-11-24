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

for f_idx in range(205):
    ret, frame = cap.read()
    if not ret:
        break
    res = model(frame, classes=[0], conf=0.25, imgsz=640, verbose=False)[0]
    dets = [Detection(int(b.xyxy[0][0]), int(b.xyxy[0][1]), int(b.xyxy[0][2]), int(b.xyxy[0][3]), float(b.conf[0])) for b in res.boxes]
    tracked = tracker.update(dets)
    
    t_map = {d.track_id: d for d in tracked}
    if 188 <= f_idx <= 200:
        print(f"=== Frame {f_idx} ===")
        if 15 in t_map and 75 in t_map:
            d1 = t_map[15]
            d2 = t_map[75]
            avg_h = (d1.height + d2.height) / 2.0
            dy2 = abs(d1.y2 - d2.y2)
            dist = math.hypot(d1.cx - d2.cx, d1.cy - d2.cy)
            iou = FightDetector._compute_iou(d1.bbox, d2.bbox)
            print(f"  Presence: 15 & 75 present. dy2={dy2:.1f} (thresh={0.35*avg_h:.1f}), dist={dist:.1f}, iou={iou:.2f}")
        else:
            print(f"  Missing IDs! 15 in t_map: {15 in t_map}, 75 in t_map: {75 in t_map}")
            print(f"  IDs in frame: {list(t_map.keys())}")
    
    f_res = fight_det.detect(tracked, frame=frame)
    if 188 <= f_idx <= 200:
        s1 = fight_det.pair_streaks.get((15, 75), 0)
        print(f"  Result: is_fight={f_res['is_fight']}, streak={s1}")

cap.release()
