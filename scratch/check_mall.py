import sys, os, tempfile
sys.path.insert(0, os.path.abspath("."))
os.environ['YOLO_CONFIG_DIR'] = tempfile.gettempdir()
import cv2, math
from ultralytics import YOLO
from src.tracker import MultiTracker
from src.detector import Detection

model = YOLO('models/yolov8n.pt')
tracker = MultiTracker()

cap = cv2.VideoCapture(r'C:/Users/Vinay/.gemini/antigravity/brain/c72e5230-35e1-4b01-8b9d-9d38f2548731/.user_uploaded/media_1794940060262_def08e8b.mp4')
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

mall_stats = []
for f_idx in range(min(120, total)):
    ret, frame = cap.read()
    if not ret: break
    res = model(frame, classes=[0], conf=0.25, imgsz=640, verbose=False)[0]
    dets = [Detection(int(b.xyxy[0][0]), int(b.xyxy[0][1]), int(b.xyxy[0][2]), int(b.xyxy[0][3]), float(b.conf[0])) for b in res.boxes]
    tracked = tracker.update(dets)
    
    n = len(tracked)
    for i in range(n):
        for j in range(i+1, n):
            d1, d2 = tracked[i], tracked[j]
            dist = math.hypot(d1.cx - d2.cx, d1.cy - d2.cy)
            if dist < 80:
                avg_h = (d1.height + d2.height) / 2.0
                dy2 = abs(d1.y2 - d2.y2)
                ratio = dy2 / max(1.0, avg_h)
                mall_stats.append((f_idx, d1.track_id, d2.track_id, dist, ratio))

cap.release()
print(f"Total close pairs in mall: {len(mall_stats)}")
print("Sample pairs (dist, dy2/avg_h ratio):")
for s in mall_stats[:15]:
    print(s)

