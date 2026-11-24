import sys, os, tempfile
sys.path.insert(0, os.path.abspath("."))
os.environ['YOLO_CONFIG_DIR'] = tempfile.gettempdir()
import cv2, math
from ultralytics import YOLO
from src.tracker import MultiTracker
from src.detector import Detection
from scratch.test_dual_benchmark import RefinedFightDetector

model = YOLO('models/yolov8n.pt')
cap_mall = cv2.VideoCapture(r'C:/Users/Vinay/.gemini/antigravity/brain/c72e5230-35e1-4b01-8b9d-9d38f2548731/.user_uploaded/media_1794940060262_def08e8b.mp4')
det_mall = RefinedFightDetector()
tracker_mall = MultiTracker()

for f in range(25):
    ret, frame = cap_mall.read()
    if not ret: break
    res = model(frame, classes=[0], conf=0.22, imgsz=640, verbose=False)[0]
    dets = [Detection(int(b.xyxy[0][0]), int(b.xyxy[0][1]), int(b.xyxy[0][2]), int(b.xyxy[0][3]), float(b.conf[0])) for b in res.boxes]
    tracked = tracker_mall.update(dets)
    res_f = det_mall.detect(tracked, frame=frame)
    if res_f["is_fight"]:
        print(f"Mall F{f}: events={res_f['events']}")
        break
cap_mall.release()

