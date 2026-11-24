import sys, os, tempfile
sys.path.insert(0, os.path.abspath("."))
os.environ['YOLO_CONFIG_DIR'] = tempfile.gettempdir()
import cv2, math
from ultralytics import YOLO
from src.tracker import MultiTracker
from src.fight_detector import FightDetector
from src.detector import Detection

model = YOLO('models/yolov8n.pt')
tracker = MultiTracker()
fight_det = FightDetector()

cap = cv2.VideoCapture(r'C:/Users/Vinay/.gemini/antigravity/brain/c72e5230-35e1-4b01-8b9d-9d38f2548731/.user_uploaded/media_1794938649770_7d23391e.mp4')

for f_idx in range(196):
    ret, frame = cap.read()
    if not ret: break
    res = model(frame, classes=[0], conf=0.25, imgsz=640, verbose=False)[0]
    dets = [Detection(int(b.xyxy[0][0]), int(b.xyxy[0][1]), int(b.xyxy[0][2]), int(b.xyxy[0][3]), float(b.conf[0])) for b in res.boxes]
    tracked = tracker.update(dets)
    
    if f_idx == 194:
        # Step through logic for (15, 75)
        d1 = [d for d in tracked if d.track_id == 15][0]
        d2 = [d for d in tracked if d.track_id == 75][0]
        avg_h = (d1.height + d2.height) / 2.0
        dy2 = abs(d1.y2 - d2.y2)
        dist = math.hypot(d1.cx - d2.cx, d1.cy - d2.cy)
        iou = FightDetector._compute_iou(d1.bbox, d2.bbox)
        hist = fight_det.pair_histories[(15, 75)]
        
        # Check step 3
        vx1 = (d1.cx - hist[-4]["p1"][0]) / 4.0
        vy1 = (d1.cy - hist[-4]["p1"][1]) / 4.0
        vx2 = (d2.cx - hist[-4]["p2"][0]) / 4.0
        vy2 = (d2.cy - hist[-4]["p2"][1]) / 4.0
        spd1 = math.hypot(vx1, vy1)
        spd2 = math.hypot(vx2, vy2)
        dot = vx1 * vx2 + vy1 * vy2
        cos_sim = dot / (spd1 * spd2) if (spd1 * spd2) > 0 else 0
        
        # Check step 4
        dists = [h["dist"] for h in list(hist)[-4:]] + [dist]
        sep_check = dists[-1] > dists[0] + 10.0 and dists[-1] > 45.0
        
        # Check step 5
        rel_xs = [h["p1"][0] - h["p2"][0] for h in list(hist)[-5:]] + [d1.cx - d2.cx]
        flips = 0
        for k in range(2, len(rel_xs)):
            s1 = rel_xs[k-1] - rel_xs[k-2]
            s2 = rel_xs[k] - rel_xs[k-1]
            if (s1 * s2) < -4.0:
                flips += 1
                
        fallen = (d1.width / max(1, d1.height) > 1.4) or (d2.width / max(1, d2.height) > 1.4)
        
        rel_speeds = []
        hist_recent = list(hist)[-4:] + [{"p1": (d1.cx, d1.cy), "p2": (d2.cx, d2.cy)}]
        for k in range(1, len(hist_recent)):
            ddx = (hist_recent[k]["p1"][0] - hist_recent[k]["p2"][0]) - (hist_recent[k-1]["p1"][0] - hist_recent[k-1]["p2"][0])
            ddy = (hist_recent[k]["p1"][1] - hist_recent[k]["p2"][1]) - (hist_recent[k-1]["p1"][1] - hist_recent[k-1]["p2"][1])
            rel_speeds.append(math.hypot(ddx, ddy))
        avg_rel_speed = float(sum(rel_speeds)/len(rel_speeds))
        
        is_combat = (flips >= 2 and avg_rel_speed > 4.0) or \
                    (fallen and avg_rel_speed > 3.0) or \
                    (flips >= 1 and avg_rel_speed > 6.5 and iou > 0.20)
                    
        print(f"F194 Details: spd1={spd1:.2f}, spd2={spd2:.2f}, cos_sim={cos_sim:.2f}")
        print(f"sep_check={sep_check}, flips={flips}, avg_rel_speed={avg_rel_speed:.2f}, fallen={fallen}, iou={iou:.2f}")
        print(f"is_combat={is_combat}")

    fight_det.detect(tracked, frame=frame)
cap.release()

