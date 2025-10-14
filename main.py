from picamera2 import Picamera2
import cv2
import numpy as np
from ultralytics import YOLO
from sort import Sort

# โหลดโมเดล YOLO
model = YOLO("yolov8n.pt")

# เปิดกล้องผ่าน Picamera2
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": "XRGB8888", "size": (640, 480)}))
picam2.start()

# สร้างตัวติดตาม
tracker = Sort(max_age=10, min_hits=3, iou_threshold=0.3)

while True:
    frame = picam2.capture_array()  # ดึงภาพจากกล้อง
    results = model(frame, verbose=False)[0]

    detections = []
    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        if cls in [2, 5, 7] and conf > 0.3:
            detections.append([x1, y1, x2, y2, conf])

    dets = np.array(detections) if len(detections) > 0 else np.empty((0, 5))
    tracks = tracker.update(dets)

    for x1, y1, x2, y2, track_id in tracks:
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Car ID: {int(track_id)}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imshow("Car Detection & Tracking (Pi Camera)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
picam2.stop()