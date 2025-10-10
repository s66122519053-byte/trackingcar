import cv2
import numpy as np
from ultralytics import YOLO
from sort import Sort  # ใช้ไฟล์ sort.py ที่คุณมี

# ==========================
# โหลดโมเดล YOLO (ตรวจจับรถ)
# ==========================
model = YOLO("yolov8n.pt")  # รุ่นเล็กสุด (เร็ว)

# ==========================
# เปิดวิดีโอ
# ==========================
cap = cv2.VideoCapture("istockphoto-1194498762-640_adpp_is.mp4")

if not cap.isOpened():
    print("❌ ไม่สามารถเปิดวิดีโอได้")
    exit()

# ==========================
# สร้างตัวติดตาม (SORT Tracker)
# ==========================
tracker = Sort(max_age=10, min_hits=3, iou_threshold=0.3)

# ==========================
# ลูปหลัก
# ==========================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)[0]

    detections = []
    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # เฉพาะรถยนต์ รถบัส และรถบรรทุก
        if cls in [2, 5, 7] and conf > 0.3:
            detections.append([x1, y1, x2, y2, conf])

    dets = np.array(detections)
    if len(dets) == 0:
        dets = np.empty((0, 5))

    # อัปเดต tracker
    tracks = tracker.update(dets)

    # วาดกรอบและ ID
    for x1, y1, x2, y2, track_id in tracks:
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Car ID: {int(track_id)}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imshow("Car Detection & Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
