# from picamera2 import Picamera2 # ไม่ใช้
import cv2
import numpy as np
from ultralytics import YOLO
from sort import Sort

# โหลดโมเดล YOLO
model = YOLO("yolov8n.pt")
# กำหนด mapping คลาสที่ต้องการติดตาม
TARGET_CLASSES = {2: "Car", 5: "Bus", 7: "Truck"}
TARGET_CLASS_IDS = list(TARGET_CLASSES.keys())


# เปิดกล้องโน้ตบุ๊ก (Built-in Webcam)
# **เพิ่ม cv2.CAP_DSHOW สำหรับ Windows เพื่อความเสถียร**
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) 


# ตรวจสอบว่าเปิดกล้องสำเร็จหรือไม่
if not cap.isOpened():
    # ลองเปิดโดยไม่มี Back-end
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open Notebook Camera (ID 0). Program exiting.")
        exit()
    print("Warning: Opened camera without CAP_DSHOW. Stability may vary.")


# ตั้งค่าความละเอียดเฟรม
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# สร้างตัวติดตาม
tracker = Sort(max_age=10, min_hits=3, iou_threshold=0.3)

print("Starting video processing. Press 'q' to stop.")

while True:
    ret, frame = cap.read()  # ดึงภาพจากกล้อง
    
    # หากอ่านเฟรมไม่ได้ (กล้องหยุดทำงาน, หรือสตรีมจบ) ให้หยุดทำงาน
    if not ret:
        print("Error: Can't receive frame (stream end or camera disconnected). Stopping.")
        break

    # 1. การตรวจจับ (Detection)
    results = model(frame, verbose=False, device='cpu')[0] # ระบุ device='cpu' เพื่อความเสถียรหากไม่ได้ใช้ GPU

    detections = []
    
    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        # ตรวจจับเฉพาะคลาสที่กำหนดไว้และมี Confidence สูงกว่า 0.3
        if cls in TARGET_CLASS_IDS and conf > 0.3:
            detections.append([x1, y1, x2, y2, conf])
            
    # 2. การติดตาม (Tracking)
    dets = np.array(detections) if len(detections) > 0 else np.empty((0, 5))
    tracks = tracker.update(dets)

    # 3. วาดผลลัพธ์ (Drawing)
    for x1, y1, x2, y2, track_id in tracks:
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        track_id = int(track_id)
        
        label = f"ID: {track_id}"
        
        # วาดกล่อง
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # วาด ID
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imshow("Car Detection & Tracking (Notebook Camera)", frame)

    # 4. การจัดการการหยุดทำงาน
    # cv2.waitKey(1) เป็นค่าที่ลื่นไหลที่สุด (ไม่หน่วง) 
    # ใช้ 1 หรือ 30 ขึ้นอยู่กับประสิทธิภาพ
    key = cv2.waitKey(1) & 0xFF 
    
    # หยุดเมื่อกดปุ่ม 'q' หรือ Esc (27)
    if key == ord('q') or key == 27:
        print("Quit key pressed. Stopping program.")
        break

# Cleanup
cv2.destroyAllWindows()
cap.release()
print("Program stopped and resources released.")