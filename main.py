import cv2

# โหลด Haar cascade สำหรับตรวจจับรถยนต์
car_cascade = cv2.CascadeClassifier("cars.xml")  # ต้องมีไฟล์ cars.xml

# เปิดกล้อง (หรือวิดีโอไฟล์ก็ได้ เช่น cv2.VideoCapture("traffic.mp4"))
cap = cv2.VideoCapture("2165-155327596_small.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # แปลงภาพเป็นขาวดำ
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ตรวจจับรถยนต์
    cars = car_cascade.detectMultiScale(gray, 1.1, 2)

    # วาดสี่เหลี่ยมรอบ ๆ รถ
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # แสดงผล
    cv2.imshow("Car Detection", frame)

    # กด q เพื่อออก
    if cv2.waitKey(30) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
