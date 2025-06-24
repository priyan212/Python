import cv2
from ultralytics import YOLO
import serial
import time

arduino = serial.Serial('COM10', 9600)
time.sleep(2)

model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture(1) 
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
x_center_target = frame_width // 2

def send(cmd):
    arduino.write(f"{cmd}\n".encode())
    print("Sent:", cmd)

TOLERANCE = 40  # Pixels around center considered "locked"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(source=frame, conf=0.3, save=False)
    boxes = results[0].boxes
    annotated = frame.copy()

    for box in boxes:
        cls = int(box.cls[0])
        label = model.names[cls]

        if label in ['airplane', 'bird', 'kite']:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Servo angle based on Y (vertical position)
            angle = int((cy / frame_height) * 180)
            send(f"AIM:{angle}")

            # Pan control based on X (horizontal center)
            diff = cx - x_center_target
            if abs(diff) <= TOLERANCE:
                send("LOCK")
            elif diff < -TOLERANCE:
                send("LEFT")
            elif diff > TOLERANCE:
                send("RIGHT")

            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.circle(annotated, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(annotated, f"{label} ({cx},{cy})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
            break  # Only handle 1 detection for now

    cv2.imshow("Targeting System", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
