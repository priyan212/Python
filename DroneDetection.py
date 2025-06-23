import cv2
from ultralytics import YOLO

# Load general YOLOv8 model (not drone-specific, but good enough for objects)
model = YOLO('yolov8n.pt')  # This will download the model automatically

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Could not access webcam.")
    exit()

print("✅ Webcam started. Press 'q' to quit.")

# Common labels that could represent drones (based on COCO classes)
DRONE_CLASSES = ['airplane', 'bird', 'kite']

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to read from webcam.")
        break

    # Get predictions
    results = model.predict(source=frame, conf=0.3, save=False)
    frame_annotated = frame.copy()

    # Go through detections and filter drone-like objects
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        class_name = model.names[cls_id]

        if class_name in DRONE_CLASSES:
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = xyxy
            cv2.rectangle(frame_annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame_annotated, f"{class_name}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display
    cv2.imshow("Drone-Like Object Detection", frame_annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("🛑 Quitting...")
        break

cap.release()
cv2.destroyAllWindows()
