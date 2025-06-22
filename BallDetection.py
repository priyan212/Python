import cv2
import numpy as np

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize for consistency and speed
    frame = cv2.resize(frame, (640, 480))

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    # Detect circles using HoughCircles
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=50,
        param1=100,
        param2=30,
        minRadius=10,
        maxRadius=100
    )

    # If any circles are found
    if circles is not None:
        circles = np.uint16(np.around(circles))  # Convert float to integer
        for (x, y, r) in circles[0, :]:
            # Draw the outer circle
            cv2.circle(frame, (x, y), r, (0, 255, 0), 2)
            # Draw the center point
            cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)
            # Show coordinates
            cv2.putText(frame, f"Ball at ({x}, {y})", (x - 50, y - r - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Display the result
    cv2.imshow("Ball Detection (Shape Only)", frame)

    if cv2.waitKey(1) & 0xFF == ord(' '):
        break

cap.release()
cv2.destroyAllWindows()
