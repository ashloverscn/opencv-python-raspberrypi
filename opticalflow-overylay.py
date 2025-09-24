import cv2
import numpy as np

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Read the first frame
ret, frame1 = cap.read()
if not ret:
    print("Error: Cannot read first frame.")
    exit()

# Convert to grayscale
prev_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

# Parameters for drawing motion vectors
step = 16  # spacing between points for visualization

while True:
    ret, frame2 = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Compute optical flow
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
                                        0.5, 3, 15, 3, 5, 1.2, 0)

    # Copy frame to draw arrows
    output = frame2.copy()

    # Draw arrows for motion vectors
    h, w = gray.shape
    for y in range(0, h, step):
        for x in range(0, w, step):
            fx, fy = flow[y, x]
            # Only draw if motion is significant
            if np.sqrt(fx**2 + fy**2) > 2:
                cv2.arrowedLine(output, (x, y), (int(x+fx), int(y+fy)), (0, 255, 0), 1, tipLength=0.3)
                print(f"Point ({x},{y}) moved to ({int(x+fx)},{int(y+fy)})")

    cv2.imshow('Optical Flow with Coordinates', output)

    prev_gray = gray.copy()

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
