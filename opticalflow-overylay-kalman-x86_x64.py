import cv2
import numpy as np

# --- Adjustable settings ---
FRAME_WIDTH  = 640   # set to 320, 640, 800, etc.
FRAME_HEIGHT = 480   # set to 240, 480, 600, etc.
SENSITIVITY  = 1.9   # lower = more sensitive, higher = less sensitive
STEP         = 11    # grid spacing for flow visualization

# --- Initialize webcam ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Read the first frame
ret, frame1 = cap.read()
if not ret:
    print("Error: Could not capture first frame.")
    exit()

frame1 = cv2.resize(frame1, (FRAME_WIDTH, FRAME_HEIGHT))
prev_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

# --- Initialize Kalman filters per sampled point ---
h, w = prev_gray.shape
kalman_filters = {}
for y in range(0, h, STEP):
    for x in range(0, w, STEP):
        kf = cv2.KalmanFilter(4, 2)  # state: [x, y, dx, dy], measurement: [x, y]
        kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                         [0, 1, 0, 0]], np.float32)
        kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                        [0, 1, 0, 1],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 1]], np.float32)
        kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5
        kf.statePre = np.array([[x], [y], [0], [0]], np.float32)
        kalman_filters[(x, y)] = kf

try:
    while True:
        ret, frame2 = cap.read()
        if not ret:
            break

        frame2 = cv2.resize(frame2, (FRAME_WIDTH, FRAME_HEIGHT))
        gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # Compute optical flow
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
                                            0.5, 3, 15, 3, 5, 1.2, 0)

        output = frame2.copy()

        # Draw arrows with Kalman-smoothed endpoints
        for y in range(0, h, STEP):
            for x in range(0, w, STEP):
                fx, fy = flow[y, x]
                magnitude = np.sqrt(fx**2 + fy**2)

                if magnitude > SENSITIVITY:
                    measured = np.array([[np.float32(x + fx)], [np.float32(y + fy)]])
                    kf = kalman_filters[(x, y)]
                    kf.correct(measured)
                    predicted = kf.predict()
                    end_x, end_y = int(predicted[0]), int(predicted[1])

                    cv2.arrowedLine(output, (x, y), (end_x, end_y),
                                    (0, 255, 0), 1, tipLength=0.3)
                    print(f"Point ({x},{y}) smoothed to ({end_x},{end_y}) with motion {magnitude:.2f}")

        cv2.imshow('Smoothed Optical Flow', output)
        prev_gray = gray.copy()

        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
