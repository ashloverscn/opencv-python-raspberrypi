from picamera2 import Picamera2
import cv2
import numpy as np
import time

# --- Adjustable settings ---
FRAME_WIDTH  = 320
FRAME_HEIGHT = 240
SENSITIVITY  = 2.0  # optical flow scaling factor

# PID-like gains for drift correction
Kp_x = 0.1
Kp_y = 0.1
Kp_z = 0.1

# --- Kalman filter for smoothing ---
class KalmanFilter1D:
    def __init__(self, process_var=1e-4, meas_var=0.1):
        self.x = 0.0
        self.P = 1.0
        self.Q = process_var
        self.R = meas_var

    def update(self, measurement):
        P_pred = self.P + self.Q
        K = P_pred / (P_pred + self.R)
        self.x = self.x + K * (measurement - self.x)
        self.P = (1 - K) * P_pred
        return self.x

kf_x = KalmanFilter1D()
kf_y = KalmanFilter1D()
kf_z = KalmanFilter1D()

# --- Initialize Picamera2 ---
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (FRAME_WIDTH, FRAME_HEIGHT)})
picam2.configure(config)
picam2.start()
time.sleep(1)

# Capture the first frame (stationary reference frame)
old_frame = picam2.capture_array()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

# Detect stationary features only in the first frame
feature_params = dict(maxCorners=200, qualityLevel=0.3, minDistance=7, blockSize=7)
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# Lucas-Kanade optical flow parameters
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Store temporary trailing dots
dots = []

while True:
    frame = picam2.capture_array()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Compute optical flow using only initial stationary features
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    if p1 is None:
        continue

    good_new = p1[st == 1]
    good_old = p0[st == 1]

    drift_x, drift_y, drift_z = 0, 0, 0

    for new, old in zip(good_new, good_old):
        a, b = new.ravel()
        c, d = old.ravel()
        drift_x += (a - c)
        drift_y += (b - d)
        drift_z += np.sqrt((a - c)**2 + (b - d)**2)
        dots.append((int(a), int(b)))

    n_points = len(good_new)
    if n_points > 0:
        drift_x = (drift_x / n_points) * Kp_x * SENSITIVITY
        drift_y = (drift_y / n_points) * Kp_y * SENSITIVITY
        drift_z = (drift_z / n_points) * Kp_z * SENSITIVITY

    # Kalman filter smoothing
    drift_x = kf_x.update(drift_x)
    drift_y = kf_y.update(drift_y)
    drift_z = kf_z.update(drift_z)

    # Draw temporary trailing dots (fade out older ones)
    for i, (x, y) in enumerate(dots):
        alpha = max(0, 1 - i / 50)
        cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)

    if len(dots) > 50:
        dots = dots[-50:]

    # Display drift values
    cv2.putText(frame, f"Drift X: {drift_x:.2f}, Y: {drift_y:.2f}, Z: {drift_z:.2f}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    cv2.imshow("Drone Drift Correction", frame)

    # Update previous frame (do not update features)
    old_gray = frame_gray.copy()

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break

cv2.destroyAllWindows()
picam2.stop()
