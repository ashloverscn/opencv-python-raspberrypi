import cv2
import numpy as np
from picamera2 import Picamera2
import time

# -----------------------------
# Camera setup
# -----------------------------
picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"size": (640, 480), "format": "XRGB8888"}
)
picam2.configure(config)
picam2.start()
time.sleep(0.5)  # camera warmup

# -----------------------------
# Parameters
# -----------------------------
MAX_CORNERS = 200
QUALITY_LEVEL = 0.3
MIN_DISTANCE = 7
BLOCK_SIZE = 7
FEATURE_REFRESH_INTERVAL = 30
STATIONARY_THRESHOLD = 2.0  # pixels

# -----------------------------
# Kalman Filter Setup (XYZ + velocity)
# State: [x, y, z, vx, vy, vz]
# Measurement: [x, y, z]
# -----------------------------
kalman = cv2.KalmanFilter(6, 3)
kalman.measurementMatrix = np.array([[1,0,0,0,0,0],
                                     [0,1,0,0,0,0],
                                     [0,0,1,0,0,0]], np.float32)
kalman.transitionMatrix = np.array([[1,0,0,1,0,0],
                                    [0,1,0,0,1,0],
                                    [0,0,1,0,0,1],
                                    [0,0,0,1,0,0],
                                    [0,0,0,0,1,0],
                                    [0,0,0,0,0,1]], np.float32)
kalman.processNoiseCov = np.eye(6, dtype=np.float32) * 0.03
kalman.measurementNoiseCov = np.eye(3, dtype=np.float32) * 0.5  # smooth measurement noise

# -----------------------------
# Initial feature detection
# -----------------------------
old_frame = picam2.capture_array()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, maxCorners=MAX_CORNERS,
                             qualityLevel=QUALITY_LEVEL,
                             minDistance=MIN_DISTANCE,
                             blockSize=BLOCK_SIZE)
frame_idx = 0

# -----------------------------
# Relative Z estimation
# -----------------------------
def estimate_z(points):
    if len(points) < 2:
        return 0
    dists = np.linalg.norm(points[:, None] - points[None, :], axis=2)
    return np.mean(dists)  # relative Z

old_z = estimate_z(p0.reshape(-1,2)) if p0 is not None else 0

# -----------------------------
# Main loop
# -----------------------------
while True:
    frame = picam2.capture_array()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if p0 is not None and len(p0) > 0:
        # Optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None)
        good_new = p1[st==1]
        good_old = p0[st==1]

        # Stationary points
        movement = np.linalg.norm(good_new - good_old, axis=1)
        stationary_points = good_new[movement < STATIONARY_THRESHOLD]

        # Draw stationary points
        for pt in stationary_points:
            x, y = pt.ravel()
            cv2.circle(frame, (int(x), int(y)), 4, (0,255,0), -1)

        # Centroid and relative Z
        if len(stationary_points) > 0:
            centroid = np.mean(stationary_points, axis=0)
            z = estimate_z(stationary_points)
            old_z = z

            measured = np.array([[np.float32(centroid[0])],
                                 [np.float32(centroid[1])],
                                 [np.float32(z)]])

            # Kalman update
            kalman.correct(measured)
            predicted = kalman.predict()
            px, py, pz, vx, vy, vz = predicted.flatten()

            # Smoothed velocity directly from Kalman
            smoothed_vx, smoothed_vy, smoothed_vz = vx, vy, vz

            # Display XYZ position
            cv2.circle(frame, (int(px), int(py)), 6, (0,0,255), 2)
            cv2.putText(frame,
                        f"X:{int(px)} Y:{int(py)} Z:{int(pz)}",
                        (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0),2)
            cv2.putText(frame,
                        f"Vx:{smoothed_vx:.2f} Vy:{smoothed_vy:.2f} Vz:{smoothed_vz:.2f}",
                        (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)

            # Here you can use smoothed_vx, vy, vz to counteract drone drift

        p0 = good_new.reshape(-1,1,2)

    old_gray = frame_gray.copy()
    frame_idx += 1

    # Refresh features
    if frame_idx % FEATURE_REFRESH_INTERVAL == 0 or p0 is None or len(p0) < 10:
        p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, maxCorners=MAX_CORNERS,
                                     qualityLevel=QUALITY_LEVEL,
                                     minDistance=MIN_DISTANCE,
                                     blockSize=BLOCK_SIZE)

    # Display
    cv2.imshow("Drone Position + Smoothed Velocity", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cv2.destroyAllWindows()
picam2.stop()
