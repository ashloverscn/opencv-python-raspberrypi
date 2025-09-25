#!/usr/bin/env python3

import cv2
import numpy as np
from picamera2 import Picamera2

# --- Adjustable settings ---
FRAME_WIDTH  = 320   # set to 320, 640, 800, etc.
FRAME_HEIGHT = 240   # set to 240, 480, 600, etc.

# Initialize Picamera2
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "RGB888"})
picam2.configure(config)
picam2.start()

# ORB setup
orb = cv2.ORB_create(nfeatures=1000)  # fewer features for faster processing
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

prev_gray = None
prev_kp = None
prev_des = None

trajectory_size = 400
trajectory = np.zeros((trajectory_size, trajectory_size, 3), dtype=np.uint8)
pos = np.array([trajectory_size // 2, trajectory_size // 2], dtype=np.int32)

try:
    while True:
        # Capture frame in RGB
        frame_rgb = picam2.capture_array()
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)

        if prev_gray is None:
            prev_gray = gray
            prev_kp, prev_des = orb.detectAndCompute(prev_gray, None)
            cv2.imshow("ORB-SLAM Picamera2", frame_rgb)
            if cv2.waitKey(1) & 0xFF == 27:
                break
            continue

        # ORB keypoints and descriptors
        kp, des = orb.detectAndCompute(gray, None)
        if des is not None and prev_des is not None and len(prev_kp) > 0 and len(kp) > 0:
            matches = bf.match(prev_des, des)
            matches = sorted(matches, key=lambda x: x.distance)

            # Compute simplified motion
            motion = np.array([0.0, 0.0])
            for m in matches[:30]:  # top 30 matches
                motion += np.array(kp[m.trainIdx].pt) - np.array(prev_kp[m.queryIdx].pt)
            motion /= max(len(matches[:30]), 1)

            # Invert Z-axis (vertical motion)
            motion[1] = -motion[1]

            # Update trajectory position
            pos += motion.astype(np.int32)
            pos = np.clip(pos, 0, trajectory_size - 1)
            cv2.circle(trajectory, tuple(pos), 2, (0, 0, 255), -1)

            # Resize trajectory to match frame size
            traj_resized = cv2.resize(trajectory, (FRAME_WIDTH, FRAME_HEIGHT))
            combined = cv2.hconcat([frame_rgb, traj_resized])
            cv2.imshow("ORB-SLAM Picamera2", combined)

        prev_gray = gray
        prev_kp = kp
        prev_des = des

        if cv2.waitKey(1) & 0xFF == 27:
            break

except KeyboardInterrupt:
    print("Exiting...")

finally:
    cv2.destroyAllWindows()
    picam2.stop()
