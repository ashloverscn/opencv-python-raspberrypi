#!/usr/bin/env python3

import cv2
import numpy as np

# Open local webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open webcam")
    exit()

# Initialize ORB detector and matcher
orb = cv2.ORB_create(nfeatures=2000)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

prev_gray = None
prev_kp = None
prev_des = None

trajectory_size = 600
trajectory = np.zeros((trajectory_size, trajectory_size, 3), dtype=np.uint8)
pos = np.array([trajectory_size // 2, trajectory_size // 2], dtype=np.int32)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_gray is None:
            prev_gray = gray
            prev_kp, prev_des = orb.detectAndCompute(prev_gray, None)
            cv2.imshow("Webcam ORB SLAM", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
            continue

        # ORB keypoints and descriptors
        kp, des = orb.detectAndCompute(gray, None)
        if des is not None and prev_des is not None and len(prev_kp) > 0 and len(kp) > 0:
            # Match features
            matches = bf.match(prev_des, des)
            matches = sorted(matches, key=lambda x: x.distance)

            # Compute simplified motion
            motion = np.array([0.0, 0.0])
            for m in matches[:50]:  # top 50 matches
                motion += np.array(kp[m.trainIdx].pt) - np.array(prev_kp[m.queryIdx].pt)
            motion /= max(len(matches[:50]), 1)

            # Invert Z-axis (invert vertical displacement)
            motion_inverted = motion * np.array([1, -1])
            pos += motion_inverted.astype(np.int32)
            pos = np.clip(pos, 0, trajectory_size - 1)

            # Draw trajectory
            cv2.circle(trajectory, tuple(pos), 2, (0, 0, 255), -1)

            # Resize trajectory to match frame height
            traj_resized = cv2.resize(trajectory, (frame.shape[1], frame.shape[0]))
            combined = cv2.hconcat([frame, traj_resized])
            cv2.imshow("Webcam ORB SLAM", combined)

        prev_gray = gray
        prev_kp = kp
        prev_des = des

        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break

except KeyboardInterrupt:
    print("Stopped")
finally:
    cap.release()
    cv2.destroyAllWindows()
