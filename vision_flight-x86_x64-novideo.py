#!/usr/bin/env python3

import time
import cv2
import numpy as np
from pymavlink import mavutil

# ================= CONFIG =================
WIDTH, HEIGHT = 640, 480
FPS = 15

CAMERA_INDEX = 0   # 0 = default webcam

MAV_PORT = "/dev/ttyUSB0"   # Windows example: "COM5"
MAV_BAUD = 115200

SEND_MODES = ["LOITER", "POSHOLD", "ALTHOLD", "GUIDED"]

FOCAL_PIX = 420.0
LANDMARK_MIN = 15
ALPHA_FLOW = 0.7

# ================= MAVLINK =================
print("[INFO] Connecting to MAVLink...")
try:
    mav = mavutil.mavlink_connection(MAV_PORT, baud=MAV_BAUD)
    mav.wait_heartbeat(timeout=5)
    print("[OK] MAVLink connected")
except Exception as e:
    print("[WARN] MAVLink not available:", e)
    mav = None

current_mode = "UNKNOWN"

# ================= CAMERA =================
print("[INFO] Opening camera...")
cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
cap.set(cv2.CAP_PROP_FPS, FPS)

if not cap.isOpened():
    print("[FATAL] Camera not detected")
    exit(1)

time.sleep(1)

# ================= VISION =================
fast = cv2.FastFeatureDetector_create(
    threshold=20,
    nonmaxSuppression=True
)

lk_params = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS |
              cv2.TERM_CRITERIA_COUNT, 30, 0.01)
)

prev_gray = None
prev_pts = None
last_time = time.time()

# ================= HELPERS =================
def get_flight_mode():
    global current_mode
    if not mav:
        return "NO_MAV"

    msg = mav.recv_match(type="HEARTBEAT", blocking=False)
    if msg:
        current_mode = mavutil.mode_string_v10(msg)
    return current_mode


def send_vision_velocity(vx, vy):
    if not mav:
        return
    mav.mav.vision_speed_estimate_send(
        int(time.time() * 1e6),
        vx,
        vy,
        0.0
    )

# ================= MAIN LOOP =================
print("[INFO] Vision loop started (PC mode)")
while True:
    ret, frame = cap.read()
    if not ret:
        print("[WARN] Camera frame missed")
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    now = time.time()
    dt = now - last_time
    last_time = now

    mode = get_flight_mode()

    if prev_gray is None:
        prev_gray = gray
        prev_pts = None
        continue

    # --- Detect landmarks ---
    if prev_pts is None or len(prev_pts) < LANDMARK_MIN:
        kps = fast.detect(prev_gray)
        prev_pts = np.array(
            [kp.pt for kp in kps],
            dtype=np.float32
        ).reshape(-1, 1, 2)

        print(f"[VISION] Re-detected landmarks: {len(prev_pts)}")
        prev_gray = gray
        continue

    # --- Track landmarks ---
    pts, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, gray, prev_pts, None, **lk_params
    )

    good_new = pts[status == 1]
    good_old = prev_pts[status == 1]

    if len(good_new) < LANDMARK_MIN:
        prev_pts = None
        prev_gray = gray
        print("[VISION] Landmark count low, reset")
        continue

    # --- Pixel flow ---
    flow = good_new - good_old
    mean_flow = np.mean(flow, axis=0)

    px, py = mean_flow
    vx_flow = px / FOCAL_PIX
    vy_flow = py / FOCAL_PIX

    # --- Fuse (placeholder for future IMU fusion) ---
    vx = ALPHA_FLOW * vx_flow + (1 - ALPHA_FLOW) * vx_flow
    vy = ALPHA_FLOW * vy_flow + (1 - ALPHA_FLOW) * vy_flow

    # --- Mode gate ---
    if mode in SEND_MODES:
        send_vision_velocity(vx, vy)
        active = "YES"
    else:
        active = "NO"

    # --- VERBOSE OUTPUT ---
    print(
        f"[MODE:{mode:8}] "
        f"LM:{len(good_new):3d} "
        f"Vx:{vx:+.3f} Vy:{vy:+.3f} "
        f"VISION_ACTIVE:{active}"
    )

    prev_gray = gray
    prev_pts = good_new.reshape(-1, 1, 2)

    time.sleep(1.0 / FPS)
    
