#!/usr/bin/env python3

import time
import cv2
import numpy as np
from picamera2 import Picamera2
from pymavlink import mavutil

# ================= CONFIG =================
WIDTH, HEIGHT = 640, 480
FPS = 15

MAV_PORT = "/dev/ttyUSB0"
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
picam = Picamera2()
picam.configure(
    picam.create_video_configuration(
        main={"format": "RGB888", "size": (WIDTH, HEIGHT)}
    )
)
picam.start()
time.sleep(1)

# ================= VISION =================
fast = cv2.FastFeatureDetector_create(threshold=20, nonmaxSuppression=True)

lk_params = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
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
print("[INFO] Vision loop started")
cv2.namedWindow("Vision Debug", cv2.WINDOW_NORMAL)

while True:
    frame = picam.capture_array()
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    vis = frame.copy()

    now = time.time()
    dt = now - last_time
    last_time = now

    mode = get_flight_mode()

    if prev_gray is None:
        prev_gray = gray
        prev_pts = None
        cv2.imshow("Vision Debug", vis)
        cv2.waitKey(1)
        continue

    # --- Detect landmarks ---
    if prev_pts is None or len(prev_pts) < LANDMARK_MIN:
        kps = fast.detect(prev_gray)
        prev_pts = np.array(
            [kp.pt for kp in kps], dtype=np.float32
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

    # --- Draw landmarks & flow ---
    for (new, old) in zip(good_new, good_old):
        x1, y1 = new
        x0, y0 = old
        cv2.circle(vis, (int(x1), int(y1)), 2, (0, 255, 0), -1)
        cv2.line(vis, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 0), 1)

    # --- Compute mean flow ---
    flow = good_new - good_old
    mean_flow = np.mean(flow, axis=0)
    px, py = mean_flow

    vx_flow = (px / FOCAL_PIX)
    vy_flow = (py / FOCAL_PIX)

    vx = vx_flow
    vy = vy_flow

    # --- Mode gate ---
    if mode in SEND_MODES:
        send_vision_velocity(vx, vy)
        active = "YES"
    else:
        active = "NO"

    # --- Draw mean flow arrow ---
    cx, cy = WIDTH // 2, HEIGHT // 2
    cv2.arrowedLine(
        vis,
        (cx, cy),
        (int(cx + px * 5), int(cy + py * 5)),
        (0, 0, 255),
        2,
        tipLength=0.3,
    )

    # --- HUD overlay ---
    cv2.putText(vis, f"MODE: {mode}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(vis, f"LM: {len(good_new)}", (10, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(vis, f"Vx: {vx:+.3f}  Vy: {vy:+.3f}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(vis, f"VISION ACTIVE: {active}", (10, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 0) if active == "YES" else (0, 0, 255), 2)

    # --- Show window ---
    cv2.imshow("Vision Debug", vis)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

    # --- Terminal verbose ---
    print(
        f"[MODE:{mode:8}] "
        f"LM:{len(good_new):3d} "
        f"Vx:{vx:+.3f} Vy:{vy:+.3f} "
        f"VISION_ACTIVE:{active}"
    )

    prev_gray = gray
    prev_pts = good_new.reshape(-1, 1, 2)

    time.sleep(1.0 / FPS)

cv2.destroyAllWindows()
