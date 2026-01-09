#!/usr/bin/env python3

import socket
import struct
import pickle
import time
import cv2
import numpy as np
from pymavlink import mavutil

# ================= CONFIG =================
VIDEO_PORT = 8485
LISTEN_IP = "0.0.0.0"

WIDTH, HEIGHT = 640, 480
FPS = 15

MAV_PORT = "/dev/ttyUSB0"
MAV_BAUD = 115200

SEND_MODES = ["LOITER", "POSHOLD", "ALTHOLD", "GUIDED"]

FOCAL_PIX = 420.0
LANDMARK_MIN = 15

WINDOW_NAME = "Vision Overlay"

# ================= MAVLINK =================
print("[INFO] Connecting MAVLink...")
try:
    mav = mavutil.mavlink_connection(MAV_PORT, baud=MAV_BAUD)
    mav.wait_heartbeat(timeout=5)
    print("[OK] MAVLink connected")
except Exception as e:
    print("[WARN] MAVLink not available:", e)
    mav = None

current_mode = "UNKNOWN"

# ================= TCP SERVER =================
def wait_for_client():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((LISTEN_IP, VIDEO_PORT))
    server.listen(1)
    print(f"[WAIT] Listening for video on {VIDEO_PORT}")
    conn, addr = server.accept()
    print(f"[OK] Video client connected: {addr}")
    return conn

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


def recvall(sock, size):
    data = b""
    while len(data) < size:
        packet = sock.recv(size - len(data))
        if not packet:
            return None
        data += packet
    return data


def draw_overlay(frame, old_pts, new_pts, vx, vy, mode, active):
    # Draw landmarks and flow vectors
    for o, n in zip(old_pts, new_pts):
        ox, oy = o.ravel().astype(int)
        nx, ny = n.ravel().astype(int)

        cv2.circle(frame, (nx, ny), 2, (0, 255, 0), -1)
        cv2.arrowedLine(
            frame,
            (ox, oy),
            (nx, ny),
            (255, 255, 0),
            1,
            tipLength=0.3
        )

    # HUD background
    cv2.rectangle(frame, (5, 5), (330, 110), (0, 0, 0), -1)

    # HUD text
    cv2.putText(frame, f"MODE: {mode}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

    cv2.putText(frame, f"LM: {len(new_pts)}", (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

    cv2.putText(frame, f"Vx: {vx:+.3f}  Vy: {vy:+.3f}", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.putText(frame, f"VISION ACTIVE: {active}", (10, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 0) if active == "YES" else (0, 0, 255), 1)


# ================= MAIN =================
print("[INFO] Vision RX started")

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

while True:
    conn = wait_for_client()

    try:
        while True:
            raw_size = recvall(conn, 4)
            if not raw_size:
                raise ConnectionError("Video stream lost")

            frame_size = struct.unpack(">L", raw_size)[0]
            frame_data = recvall(conn, frame_size)
            if frame_data is None:
                raise ConnectionError("Frame receive failed")

            buffer = pickle.loads(frame_data)
            frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            now = time.time()
            dt = now - last_time
            last_time = now

            mode = get_flight_mode()

            if prev_gray is None:
                prev_gray = gray
                prev_pts = None
                cv2.imshow(WINDOW_NAME, frame)
                cv2.waitKey(1)
                continue

            if prev_pts is None or len(prev_pts) < LANDMARK_MIN:
                kps = fast.detect(prev_gray)
                prev_pts = np.array([kp.pt for kp in kps],
                                    dtype=np.float32).reshape(-1, 1, 2)
                prev_gray = gray
                cv2.imshow(WINDOW_NAME, frame)
                cv2.waitKey(1)
                continue

            pts, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray, gray, prev_pts, None, **lk_params
            )

            good_new = pts[status == 1]
            good_old = prev_pts[status == 1]

            if len(good_new) < LANDMARK_MIN:
                prev_pts = None
                prev_gray = gray
                cv2.imshow(WINDOW_NAME, frame)
                cv2.waitKey(1)
                continue

            flow = good_new - good_old
            mean_flow = np.mean(flow, axis=0)

            px, py = mean_flow
            vx = px / FOCAL_PIX
            vy = py / FOCAL_PIX

            if mode in SEND_MODES:
                send_vision_velocity(vx, vy)
                active = "YES"
            else:
                active = "NO"

            print(
                f"[MODE:{mode:8}] "
                f"LM:{len(good_new):3d} "
                f"Vx:{vx:+.3f} Vy:{vy:+.3f} "
                f"VISION_ACTIVE:{active}"
            )

            draw_overlay(frame, good_old, good_new, vx, vy, mode, active)

            cv2.imshow(WINDOW_NAME, frame)
            if cv2.waitKey(1) & 0xFF == 27:
                raise KeyboardInterrupt

            prev_gray = gray
            prev_pts = good_new.reshape(-1, 1, 2)
            time.sleep(1.0 / FPS)

    except Exception as e:
        print(f"[DISCONNECTED] {e}")
        conn.close()
        prev_gray = None
        prev_pts = None
        time.sleep(1)
