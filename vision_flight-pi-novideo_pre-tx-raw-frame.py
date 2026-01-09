#!/usr/bin/env python3

import time
import cv2
import numpy as np
import socket
import struct
import pickle
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

# TCP streaming config
HOST = "192.168.29.165"  # workstation IP
PORT = 8485

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
    picam.create_preview_configuration(
        main={"format": "RGB888", "size": (WIDTH, HEIGHT)}
    )
)
picam.start()
time.sleep(0.1)

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

# ================= TCP STREAMING =================
def connect_to_server():
    while True:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((HOST, PORT))
            print(f"[OK] Connected to {HOST}:{PORT}")
            return sock
        except Exception as e:
            print(f"[WAIT] Server not available, retrying... ({e})")
            time.sleep(2)

client_socket = None

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

try:
    while True:
        # connect/reconnect TCP if needed
        if client_socket is None:
            client_socket = connect_to_server()

        frame = picam.capture_array()
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # --- Send frame over TCP ---
        try:
            _, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            data = pickle.dumps(buffer)
            size = len(data)
            client_socket.sendall(struct.pack(">L", size) + data)
        except Exception as e:
            print(f"[DISCONNECTED] TCP streaming lost: {e}")
            client_socket.close()
            client_socket = None

        now = time.time()
        dt = now - last_time
        last_time = now

        mode = get_flight_mode()

        if prev_gray is None:
            prev_gray = gray
            prev_pts = None
            continue

        # --- Detect landmarks if needed ---
        if prev_pts is None or len(prev_pts) < LANDMARK_MIN:
            kps = fast.detect(prev_gray)
            prev_pts = np.array([kp.pt for kp in kps], dtype=np.float32).reshape(-1, 1, 2)
            print(f"[VISION] Re-detected landmarks: {len(prev_pts)}")
            prev_gray = gray
            continue

        # --- Track landmarks ---
        pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None, **lk_params)
        good_new = pts[status == 1]
        good_old = prev_pts[status == 1]

        if len(good_new) < LANDMARK_MIN:
            prev_pts = None
            prev_gray = gray
            print("[VISION] Landmark count low, reset")
            continue

        # --- Compute pixel motion ---
        flow = good_new - good_old
        mean_flow = np.mean(flow, axis=0)
        px, py = mean_flow
        vx_flow = (px / FOCAL_PIX) * 1.0
        vy_flow = (py / FOCAL_PIX) * 1.0

        # --- Landmark velocity ---
        vx_land = vx_flow
        vy_land = vy_flow

        # --- Fuse ---
        alpha = ALPHA_FLOW
        vx = alpha * vx_flow + (1 - alpha) * vx_land
        vy = alpha * vy_flow + (1 - alpha) * vy_land

        # --- Mode gate ---
        if mode in SEND_MODES:
            send_vision_velocity(vx, vy)
            active = "YES"
        else:
            active = "NO"

        # --- VERBOSE OUTPUT ---
        print(f"[MODE:{mode:8}] LM:{len(good_new):3d} Vx:{vx:+.3f} Vy:{vy:+.3f} VISION_ACTIVE:{active}")

        prev_gray = gray
        prev_pts = good_new.reshape(-1, 1, 2)

        time.sleep(1.0 / FPS)

except KeyboardInterrupt:
    print("Stopped streaming")

finally:
    if client_socket:
        client_socket.close()
    picam.stop()

