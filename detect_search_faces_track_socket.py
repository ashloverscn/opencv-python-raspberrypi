#!/usr/bin/env python3

import cv2
import socket
import struct
import pickle
import numpy as np

# ---------------- CONFIG ----------------
HOST = "0.0.0.0"
PORT = 8485
CONF_THRESHOLD = 0.6
# ---------------------------------------

# Load MobileNet SSD face detector
net = cv2.dnn.readNetFromCaffe(
    "deploy.prototxt",
    "res10_300x300_ssd_iter_140000.caffemodel"
)

tracker = None
tracking = False
bbox = None

# TCP server
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind((HOST, PORT))
server.listen(1)

print(f"[INFO] Listening on {HOST}:{PORT}")
conn, addr = server.accept()
print("[INFO] Connected:", addr)

data = b""
payload_size = struct.calcsize(">L")

def detect_face(frame):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)),
        1.0,
        (300, 300),
        (104.0, 177.0, 123.0)
    )
    net.setInput(blob)
    detections = net.forward()

    best_conf = 0
    best_box = None

    for i in range(detections.shape[2]):
        conf = detections[0, 0, i, 2]
        if conf > CONF_THRESHOLD and conf > best_conf:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            best_box = box.astype(int)
            best_conf = conf

    if best_box is None:
        return None

    x1, y1, x2, y2 = best_box
    return (x1, y1, x2 - x1, y2 - y1)

try:
    while True:
        # ---------- RECEIVE FRAME ----------
        while len(data) < payload_size:
            packet = conn.recv(4096)
            if not packet:
                raise ConnectionError
            data += packet

        packed_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack(">L", packed_size)[0]

        while len(data) < msg_size:
            data += conn.recv(4096)

        frame_data = data[:msg_size]
        data = data[msg_size:]

        # ---------- FIXED DECODE ----------
        payload = pickle.loads(frame_data)

        # Handle bytes OR numpy array safely
        if isinstance(payload, bytes):
            buffer = np.frombuffer(payload, dtype=np.uint8)
        elif isinstance(payload, np.ndarray):
            buffer = payload.astype(np.uint8)
        else:
            continue

        frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if frame is None or frame.size == 0:
            continue

        # ---------- DETECT → TRACK ----------
        if not tracking:
            bbox = detect_face(frame)
            if bbox:
                tracker = cv2.TrackerKCF_create()
                tracker.init(frame, bbox)
                tracking = True
        else:
            ok, bbox = tracker.update(frame)
            if not ok:
                tracking = False
                bbox = None

        # ---------- DRAW ----------
        if tracking and bbox:
            x, y, w, h = map(int, bbox)
            cx, cy = x + w // 2, y + h // 2

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
            cv2.putText(frame, "FACE LOCK",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "SEARCHING",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 255), 2)

        cv2.imshow("Face Tracker (Fixed)", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

except (KeyboardInterrupt, ConnectionError):
    print("[INFO] Stopped")

finally:
    conn.close()
    server.close()
    cv2.destroyAllWindows()
