#!/usr/bin/env python3

import cv2
import socket
import struct
import pickle
import numpy as np

HOST = "0.0.0.0"
PORT = 8485

# --- Adjustable settings ---
FRAME_WIDTH  = 320
FRAME_HEIGHT = 240
SENSITIVITY  = 1.9     # motion threshold
STEP         = 11      # grid spacing

# --- Setup socket server ---
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))
server_socket.listen(1)

print(f"Listening on {HOST}:{PORT}...")
conn, addr = server_socket.accept()
print("Connected by", addr)

data = b""
payload_size = struct.calcsize(">L")

# --- Get first frame ---
frame = None
while frame is None:
    while len(data) < payload_size:
        data += conn.recv(4096)

    packed_size = data[:payload_size]
    data = data[payload_size:]
    msg_size = struct.unpack(">L", packed_size)[0]

    while len(data) < msg_size:
        data += conn.recv(4096)

    frame_data = data[:msg_size]
    data = data[msg_size:]

    buffer = pickle.loads(frame_data)
    frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)

if frame is None:
    print("Error: Cannot capture first frame.")
    exit()

# --- Resize & grayscale ---
frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

try:
    while True:
        # --- Receive next frame ---
        while len(data) < payload_size:
            packet = conn.recv(4096)
            if not packet:
                raise ConnectionError("Client disconnected")
            data += packet

        packed_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack(">L", packed_size)[0]

        while len(data) < msg_size:
            packet = conn.recv(4096)
            if not packet:
                raise ConnectionError("Client disconnected")
            data += packet

        frame_data = data[:msg_size]
        data = data[msg_size:]

        buffer = pickle.loads(frame_data)
        frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)

        # --- Force 320x240 ---
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # --- Optical Flow ---
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None,
            0.5, 3, 15, 3, 5, 1.2, 0
        )

        output = frame.copy()
        h, w = gray.shape

        # --- Draw motion vectors ---
        for y in range(0, h, STEP):
            for x in range(0, w, STEP):
                fx, fy = flow[y, x]
                mag = np.sqrt(fx * fx + fy * fy)

                if mag > SENSITIVITY:
                    end_x = int(x + fx)
                    end_y = int(y + fy)

                    end_x = np.clip(end_x, 0, w - 1)
                    end_y = np.clip(end_y, 0, h - 1)

                    cv2.arrowedLine(
                        output,
                        (x, y),
                        (end_x, end_y),
                        (0, 255, 0),
                        1,
                        tipLength=0.3
                    )

                    print(f"Point ({x},{y}) → ({end_x},{end_y})")

        cv2.imshow(
            f"Optical Flow ({FRAME_WIDTH}x{FRAME_HEIGHT}, Sens={SENSITIVITY})",
            output
        )

        prev_gray = gray.copy()

        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

except Exception as e:
    print("Error:", e)

finally:
    conn.close()
    server_socket.close()
    cv2.destroyAllWindows()
