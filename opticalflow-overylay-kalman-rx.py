#!/usr/bin/env python3
import cv2
import numpy as np
import socket
import struct
import pickle

HOST = "0.0.0.0"   # listen on all interfaces
PORT = 8485

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

# Resize to 320x240
frame = cv2.resize(frame, (320, 240))
prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Parameters
step = 16
threshold = 1.0   # lower threshold = more sensitive
h, w = prev_gray.shape

# --- Initialize Kalman filters ---
kalman_filters = {}
for y in range(0, h, step):
    for x in range(0, w, step):
        kf = cv2.KalmanFilter(4, 2)
        kf.measurementMatrix = np.array([[1,0,0,0],
                                         [0,1,0,0]], np.float32)
        kf.transitionMatrix = np.array([[1,0,1,0],
                                        [0,1,0,1],
                                        [0,0,1,0],
                                        [0,0,0,1]], np.float32)
        kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5
        kf.statePre = np.array([[x],[y],[0],[0]], np.float32)
        kalman_filters[(x,y)] = kf

try:
    while True:
        # --- Receive new frame ---
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
        frame = cv2.resize(frame, (320, 240))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Compute optical flow (slightly more sensitive params)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
                                            0.5, 3, 9, 2, 5, 1.1, 0)
        output = frame.copy()

        # Draw smoothed motion vectors
        for y in range(0, h, step):
            for x in range(0, w, step):
                fx, fy = flow[y, x]
                if np.sqrt(fx**2 + fy**2) > threshold:
                    measured = np.array([[np.float32(x + fx)], [np.float32(y + fy)]])
                    kf = kalman_filters[(x,y)]
                    kf.correct(measured)
                    predicted = kf.predict()
                    end_x, end_y = int(predicted[0]), int(predicted[1])

                    cv2.arrowedLine(output, (x, y), (end_x, end_y),
                                    (0, 255, 0), 1, tipLength=0.3)
                    print(f"Point ({x},{y}) → Smoothed ({end_x},{end_y})")

        cv2.imshow("Smoothed Optical Flow (320x240, sensitive)", output)
        prev_gray = gray.copy()

        if cv2.waitKey(1) & 0xFF == 27:
            break

except Exception as e:
    print("Error:", e)

finally:
    conn.close()
    server_socket.close()
    cv2.destroyAllWindows()
