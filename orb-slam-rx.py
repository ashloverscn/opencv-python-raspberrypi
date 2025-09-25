#!/usr/bin/env python3

import cv2
import socket
import struct
import pickle
import numpy as np

HOST = "0.0.0.0"   # listen on all interfaces
PORT = 8485

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))
server_socket.listen(1)

print(f"Listening on {HOST}:{PORT}...")
conn, addr = server_socket.accept()
print("Connected by", addr)

data = b""
payload_size = struct.calcsize(">L")

# Initialize ORB detector
orb = cv2.ORB_create(nfeatures=2000)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

prev_gray = None
prev_kp = None
prev_des = None

trajectory_size = 600
trajectory = np.zeros((trajectory_size, trajectory_size, 3), dtype=np.uint8)  # blank trajectory image
pos = np.array([trajectory_size // 2, trajectory_size // 2], dtype=np.int32)  # start in center

try:
    while True:
        # Receive message size
        while len(data) < payload_size:
            data += conn.recv(4096)
        packed_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack(">L", packed_size)[0]

        # Receive frame data
        while len(data) < msg_size:
            data += conn.recv(4096)
        frame_data = data[:msg_size]
        data = data[msg_size:]

        # Deserialize
        buffer = pickle.loads(frame_data)
        frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_gray is None:
            prev_gray = gray
            prev_kp, prev_des = orb.detectAndCompute(prev_gray, None)
            cv2.imshow("Remote Pi Camera", frame)
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

            # Update trajectory position
            pos += motion.astype(np.int32)
            pos = np.clip(pos, 0, trajectory_size - 1)

            # Draw trajectory
            cv2.circle(trajectory, tuple(pos), 2, (0, 0, 255), -1)

            # Resize trajectory to match frame height
            traj_resized = cv2.resize(trajectory, (frame.shape[1], frame.shape[0]))
            combined = cv2.hconcat([frame, traj_resized])
            cv2.imshow("Remote Pi Camera + ORB SLAM Trajectory", combined)

        prev_gray = gray
        prev_kp = kp
        prev_des = des

        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break

except KeyboardInterrupt:
    print("Stopped receiving")
finally:
    conn.close()
    server_socket.close()
    cv2.destroyAllWindows()
