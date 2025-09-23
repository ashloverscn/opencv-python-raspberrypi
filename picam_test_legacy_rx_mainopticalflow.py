#!/usr/bin/env python3

import cv2
import socket
import struct
import pickle
import numpy as np

# TCP server
HOST = "0.0.0.0"
PORT = 8485

# Lucas-Kanade optical flow parameters
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Feature detection parameters
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

# TCP setup
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))
server_socket.listen(1)
print(f"Listening on {HOST}:{PORT}...")
conn, addr = server_socket.accept()
print("Connected by", addr)

data = b""
payload_size = struct.calcsize(">L")

# Initialize tracking variables
old_gray = None
p0 = None
mask = None  # for drawing

try:
    while True:
        # Receive message size
        while len(data) < payload_size:
            packet = conn.recv(4096)
            if not packet:
                raise ConnectionError("Client disconnected")
            data += packet

        packed_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack(">L", packed_size)[0]

        # Receive frame data
        while len(data) < msg_size:
            packet = conn.recv(4096)
            if not packet:
                raise ConnectionError("Client disconnected")
            data += packet

        frame_data = data[:msg_size]
        data = data[msg_size:]

        # Deserialize frame
        buffer = pickle.loads(frame_data)
        frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Initialize tracking
        if old_gray is None:
            old_gray = gray.copy()
            p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
            mask = np.zeros_like(frame)
            continue

        # Calculate optical flow
        if p0 is not None and len(p0) > 0:
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, gray, p0, None, **lk_params)
            if p1 is not None:
                good_new = p1[st == 1]
                good_old = p0[st == 1]

                # Draw tracks
                for new, old in zip(good_new, good_old):
                    a, b = new.ravel()
                    c, d = old.ravel()
                    a, b, c, d = int(a), int(b), int(c), int(d)  # cast to int
                    mask = cv2.line(mask, (a, b), (c, d), (0, 255, 0), 2)
                    frame = cv2.circle(frame, (a, b), 5, (0, 0, 255), -1)

                frame = cv2.add(frame, mask)

                # Update points
                p0 = good_new.reshape(-1, 1, 2)
                old_gray = gray.copy()
            else:
                # Re-initialize if lost points
                p0 = cv2.goodFeaturesToTrack(gray, mask=None, **feature_params)
                mask = np.zeros_like(frame)

        # Display
        cv2.imshow("Remote Pi Camera - Optical Flow", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break

except (KeyboardInterrupt, ConnectionError):
    print("Stopped receiving")
finally:
    conn.close()
    server_socket.close()
    cv2.destroyAllWindows()
