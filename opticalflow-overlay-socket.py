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
payload_size = struct.calcsize(">L")  # 4-byte length prefix

prev_gray = None
step = 16  # spacing between sampled points for arrows

try:
    while True:
        # Receive message size
        while len(data) < payload_size:
            packet = conn.recv(4096)
            if not packet:
                break
            data += packet
        if len(data) < payload_size:
            break

        packed_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack(">L", packed_size)[0]

        # Receive frame data
        while len(data) < msg_size:
            data += conn.recv(4096)
        frame_data = data[:msg_size]
        data = data[msg_size:]

        # Deserialize frame
        buffer = pickle.loads(frame_data)
        frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Initialize previous frame
        if prev_gray is None:
            prev_gray = gray
            continue

        # Compute Farneback optical flow
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
                                            0.5, 3, 15, 3, 5, 1.2, 0)

        output = frame.copy()
        h, w = gray.shape

        # Draw arrows and print coordinates
        for y in range(0, h, step):
            for x in range(0, w, step):
                fx, fy = flow[y, x]
                if np.sqrt(fx**2 + fy**2) > 2:  # filter small motions
                    cv2.arrowedLine(output, (x, y), (int(x + fx), int(y + fy)),
                                    (0, 255, 0), 1, tipLength=0.3)
                    print(f"Point ({x},{y}) moved to ({int(x + fx)},{int(y + fy)})")

        cv2.imshow("Optical Flow from Socket", output)
        prev_gray = gray.copy()

        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break

except KeyboardInterrupt:
    print("Stopped receiving")
finally:
    conn.close()
    server_socket.close()
    cv2.destroyAllWindows()
