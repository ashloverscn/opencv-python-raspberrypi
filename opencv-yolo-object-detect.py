#!/usr/bin/env python3

import cv2
import socket
import struct
import pickle
from ultralytics import YOLO

# Listen on all interfaces
HOST = "0.0.0.0"
PORT = 8485

# Load YOLO model (choose "yolov8n.pt" for lightweight, or yolov8s/m/l)
print("[INFO] Loading YOLO model...")
model = YOLO("yolov8n.pt")  # make sure the weights file is available
print("[OK] Model loaded.")

# Setup server socket
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))
server_socket.listen(1)

print(f"[INFO] Listening on {HOST}:{PORT}...")
conn, addr = server_socket.accept()
print("[INFO] Connected by", addr)

data = b""
payload_size = struct.calcsize(">L")

try:
    while True:
        # Receive message size
        while len(data) < payload_size:
            packet = conn.recv(4096)
            if not packet:
                print("[ERROR] Connection closed by sender.")
                break
            data += packet
        if len(data) < payload_size:
            break

        packed_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack(">L", packed_size)[0]

        # Receive frame data
        while len(data) < msg_size:
            packet = conn.recv(4096)
            if not packet:
                print("[ERROR] Connection closed while receiving frame.")
                break
            data += packet
        if len(data) < msg_size:
            break

        frame_data = data[:msg_size]
        data = data[msg_size:]

        # Deserialize -> buffer -> image
        buffer = pickle.loads(frame_data)
        frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)

        if frame is None:
            print("[ERROR] Failed to decode frame")
            continue

        # Run YOLO inference
        results = model(frame, verbose=False)  # disable console spam

        # Annotated frame
        annotated = results[0].plot()

        # Show frame
        cv2.imshow("YOLO Remote Stream", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\n[INFO] Stopped by user")
finally:
    conn.close()
    server_socket.close()
    cv2.destroyAllWindows()
    print("[INFO] Connection closed, exiting.")
