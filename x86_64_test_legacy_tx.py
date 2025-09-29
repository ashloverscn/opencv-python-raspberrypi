#!/usr/bin/env python3

import cv2
import socket
import struct
import pickle
import time

# Workstation IP + Port
HOST = "localhost"  
PORT = 8488

# Setup webcam (device 0)
camera = cv2.VideoCapture(0)  # /dev/video0
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

def connect_to_server():
    """Try to connect until success."""
    while True:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((HOST, PORT))
            print(f"[OK] Connected to {HOST}:{PORT}")
            return sock
        except Exception as e:
            print(f"[WAIT] Server not available, retrying... ({e})")
            time.sleep(2)  # wait before retry

try:
    while True:
        # connect/reconnect
        client_socket = connect_to_server()
        try:
            while True:
                ret, frame = camera.read()
                if not ret:
                    print("[ERROR] Failed to grab frame from webcam")
                    break

                # Encode frame as JPEG
                _, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])

                # Serialize
                data = pickle.dumps(buffer)
                size = len(data)

                # Send size + data
                client_socket.sendall(struct.pack(">L", size) + data)

        except Exception as e:
            print(f"[DISCONNECTED] {e}, reconnecting...")
            client_socket.close()
            time.sleep(1)

except KeyboardInterrupt:
    print("Stopped streaming")
finally:
    camera.release()
