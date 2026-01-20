#!/usr/bin/env python3

import cv2
import socket
import struct
import pickle
import time
from picamera2 import Picamera2

# Workstation IP + Port
HOST = "192.168.29.165"  
PORT = 8485

# Setup camera
camera = Picamera2()
#config = camera.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)})
config = camera.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)}, raw={"size": (2592, 1944)})
camera.configure(config)
camera.start()
time.sleep(0.1)

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
                frame = camera.capture_array()
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
    camera.stop()
