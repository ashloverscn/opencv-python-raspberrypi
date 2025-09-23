#!/usr/bin/env python3

import cv2
import socket
import struct
import pickle
from picamera2 import Picamera2
import time

# Setup camera
camera = Picamera2()
config = camera.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)})
camera.configure(config)
camera.start()
time.sleep(0.1)

# Setup socket (replace with workstation IP)
HOST = "192.168.29.165"  # workstation IP
PORT = 8485

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((HOST, PORT))
conn = client_socket.makefile("wb")

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

except KeyboardInterrupt:
    print("Stopped streaming")
finally:
    camera.stop()
    client_socket.close()
