#!/usr/bin/env python3

import cv2
import socket
import struct
import pickle

# TCP server
HOST = "0.0.0.0"
PORT = 8485

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))
server_socket.listen(1)

print(f"Listening on {HOST}:{PORT}...")
conn, addr = server_socket.accept()
print("Connected by", addr)

data = b""
payload_size = struct.calcsize(">L")

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

        # ----- Face detection -----
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Draw rectangles around faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display
        cv2.imshow("Remote Pi Camera - Face Detection", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break

except (KeyboardInterrupt, ConnectionError):
    print("Stopped receiving")
finally:
    conn.close()
    server_socket.close()
    cv2.destroyAllWindows()
