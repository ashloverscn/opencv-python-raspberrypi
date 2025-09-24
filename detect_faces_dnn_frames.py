#!/usr/bin/env python3

import cv2
import socket
import struct
import pickle
import numpy as np

# TCP server
HOST = "0.0.0.0"
PORT = 8485

# --- Load DNN Caffe face detector ---
prototxt = "deploy.prototxt"
model = "res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNetFromCaffe(prototxt, model)

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

        # ----- Face detection (DNN) -----
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0,
            (300, 300), (104.0, 177.0, 123.0)
        )
        net.setInput(blob)
        detections = net.forward()

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:  # confidence threshold
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{confidence:.2f}"
                y = y1 - 10 if y1 - 10 > 10 else y1 + 10
                cv2.putText(frame, label, (x1, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

        # Display
        cv2.imshow("Remote Pi Camera - DNN Face Detection", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break

except (KeyboardInterrupt, ConnectionError):
    print("Stopped receiving")
finally:
    conn.close()
    server_socket.close()
    cv2.destroyAllWindows()
