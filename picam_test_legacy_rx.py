#!/usr/bin/env python3

import cv2
import socket
import struct
import pickle

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

        cv2.imshow("Remote Pi Camera", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break

except KeyboardInterrupt:
    print("Stopped receiving")
finally:
    conn.close()
    server_socket.close()
    cv2.destroyAllWindows()
