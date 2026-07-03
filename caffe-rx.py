#!/usr/bin/env python3

import cv2
import socket
import struct
import pickle
import numpy as np

# --- 1. Load Model via OpenCV DNN Module ---
print("Loading optimized MobileNet SSD model via OpenCV DNN...")
try:
    # Using your exact file names as requested
    net = cv2.dnn.readNetFromCaffe("mobilenet_iter_73000.txt", "mobilenet_iter_73000.caffemodel")
    
    # Configure OpenCV to optimize calculations for CPU
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model files: {e}")
    print("Please ensure 'mobilenet_iter_73000.txt' and 'mobilenet_iter_73000.caffemodel' are in this directory.")
    exit()

# In the Caffe MobileNet-SSD dataset mapping, Class ID 15 is 'person'
PERSON_CLASS_ID = 15
CONFIDENCE_THRESHOLD = 0.45  # Ignore weak predictions

# --- 2. Network Socket Setup ---
HOST = "0.0.0.0"   # Listen on all networks
PORT = 8485

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind((HOST, PORT))
server_socket.listen(1)

print(f"Server is listening for frames on {HOST}:{PORT}...")
conn, addr = server_socket.accept()
print(f"Connected to client at: {addr}")

data = b""
payload_size = struct.calcsize(">L")

try:
    while True:
        # --- 3. Extract Frame Header Size ---
        while len(data) < payload_size:
            packet = conn.recv(4096)
            if not packet: break
            data += packet
        if not data: break
        
        packed_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack(">L", packed_size)[0]

        # --- 4. Extract Image Frame Content ---
        while len(data) < msg_size:
            packet = conn.recv(4096)
            if not packet: break
            data += packet
        if not data: break
        
        frame_data = data[:msg_size]
        data = data[msg_size:]

        # --- 5. Aggressive Backlog Buffer Clearing ---
        # If the inference drops frames, drop the older network data instantly to catch up
        if len(data) > msg_size * 2:
            data = b""
            continue

        # --- 6. Decode Image Frame ---
        buffer = pickle.loads(frame_data)
        frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if frame is None:
            continue

        # --- 7. Fast Inference Pass ---
        h, w = frame.shape[:2]
        
        # Convert image frame into a standardized 300x300 structural blob for the DNN engine
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()

        person_count = 0

        # --- 8. Parse Target Bounding Boxes ---
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            # Keep it if it matches our minimum precision criteria
            if confidence >= CONFIDENCE_THRESHOLD:
                class_id = int(detections[0, 0, i, 1])
                
                # Check explicitly for human match
                if class_id == PERSON_CLASS_ID:
                    person_count += 1
                    
                    # Convert percentages back into frame coordinates
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (left, top, right, bottom) = box.astype("int")

                    label = f"Person: {int(confidence * 100)}%"
                    
                    # Render visual frame identifiers
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.putText(frame, label, (left, max(20, top - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Draw Live Frame Counter Overlay
        cv2.putText(frame, f"People Live Count: {person_count}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # --- 9. Render Stream ---
        cv2.imshow("High-Speed Caffe Inference Feed", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # Hit ESC to terminate cleanly
            break

except KeyboardInterrupt:
    print("\nShutting down stream server.")
finally:
    conn.close()
    server_socket.close()
    cv2.destroyAllWindows()
    print("All stream hooks detached cleanly.")