#!/usr/bin/env python3

import cv2
import numpy as np
import time
from picamera2 import Picamera2

# --- 1. Load Model via OpenCV DNN Module ---
print("Loading optimized MobileNet SSD model...")
try:
    net = cv2.dnn.readNetFromCaffe("mobilenet_iter_73000.txt", "mobilenet_iter_73000.caffemodel")
    
    # Crucial for Pi 3B performance optimization
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model files: {e}")
    print("Ensure 'mobilenet_iter_73000.txt' and 'mobilenet_iter_73000.caffemodel' are in this folder.")
    exit()

# Caffe MobileNet-SSD configuration
PERSON_CLASS_ID = 15
CONFIDENCE_THRESHOLD = 0.40  # 40% confidence threshold

# --- 2. Initialize Picamera2 ---
print("Initializing Picamera2 hardware...")
picam = Picamera2()

# THE FIX: Changed format from 'BGR24' to 'RGB888'
cam_config = picam.create_preview_configuration(main={"format": "RGB888", "size": (320, 240)})
picam.configure(cam_config)

# Start the camera stream capture
picam.start()
print("Camera streaming started.")

# Variables to calculate real-time FPS
prev_frame_time = 0
new_frame_time = 0

try:
    while True:
        # --- 3. Grab Frame via Picamera2 ---
        # Capture an array frame directly in RGB format
        rgb_frame = picam.capture_array()
        
        if rgb_frame is None:
            continue

        # Convert RGB to BGR so OpenCV's imshow and drawing functions display colors correctly
        frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

        h, w = frame.shape[:2]

        # --- 4. Fast Inference Pass ---
        # Convert image frame into a standardized 300x300 structural blob
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()

        person_count = 0

        # --- 5. Parse Target Bounding Boxes ---
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence >= CONFIDENCE_THRESHOLD:
                class_id = int(detections[0, 0, i, 1])
                
                # Filter specifically for human matches
                if class_id == PERSON_CLASS_ID:
                    person_count += 1
                    
                    # Convert coordinates back into pixel dimensions
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (left, top, right, bottom) = box.astype("int")

                    label = f"Person: {int(confidence * 100)}%"
                    
                    # Draw visual boxes on screen
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.putText(frame, label, (left, max(15, top - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # --- 6. Calculate and Display Performance Stats ---
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time) if (new_frame_time - prev_frame_time) > 0 else 0
        prev_frame_time = new_frame_time

        # Render Stats Overlay
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.putText(frame, f"People: {person_count}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # --- 7. Render Window Display ---
        cv2.imshow("Raspberry Pi 3B - Picamera2 Live Detection", frame)
        
        if cv2.waitKey(1) & 0xFF == 27:  # Press 'ESC' key to quit cleanly
            break

except KeyboardInterrupt:
    print("\nStopping script manually.")
finally:
    # Safely close the Picamera2 capture engine and OpenCV windows
    picam.stop()
    cv2.destroyAllWindows()
    print("Camera hardware closed safely.")