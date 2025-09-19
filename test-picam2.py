#!/usr/bin/env python3
import os
os.environ["PICAMERA2_NO_PREVIEW"] = "1"
from picamera2 import Picamera2
from ultralytics import YOLO
import cv2

# load small model (nano)
model = YOLO("yolov8n.pt")

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration({"size": (640, 480)}))
picam2.start()

print("[INFO] YOLOv8 Detection started. Press 'q' to quit.")
try:
    while True:
        frame = picam2.capture_array()
        if frame is None:
            continue
        # convert 4-channel -> 3-channel if necessary
        if frame.ndim == 3 and frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        results = model(frame)        # inference (may download model first run)
        annotated = results[0].plot()
        cv2.imshow("YOLOv8 Detection", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    picam2.stop()
    cv2.destroyAllWindows()
