#!/usr/bin/env python3
import os
os.environ["PICAMERA2_NO_PREVIEW"] = "1"
from picamera2 import Picamera2
import cv2

picam2 = Picamera2()
picam2.configure(picam2.create_still_configuration({"size": (640, 480)}))
picam2.start()

print("[INFO] Camera started. Press 'q' to quit.")
while True:
    frame = picam2.capture_array()
    if frame is None:
        print("[ERROR] No frame received")
        break
    # convert 4-channel -> 3-channel if necessary
    if frame.ndim == 3 and frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    cv2.imshow("Picamera2 Test 640x480", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

picam2.stop()
cv2.destroyAllWindows()
