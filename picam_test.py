#!/usr/bin/env python3
import cv2
from picamera2 import Picamera2

# Initialize Pi Camera
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"format": "RGB888", "size": >
picam2.configure(config)
picam2.start()

while True:
    frame = picam2.capture_array()
    cv2.imshow("PiCamera OpenCV", frame)

    # Exit on ESC key
    if cv2.waitKey(1) & 0xFF == 27:
        break

picam2.stop()
cv2.destroyAllWindows()
