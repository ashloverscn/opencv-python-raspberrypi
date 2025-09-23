#!/usr/bin/env python3

import cv2
import numpy as np
from picamera2 import Picamera2
import time

# Initialize camera
camera = Picamera2()
#config = camera.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"})
config = picam2.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)})
camera.configure(config)
camera.start()

time.sleep(0.1)

try:
    while True:
        frame = camera.capture_array()
        # Convert RGB → BGR for OpenCV display
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        cv2.imshow("PiCamera2", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC to quit
            break
finally:
    camera.stop()
    cv2.destroyAllWindows()
