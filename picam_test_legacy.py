#!/usr/bin/env python3

import cv2
import numpy as np
from picamera2 import Picamera2
import time

# Initialize camera
camera = Picamera2()
camera_config = camera.create_preview_configuration(main={"size": (640, 480), "format": "BGR888"})
camera.configure(camera_config)
camera.start()

time.sleep(0.1)

try:
    while True:
        frame = camera.capture_array()
        cv2.imshow("PiCamera2", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC to quit
            break
finally:
    cv2.destroyAllWindows()
    camera.stop()
