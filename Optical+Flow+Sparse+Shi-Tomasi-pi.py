import cv2
import numpy as np
from picamera2 import Picamera2

# --- Adjustable settings ---
FRAME_WIDTH  = 320   # set to 320, 640, etc. (smaller = faster)
FRAME_HEIGHT = 240
MAX_FEATURES = 200   # number of features to track
QUALITY_LEVEL = 0.3  # quality for feature detection
MIN_DISTANCE  = 7    # minimum distance between features

# Initialize PiCamera2
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (FRAME_WIDTH, FRAME_HEIGHT)})
picam2.configure(config)
picam2.start()

# Params for Shi-Tomasi corner detection
feature_params = dict(maxCorners=MAX_FEATURES,
                      qualityLevel=QUALITY_LEVEL,
                      minDistance=MIN_DISTANCE,
                      blockSize=7)

# Params for Lucas-Kanade optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Get first frame
frame0 = picam2.capture_array()
old_gray = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# Create random colors for drawing
color = np.random.randint(0, 255, (MAX_FEATURES, 3))

# Mask for drawing lines
mask = np.zeros_like(frame0)

while True:
    frame = picam2.capture_array()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow
    if p0 is not None:
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        if p1 is not None:
            good_new = p1[st == 1]
            good_old = p0[st == 1]

            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
                frame = cv2.circle(frame, (int(a), int(b)), 3, color[i].tolist(), -1)

            img = cv2.add(frame, mask)
        else:
            img = frame

        # Update previous frame and points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2) if len(good_new) > 0 else None
    else:
        # Re-detect features if lost
        p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)
        old_gray = frame_gray.copy()
        mask = np.zeros_like(frame)

        img = frame

    # Show result
    cv2.imshow("Optical Flow - Lucas Kanade", img)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
picam2.stop()
