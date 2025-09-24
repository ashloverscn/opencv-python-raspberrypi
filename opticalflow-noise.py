import cv2
import numpy as np

# Initialize webcam
cap = cv2.VideoCapture(0)  # 0 is usually the default webcam

# Check if webcam opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Read the first frame
ret, frame1 = cap.read()
if not ret:
    print("Error: Cannot read first frame.")
    exit()

# Convert to grayscale
prev_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

# Create mask for visualization
mask = np.zeros_like(frame1)
mask[..., 1] = 255  # Set saturation to maximum for HSV

while True:
    ret, frame2 = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow using Farneback method
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray,
                                        None,
                                        0.5, 3, 15, 3, 5, 1.2, 0)

    # Convert flow to polar coordinates (magnitude and angle)
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Set hue according to the angle of optical flow
    mask[..., 0] = angle * 180 / np.pi / 2

    # Set value according to the normalized magnitude
    mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

    # Convert HSV to BGR for visualization
    rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)

    # Show the optical flow
    cv2.imshow('Optical Flow', rgb)

    # Update previous frame
    prev_gray = gray.copy()

    # Press ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
