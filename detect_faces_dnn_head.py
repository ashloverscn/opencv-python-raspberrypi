#!/usr/bin/env python3
"""
local_head_detect.py
Detects heads from local webcam:
- Frontal face (DNN)
- Profile face (Haar)
- Upper body (Haar)
Press 'q' to quit.
"""

import cv2
import numpy as np

# ---------- Load detectors ----------
print("OpenCV version:", cv2.__version__)

# DNN frontal face
dnn_proto = "deploy.prototxt"
dnn_model = "res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNetFromCaffe(dnn_proto, dnn_model)

# Haar cascades
profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")
upper_cascade   = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_upperbody.xml")

# ---------- Open webcam ----------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("❌ Cannot open webcam")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    (h, w) = frame.shape[:2]

    # ---------- 1. DNN FRONTAL FACE ----------
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    for i in range(0, detections.shape[2]):
        conf = float(detections[0, 0, i, 2])
        if conf > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Face {conf:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # ---------- 2. PROFILE FACE ----------
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    profiles = profile_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w1, h1) in profiles:
        cv2.rectangle(frame, (x, y), (x + w1, y + h1), (255, 255, 0), 2)
        cv2.putText(frame, "Profile", (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    # ---------- 3. UPPER BODY (HEAD APPROX) ----------
    uppers = upper_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w1, h1) in uppers:
        cv2.rectangle(frame, (x, y), (x + w1, y + h1), (255, 0, 0), 2)
        cv2.putText(frame, "Upper Body / Head", (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # ---------- Show ----------
    cv2.imshow("Local Head Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
