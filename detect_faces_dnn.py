#!/usr/bin/env python3
"""
detect_faces_dnn.py
Usage:
  python detect_faces_dnn.py --prototxt deploy.prototxt --model res10_300x300_ssd_iter_140000.caffemodel
Press 'q' to quit.
"""
import cv2
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--prototxt", default="deploy.prototxt", help="path to Caffe prototxt")
ap.add_argument("--model", default="res10_300x300_ssd_iter_140000.caffemodel", help="path to Caffe model")
ap.add_argument("--confidence", type=float, default=0.5, help="min probability to filter weak detections")
ap.add_argument("--camera", type=int, default=0, help="camera device index")
args = ap.parse_args()

print("OpenCV version:", cv2.__version__)
net = cv2.dnn.readNetFromCaffe(args.prototxt, args.model)

# --- Optional: prefer GPU if OpenCV built with CUDA / OpenCL support ---
# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
# OR for OpenCL:
# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)

cap = cv2.VideoCapture(args.camera)
if not cap.isOpened():
    raise SystemExit(f"Cannot open camera {args.camera}")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    for i in range(0, detections.shape[2]):
        conf = float(detections[0, 0, i, 2])
        if conf < args.confidence:
            continue
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (x1, y1, x2, y2) = box.astype("int")
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{conf:.2f}"
        y = y1 - 10 if y1 - 10 > 10 else y1 + 10
        cv2.putText(frame, label, (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,0), 2)

    cv2.imshow("DNN Face Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
