import asyncio
import cv2
import numpy as np
import threading
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from picamera2 import Picamera2
import websockets
import onnxruntime as ort

# Suppress noisy network logging
logging.getLogger('websockets').setLevel(logging.ERROR)

# --- Configuration ---
WIDTH, HEIGHT = 640, 480
HTTP_PORT = 8080  
WS_PORT = 8001
CONFIDENCE_THRESHOLD = 0.45

# Initialize Picamera2 Camera Pipeline
picam = Picamera2()
config = picam.create_preview_configuration(main={"format": "RGB888", "size": (WIDTH, HEIGHT)})
picam.configure(config)
picam.start()

# Load Pre-trained YOLOv8-Face Model via ONNX Runtime CPU Provider
model_path = "yolov8n-face.onnx"
session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name

# Synchronized thread arrays
connected_clients = set()
main_loop = None

# --- HTML Canvas Dashboard ---
HTML_PAGE = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>YOLOv8-Face Streaming Server</title>
    <style>
        body {{ font-family: 'Segoe UI', sans-serif; background-color: #0c0c0e; color: #e1e1e6; margin: 0; display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 100vh; }}
        .container {{ text-align: center; background-color: #16161a; padding: 25px; border-radius: 12px; box-shadow: 0 10px 30px rgba(0,0,0,0.7); }}
        h1 {{ margin-top: 0; color: #a370f7; font-size: 1.6rem; }}
        .video-box {{ border: 4px solid #232329; border-radius: 8px; overflow: hidden; background: #000; width: {WIDTH}px; height: {HEIGHT}px; }}
        canvas {{ width: 100%; height: 100%; display: block; }}
        .status {{ margin-top: 12px; font-size: 0.85rem; color: #7c7c8a; }}
    </style>
</head>
<body>
<div class="container">
    <h1>YOLOv8-Face Pre-Trained Deep Network</h1>
    <div class="video-box">
        <canvas id="player" width="{WIDTH}" height="{HEIGHT}"></canvas>
    </div>
    <div class="status">Socket Connection Link: <span id="socket-state" style="color: #ecc94b;">Connecting...</span></div>
</div>

<script>
    const canvas = document.getElementById('player');
    const ctx = canvas.getContext('2d');
    const stateText = document.getElementById('socket-state');
    
    const ws = new WebSocket('ws://' + window.location.hostname + ':{WS_PORT}');
    
    ws.onopen = function() {{
        stateText.innerText = "Connected & Active";
        stateText.style.color = "#04d361";
    }};

    ws.onmessage = function(event) {{
        const blob = event.data;
        const img = new Image();
        img.onload = function() {{
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
            URL.revokeObjectURL(img.src);
        }};
        img.src = URL.createObjectURL(blob);
    }};

    ws.onclose = function() {{
        stateText.innerText = "Disconnected";
        stateText.style.color = "#f75a68";
    }};
</script>
</body>
</html>
"""

# --- Non-Maximum Suppression (NMS) for clean boxes ---
def nms(boxes, scores, iou_threshold):
    if len(boxes) == 0: return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]
    return keep

# --- Video Processing Engine Execution Loop ---
def camera_processing_loop():
    global main_loop
    print("[INFO] Pre-trained YOLOv8-Face tracking runtime pipeline online...")
    while True:
        try:
            # Capture natively from hardware layer
            frame_rgb = picam.capture_array()
            
            # YOLOv8 expectation: Resized to 640x640, normalized to 0.0-1.0, shape (1, 3, 640, 640)
            input_img = cv2.resize(frame_rgb, (640, 640))
            input_img = input_img.astype(np.float32) / 255.0
            input_img = input_img.transpose(2, 0, 1)
            input_tensor = np.expand_dims(input_img, axis=0)
            
            # Execute Model Inference Pass
            outputs = session.run(None, {input_name: input_tensor})
            predictions = np.squeeze(outputs[0]).T  # Shape converts to [Num_Boxes, Elements]
            
            # Filter output arrays
            boxes, scores = [], []
            for pred in predictions:
                score = pred[4]  # Class score index for Face
                if score > CONFIDENCE_THRESHOLD:
                    # Convert bounding anchors back to real video viewport scale
                    xc, yc, w, h = pred[0], pred[1], pred[2], pred[3]
                    x1 = int((xc - w / 2) * (WIDTH / 640))
                    y1 = int((yc - h / 2) * (HEIGHT / 640))
                    x2 = int((xc + w / 2) * (WIDTH / 640))
                    y2 = int((yc + h / 2) * (HEIGHT / 640))
                    boxes.append([x1, y1, x2, y2])
                    scores.append(score)
                    
            # Apply NMS to strip out overlapping duplicates
            indices = nms(np.array(boxes), np.array(scores), iou_threshold=0.45)
            
            # Format output frame buffer for streaming out
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            for idx in indices:
                x1, y1, x2, y2 = boxes[idx]
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (247, 112, 163), 2)
                cv2.putText(frame_bgr, f"Face: {scores[idx]*100:.0f}%", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (247, 112, 163), 2)
                
            # Compress and pipe payload over to async network loop
            ret, buffer = cv2.imencode('.jpg', frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
            if ret and len(connected_clients) > 0 and main_loop is not None:
                asyncio.run_coroutine_threadsafe(broadcast(buffer.tobytes()), main_loop)
                
        except Exception as e:
            pass

async def broadcast(data):
    if connected_clients:
        await asyncio.gather(*[client.send(data) for client in connected_clients], return_exceptions=True)

async def ws_handler(websocket):
    connected_clients.add(websocket)
    try:
        await websocket.wait_closed()
    finally:
        connected_clients.remove(websocket)

# --- HTTP Asset Route Handler ---
class HTTPHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode('utf-8'))
    def log_message(self, format, *args):
        return

def run_http_server():
    server = ThreadingHTTPServer(('0.0.0.0', HTTP_PORT), HTTPHandler)
    server.serve_forever()

async def main():
    global main_loop
    main_loop = asyncio.get_running_loop()
    
    threading.Thread(target=run_http_server, daemon=True).start()
    threading.Thread(target=camera_processing_loop, daemon=True).start()
    
    print(f"\n[READY] YOLOv8 UI Dashboard Live at: http://localhost:{HTTP_PORT}")
    
    async with websockets.serve(ws_handler, '0.0.0.0', WS_PORT):
        print(f"[READY] WebSocket Frame Pipe Active via Port: {WS_PORT}\n")
        await asyncio.Future()

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nTearing down live YOLO execution pipelines...")
        try:
            picam.stop()
        except:
            pass
