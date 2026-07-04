import asyncio
import cv2
import numpy as np
import threading
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from picamera2 import Picamera2
import websockets

# Suppress background network logging noise
logging.getLogger('websockets').setLevel(logging.ERROR)

# --- Configuration optimized for Pi 3B ---
WIDTH, HEIGHT = 640, 480
HTTP_PORT = 8080  
WS_PORT = 8001

# Initialize Picamera2 Layer
picam = Picamera2()
config = picam.create_preview_configuration(main={"format": "RGB888", "size": (WIDTH, HEIGHT)})
picam.configure(config)
picam.start()

# Load the modern YuNet Face Detector natively inside OpenCV
model_path = "face_detection_yunet.onnx"
detector = cv2.FaceDetectorYN.create(
    model=model_path,
    config="",
    input_size=(WIDTH, HEIGHT),
    score_threshold=0.6,  # Filter out weak false positives
    nms_threshold=0.3      # Suppress overlapping boxes
)

connected_clients = set()
main_loop = None

# --- HTML Canvas UI Layer ---
HTML_PAGE = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OpenCV Pure Face Stream</title>
    <style>
        body {{ font-family: sans-serif; background-color: #0b0c10; color: #c5c6c7; margin: 0; display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 100vh; }}
        .container {{ text-align: center; background-color: #1f2833; padding: 25px; border-radius: 12px; box-shadow: 0 10px 30px rgba(0,0,0,0.7); }}
        h1 {{ margin-top: 0; color: #66fcf1; font-size: 1.5rem; }}
        .video-box {{ border: 4px solid #45a29e; border-radius: 8px; overflow: hidden; background: #000; width: {WIDTH}px; height: {HEIGHT}px; }}
        canvas {{ width: 100%; height: 100%; display: block; }}
    </style>
</head>
<body>
<div class="container">
    <h1>Pure OpenCV Face Detection Engine</h1>
    <div class="video-box">
        <canvas id="player" width="{WIDTH}" height="{HEIGHT}"></canvas>
    </div>
</div>
<script>
    const canvas = document.getElementById('player');
    const ctx = canvas.getContext('2d');
    const ws = new WebSocket('ws://' + window.location.hostname + ':{WS_PORT}');
    ws.onmessage = function(event) {{
        const blob = event.data;
        const img = new Image();
        img.onload = function() {{
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
            URL.revokeObjectURL(img.src);
        }};
        img.src = URL.createObjectURL(blob);
    }};
</script>
</body>
</html>
"""

# --- Core Real-Time Camera Execution Loop ---
def camera_processing_loop():
    global main_loop
    print("[INFO] OpenCV native face detection loop active...")
    while True:
        try:
            # Capture array from camera hardware
            frame_rgb = picam.capture_array()
            
            # Convert to BGR format for OpenCV's internal DNN processor
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            
            # Execute inference directly through OpenCV's internal C++ tracking module
            _, faces = detector.detect(frame_bgr)
            
            if faces is not None:
                for face in faces:
                    # YuNet box parameters: [x, y, width, height]
                    box = face[0:4].astype(int)
                    x, y, w, h = box[0], box[1], box[2], box[3]
                    confidence = face[14] * 100
                    
                    # Draw bounding box and confidence score
                    cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), (241, 252, 102), 2)
                    cv2.putText(frame_bgr, f"Face: {confidence:.0f}%", (x, y - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (241, 252, 102), 2)
            
            # Fast JPEG compression for network streaming
            ret, buffer = cv2.imencode('.jpg', frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
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

class HTTPHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode('utf-8'))
    def log_message(self, format, *args): return

def run_http_server():
    ThreadingHTTPServer(('0.0.0.0', HTTP_PORT), HTTPHandler).serve_forever()

async def main():
    global main_loop
    main_loop = asyncio.get_running_loop()
    
    threading.Thread(target=run_http_server, daemon=True).start()
    threading.Thread(target=camera_processing_loop, daemon=True).start()
    
    print(f"\n[READY] Clean OpenCV Stream Live at: http://localhost:{HTTP_PORT}")
    async with websockets.serve(ws_handler, '0.0.0.0', WS_PORT):
        await asyncio.Future()

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        try:
            picam.stop()
        except:
            pass
