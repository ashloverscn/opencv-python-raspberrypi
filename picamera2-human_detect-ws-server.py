import asyncio
import cv2
import numpy as np
import threading
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from picamera2 import Picamera2
import websockets

# Suppress noisy handshake warnings from browser refreshes
logging.getLogger('websockets').setLevel(logging.ERROR)

# --- Configuration ---
WIDTH, HEIGHT = 640, 480
HTTP_PORT = 8080  # Changed to non-privileged port
WS_PORT = 8001

# Initialize Picamera2
picam = Picamera2()
config = picam.create_preview_configuration(main={"format": "RGB888", "size": (WIDTH, HEIGHT)})
picam.configure(config)
picam.start()

# Initialize Caffe Model
net = cv2.dnn.readNetFromCaffe("mobilenet.prototxt", "mobilenet.caffemodel")
PERSON_CLASS_ID = 15
CONFIDENCE_THRESHOLD = 0.4

# Thread synchronization
connected_clients = set()
main_loop = None

# --- HTML Interface ---
HTML_PAGE = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>High-Speed WebSocket Stream</title>
    <style>
        body {{ font-family: sans-serif; background-color: #1e1e24; color: #fff; margin: 0; display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 100vh; }}
        .container {{ text-align: center; background-color: #2a2a35; padding: 20px; border-radius: 12px; box-shadow: 0 8px 24px rgba(0,0,0,0.3); }}
        h1 {{ margin-top: 0; color: #4caf50; font-size: 1.5rem; }}
        .video-box {{ border: 4px solid #3f3f50; border-radius: 8px; overflow: hidden; background: #000; width: {WIDTH}px; height: {HEIGHT}px; }}
        canvas {{ width: 100%; height: 100%; display: block; }}
    </style>
</head>
<body>
<div class="container">
    <h1>WebSocket Canvas Stream (Port 8080)</h1>
    <div class="video-box">
        <canvas id="player" width="{WIDTH}" height="{HEIGHT}"></canvas>
    </div>
</div>

<script>
    const canvas = document.getElementById('player');
    const ctx = canvas.getContext('2d');
    
    // Connect to WebSocket pipe
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

    ws.onerror = function(err) {{ console.error('WebSocket Error: ', err); }};
</script>
</body>
</html>
"""

# --- Frame Processing Worker Thread ---
def camera_processing_loop():
    global main_loop
    while True:
        try:
            frame = picam.capture_array()
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            h, w = frame_bgr.shape[:2]
            
            # Run Caffe Detection
            blob = cv2.dnn.blobFromImage(cv2.resize(frame_bgr, (300, 300)), 0.007843, (300, 300), 127.5)
            net.setInput(blob)
            detections = net.forward()
            
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > CONFIDENCE_THRESHOLD:
                    if int(detections[0, 0, i, 1]) == PERSON_CLASS_ID:
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")
                        cv2.rectangle(frame_bgr, (startX, startY), (endX, endY), (0, 255, 0), 2)
                        cv2.putText(frame_bgr, "Human", (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Encode frame directly to a fast compressed JPEG payload
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

# --- HTTP Server Asset Handler ---
class HTTPHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode('utf-8'))
    def log_message(self, format, *args):
        return  # Quiet down console logging

def run_http_server():
    server = ThreadingHTTPServer(('0.0.0.0', HTTP_PORT), HTTPHandler)
    server.serve_forever()

async def main():
    global main_loop
    main_loop = asyncio.get_running_loop()
    
    # Run hardware threads
    threading.Thread(target=run_http_server, daemon=True).start()
    threading.Thread(target=camera_processing_loop, daemon=True).start()
    
    print(f"HTTP Interface hosting on http://localhost:{HTTP_PORT}")
    
    async with websockets.serve(ws_handler, '0.0.0.0', WS_PORT):
        print(f"WebSocket core pipeline online on port {WS_PORT}")
        await asyncio.Future()  # run forever

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutting down streaming engine...")
        try:
            picam.stop()
        except:
            pass
