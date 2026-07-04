import asyncio
import cv2
import numpy as np
import threading
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from picamera2 import Picamera2
import websockets

# Suppress noisy handshake warnings from browser refreshes or dropped sockets
logging.getLogger('websockets').setLevel(logging.ERROR)

# --- Configuration ---
WIDTH, HEIGHT = 640, 480
HTTP_PORT = 8080  
WS_PORT = 8001

# Initialize Picamera2 pipeline
picam = Picamera2()
config = picam.create_preview_configuration(main={"format": "RGB888", "size": (WIDTH, HEIGHT)})
picam.configure(config)
picam.start()

# Initialize OpenCV YuNet Face Detector
# Using the local face_detection_yunet.onnx file found in your repository
model_path = "face_detection_yunet.onnx"
detector = cv2.FaceDetectorYN.create(
    model=model_path,
    config="",
    input_size=(WIDTH, HEIGHT),
    score_threshold=0.6,    # Filter out weak detections
    nms_threshold=0.3,      # Non-maximum suppression threshold
    top_k=500
)

# Thread synchronization variables
connected_clients = set()
main_loop = None

# --- Responsive HTML UI Layout ---
HTML_PAGE = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PiCamera2 Realtime Face Detection</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #121214; color: #e1e1e6; margin: 0; display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 100vh; }}
        .container {{ text-align: center; background-color: #202024; padding: 25px; border-radius: 12px; box-shadow: 0 10px 30px rgba(0,0,0,0.5); }}
        h1 {{ margin-top: 0; color: #00adb5; font-size: 1.6rem; letter-spacing: 0.5px; }}
        .video-box {{ border: 4px solid #29292e; border-radius: 8px; overflow: hidden; background: #000; width: {WIDTH}px; height: {HEIGHT}px; }}
        canvas {{ width: 100%; height: 100%; display: block; }}
        .status {{ margin-top: 12px; font-size: 0.85rem; color: #7c7c8a; }}
    </style>
</head>
<body>
<div class="container">
    <h1>YuNet Face Detection Pipeline</h1>
    <div class="video-box">
        <canvas id="player" width="{WIDTH}" height="{HEIGHT}"></canvas>
    </div>
    <div class="status">WebSocket Connection Protocol: <span id="socket-state" style="color: #ecc94b;">Connecting...</span></div>
</div>

<script>
    const canvas = document.getElementById('player');
    const ctx = canvas.getContext('2d');
    const stateText = document.getElementById('socket-state');
    
    // Create bidirectional binary pipeline over port 8001
    const ws = new WebSocket('ws://' + window.location.hostname + ':{WS_PORT}');
    
    ws.onopen = function() {{
        stateText.innerText = "Online (Active)";
        stateText.style.color = "#04d361";
    }};

    ws.onmessage = function(event) {{
        const blob = event.data;
        const img = new Image();
        img.onload = function() {{
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
            URL.revokeObjectURL(img.src); // Essential memory leak cleanup
        }};
        img.src = URL.createObjectURL(blob);
    }};

    ws.onclose = function() {{
        stateText.innerText = "Disconnected";
        stateText.style.color = "#f75a68";
    }};

    ws.onerror = function(err) {{ console.error('WebSocket Error: ', err); }};
</script>
</body>
</html>
"""

# --- Video Processing Engine (Worker Thread) ---
def camera_processing_loop():
    global main_loop
    while True:
        try:
            # Capture the layout array directly from hardware interface
            frame = picam.capture_array()
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # YuNet requires explicitly setting input size if layout dynamic variables change
            detector.setInputSize((WIDTH, HEIGHT))
            
            # Run inference pass
            _, faces = detector.detect(frame_bgr)
            
            # Draw bounding boxes around detected faces
            if faces is not None:
                for face in faces:
                    # Face box dimensions are stored in the first 4 array slots
                    box = face[0:4].astype(int)
                    x, y, w, h = box[0], box[1], box[2], box[3]
                    
                    # Draw a crisp cyan bounding box
                    cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), (245, 173, 0), 2)
                    cv2.putText(frame_bgr, "Face", (x, y - 8), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (245, 173, 0), 2)
            
            # Compact encoding pass into a tight compressed JPEG payload
            ret, buffer = cv2.imencode('.jpg', frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
            if ret and len(connected_clients) > 0 and main_loop is not None:
                # Safely ship the raw bytes over to the async network event loop
                asyncio.run_coroutine_threadsafe(broadcast(buffer.tobytes()), main_loop)
                
        except Exception as e:
            # Silently catch frame errors to protect the streaming loop integrity
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

# --- Standard UI Page Server Asset Route ---
class HTTPHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode('utf-8'))
    def log_message(self, format, *args):
        return  # Keep console logs completely clean

def run_http_server():
    server = ThreadingHTTPServer(('0.0.0.0', HTTP_PORT), HTTPHandler)
    server.serve_forever()

async def main():
    global main_loop
    main_loop = asyncio.get_running_loop()
    
    # Spawn thread engines concurrently
    threading.Thread(target=run_http_server, daemon=True).start()
    threading.Thread(target=camera_processing_loop, daemon=True).start()
    
    print(f"HTTP Server dashboard hosting on http://localhost:{HTTP_PORT}")
    
    async with websockets.serve(ws_handler, '0.0.0.0', WS_PORT):
        print(f"WebSocket real-time channel active on port {WS_PORT}")
        await asyncio.Future()  # Keep runtime active indefinitely

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nTearing down live server elements...")
        try:
            picam.stop()
        except:
            pass
