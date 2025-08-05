"""
App to stream video from a client's webcam, process it with YOLOv5 on the server,
and stream the results back to the client.
"""
import argparse
import io
import os
from PIL import Image
import cv2
import numpy as np
import torch
import base64
from flask import Flask, render_template, request, redirect, Response
from flask_socketio import SocketIO, emit

app = Flask(__name__)
# Wrap the app with SocketIO
socketio = SocketIO(app)

# Load YOLOv5 Model
try:
    model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True, force_reload=False)
except Exception as e:
    # If a custom model is needed, load it here.
    # For example: model = torch.hub.load("ultralytics/yolov5", "custom", path="./best_damage.pt", force_reload=True)
    print(f"Error loading model: {e}")
    # Fallback or exit if necessary
    model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True, force_reload=True)


# Set Model Settings
model.eval()
model.conf = 0.6  # confidence threshold (0-1)
model.iou = 0.45  # NMS IoU threshold (0-1)

def process_and_encode_image(data_image):
    """
    Receives a base64 encoded image, decodes it, runs YOLO inference,
    and returns the processed image encoded in base64.
    """
    # Decode the base64 string
    # The string might contain a data URL prefix 'data:image/jpeg;base64,', remove it.
    if "," in data_image:
        header, data = data_image.split(',', 1)
    else:
        data = data_image
    
    img_bytes = base64.b64decode(data)
    img = Image.open(io.BytesIO(img_bytes))

    # Run inference
    results = model(img, size=640)

    # Render results on the image
    results.render()  # updates results.imgs with boxes and labels
    
    # Squeeze the image array to remove single-dimensional entries
    processed_img_arr = np.squeeze(results.render())

    # Convert the processed image (which is in RGB format) to a PIL Image
    processed_img = Image.fromarray(processed_img_arr)
    
    # Encode the processed image to JPEG in a byte buffer
    buffered = io.BytesIO()
    processed_img.save(buffered, format="JPEG")
    
    # Get the base64-encoded string
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    return img_str


@socketio.on('image')
def handle_image(data_image):
    """
    This function is called when a client sends an 'image' event.
    It processes the image and emits a 'response_back' event with the result.
    """
    processed_img_str = process_and_encode_image(data_image)
    # Emit the processed image back to the client
    emit('response_back', processed_img_str)

@app.route('/')
def index():
    """Home page."""
    return render_template('index.html')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask app exposing YOLOv5 models")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()

    # Use socketio.run for development.
    # It will use the best available async server (eventlet or gevent).
    socketio.run(app, host="0.0.0.0", port=args.port)
