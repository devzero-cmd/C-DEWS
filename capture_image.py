from flask import Blueprint, render_template, request, redirect, url_for
import urllib.request
import numpy as np
import cv2
import os, time

# Make a blueprint to integrate into main Flask app
capture_bp = Blueprint("capture_bp", __name__)

# Directories (reuse the ones in your main app)
CAPTURE_DIR = "captured_images"
os.makedirs(CAPTURE_DIR, exist_ok=True)

# Snapshot endpoint (your camera IP)
SHOT_URL = "http://192.168.1.16:8080/shot.jpg"
STREAM_URL = "http://192.168.1.16:8080/video"


@capture_bp.route("/take_picture", methods=["GET", "POST"])
def take_picture():
    if request.method == "POST":
        try:
            # Fetch image from camera
            img_resp = urllib.request.urlopen(SHOT_URL)
            img_np = np.array(bytearray(img_resp.read()), dtype=np.uint8)
            frame = cv2.imdecode(img_np, -1)

            # Save with timestamp
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = os.path.join(CAPTURE_DIR, f"capture_{timestamp}.jpg")
            cv2.imwrite(filename, frame)

            return f"Image captured and saved as {filename}"

        except Exception as e:
            return f"Capture failed: {e}"

    # Render a simple form for capture
    return render_template("capture_image.html")
