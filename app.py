from flask import Flask, Response, render_template, request, url_for
from utils.camera import Camera
from ultralytics import YOLO
import numpy as np
import cv2
import os, time, csv
import re
import requests
import threading

STREAM_URL = 0
STATIC_CAPTURE_DIR = os.path.join("static", "captured_images")
STATIC_CAPTURE_SAMPLES_DIR = os.path.join("static", "captured_samples")
os.makedirs(STATIC_CAPTURE_DIR, exist_ok=True)
os.makedirs(STATIC_CAPTURE_SAMPLES_DIR, exist_ok=True)

DATA_DIR = "data"
QR_CROP_DIR = os.path.join(DATA_DIR, "qr_crops")
RESULTS_DIR = os.path.join(DATA_DIR, "results")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(QR_CROP_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

CSV_FILE = os.path.join(DATA_DIR, "qr_data.csv")
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "timestamp",
                "filename",
                "qr_text",
                "collector",
                "species",
                "location",
                "notes",
                "egg_count",
            ]
        )

COLAB_UPLOAD_URL = "https://8af2dc0ac23d.ngrok-free.app/upload"

scanned_qrs = set()
last_qr_text = None
app = Flask(__name__)
camera = Camera(STREAM_URL)
qr_detector = cv2.QRCodeDetector()
model = YOLO("yolov8n.pt")


def extract_name_from_vcard(qr_text):
    if not qr_text:
        return None
    match = re.search(r"N:([^\n\r]+)", qr_text)
    if match:
        name = match.group(1).strip()
        return re.sub(r"[^\w\-_. ]", "_", name)
    return None


def sample_filename(qr_text):
    name = extract_name_from_vcard(qr_text)
    if name:
        return f"{name}.jpg"
    return f"sample_{time.strftime('%Y%m%d-%H%M%S')}.jpg"


def capture_filename():
    return f"qr_code_{time.strftime('%Y%m%d-%H%M%S')}.jpg"


def generate_frames():
    global last_qr_text, scanned_qrs
    while True:
        frame_bytes = camera.get_frame()
        if frame_bytes is None:
            continue
        img = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)
        data, bbox, _ = qr_detector.detectAndDecode(img)
        if bbox is not None and len(bbox) > 0:
            bbox = np.int32(bbox).reshape(-1, 2)
            for i in range(len(bbox)):
                pt1 = tuple(bbox[i])
                pt2 = tuple(bbox[(i + 1) % len(bbox)])
                cv2.line(img, pt1, pt2, (0, 255, 0), 2)
            if data:
                last_qr_text = data
                cv2.putText(
                    img,
                    data.splitlines()[0],
                    (bbox[0][0], bbox[0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 0, 0),
                    2,
                )
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                x_min, y_min = np.min(bbox, axis=0)
                x_max, y_max = np.max(bbox, axis=0)
                qr_crop = img[y_min:y_max, x_min:x_max]
                crop_filename = f"qr_{timestamp}.jpg"
                cv2.imwrite(os.path.join(QR_CROP_DIR, crop_filename), qr_crop)
                if data not in scanned_qrs:
                    scanned_qrs.add(data)
                    with open(CSV_FILE, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(
                            [timestamp, crop_filename, data, "", "", "", "", ""]
                        )
        _, buffer = cv2.imencode(".jpg", img)
        yield (
            b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        )


@app.route("/")
def index():
    from datetime import datetime

    return render_template("index.html", current_year=datetime.now().year)


@app.route("/video")
def video():
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/last_qr")
def last_qr():
    return last_qr_text or ""


@app.route("/capture", methods=["POST"])
def capture():
    frame_bytes = camera.get_frame()
    if frame_bytes is None:
        return "Capture failed"
    img = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)
    filename = capture_filename()
    filepath = os.path.join(STATIC_CAPTURE_DIR, filename)
    cv2.imwrite(filepath, img)
    try:
        with open(filepath, "rb") as f:
            r = requests.post(COLAB_UPLOAD_URL, files={"file": f})
        if r.status_code == 200:
            return f"Saved & uploaded: {filename}"
        else:
            return f"Upload failed: {r.text}"
    except Exception as e:
        return f"Error uploading: {str(e)}"


@app.route("/capture_image_page", methods=["GET", "POST"])
def capture_image_page():
    image_url = None
    if request.method == "POST":
        frame_bytes = camera.get_frame()
        if frame_bytes:
            img = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)
            filename = sample_filename(last_qr_text)
            filepath = os.path.join(STATIC_CAPTURE_SAMPLES_DIR, filename)
            cv2.imwrite(filepath, img)
            image_url = url_for("static", filename=f"captured_samples/{filename}")
            try:
                with open(filepath, "rb") as f:
                    r = requests.post(COLAB_UPLOAD_URL, files={"file": f})
                if r.status_code != 200:
                    print("Upload failed:", r.text)
            except Exception as e:
                print("Error uploading:", str(e))
            if last_qr_text and last_qr_text not in scanned_qrs:
                scanned_qrs.add(last_qr_text)
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                with open(CSV_FILE, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        [timestamp, filename, last_qr_text, "", "", "", "", ""]
                    )
    return render_template("capture_image.html", image_url=image_url, time=time)


def qr_scanner_listener_windows():
    global last_qr_text, scanned_qrs
    print("Ready for QR scans (Windows Keyboard mode)...")
    while True:
        try:
            qr_data = input().strip()
            if qr_data.startswith("IOLT-") and len(qr_data) == 9:
                last_qr_text = qr_data
                if qr_data not in scanned_qrs:
                    scanned_qrs.add(qr_data)
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    frame_bytes = camera.get_frame()
                    if frame_bytes:
                        img = cv2.imdecode(
                            np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR
                        )
                        cv2.putText(
                            img,
                            qr_data,
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0,
                            (0, 0, 255),
                            2,
                        )
                        filename = f"{qr_data}_{timestamp}.jpg"
                        filepath = os.path.join(STATIC_CAPTURE_DIR, filename)
                        cv2.imwrite(filepath, img)
                        with open(CSV_FILE, "a", newline="") as f:
                            writer = csv.writer(f)
                            writer.writerow(
                                [timestamp, filename, qr_data, "", "", "", "", ""]
                            )
        except Exception as e:
            print("QR listener error:", e)


if __name__ == "__main__":
    threading.Thread(target=qr_scanner_listener_windows, daemon=True).start()
    app.run(host="0.0.0.0", port=5000, debug=True)
