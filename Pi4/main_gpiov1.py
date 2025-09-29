from flask import Flask, Response, render_template, request, url_for
from utils.camera import Camera
from ultralytics import YOLO
import numpy as np
import cv2
import os, time, csv, re

# === Optional Pi GPIO ===
try:
    import RPi.GPIO as GPIO

    GPIO.setmode(GPIO.BCM)
    LED_PIN = 18
    GPIO.setup(LED_PIN, GPIO.OUT)
    USE_GPIO = True
except ImportError:
    USE_GPIO = False
    print("RPi.GPIO not available. Running without hardware control.")

# --------------------------
# CONFIG & FOLDERS
# --------------------------
STREAM_URL = 0  # 0 = default laptop webcam

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

scanned_qrs = set()

# --------------------------
# INIT
# --------------------------
app = Flask(__name__)
camera = Camera(STREAM_URL)
qr_detector = cv2.QRCodeDetector()
last_qr_text = None
model = YOLO("yolov8n.pt")


# --------------------------
# HELPERS
# --------------------------
def extract_name_from_vcard(qr_text):
    if not qr_text:
        return None
    match = re.search(r"N:([^\n\r]+)", qr_text)
    if match:
        name = match.group(1).strip()
        safe_name = re.sub(r"[^\w\-_. ]", "_", name)
        return safe_name
    return None


def sample_filename(qr_text):
    name = extract_name_from_vcard(qr_text)
    if name:
        return f"{name}.jpg"
    return f"sample_{time.strftime('%Y%m%d-%H%M%S')}.jpg"


def capture_filename():
    return f"qr_code_{time.strftime('%Y%m%d-%H%M%S')}.jpg"


def trigger_led():
    if USE_GPIO:
        GPIO.output(LED_PIN, GPIO.HIGH)
        time.sleep(0.3)
        GPIO.output(LED_PIN, GPIO.LOW)


# --------------------------
# STREAMING & QR DETECTION
# --------------------------
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

            if data and data not in scanned_qrs:
                last_qr_text = data
                scanned_qrs.add(data)
                timestamp = time.strftime("%Y%m%d-%H%M%S")

                # Crop QR and save
                x_min, y_min = np.min(bbox, axis=0)
                x_max, y_max = np.max(bbox, axis=0)
                qr_crop = img[y_min:y_max, x_min:x_max]
                crop_filename = f"qr_{timestamp}.jpg"
                cv2.imwrite(os.path.join(QR_CROP_DIR, crop_filename), qr_crop)

                # Save to CSV
                with open(CSV_FILE, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        [timestamp, crop_filename, data, "", "", "", "", ""]
                    )

                # Trigger LED on Pi
                trigger_led()

                cv2.putText(
                    img,
                    data.splitlines()[0],
                    (bbox[0][0], bbox[0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 0, 0),
                    2,
                )

        _, buffer = cv2.imencode(".jpg", img)
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"


# --------------------------
# ROUTES
# --------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video")
def video():
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


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
    return render_template("capture_image.html", image_url=image_url)


@app.route("/capture", methods=["POST"])
def capture():
    frame_bytes = camera.get_frame()
    if frame_bytes is None:
        return "Capture failed: cannot read from camera"

    img = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)
    filename = capture_filename()
    filepath = os.path.join(STATIC_CAPTURE_DIR, filename)
    cv2.imwrite(filepath, img)
    return f"Image saved as {filepath}"


# --------------------------
# MAIN
# --------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
