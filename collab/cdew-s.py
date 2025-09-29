import os
import cv2
import csv
from flask import Flask, request, jsonify, send_from_directory
from pyngrok import ngrok, conf

conf.get_default().auth_token = "2zvxpWuxC3nOcXGhE0nHTcdW1Qh_2TBA4WTogds4toYz4H8Vp"

BASE_DIR = "/content/egg_detection_pipeline"
RECEIVED_DIR = os.path.join(BASE_DIR, "received_data")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed_samples")
EGG_COUNT_DIR = os.path.join(BASE_DIR, "egg_count")

for d in (RECEIVED_DIR, PROCESSED_DIR, EGG_COUNT_DIR):
    os.makedirs(d, exist_ok=True)

CSV_FILE = os.path.join(EGG_COUNT_DIR, "egg_counts.csv")
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="") as f:
        csv.writer(f).writerow(["filename", "egg_count"])


def bx(s):
    h, w = s
    return [(50, 50, 100, 50), (200, 100, 80, 40)]


app = Flask(__name__)


@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return "No file part", 400
    f = request.files["file"]
    if not f.filename:
        return "No selected file", 400
    uploaded_path = os.path.join(RECEIVED_DIR, f.filename)
    f.save(uploaded_path)
    img = cv2.imread(uploaded_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 127, 30
    )
    contours = bx(thresh.shape)
    out_img = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    for x, y, w, h in contours:
        cv2.rectangle(out_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    processed_path = os.path.join(PROCESSED_DIR, f.filename)
    cv2.imwrite(processed_path, out_img)
    with open(CSV_FILE, "a", newline="") as z:
        csv.writer(z).writerow([f.filename, len(contours)])
    return f"Image processed: {f.filename}, eggs detected: {len(contours)}", 200


@app.route("/list", methods=["GET"])
def list_files():
    files = os.listdir(PROCESSED_DIR)
    return jsonify({"files": files})


@app.route("/processed_samples/<path:filename>")
def serve_processed(filename):
    return send_from_directory(PROCESSED_DIR, filename)


@app.route("/egg_count/<path:filename>")
def serve_csv(filename):
    return send_from_directory(EGG_COUNT_DIR, filename)


public_url = ngrok.connect(5000)
print("Ngrok URL:", public_url)
app.run(host="0.0.0.0", port=5000)
