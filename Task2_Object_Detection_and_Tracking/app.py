"""
app.py
Flask backend for VisionTrack.

Endpoints:
    GET  /                  -> renders index.html
    GET  /video_feed        -> MJPEG stream of processed frames
    GET  /detections        -> JSON list of current tracked objects (for sidebar)
    GET  /stats             -> JSON session stats (FPS, total IDs, class counts)
    POST /start              -> body: {"source": "webcam"|"video"|"image"|"rtsp", "path": "...", "url": "..."}
    POST /stop               -> stop current source
    POST /settings           -> body: {"conf": 0.45, "iou": 0.5, "max_det": 100}
    POST /upload              -> multipart file upload (video or image), returns saved path
"""

import os
import sys
import time
import threading

import cv2
from flask import Flask, render_template, Response, jsonify, request
from werkzeug.utils import secure_filename

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from detector import Detector
from sort import Sort
from utils import draw_tracks, draw_fps, FPSCounter

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join("data", "input")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
ALLOWED_EXT = {"mp4", "avi", "mov", "mkv", "jpg", "jpeg", "png", "bmp"}


# ---------------------------------------------------------------------------
# Shared application state (single global pipeline; fine for a local demo app)
# ---------------------------------------------------------------------------
class PipelineState:
    def __init__(self):
        self.detector = Detector(model_path="yolov8n.pt", conf_threshold=0.45)
        self.tracker = Sort(max_age=15, min_hits=3, iou_threshold=0.3)
        self.fps_counter = FPSCounter()

        self.lock = threading.Lock()
        self.cap = None
        self.source_type = None      # "webcam" | "video" | "image" | "rtsp"
        self.running = False
        self.latest_frame = None     # last processed frame (JPEG bytes)
        self.latest_tracks = []      # list of dicts for /detections
        self.latest_fps = 0.0
        self.total_ids_seen = set()
        self.class_counts = {}

        self.static_image = None     # for image source: hold single processed frame

    def open_source(self, source_type, path=None, url=None):
        self.stop()
        with self.lock:
            self.source_type = source_type

            if source_type == "webcam":
                # CAP_DSHOW avoids silent open failures / long delays on Windows
                self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                if not self.cap.isOpened():
                    self.cap = cv2.VideoCapture(0)  # fallback to default backend
            elif source_type == "video":
                self.cap = cv2.VideoCapture(path)
            elif source_type == "rtsp":
                self.cap = cv2.VideoCapture(url)
            elif source_type == "image":
                self.cap = None
                img = cv2.imread(path)
                if img is None:
                    raise ValueError(f"Could not read image: {path}")
                self.static_image = img
            else:
                raise ValueError(f"Unknown source_type: {source_type}")

            if source_type in ("webcam", "video", "rtsp"):
                if self.cap is None or not self.cap.isOpened():
                    self.running = False
                    raise ValueError(f"Could not open {source_type} source.")

            # reset tracker/state for a fresh session
            self.tracker = Sort(max_age=15, min_hits=3, iou_threshold=self.tracker.iou_threshold)
            self.fps_counter = FPSCounter()
            self.total_ids_seen = set()
            self.class_counts = {}
            self.running = True

    def stop(self):
        with self.lock:
            self.running = False
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            self.static_image = None

    def update_settings(self, conf=None, iou=None, max_det=None):
        with self.lock:
            if conf is not None:
                self.detector.conf_threshold = float(conf)
            if iou is not None:
                self.tracker.iou_threshold = float(iou)
            # max_det is enforced when reading detections (see process_frame)
            if max_det is not None:
                self.max_det = int(max_det)

    def process_frame(self, frame):
        """Run detection + tracking on a single frame, update shared state."""
        detections = self.detector.detect(frame)

        max_det = getattr(self, "max_det", 100)
        if len(detections) > max_det:
            # keep the highest-confidence detections
            order = detections[:, 4].argsort()[::-1][:max_det]
            detections = detections[order]

        tracks = self.tracker.update(detections)
        frame = draw_tracks(frame, tracks, self.detector.class_names)

        fps = self.fps_counter.tick()
        frame = draw_fps(frame, fps)

        # Build JSON-friendly track list + stats
        track_list = []
        class_counts = {}
        conf_sum = 0.0
        for trk in tracks:
            x1, y1, x2, y2, track_id, conf, class_id = trk
            class_name = self.detector.get_class_name(class_id)
            track_list.append({
                "id": int(track_id),
                "class": class_name,
                "confidence": round(float(conf), 2),
                "x": int(x1), "y": int(y1),
                "w": int(x2 - x1), "h": int(y2 - y1),
            })
            self.total_ids_seen.add(int(track_id))
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            conf_sum += conf

        with self.lock:
            self.latest_tracks = track_list
            self.latest_fps = fps
            self.class_counts = class_counts
            self.avg_conf = (conf_sum / len(tracks)) if len(tracks) > 0 else 0.0

        return frame


state = PipelineState()


def gen_frames():
    """Generator yielding MJPEG frames for the /video_feed route."""
    while True:
        if not state.running:
            time.sleep(0.05)
            continue

        if state.source_type == "image":
            if state.static_image is None:
                time.sleep(0.1)
                continue
            frame = state.process_frame(state.static_image.copy())
        else:
            if state.cap is None or not state.cap.isOpened():
                time.sleep(0.05)
                continue
            ret, frame = state.cap.read()
            if not ret:
                # video file ended, or webcam/rtsp dropped a frame
                if state.source_type == "video":
                    state.running = False
                time.sleep(0.05)
                continue
            frame = state.process_frame(frame)

        ok, buffer = cv2.imencode(".jpg", frame)
        if not ok:
            continue

        frame_bytes = buffer.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

        if state.source_type == "image":
            time.sleep(0.1)  # avoid pegging CPU re-running detection on a still image


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/detections")
def detections():
    with state.lock:
        return jsonify(state.latest_tracks)


@app.route("/stats")
def stats():
    with state.lock:
        return jsonify({
            "active": len(state.latest_tracks),
            "total_ids": len(state.total_ids_seen),
            "fps": round(state.latest_fps, 1),
            "avg_conf": round(getattr(state, "avg_conf", 0.0) * 100, 1),
            "class_counts": state.class_counts,
            "running": state.running,
        })


@app.route("/start", methods=["POST"])
def start():
    data = request.get_json(force=True)
    source_type = data.get("source", "webcam")
    path = data.get("path")
    url = data.get("url")

    try:
        state.open_source(source_type, path=path, url=url)
        return jsonify({"ok": True, "source": source_type})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@app.route("/stop", methods=["POST"])
def stop():
    state.stop()
    return jsonify({"ok": True})


@app.route("/settings", methods=["POST"])
def settings():
    data = request.get_json(force=True)
    state.update_settings(
        conf=data.get("conf"),
        iou=data.get("iou"),
        max_det=data.get("max_det"),
    )
    return jsonify({"ok": True})


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"ok": False, "error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"ok": False, "error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"ok": False, "error": "File type not allowed"}), 400

    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(save_path)

    ext = filename.rsplit(".", 1)[1].lower()
    file_kind = "image" if ext in {"jpg", "jpeg", "png", "bmp"} else "video"

    return jsonify({"ok": True, "path": save_path, "kind": file_kind})


if __name__ == "__main__":
    app.run(debug=True, threaded=True, host="0.0.0.0", port=5000)