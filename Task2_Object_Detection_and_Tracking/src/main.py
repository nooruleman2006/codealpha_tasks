"""
main.py
Entry point: webcam capture -> YOLOv8 detection -> SORT tracking -> display.

Run from project root:
    python main.py

Press 'q' to quit.
"""

import sys
import os
import cv2

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from detector import Detector
from sort import Sort
from utils import draw_tracks, draw_fps, FPSCounter


def main():
    # --- Configuration ---
    MODEL_PATH = "yolov8n.pt"
    CONF_THRESHOLD = 0.4
    MAX_AGE = 15          # frames a track survives without a detection
    MIN_HITS = 3           # consecutive hits before a track is shown
    IOU_THRESHOLD = 0.3    # min overlap to match detection -> track

    # --- Initialize components ---
    detector = Detector(model_path=MODEL_PATH, conf_threshold=CONF_THRESHOLD)
    tracker = Sort(max_age=MAX_AGE, min_hits=MIN_HITS, iou_threshold=IOU_THRESHOLD)
    fps_counter = FPSCounter()

    cap = cv2.VideoCapture(0)  # 0 = default webcam
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Webcam opened. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # 1. Detect objects in current frame
        detections = detector.detect(frame)

        # 2. Update tracker with new detections
        tracks = tracker.update(detections)

        # 3. Draw boxes, IDs, labels
        frame = draw_tracks(frame, tracks, detector.class_names)

        # 4. Draw FPS
        fps = fps_counter.tick()
        frame = draw_fps(frame, fps)

        # 5. Display
        cv2.imshow("Object Detection and Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()