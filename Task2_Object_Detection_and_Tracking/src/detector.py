"""
detector.py
Wraps YOLOv8 model to perform object detection on individual frames.
"""

from ultralytics import YOLO
import numpy as np


class Detector:
    def __init__(self, model_path="yolov8n.pt", conf_threshold=0.4):
        """
        Initialize the YOLOv8 detector.

        Args:
            model_path (str): Path to YOLO weights. If not found locally,
                               ultralytics auto-downloads it.
            conf_threshold (float): Minimum confidence to keep a detection.
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.class_names = self.model.names  # dict: {0: 'person', 1: 'bicycle', ...}

    def detect(self, frame):
        """
        Run detection on a single frame.

        Args:
            frame (np.ndarray): BGR image (from cv2.VideoCapture).

        Returns:
            np.ndarray: detections of shape (N, 6) ->
                        [x1, y1, x2, y2, confidence, class_id]
                        Empty array if no detections.
        """
        results = self.model(frame, conf=self.conf_threshold, verbose=False)[0]

        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())
            detections.append([x1, y1, x2, y2, conf, cls_id])

        if len(detections) == 0:
            return np.empty((0, 6))

        return np.array(detections)

    def get_class_name(self, class_id):
        """Return human-readable class name for a class id."""
        return self.class_names.get(int(class_id), "unknown")