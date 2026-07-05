"""
utils.py
Helper functions for visualization: consistent colors per track ID,
drawing boxes/labels, and an FPS counter.
"""

import cv2
import time
import numpy as np


def get_color(track_id):
    """
    Generate a consistent BGR color for a given track ID.
    Same ID always produces the same color (deterministic hash).
    """
    np.random.seed(int(track_id) * 37 % (2**32 - 1))
    color = tuple(int(c) for c in np.random.randint(50, 255, size=3))
    return color


def draw_tracks(frame, tracks, class_names):
    """
    Draw bounding boxes, track IDs, and class labels on the frame.

    Args:
        frame: the image to draw on (modified in place)
        tracks: np.ndarray of shape (N, 7) ->
                [x1, y1, x2, y2, track_id, conf, class_id]
        class_names: dict mapping class_id -> class name string

    Returns:
        frame with drawings applied
    """
    for trk in tracks:
        x1, y1, x2, y2, track_id, conf, class_id = trk
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        track_id = int(track_id)

        color = get_color(track_id)
        label = f"ID {track_id} {class_names.get(int(class_id), 'obj')} {conf:.2f}"

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Label background for readability
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(frame, (x1, y1 - text_h - 10), (x1 + text_w + 6, y1), color, -1)
        cv2.putText(frame, label, (x1 + 3, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return frame


class FPSCounter:
    """
    Simple FPS counter using a rolling average over the last N frames.
    """

    def __init__(self, avg_frames=30):
        self.avg_frames = avg_frames
        self.timestamps = []

    def tick(self):
        """Call once per frame. Returns the current estimated FPS."""
        now = time.time()
        self.timestamps.append(now)
        if len(self.timestamps) > self.avg_frames:
            self.timestamps.pop(0)

        if len(self.timestamps) < 2:
            return 0.0

        elapsed = self.timestamps[-1] - self.timestamps[0]
        if elapsed <= 0:
            return 0.0
        return (len(self.timestamps) - 1) / elapsed


def draw_fps(frame, fps):
    """Draw FPS counter text on the top-left of the frame."""
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return frame
