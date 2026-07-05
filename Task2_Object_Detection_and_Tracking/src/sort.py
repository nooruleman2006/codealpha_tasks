"""
sort.py
SORT (Simple Online and Realtime Tracking).

Maintains a set of KalmanBoxTracker objects (one per tracked entity).
Each frame:
    1. Predict new positions for all existing trackers.
    2. Match new detections to existing trackers using IoU + Hungarian algorithm.
    3. Update matched trackers, create new trackers for unmatched detections,
       and remove trackers that haven't been matched for too long.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from kalman_filter import KalmanBoxTracker


def iou(bbox1, bbox2):
    """
    Compute Intersection-over-Union between two boxes [x1, y1, x2, y2].
    """
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, bbox1[2] - bbox1[0]) * max(0.0, bbox1[3] - bbox1[1])
    area2 = max(0.0, bbox2[2] - bbox2[0]) * max(0.0, bbox2[3] - bbox2[1])

    union = area1 + area2 - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """
    Match detections to existing trackers based on IoU, using the
    Hungarian algorithm to find the optimal assignment.

    Args:
        detections: list/array of [x1, y1, x2, y2, conf, class_id]
        trackers: list of predicted bboxes [x1, y1, x2, y2] (one per tracker)
        iou_threshold: minimum IoU to consider a match valid

    Returns:
        matches: list of (detection_idx, tracker_idx)
        unmatched_detections: list of detection indices
        unmatched_trackers: list of tracker indices
    """
    if len(trackers) == 0:
        return [], list(range(len(detections))), []

    if len(detections) == 0:
        return [], [], list(range(len(trackers)))

    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            iou_matrix[d, t] = iou(det[:4], trk)

    # Hungarian algorithm minimizes cost, so we negate IoU (maximize IoU)
    row_idx, col_idx = linear_sum_assignment(-iou_matrix)

    matches = []
    unmatched_detections = []
    unmatched_trackers = []

    matched_det_set = set()
    matched_trk_set = set()

    for d, t in zip(row_idx, col_idx):
        if iou_matrix[d, t] >= iou_threshold:
            matches.append((d, t))
            matched_det_set.add(d)
            matched_trk_set.add(t)

    for d in range(len(detections)):
        if d not in matched_det_set:
            unmatched_detections.append(d)

    for t in range(len(trackers)):
        if t not in matched_trk_set:
            unmatched_trackers.append(t)

    return matches, unmatched_detections, unmatched_trackers


class Sort:
    """
    Main SORT tracker. Call update() once per frame with new detections.
    """

    def __init__(self, max_age=15, min_hits=3, iou_threshold=0.3):
        """
        Args:
            max_age: max frames a tracker survives without a matched detection
            min_hits: min consecutive hits before a tracker is reported
                      (avoids flickering IDs on noisy single-frame detections)
            iou_threshold: min IoU for a valid detection-tracker match
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, detections):
        """
        Update trackers with new detections for the current frame.

        Args:
            detections: np.ndarray of shape (N, 6) ->
                        [x1, y1, x2, y2, conf, class_id]

        Returns:
            np.ndarray of shape (M, 7) ->
                        [x1, y1, x2, y2, track_id, conf, class_id]
                        for all currently confirmed tracks.
        """
        self.frame_count += 1

        # Step 1: predict new locations for existing trackers
        predicted_boxes = []
        valid_trackers = []
        for trk in self.trackers:
            pred_box = trk.predict()
            if np.any(np.isnan(pred_box)):
                continue
            predicted_boxes.append(pred_box)
            valid_trackers.append(trk)
        self.trackers = valid_trackers

        # Step 2: match detections to predicted tracker boxes
        matches, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            detections, predicted_boxes, self.iou_threshold
        )

        # Step 3: update matched trackers with their assigned detection
        for d_idx, t_idx in matches:
            self.trackers[t_idx].update(detections[d_idx][:4])
            self.trackers[t_idx].last_class_id = detections[d_idx][5]
            self.trackers[t_idx].last_conf = detections[d_idx][4]

        # Step 4: create new trackers for unmatched detections
        for d_idx in unmatched_dets:
            new_trk = KalmanBoxTracker(detections[d_idx][:4])
            new_trk.last_class_id = detections[d_idx][5]
            new_trk.last_conf = detections[d_idx][4]
            self.trackers.append(new_trk)

        # Step 5: build output and remove dead trackers
        results = []
        alive_trackers = []
        for trk in self.trackers:
            if trk.time_since_update <= self.max_age:
                alive_trackers.append(trk)

                if trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits:
                    bbox = trk.get_state()
                    class_id = getattr(trk, "last_class_id", -1)
                    conf = getattr(trk, "last_conf", 0.0)
                    results.append([
                        bbox[0], bbox[1], bbox[2], bbox[3],
                        trk.id, conf, class_id
                    ])

        self.trackers = alive_trackers

        if len(results) == 0:
            return np.empty((0, 7))
        return np.array(results)