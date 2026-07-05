"""
kalman_filter.py
A simple Kalman Filter for tracking a single object's bounding box over time.

State vector (7D): [cx, cy, s, r, vcx, vcy, vs]
    cx, cy = center x, y of the bounding box
    s      = scale (area of the box)
    r      = aspect ratio (width / height) -- assumed constant
    vcx, vcy, vs = velocities of cx, cy, s

We predict the next state using a constant-velocity motion model, then
update/correct it whenever a matching detection arrives.
"""

import numpy as np


def bbox_to_state(bbox):
    """
    Convert [x1, y1, x2, y2] to [cx, cy, s, r].
    s = area, r = aspect ratio.
    """
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0
    s = w * h
    r = w / float(h) if h != 0 else 0
    return np.array([cx, cy, s, r]).reshape((4, 1))


def state_to_bbox(state):
    """
    Convert [cx, cy, s, r, ...] back to [x1, y1, x2, y2].
    """
    cx, cy, s, r = state[0], state[1], state[2], state[3]
    w = np.sqrt(max(s * r, 0))
    h = s / w if w != 0 else 0
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return np.array([x1, y1, x2, y2]).flatten()


class KalmanBoxTracker:
    """
    Tracks a single object's bounding box using a Kalman Filter
    with a constant-velocity model.
    """

    count = 0  # class-level counter to assign unique IDs

    def __init__(self, bbox):
        """
        Initialize tracker with an initial bounding box [x1, y1, x2, y2].
        """
        # State transition matrix (7x7): constant velocity model
        self.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
        ], dtype=float)

        # Measurement matrix (4x7): we only directly observe cx, cy, s, r
        self.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
        ], dtype=float)

        # Measurement noise covariance
        self.R = np.eye(4) * 1.0
        self.R[2:, 2:] *= 10.0  # less confident about scale/ratio noise

        # Process noise covariance
        self.Q = np.eye(7) * 1.0
        self.Q[4:, 4:] *= 0.01  # velocities change slowly
        self.Q[2, 2] *= 0.01
        self.Q[6, 6] *= 0.01

        # Initial state covariance (high uncertainty for velocities)
        self.P = np.eye(7) * 10.0
        self.P[4:, 4:] *= 1000.0

        # Initial state
        self.x = np.zeros((7, 1))
        self.x[:4] = bbox_to_state(bbox)

        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1

        self.time_since_update = 0
        self.hits = 0           # total number of successful updates
        self.hit_streak = 0     # consecutive updates without a miss
        self.age = 0            # total frames since creation

    def predict(self):
        """
        Advance the state one time step. Call once per frame.
        Returns the predicted bounding box [x1, y1, x2, y2].
        """
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1

        return state_to_bbox(self.x.flatten())

    def update(self, bbox):
        """
        Correct the state using a newly matched detection bbox [x1,y1,x2,y2].
        """
        z = bbox_to_state(bbox)

        y = z - self.H @ self.x                      # innovation
        S = self.H @ self.P @ self.H.T + self.R       # innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)      # Kalman gain

        self.x = self.x + K @ y
        self.P = (np.eye(7) - K @ self.H) @ self.P

        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1

    def get_state(self):
        """Return the current estimated bounding box [x1, y1, x2, y2]."""
        return state_to_bbox(self.x.flatten())