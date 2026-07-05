"""
test_detector.py
Quick standalone test: opens webcam, runs YOLOv8 detection, draws raw boxes.
Run this from the project root: python test_detector.py
Press 'q' to quit.
"""

import cv2
import sys
sys.path.append("src")
from detector import Detector

def main():
    detector = Detector(model_path="yolov8n.pt", conf_threshold=0.4)
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

        detections = detector.detect(frame)

        for det in detections:
            x1, y1, x2, y2, conf, cls_id = det
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            label = f"{detector.get_class_name(cls_id)} {conf:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("Detector Test", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()