# detect.py
from ultralytics import YOLO
import cv2
import cvzone

# ------------------------------
# Load YOLO model
# ------------------------------
def load_model(model_path="weights/yolov8n.pt"):
    return YOLO(model_path)

# ------------------------------
# Detect vehicles in a frame
# ------------------------------
def detect_vehicles(frame, model, class_filter=None, conf_threshold=0.3):
    """
    Detects vehicles in a frame using YOLO model.

    Args:
        frame: Input video frame
        model: YOLO model object
        class_filter: list of allowed class names (e.g., ["car", "truck", "bus"])
        conf_threshold: minimum confidence threshold

    Returns:
        frame: frame with bounding boxes drawn
        count: number of detected vehicles
    """
    results = model(frame)
    count = 0
    classNames = model.names

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = classNames[cls]

            if (class_filter is None or label in class_filter) and conf > conf_threshold:
                count += 1
                cvzone.cornerRect(frame, (x1, y1, x2 - x1, y2 - y1), l=8)
                cvzone.putTextRect(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                                   scale=1, thickness=2, offset=3)

    return frame, count
