from ultralytics import YOLO
import cv2
import cvzone
import os
import urllib.request
import streamlit as st


# ------------------------------
# Load YOLO model
# ------------------------------
@st.cache_resource
def load_model(model_path="weights/yolov8n.pt"):
    """Load YOLO model with caching for better Streamlit performance"""
    try:
        os.makedirs("weights", exist_ok=True)

        if not os.path.exists(model_path):
            st.info("🔄 Downloading YOLOv8n model weights (first time only)...")
            url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"

            def download_progress(block_num, block_size, total_size):
                downloaded = block_num * block_size
                if total_size > 0:
                    percent = min(100, (downloaded * 100) / total_size)
                    st.progress(percent / 100, text=f"Downloading... {percent:.1f}%")

            urllib.request.urlretrieve(url, model_path, reporthook=download_progress)
            st.success("✅ Model downloaded successfully!")

        model = YOLO(model_path)

        if hasattr(model, "names"):
            st.success(f"✅ Model loaded! Can detect {len(model.names)} classes.")

        return model

    except Exception as e:
        st.error(f"❌ Error loading YOLO model: {str(e)}")
        st.info("Please check your internet or manually place yolov8n.pt in weights/")
        raise e


# ------------------------------
# Detect vehicles in a frame
# ------------------------------
def detect_vehicles(frame, model, class_filter=None, conf_threshold=0.3):
    """
    Detect vehicles in a frame and return annotated frame with count
    """
    try:
        results = model(frame, verbose=False, conf=conf_threshold)
        count = 0

        classNames = getattr(model, "names", {
            0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle',
            5: 'bus', 7: 'truck'
        })

        if class_filter is None:
            class_filter = ["car", "motorcycle", "bus", "truck", "bicycle"]

        for r in results:
            if not hasattr(r, "boxes") or r.boxes is None:
                continue

            for box in r.boxes:
                try:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    conf = float(box.conf[0].cpu().numpy())
                    cls = int(box.cls[0].cpu().numpy())

                    label = classNames.get(cls, None)
                    if not label:
                        continue

                    if label in class_filter and conf >= conf_threshold:
                        count += 1
                        box_width, box_height = x2 - x1, y2 - y1

                        try:
                            cvzone.cornerRect(frame, (x1, y1, box_width, box_height),
                                              l=15, t=2, colorR=(255, 0, 255), colorC=(0, 255, 0))
                            cvzone.putTextRect(frame, f"{label} {conf:.2f}",
                                               (x1, max(y1 - 10, 20)),
                                               scale=1, thickness=2, offset=5,
                                               colorT=(255, 255, 255), colorR=(255, 0, 255))
                        except Exception:
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                            cv2.putText(frame, f"{label} {conf:.2f}", (x1, max(y1 - 10, 20)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                except Exception:
                    continue

        return frame, count

    except Exception as e:
        st.warning(f"⚠️ Detection error: {str(e)}")
        return frame, 0


def get_supported_vehicle_classes():
    return ["car", "motorcycle", "bus", "truck", "bicycle", "train", "airplane", "boat"]


def validate_video_file(file_path):
    try:
        cap = cv2.VideoCapture(file_path)
        is_valid = cap.isOpened()
        if is_valid:
            ret, frame = cap.read()
            is_valid = ret and frame is not None
        cap.release()
        return is_valid
    except Exception:
        return False


def get_video_info(file_path):
    try:
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            return None
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0
        cap.release()
        return {
            "fps": fps,
            "frame_count": frame_count,
            "duration": duration,
            "width": width,
            "height": height,
            "resolution": f"{width}x{height}"
        }
    except Exception:
        return None
