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
        # Ensure weights directory exists
        os.makedirs("weights", exist_ok=True)

        # Download model if it doesn't exist
        if not os.path.exists(model_path):
            with st.spinner("🔄 Downloading YOLOv8n model weights (first time only)..."):
                url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"

                try:
                    # Download without showing progress bars to avoid multiple bars
                    urllib.request.urlretrieve(url, model_path)
                    st.success("✅ Model downloaded successfully!")

                except Exception as download_error:
                    st.error(f"❌ Failed to download model: {download_error}")
                    st.info("Please manually download yolov8n.pt to the weights/ folder")
                    raise download_error

        # Load the model (suppress YOLO verbose output)
        model = YOLO(model_path, verbose=False)

        # Verify model loaded correctly
        if hasattr(model, 'names'):
            st.success(f"✅ Model loaded successfully! Can detect {len(model.names)} classes.")

        return model

    except Exception as e:
        st.error(f"❌ Error loading YOLO model: {str(e)}")
        st.info("Make sure you have proper internet connection for first-time model download.")
        raise e


# ------------------------------
# Detect vehicles in a frame
# ------------------------------
def detect_vehicles(frame, model, class_filter=None, conf_threshold=0.3):
    """
    Detect vehicles in a frame and return annotated frame with count

    Args:
        frame: Input frame (numpy array)
        model: YOLO model instance
        class_filter: List of classes to detect (e.g., ["car", "truck", "bus"])
        conf_threshold: Confidence threshold for detections (0.0 to 1.0)

    Returns:
        tuple: (annotated_frame, vehicle_count)
    """
    try:
        # Run YOLO inference (suppress verbose output)
        results = model(frame, verbose=False, conf=conf_threshold)
        count = 0

        # Get class names from model
        if hasattr(model, 'names'):
            classNames = model.names
        else:
            # Fallback class names for COCO dataset
            classNames = {
                0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
                5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light'
                # ... (simplified for key vehicle classes)
            }

        # Default vehicle classes if no filter specified
        if class_filter is None:
            class_filter = ["car", "motorcycle", "bus", "truck", "bicycle"]

        # Process each detection result
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    try:
                        # Extract box data
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                        conf = float(box.conf[0].cpu().numpy())
                        cls = int(box.cls[0].cpu().numpy())

                        # Get class name
                        if cls in classNames:
                            label = classNames[cls]
                        else:
                            continue  # Skip unknown classes

                        # Filter by class and confidence
                        if label in class_filter and conf >= conf_threshold:
                            count += 1

                            # Draw bounding box and label
                            try:
                                # Use cvzone for styled boxes
                                box_width = x2 - x1
                                box_height = y2 - y1

                                cvzone.cornerRect(frame, (x1, y1, box_width, box_height),
                                                  l=15, t=2, colorR=(255, 0, 255), colorC=(0, 255, 0))

                                # Add label with confidence
                                label_text = f"{label} {conf:.2f}"
                                cvzone.putTextRect(frame, label_text, (x1, max(y1 - 10, 20)),
                                                   scale=1, thickness=2, offset=5,
                                                   colorT=(255, 255, 255), colorR=(255, 0, 255))

                            except Exception as cvzone_error:
                                # Fallback to OpenCV drawing if cvzone fails
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                                cv2.putText(frame, f"{label} {conf:.2f}",
                                            (x1, max(y1 - 10, 20)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                    except Exception as box_error:
                        # Skip problematic boxes
                        continue

        return frame, count

    except Exception as e:
        # Return original frame with zero count if detection fails
        st.warning(f"⚠️ Detection error: {str(e)}")
        return frame, 0


# ------------------------------
# Utility functions
# ------------------------------
def get_supported_vehicle_classes():
    """Return list of vehicle classes supported by YOLO"""
    return ["car", "motorcycle", "bus", "truck", "bicycle", "train", "airplane", "boat"]


def validate_video_file(file_path):
    """
    Validate if video file can be opened and read

    Args:
        file_path: Path to video file

    Returns:
        bool: True if video is valid, False otherwise
    """
    try:
        cap = cv2.VideoCapture(file_path)
        is_valid = cap.isOpened()

        if is_valid:
            # Try to read first frame
            ret, frame = cap.read()
            is_valid = ret and frame is not None

        cap.release()
        return is_valid

    except Exception:
        return False


def get_video_info(file_path):
    """
    Get basic information about a video file

    Args:
        file_path: Path to video file

    Returns:
        dict: Video information (fps, frame_count, duration, resolution)
    """
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
            'fps': fps,
            'frame_count': frame_count,
            'duration': duration,
            'width': width,
            'height': height,
            'resolution': f"{width}x{height}"
        }

    except Exception:
        return None