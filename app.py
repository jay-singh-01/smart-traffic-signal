import streamlit as st
import os
import cv2
from detect import load_model, detect_vehicles, get_supported_vehicle_classes, validate_video_file, get_video_info
import main


# ------------------------------
# Streamlit Page Config
# ------------------------------
st.set_page_config(
    page_title="Smart Traffic Management",
    page_icon="🚦",
    layout="wide"
)

st.title("🚦 AI-Powered Smart Traffic Signal Controller")
st.markdown("Using **YOLOv8 + Streamlit** to detect vehicles and optimize signals.")


# ------------------------------
# Initialize Session State
# ------------------------------
if "video_path" not in st.session_state:
    st.session_state.video_path = None

if "detection_results" not in st.session_state:
    st.session_state.detection_results = None

if "model" not in st.session_state:
    st.session_state.model = None


# ------------------------------
# Load Model
# ------------------------------
with st.spinner("Loading YOLOv8 model..."):
    try:
        if st.session_state.model is None:
            st.session_state.model = load_model()
    except Exception as e:
        st.error("❌ Could not load YOLO model.")
        st.stop()


# ------------------------------
# Sidebar Controls
# ------------------------------
st.sidebar.header("⚙️ Settings")

vehicle_classes = get_supported_vehicle_classes()
selected_classes = st.sidebar.multiselect(
    "Select vehicle types to detect",
    options=vehicle_classes,
    default=["car", "motorcycle", "bus", "truck"]
)

confidence = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.3, 0.05)


# ------------------------------
# File Upload Section
# ------------------------------
st.subheader("📂 Upload a Traffic Video")

uploaded_file = st.file_uploader(
    "Choose a video file",
    type=["mp4", "avi", "mov", "mkv"],
    help="Upload a short traffic clip for analysis"
)

if uploaded_file is not None:
    save_dir = "uploads"
    os.makedirs(save_dir, exist_ok=True)

    file_path = os.path.join(save_dir, uploaded_file.name)

    # Save uploaded file
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    # Validate video
    if not validate_video_file(file_path):
        st.error("❌ Invalid or corrupted video file. Please try another.")
        st.session_state.video_path = None
    else:
        st.session_state.video_path = file_path
        st.success(f"✅ Uploaded: {uploaded_file.name}")

        # Show video details
        info = get_video_info(file_path)
        if info:
            st.info(f"""
                **Video Info**  
                - Resolution: {info['resolution']}  
                - FPS: {info['fps']:.2f}  
                - Frames: {info['frame_count']}  
                - Duration: {info['duration']:.1f}s
            """)


# ------------------------------
# Run Detection
# ------------------------------
if st.session_state.video_path:
    st.subheader("▶️ Process Video")

    if st.button("Run Vehicle Detection"):
        st.session_state.detection_results = main.process_video(
            st.session_state.video_path,
            st.session_state.model,
            selected_classes,
            confidence
        )

    if st.session_state.detection_results:
        st.video(st.session_state.detection_results["output_video"])
        st.success(f"Total Vehicles Detected: {st.session_state.detection_results['total_count']}")
        st.json(st.session_state.detection_results["counts_per_class"])
