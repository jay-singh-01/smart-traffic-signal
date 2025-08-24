import streamlit as st
import cv2
import tempfile
import time
import os
import cvzone
import numpy as np
from detect import load_model, detect_vehicles

# Compatibility function for different Streamlit versions
def display_image_safe(container, image):
    """Safe image display that works with different Streamlit versions"""
    try:
        # Try with use_container_width (Streamlit >= 1.18.0)
        container.image(image, use_container_width=True)
    except TypeError:
        # Fallback for older versions
        container.image(image)

# Page config (wide layout)
st.set_page_config(page_title="Smart Traffic Signal", layout="wide")
st.title("🚦 AI-Powered Smart Traffic Signal Controller")

st.markdown("""
This app uses **YOLOv8 + OpenCV** to detect vehicles and simulate adaptive traffic signals.  
- Counts vehicles frame-by-frame  
- Switches signal **GREEN** when density exceeds threshold  
- Shows live detection and provides processed video download
""")

# Sidebar for parameters
st.sidebar.header("🔧 Configuration")
VEHICLE_THRESHOLD = st.sidebar.slider("Vehicle Threshold for Green Signal", 1, 50, 20)
GREEN_SIGNAL_DURATION = st.sidebar.slider("Green Signal Duration (seconds)", 5, 30, 15)
PROCESS_EVERY_N_FRAMES = st.sidebar.slider("Process every N frames (for speed)", 1, 10, 3)
CONF_THRESHOLD = st.sidebar.slider("Detection Confidence", 0.1, 1.0, 0.3, 0.1)

# Initialize session state
if 'signal_state' not in st.session_state:
    st.session_state.signal_state = "RED"
if 'green_start_time' not in st.session_state:
    st.session_state.green_start_time = None
if 'total_vehicle_count' not in st.session_state:
    st.session_state.total_vehicle_count = 0
if 'processing' not in st.session_state:
    st.session_state.processing = False

# Option to use sample video or upload
st.subheader("📹 Choose Video Source")
video_option = st.radio(
    "Select video source:",
    ["Upload your own video", "Use sample video (Traffic_Flow.mp4)"]
)

video_file = None
if video_option == "Upload your own video":
    uploaded_file = st.file_uploader("📂 Upload a traffic video", type=["mp4", "avi", "mov"])
    video_file = uploaded_file
elif video_option == "Use sample video (Traffic_Flow.mp4)":
    sample_path = "data/Traffic_Flow.mp4"
    if os.path.exists(sample_path):
        video_file = sample_path
        st.success("✅ Using sample video: Traffic_Flow.mp4")
    else:
        st.error("❌ Sample video not found. Please upload your own video.")

if video_file and not st.session_state.processing:
    if st.button("🚀 Start Processing"):
        st.session_state.processing = True
        st.rerun()


if video_file and st.session_state.processing:
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📺 Live Processing")
        stframe = st.empty()

    with col2:
        st.subheader("📊 Statistics")
        stats_placeholder = st.empty()
        signal_placeholder = st.empty()

    # Handle file path based on source
    if isinstance(video_file, str):  # Sample video
        temp_file_path = video_file
        filename = "Traffic_Flow.mp4"
    else:  # Uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
            tfile.write(video_file.read())
            temp_file_path = tfile.name
            filename = video_file.name

    try:
        # Load YOLO model
        with st.spinner("🤖 Loading YOLO model..."):
            model = load_model()

        cap = cv2.VideoCapture(temp_file_path)

        if not cap.isOpened():
            st.error("❌ Error: Could not open video file.")
            st.session_state.processing = False
            st.stop()

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width, height = 1280, 720

        # Create output directory and file
        os.makedirs("outputs", exist_ok=True)
        out_path = os.path.join("outputs", f"processed_{filename}")

        # Video writer setup
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

        # Progress tracking
        progress_bar = st.progress(0)
        progress_text = st.empty()

        frame_count = 0
        processed_frames = 0
        start_time = time.time()

        st.info("▶️ Processing video... Watch the live preview!")

        # Reset counters for new video
        st.session_state.signal_state = "RED"
        st.session_state.green_start_time = None
        st.session_state.total_vehicle_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Update progress
            if total_frames > 0:
                progress = frame_count / total_frames
                progress_bar.progress(progress)
                progress_text.text(f"Processing frame {frame_count}/{total_frames}")

            # Resize frame
            frame = cv2.resize(frame, (width, height))

            # Process detection every N frames
            count = 0
            if frame_count % PROCESS_EVERY_N_FRAMES == 0:
                # Run detection
                frame, count = detect_vehicles(
                    frame, model,
                    class_filter=["car", "truck", "bus", "motorcycle"],
                    conf_threshold=CONF_THRESHOLD
                )
                st.session_state.total_vehicle_count += count
                processed_frames += 1

                # Traffic signal logic
                current_time = time.time()

                if st.session_state.signal_state == "GREEN":
                    if (st.session_state.green_start_time and
                            current_time - st.session_state.green_start_time >= GREEN_SIGNAL_DURATION):
                        st.session_state.signal_state = "RED"
                        st.session_state.green_start_time = None
                else:  # RED signal
                    if count >= VEHICLE_THRESHOLD:
                        st.session_state.signal_state = "GREEN"
                        st.session_state.green_start_time = time.time()

                # Prepare display info
                signal_color = (0, 255, 0) if st.session_state.signal_state == "GREEN" else (0, 0, 255)
                status_text = f"Signal: {st.session_state.signal_state}"

                if (st.session_state.signal_state == "GREEN" and
                        st.session_state.green_start_time):
                    remaining = GREEN_SIGNAL_DURATION - int(current_time - st.session_state.green_start_time)
                    remaining = max(0, remaining)
                    status_text += f" ({remaining}s)"

            # Add overlays to frame (even on non-processed frames)
            try:
                # Left side info
                cvzone.putTextRect(frame, f"Frame Vehicles: {count}", (50, 60),
                                   scale=2, thickness=3, offset=5, colorR=(0, 0, 0))
                cvzone.putTextRect(frame, f"Total Vehicles: {st.session_state.total_vehicle_count}",
                                   (50, 120), scale=2, thickness=3, offset=5, colorR=(0, 0, 0))

                # Right side signal status
                signal_color = (0, 255, 0) if st.session_state.signal_state == "GREEN" else (0, 0, 255)
                status_text = f"Signal: {st.session_state.signal_state}"

                if (st.session_state.signal_state == "GREEN" and
                        st.session_state.green_start_time):
                    remaining = GREEN_SIGNAL_DURATION - int(time.time() - st.session_state.green_start_time)
                    remaining = max(0, remaining)
                    status_text += f" ({remaining}s)"

                text_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 4)[0]
                text_x = max(width - text_size[0] - 50, 50)
                text_y = 100
                cv2.putText(frame, status_text, (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, signal_color, 4)

            except Exception as e:
                # Fallback rendering
                cv2.putText(frame, f"Vehicles: {count}", (50, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Update live preview (every few frames for performance)
            # FIXED: This is the line that was causing the error
            if frame_count % (PROCESS_EVERY_N_FRAMES * 2) == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                display_image_safe(stframe, frame_rgb)  # Using safe function

            # Update statistics
            if frame_count % PROCESS_EVERY_N_FRAMES == 0:
                with stats_placeholder.container():
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Current Frame", count)
                        st.metric("Total Count", st.session_state.total_vehicle_count)
                    with col_b:
                        st.metric("Processed", f"{processed_frames}")
                        elapsed = time.time() - start_time
                        st.metric("Elapsed", f"{elapsed:.1f}s")

                with signal_placeholder.container():
                    if st.session_state.signal_state == "GREEN":
                        remaining_time = ""
                        if st.session_state.green_start_time:
                            remaining = GREEN_SIGNAL_DURATION - int(time.time() - st.session_state.green_start_time)
                            remaining = max(0, remaining)
                            remaining_time = f" ({remaining}s remaining)"
                        st.success(f"🟢 GREEN SIGNAL{remaining_time}")
                    else:
                        st.error(f"🔴 RED SIGNAL")

            # Write frame to output video
            out.write(frame)

            # Safety break
            if frame_count > total_frames and total_frames > 0:
                break

        # Cleanup
        cap.release()
        out.release()

        progress_bar.progress(1.0)
        progress_text.text("✅ Processing complete!")

        total_elapsed = time.time() - start_time
        st.success(f"✅ Processing complete! Processed {processed_frames} frames in {total_elapsed:.1f} seconds")

        # Final statistics
        st.subheader("📈 Final Results")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Frames", total_frames)
        with col2:
            st.metric("Vehicles Detected", st.session_state.total_vehicle_count)
        with col3:
            st.metric("Final Signal", st.session_state.signal_state)
        with col4:
            st.metric("Processing Time", f"{total_elapsed:.1f}s")

        # Show processed video
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            st.subheader("🎬 Processed Video")
            st.video(out_path)

            # Download button
            with open(out_path, "rb") as f:
                st.download_button(
                    "⬇️ Download Processed Video",
                    f,
                    file_name=f"processed_{filename}",
                    mime="video/mp4"
                )
        else:
            st.warning("⚠️ Output video file was not created properly.")

    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")
        st.exception(e)

    finally:
        # Clean up temp file (only if it was uploaded)
        if not isinstance(video_file, str):
            try:
                os.unlink(temp_file_path)
            except:
                pass

        st.session_state.processing = False

# Control buttons
if st.session_state.processing:
    if st.button("⏹️ Stop Processing"):
        st.session_state.processing = False
        st.rerun()

if st.sidebar.button("🔄 Reset All"):
    st.session_state.signal_state = "RED"
    st.session_state.green_start_time = None
    st.session_state.total_vehicle_count = 0
    st.session_state.processing = False
    st.rerun()

# Information section
with st.expander("ℹ️ How it works"):
    st.markdown("""
    **Smart Traffic Signal Logic:**
    1. **Vehicle Detection**: Uses YOLOv8 to detect cars, trucks, buses, and motorcycles
    2. **Signal Control**: 
       - Signal stays RED by default
       - Switches to GREEN when vehicle count ≥ threshold
       - GREEN duration is configurable (default: 15 seconds)
       - Returns to RED after timer expires
    3. **Performance**: Processes every N frames to balance accuracy vs speed

    **Controls:**
    - Adjust vehicle threshold and green signal duration in sidebar
    - Use sample video or upload your own
    - Download processed video with detections and signal states
    """)

if not video_file:
    st.info("👆 Please select a video source to begin processing")