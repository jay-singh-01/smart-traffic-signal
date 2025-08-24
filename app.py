# app.py
import streamlit as st
import cv2
import tempfile
import time
import os
import cvzone
from detect import load_model, detect_vehicles

# Page config (wide layout)
st.set_page_config(page_title="Smart Traffic Signal", layout="wide")
st.title("🚦 AI-Powered Smart Traffic Signal Controller")

st.markdown("""
This app uses **YOLOv8 + OpenCV** to detect vehicles and simulate adaptive traffic signals.  
- Counts vehicles frame-by-frame  
- Switches signal **GREEN** when density exceeds threshold  
- Lets you preview live detection or download a fully processed video
""")

# Parameters
VEHICLE_THRESHOLD = 20
GREEN_SIGNAL_DURATION = 15

uploaded_file = st.file_uploader("📂 Upload a traffic video", type=["mp4", "avi", "mov"])

if uploaded_file:
    # Save uploaded file to temp
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())

    # Load YOLO model
    model = load_model("weights/yolov8n.pt")
    cap = cv2.VideoCapture(tfile.name)

    # Streamlit live frame
    stframe = st.empty()

    # Signal state vars
    signal_state = "RED"
    green_start_time = None
    total_vehicle_count = 0

    # Save processed video
    out_path = os.path.join("outputs", f"processed_{uploaded_file.name}")
    os.makedirs("outputs", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width, height = 1280, 720  # force resolution
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    st.info("▶ Processing video... please wait or watch live preview below")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame for consistent display
        frame = cv2.resize(frame, (width, height))

        # Run detection
        frame, count = detect_vehicles(frame, model, class_filter=["car", "truck", "bus"])
        total_vehicle_count += count

        # Traffic signal logic
        current_time = time.time()
        if signal_state == "GREEN":
            if current_time - green_start_time >= GREEN_SIGNAL_DURATION:
                signal_state = "RED"
        else:
            if count > VEHICLE_THRESHOLD:
                signal_state = "GREEN"
                green_start_time = time.time()

        # ---------------- DISPLAY INFO ----------------
        signal_color = (0, 255, 0) if signal_state == "GREEN" else (0, 0, 255)
        status_text = f"Signal: {signal_state}"
        if signal_state == "GREEN":
            remaining = GREEN_SIGNAL_DURATION - int(current_time - green_start_time)
            status_text += f" ({remaining}s)"

        # Left side text
        cvzone.putTextRect(frame, f"Frame Vehicles: {count}", (50, 60),
                           scale=2, thickness=3, offset=5)
        cvzone.putTextRect(frame, f"Total Vehicles: {total_vehicle_count}", (50, 120),
                           scale=2, thickness=3, offset=5)

        # Right side text (aligned dynamically)
        text_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 4)[0]
        text_x = width - text_size[0] - 50
        text_y = 100
        cv2.putText(frame, status_text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, signal_color, 4)

        # Show live preview (fullscreen effect with use_column_width)
        stframe.image(frame, channels="BGR", use_column_width=True)

        # Write to processed video
        out.write(frame)

    cap.release()
    out.release()

    st.success("✅ Processing complete!")

    # Show final processed video
    st.video(out_path)

    # Download option
    with open(out_path, "rb") as f:
        st.download_button("⬇ Download Processed Video", f, file_name=f"processed_{uploaded_file.name}")
