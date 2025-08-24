import cv2
import cvzone
import time
from detect import load_model, detect_vehicles

# ------------------------------
# CONFIGURATION
# ------------------------------
VIDEO_PATH = "data/Traffic_Flow.mp4"
MODEL_PATH = "weights/yolov8n.pt"

VEHICLE_THRESHOLD = 20
GREEN_SIGNAL_DURATION = 15

SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720

# ------------------------------
# INITIALIZATION
# ------------------------------
model = load_model(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("❌ Error: Could not open video.")
    exit()

signal_state = "RED"
green_start_time = None
total_vehicle_count = 0

# ------------------------------
# MAIN LOOP
# ------------------------------
while True:
    success, frame = cap.read()
    if not success:
        print("✅ Video processing finished.")
        break

    frame = cv2.resize(frame, (SCREEN_WIDTH, SCREEN_HEIGHT))

    # Detect vehicles
    frame, frame_vehicle_count = detect_vehicles(
        frame, model, class_filter=["car", "bus", "truck"], conf_threshold=0.3
    )

    total_vehicle_count += frame_vehicle_count
    current_time = time.time()

    # ---------------- SIGNAL LOGIC ----------------
    if signal_state == "GREEN":
        if current_time - green_start_time >= GREEN_SIGNAL_DURATION:
            signal_state = "RED"
    else:
        if frame_vehicle_count > VEHICLE_THRESHOLD:
            signal_state = "GREEN"
            green_start_time = time.time()

    # ---------------- DISPLAY INFO ----------------
    signal_color = (0, 255, 0) if signal_state == "GREEN" else (0, 0, 255)
    status_text = f"Signal: {signal_state}"

    if signal_state == "GREEN":
        remaining = GREEN_SIGNAL_DURATION - int(current_time - green_start_time)
        status_text += f" ({remaining}s)"

    # Left side info
    cvzone.putTextRect(frame, f"Frame Vehicles: {frame_vehicle_count}", (50, 50),
                       scale=2, thickness=3, offset=5)
    cvzone.putTextRect(frame, f"Total Vehicles: {total_vehicle_count}", (50, 110),
                       scale=2, thickness=3, offset=5)

    # Right side info (aligned near the right edge)
    text_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 4)[0]
    text_x = SCREEN_WIDTH - text_size[0] - 50  # 50px padding from right edge
    text_y = 80
    cv2.putText(frame, status_text, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, signal_color, 4)

    # Show window
    cv2.imshow("Smart Traffic Signal", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
