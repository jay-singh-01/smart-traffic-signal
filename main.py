import cv2
import cvzone
import time
import os
import sys
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

# Vehicle classes to detect
VEHICLE_CLASSES = ["car", "bus", "truck", "motorcycle"]
CONFIDENCE_THRESHOLD = 0.3


# ------------------------------
# INITIALIZATION
# ------------------------------
def initialize_system():
    """Initialize the traffic signal system"""
    print("🚦 Smart Traffic Signal System - Local Demo")
    print("=" * 50)

    # Check if video file exists
    if not os.path.exists(VIDEO_PATH):
        print(f"❌ Error: Video file not found at {VIDEO_PATH}")
        print("Please ensure TrafficFlow.mp4 is in the data/ folder")
        return None, None

    # Load YOLO model
    try:
        print("🤖 Loading YOLO model...")
        model = load_model(MODEL_PATH)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None

    # Initialize video capture
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("❌ Error: Could not open video.")
        return None, None

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    print(f"📹 Video Info:")
    print(f"   - FPS: {fps}")
    print(f"   - Total Frames: {total_frames}")
    print(f"   - Duration: {duration:.1f} seconds")
    print(f"   - Resolution: {SCREEN_WIDTH}x{SCREEN_HEIGHT} (resized)")
    print()
    print("🎮 Controls:")
    print("   - Press 'q' to quit")
    print("   - Press 'r' to restart video")
    print("   - Press 'p' to pause/resume")
    print("   - Press SPACE to reset counters")
    print("=" * 50)

    return model, cap


def draw_info_panel(frame, frame_vehicle_count, total_vehicle_count, signal_state,
                    green_start_time, current_time):
    """Draw information panel on the frame"""

    # Signal color and status
    signal_color = (0, 255, 0) if signal_state == "GREEN" else (0, 0, 255)
    status_text = f"Signal: {signal_state}"

    if signal_state == "GREEN" and green_start_time:
        remaining = GREEN_SIGNAL_DURATION - int(current_time - green_start_time)
        remaining = max(0, remaining)
        status_text += f" ({remaining}s)"

    # Left side information
    try:
        cvzone.putTextRect(frame, f"Frame Vehicles: {frame_vehicle_count}", (50, 50),
                           scale=2, thickness=3, offset=5, colorR=(0, 0, 0), colorT=(255, 255, 255))
        cvzone.putTextRect(frame, f"Total Vehicles: {total_vehicle_count}", (50, 110),
                           scale=2, thickness=3, offset=5, colorR=(0, 0, 0), colorT=(255, 255, 255))
        cvzone.putTextRect(frame, f"Threshold: {VEHICLE_THRESHOLD}", (50, 170),
                           scale=1.5, thickness=2, offset=5, colorR=(100, 100, 100), colorT=(255, 255, 255))
    except Exception:
        # Fallback to basic OpenCV text
        cv2.putText(frame, f"Frame: {frame_vehicle_count}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Total: {total_vehicle_count}", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Right side signal status
    text_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 4)[0]
    text_x = SCREEN_WIDTH - text_size[0] - 50
    text_y = 80
    cv2.putText(frame, status_text, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, signal_color, 4)

    # Draw signal indicator circle
    circle_center = (SCREEN_WIDTH - 100, 150)
    cv2.circle(frame, circle_center, 30, signal_color, -1)
    cv2.circle(frame, circle_center, 30, (255, 255, 255), 3)

    return frame


# ------------------------------
# MAIN FUNCTION
# ------------------------------
def main():
    """Main execution function"""

    # Initialize system
    model, cap = initialize_system()
    if model is None or cap is None:
        return

    # State variables
    signal_state = "RED"
    green_start_time = None
    total_vehicle_count = 0
    frame_count = 0
    paused = False

    # Statistics
    start_time = time.time()
    detection_times = []

    try:
        # Main processing loop
        while True:
            if not paused:
                success, frame = cap.read()
                if not success:
                    print("\n✅ Video processing finished.")
                    print("Press 'r' to restart or 'q' to quit")

                    # Wait for restart or quit
                    while True:
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('r'):
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Restart video
                            total_vehicle_count = 0
                            signal_state = "RED"
                            green_start_time = None
                            frame_count = 0
                            break
                        elif key == ord('q'):
                            return
                    continue

                frame_count += 1

                # Resize frame for consistent display
                frame = cv2.resize(frame, (SCREEN_WIDTH, SCREEN_HEIGHT))

                # Vehicle detection
                detection_start = time.time()
                frame, frame_vehicle_count = detect_vehicles(
                    frame, model,
                    class_filter=VEHICLE_CLASSES,
                    conf_threshold=CONFIDENCE_THRESHOLD
                )
                detection_end = time.time()
                detection_times.append(detection_end - detection_start)

                total_vehicle_count += frame_vehicle_count
                current_time = time.time()

                # Traffic signal logic
                if signal_state == "GREEN":
                    if green_start_time and current_time - green_start_time >= GREEN_SIGNAL_DURATION:
                        signal_state = "RED"
                        green_start_time = None
                        print(f"🔴 Signal changed to RED (timer expired)")
                else:  # RED signal
                    if frame_vehicle_count >= VEHICLE_THRESHOLD:
                        signal_state = "GREEN"
                        green_start_time = time.time()
                        print(f"🟢 Signal changed to GREEN ({frame_vehicle_count} vehicles detected)")

                # Draw information panel
                frame = draw_info_panel(frame, frame_vehicle_count, total_vehicle_count,
                                        signal_state, green_start_time, current_time)

                # Add performance info
                if detection_times:
                    avg_detection_time = sum(detection_times[-30:]) / len(detection_times[-30:])
                    fps_text = f"Avg Detection: {avg_detection_time * 1000:.1f}ms"
                    cv2.putText(frame, fps_text, (50, SCREEN_HEIGHT - 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # Show frame
                cv2.imshow("Smart Traffic Signal - Local Demo", frame)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('r'):
                # Restart video
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                total_vehicle_count = 0
                signal_state = "RED"
                green_start_time = None
                frame_count = 0
                detection_times.clear()
                print("🔄 Video restarted")
            elif key == ord('p'):
                # Pause/resume
                paused = not paused
                print("⏸️ Paused" if paused else "▶️ Resumed")
            elif key == ord(' '):
                # Reset counters
                total_vehicle_count = 0
                signal_state = "RED"
                green_start_time = None
                detection_times.clear()
                print("🔄 Counters reset")

    except KeyboardInterrupt:
        print("\n⏹️ Interrupted by user")

    except Exception as e:
        print(f"\n❌ An error occurred: {e}")

    finally:
        # Cleanup and show statistics
        total_time = time.time() - start_time
        print(f"\n📊 Session Statistics:")
        print(f"   - Total runtime: {total_time:.1f} seconds")
        print(f"   - Frames processed: {frame_count}")
        print(f"   - Total vehicles detected: {total_vehicle_count}")
        if detection_times:
            avg_detection = sum(detection_times) / len(detection_times)
            print(f"   - Average detection time: {avg_detection * 1000:.1f}ms")
        print(f"   - Final signal state: {signal_state}")
        print("\n🚦 Thank you for using Smart Traffic Signal!")

        cap.release()
        cv2.destroyAllWindows()


# ------------------------------
# ENTRY POINT
# ------------------------------
if __name__ == "__main__":
    main()