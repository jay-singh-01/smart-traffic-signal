import cv2
import os
from detect import detect_vehicles, validate_video_file

def process_video(video_path, model, selected_classes, confidence=0.3, output_dir="outputs"):
    """
    Process the uploaded video using YOLO model and return results.
    """
    if not validate_video_file(video_path):
        raise ValueError("Invalid or corrupted video file.")

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "output_" + os.path.basename(video_path))

    cap = cv2.VideoCapture(video_path)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    total_counts = {cls: 0 for cls in selected_classes}

    frame_no = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_no += 1
        processed_frame, counts = detect_vehicles(frame, model, selected_classes, confidence)

        # Update totals
        for cls, cnt in counts.items():
            total_counts[cls] += cnt

        out.write(processed_frame)

    cap.release()
    out.release()

    return {
        "output_video": output_path,
        "total_count": sum(total_counts.values()),
        "counts_per_class": total_counts,
        "frame_count": total_frames,
        "fps": fps,
        "resolution": (width, height)
    }

if __name__ == "__main__":
    print("⚡ Main module ready. Run app.py for Streamlit UI.")
