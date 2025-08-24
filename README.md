# 🚦 AI-Powered Smart Traffic Signal Controller

This project uses **YOLOv8 + OpenCV + Streamlit** to detect vehicles from a video feed  
and control traffic lights adaptively based on real-time traffic density.

## Features
- Vehicle detection (cars, buses, trucks)  
- Frame-wise vehicle counting  
- Adaptive traffic light switching  
- Streamlit web app for demo  

## Setup & Installation :-

1. Clone the repo:

bash
git clone https://github.com/yourusername/smart-traffic-signal.git
cd smart-traffic-signal

2. Create a virtual environment (recommended):

bash
python -m venv .venv
source .venv/bin/activate   # Mac/Linux
.venv\Scripts\activate      # Windows

3. Install dependencies:

bash
pip install -r requirements.txt

4. Running the Project
Local Demo (OpenCV)

bash
python main.py

5. Streamlit Web App

bash
streamlit run app.py

## Example Output:-

* Counts vehicles in video frames
* Displays adaptive traffic signal state (RED/GREEN)
* Shows processed video in real time

## Future Enhancements:-

* Integrate with live traffic camera feed
* Add database/API for smart city integration


## Project Structure
```plaintext
smart-traffic-signal/
├── app.py              # Streamlit web app (frontend + UI)
├── main.py             # Local demo (OpenCV fullscreen mode)
├── detect.py           # YOLO-based vehicle detection logic
├── requirements.txt    # Dependencies list
├── README.md           # Project documentation
│
├── data/               # Input video samples
│   └── TrafficFlow.mp4
│
├── outputs/            # Processed output videos/images
│   └── sample_output.mp4
│
├── weights/            # Pre-trained YOLO weights
│   └── yolov8n.pt
│
└── utils/              # (Optional) helper functions in future
    └── __init__.py
