# 🚦 AI-Powered Smart Traffic Signal Controller

This project uses **YOLOv8 + OpenCV + Streamlit** to detect vehicles from a video feed  
and control traffic lights adaptively based on real-time traffic density.

## Features
- Vehicle detection (cars, buses, trucks)  
- Frame-wise vehicle counting  
- Adaptive traffic light switching  
- Streamlit web app for demo  

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
Setup & Installation
Clone the repo:

bash
Copy
Edit
git clone https://github.com/yourusername/smart-traffic-signal.git
cd smart-traffic-signal
Create a virtual environment (recommended):

bash
Copy
Edit
python -m venv .venv
source .venv/bin/activate   # Mac/Linux
.venv\Scripts\activate      # Windows
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Running the Project
Local Demo (OpenCV)

bash
Copy
Edit
python main.py
Streamlit Web App

bash
Copy
Edit
streamlit run app.py
Example Output
Counts vehicles in video frames

Displays adaptive traffic signal state (RED/GREEN)

Shows processed video in real time

Future Enhancements
Integrate with live traffic camera feed

Add database/API for smart city integration