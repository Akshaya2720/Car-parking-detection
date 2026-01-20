# Boundary-less Parking Slot Detection System

A lightweight, real-time computer vision system to detect parking slot availability using YOLOv8 and geometric gap analysis.

## Features
- **Real-time Detection**: Uses YOLOv8-Nano for fast inference on CPU.
- **Dynamic Input**: Supports Webcams, IP Cameras, Video Files, and Images.
- **Boundary-less Logic**: Does not require painting lines or fixed polygons; calculates availability based on gaps between vehicles.
- **User Interface**: Clean Streamlit dashboard for visualization and control.

## Prerequisites
- Python 3.8+
- Hardware: CPU (Intel i3 12th Gen or better recommended) or GPU.

## Installation

1.  Clone this repository or copy the files to a local directory:
    - `app.py`
    - `camera.py`
    - `detector.py`
    - `gap_logic.py`
    - `train.py`
    - `requirements.txt`

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Running the Application
Run the Streamlit app:
```bash
streamlit run app.py
```

### Using the Application
1.  **Select Input Mode**: Choose between "Live Camera", "Upload Video", or "Upload Image" in the sidebar.
2.  **Configuration**:
    - **Confidence Threshold**: Adjust how sure the model needs to be to detect a car. Default is 0.25.
    - **Min Gap Width**: This is the critical calibration parameter.

### ⚠️ Calibration Guide
Since this is a 2D camera system without depth sensors, "distance" is measured in pixels.
1.  Point your camera at the parking area.
2.  Identify a car in the view.
3.  Estimate or measure the width of that car in pixels on your screen.
    - You can use a screenshot and an image editor to measure valid crop.
    - Or trial and error: Set the "Min Gap Width" to a value (e.g., 200). If the system says "Available" for a gap that is clearly too small for a car, **increase** the value. If it says "No Slot" when there is space, **decrease** the value.
    - **Tip**: Cars further away look smaller. This system uses a single threshold, so it works best if cars are roughly at the same distance from the camera (e.g., a side view of a row of cars).

## Training Custom Model
If the default YOLOv8 model does not detect your vehicles well, you can train it on your own dataset.
1.  Prepare your dataset in YOLO format (images and txt labels).
2.  Create a `data.yaml` file pointing to your train/val paths.
3.  Run the training script:
    ```bash
    python train.py --data path/to/data.yaml --epochs 50
    ```
4.  After training, update `detector.py` to point to your new model path (e.g., `runs/detect/train/weights/best.pt`) or pass it in code.

## Troubleshooting
- **Backend Unreachable**: Ensure no other process is using the camera.
- **Slow Performance**: Lower the resolution or ensure strict CPU usage restrictions are met.
