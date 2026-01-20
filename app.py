import streamlit as st
import cv2
import tempfile
import numpy as np
import os
from camera import CameraHandler
from detector import ObjectDetector
from gap_logic import ParkingGapAnalyzer

st.set_page_config(page_title="Parking Slot Detector", layout="wide")

st.title("🚗 Real-time Parking Slot Scanner")
st.markdown("### Detect parking availability using generic vehicle detection and geometric gap analysis.")

# Sidebar Configuration
st.sidebar.header("⚙️ Configuration")

# Input Mode Selection
input_mode = st.sidebar.selectbox("Input Mode", ["Upload Video", "Upload Image", "Live Camera"])

source = None
if input_mode == "Upload Video":
    uploaded_file = st.sidebar.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        tfile.close() # Important for Windows
        source = tfile.name
elif input_mode == "Upload Image":
    uploaded_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        # Save to temp to use with CameraHandler (or just read directly, but Handler expects path or ID)
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        tfile.write(uploaded_file.read())
        tfile.close() # Important for Windows
        source = tfile.name
elif input_mode == "Live Camera":
    cam_id = st.sidebar.number_input("Camera ID (default 0)", value=0, step=1)
    source = int(cam_id)

# Detection Settings
st.sidebar.subheader("Detection Parameters")
# Model Selection
model_files = [f for f in os.listdir('.') if f.endswith('.pt')]
# Also check runs/detect for best.pt
for root, dirs, files in os.walk("runs"):
    for file in files:
        if file.endswith(".pt") and "best" in file:
            model_files.append(os.path.join(root, file))

if not model_files:
    model_files = ["yolov8n.pt"]

selected_model = st.sidebar.selectbox("Select Model Source", model_files)

conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.25, 0.05)
min_gap_width = st.sidebar.slider("Min Gap Width (Pixels)", 10, 500, 100, 10, help="Minimum width in pixels required for a parking spot. Calibrate this based on your camera view.")

start_button = st.sidebar.button("Start / Restart Processing")

# Main Display Area
st_frame = st.empty()
st_status = st.empty()

if start_button and source is not None:
    # Initialize Modules
    try:
        detector = ObjectDetector(model_path=selected_model, conf_threshold=conf_threshold)

        gap_analyzer = ParkingGapAnalyzer(min_gap_width=min_gap_width)
        camera = CameraHandler(source)
        
        st_status.info("Starting processing...")
        
        for frame in camera.get_frame():
            # Run Detection
            detections = detector.detect(frame)
            
            # Analyze Gaps
            height, width, _ = frame.shape
            is_available, gaps = gap_analyzer.analyze_availability(detections, width)
            
            # Visualization
            # Draw Bounding Boxes
            for det in detections:
                x1, y1, x2, y2 = map(int, det['box'])
                label_text = f"{det['name']} {det['conf']:.2f}"
                
                # Color coding: Green for empty (1), Red for others
                # Check class ID or name. In our custom data 1=empty.
                color = (0, 0, 255) # Red default
                if det['class'] == 1 or det['name'] == 'empty':
                    color = (0, 255, 0) # Green
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
            # Draw Gaps
            for gap in gaps:
                gx1, gx2 = int(gap['start']), int(gap['end'])
                # Draw a green overlay or line for gaps
                # We'll calculate a 'floor' y-coordinate. 
                # Since we don't have 3D info, we'll just draw a strip at the bottom or middle.
                # Let's draw a semi-transparent green box across the whole height for the gap region
                overlay = frame.copy()
                cv2.rectangle(overlay, (gx1, 0), (gx2, height), (0, 255, 0), -1)
                cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
                
                # Draw text
                center_x = (gx1 + gx2) // 2
                cv2.putText(frame, "FREE", (center_x - 20, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

            # Update Status Text
            # Update Status Text
            # If we found any "empty" class detections, report available.
            # ALSO consider gap_logic results if desired.
            empty_spots = [d for d in detections if d['class'] == 1 or d['name'] == 'empty']
            
            if len(empty_spots) > 0 or is_available:
                count = len(empty_spots) + len(gaps)
                st_status.success(f"**PARKING SLOT AVAILABLE** (Found {count} slots)")
            else:
                st_status.error("**NO PARKING SLOT AVAILABLE**")
                
            # Convert BGR to RGB for Streamlit
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st_frame.image(frame_rgb, channels="RGB")
            
            # Stop condition for single images to avoid flicker
            if camera.is_image:
                break
                
        camera.release()
        
    except Exception as e:
        st.error(f"Error: {e}")
elif start_button and source is None:
    st.warning("Please select a valid input source.")
else:
    st.info("Configure settings and click 'Start' to begin.")
