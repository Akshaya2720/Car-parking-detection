import streamlit as st
import os
import glob
from PIL import Image
import json
import pandas as pd

# Try to import the canvas component
try:
    from streamlit_drawable_canvas import st_canvas
except ImportError:
    st.error("Please install the canvas component: pip install streamlit-drawable-canvas")
    st.stop()

st.set_page_config(layout="wide")
st.title("🏷️ Easy Toy Car Labeler")

# --- session state setup ---
if 'image_index' not in st.session_state:
    st.session_state.image_index = 0
if 'saved_labels' not in st.session_state:
    st.session_state.saved_labels = 0

# --- Load Images ---
IMAGE_DIR = r"C:\Users\akshaya\Desktop\mini\dataset"  # Start from root
# Scan safely
image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
all_images = []
for ext in image_extensions:
    all_images.extend(glob.glob(os.path.join(IMAGE_DIR, '**', ext), recursive=True))

# Sort for consistency
all_images = sorted(all_images)

if not all_images:
    st.error(f"No images found in {IMAGE_DIR}")
    st.stop()

# --- Sidebar Controls ---
st.sidebar.header("Controls")
st.sidebar.info(f"Total Images: {len(all_images)}")

# Navigation
idx = st.session_state.image_index
col1, col2 = st.sidebar.columns(2)
if col1.button("⬅️ Previous"):
    st.session_state.image_index = max(0, idx - 1)
    st.rerun()
if col2.button("Next ➡️"):
    st.session_state.image_index = min(len(all_images) - 1, idx + 1)
    st.rerun()

# Jump to
new_idx = st.sidebar.number_input("Jump to Image #", 0, len(all_images)-1, st.session_state.image_index)
if new_idx != st.session_state.image_index:
    st.session_state.image_index = new_idx
    st.rerun()

# --- Display Image ---
current_image_path = all_images[st.session_state.image_index]
display_name = os.path.relpath(current_image_path, IMAGE_DIR)
st.subheader(f"Image {st.session_state.image_index + 1}/{len(all_images)}: {display_name}")

# Check local label file
label_path = os.path.splitext(current_image_path)[0] + ".txt"
has_label = os.path.exists(label_path)
if has_label:
    st.sidebar.success("✅ Already Labeled")
else:
    st.sidebar.warning("⚠️ Not Labeled")

img = Image.open(current_image_path)
width, height = img.size

# --- Canvas ---
# Scale image for display if too big, but keep note of scale
# Canvas works best with fixed width?
canvas_width = 700
scale_factor = canvas_width / width
canvas_height = int(height * scale_factor)

st.write("Draw a box around the toy car:")

# Load existing labels to pre-fill canvas? 
# (Complex to reverse engineer perfectly, let's keep it simple: manual drawing)
initial_drawing = None

# Create canvas
canvas_result = st_canvas(
    fill_color="rgba(0, 255, 0, 0.3)",  # Fixed fill color with some opacity
    stroke_width=2,
    stroke_color="#00FF00",
    background_image=img,
    update_streamlit=True,
    height=canvas_height,
    width=canvas_width,
    drawing_mode="rect",
    point_display_radius=0,
    key="canvas",
)

# --- Save Logic ---
if canvas_result.json_data is not None:
    objects = canvas_result.json_data["objects"]
    
    # Calculate YOLO format labels
    yolo_labels = []
    
    if len(objects) > 0:
        st.write(f"Found {len(objects)} boxes.")
        
        for obj in objects:
            # Coords relative to canvas size
            left = obj["left"]
            top = obj["top"]
            w_box = obj["width"]
            h_box = obj["height"]
            
            # Scale back to original image size
            # x_center, y_center, w, h (normalized 0-1)
            
            # Real coords
            real_x = left / scale_factor
            real_y = top / scale_factor
            real_w = w_box / scale_factor
            real_h = h_box / scale_factor
            
            # Normalize
            norm_x = (real_x + real_w / 2) / width
            norm_y = (real_y + real_h / 2) / height
            norm_w = real_w / width
            norm_h = real_h / height
            
            # Class 0 (car)
            yolo_labels.append(f"0 {norm_x:.6f} {norm_y:.6f} {norm_w:.6f} {norm_h:.6f}")
        
    if st.button("💾 Save Labels"):
        if yolo_labels:
            with open(label_path, "w") as f:
                f.write("\n".join(yolo_labels))
            st.success(f"Saved {len(yolo_labels)} labels to {os.path.basename(label_path)}")
            # Auto advance option?
            # st.session_state.image_index += 1
            # st.rerun()
        else:
            st.warning("No boxes drawn! If image is empty, saving will create an empty file (which is correct behavior for no cars).")
            # Create empty file
            with open(label_path, "w") as f:
                pass
            st.success("Saved as empty.")

