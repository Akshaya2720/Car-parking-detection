import os
import shutil
from ultralytics import YOLO
import cv2
from pathlib import Path
from tqdm import tqdm

def prepare_dataset(source_root, target_root):
    """
    Converts a classification-style dataset to YOLO detection format.
    Structure expected:
    source_root/
        Normal/
            parked/ (images with cars)
            empty/  (images without cars)
        Rainy/
            ...
    
    Target:
    target_root/
        images/train
        labels/train
    """
    
    # Initialize auto-labeler (using the base model)
    model = YOLO('yolov8n.pt') 
    
    # Create target directories
    for split in ['train', 'val']:
        os.makedirs(os.path.join(target_root, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(target_root, 'labels', split), exist_ok=True)

    # Collect all image paths
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    all_images = []
    
    source_path = Path(source_root)
    if not source_path.exists():
        print(f"Error: Source path {source_root} does not exist.")
        return

    print("Scanning dataset...")
    for file_path in source_path.rglob('*'):
        if file_path.suffix.lower() in image_extensions:
            all_images.append(file_path)

    print(f"Found {len(all_images)} images. Processing...")

    # We will put everything in 'train' for now, user can split manually if desired or we can do random split
    # For simplicity, let's put 80% in train, 20% in val
    import random
    random.shuffle(all_images)
    split_idx = int(len(all_images) * 0.8)
    
    # Process
    for i, img_path in enumerate(tqdm(all_images)):
        subset = 'train' if i < split_idx else 'val'
        
        # Determine if it's likely "empty" or "parked" based on parent folder name
        # User said structure is "include empty and parked folders"
        parent_name = img_path.parent.name.lower()
        is_empty = 'empty' in parent_name
        
        target_img_dir = os.path.join(target_root, 'images', subset)
        target_lbl_dir = os.path.join(target_root, 'labels', subset)
        
        # Copy image
        # Use a unique name to avoid collisions from different subfolders
        unique_name = f"{img_path.parent.parent.name}_{img_path.parent.name}_{img_path.name}"
        unique_name = unique_name.replace(" ", "_").lower()
        shutil.copy(img_path, os.path.join(target_img_dir, unique_name))
        
        label_file = os.path.splitext(unique_name)[0] + ".txt"
        label_path = os.path.join(target_lbl_dir, label_file)
        
        # Check for manual label file (same basename, .txt extension)
        manual_label_src = img_path.with_suffix('.txt')
        
        if manual_label_src.exists():
            # If user manually labeled it, respect that!
            shutil.copy(manual_label_src, label_path)
            
        elif is_empty:
            # Create label for Empty Slot (Class 1)
            # Assuming the image IS the slot (patch-based), we label the whole image.
            with open(label_path, 'w') as f:
                # class x_c y_c w h -> 1 0.5 0.5 1.0 1.0
                f.write("1 0.5 0.5 1.0 1.0\n")
        else:
            # "Parked" folder - we need valid bounding boxes.
            # Assuming user DOES NOT have labels, we use the model to AUTO-LABEL.
            # This is a starting point ("Pseudo-labeling").
            results = model(img_path, verbose=False)
            
            with open(label_path, 'w') as f:
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        cls_id = int(box.cls[0])
                        # Check standard vehicle classes (car, motorcycle, bus, truck)
                        if cls_id in [2, 3, 5, 7]: 
                            # Convert to YOLO format (normalized x_center, y_center, width, height)
                            # box.xywhn returns [x_c, y_c, w, h] normalized
                            x, y, w, h = box.xywhn[0].tolist()
                            
                            # We map all vehicles to class 0 for our custom single-class training if desired,
                            # OR we keep them as is. 
                            # If data.yaml has "0: car", we typically map all vehicles to 0.
                            f.write(f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

    print(f"\nDataset prepared at {target_root}")
    print("IMPORTANT: The 'parked' images were auto-labeled using the standard YOLOv8n model.")
    print("This is good for domain adaptation (teaching the model your specific camera angles and lighting).")
    print("However, if the standard model completely fails to see cars in your 'night' or 'rainy' images,")
    print("the labels will be missing! Inspect the 'labels' folder or use a tool like LabelImg to verify.")

if __name__ == "__main__":
    # Hardcoded paths based on user conversation
    SOURCE = r"C:\Users\akshaya\Desktop\mini\dataset"
    TARGET = r"C:\Users\akshaya\Desktop\mini\formatted_dataset"
    
    prepare_dataset(SOURCE, TARGET)
