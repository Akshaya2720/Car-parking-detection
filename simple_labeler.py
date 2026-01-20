import cv2
import os
import argparse
from glob import glob

def mouse_callback(event, x, y, flags, param):
    global drawing, ix, iy, current_img, temp_img
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            temp_img = current_img.copy()
            cv2.rectangle(temp_img, (ix, iy), (x, y), (0, 255, 0), 2)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.rectangle(current_img, (ix, iy), (x, y), (0, 255, 0), 2)
        
        # Save normalized coordinates
        h, w, _ = current_img.shape
        # Normalize
        x_center = ((ix + x) / 2) / w
        y_center = ((iy + y) / 2) / h
        width = abs(x - ix) / w
        height = abs(y - iy) / h
        
        # Append to labels
        # Class 0 = car (assuming user wants to label cars)
        current_labels.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        print(f"Recorded box: Class 0 (car)")

def label_images(source_dir):
    global current_img, temp_img, drawing, ix, iy, current_labels
    
    extensions = ['*.jpg', '*.jpeg', '*.png']
    files = []
    for ext in extensions:
        files.extend(glob(os.path.join(source_dir, '**', ext), recursive=True))
    
    print(f"Found {len(files)} images.")
    print("Controls:")
    print("  Draw box with mouse")
    print("  's' to SAVE and go to next image")
    print("  'd' to SKIP and go to next image")
    print("  'c' to CLEAR current boxes on image")
    print("  'q' to QUIT")
    
    cv2.namedWindow('Labeler')
    cv2.setMouseCallback('Labeler', mouse_callback)
    
    for file_path in files:
        # Skip if label already exists?
        label_path = os.path.splitext(file_path)[0] + ".txt"
        if os.path.exists(label_path):
            print(f"Skipping {os.path.basename(file_path)} (already labeled)")
            continue

        current_img = cv2.imread(file_path)
        if current_img is None:
            continue
            
        temp_img = current_img.copy()
        current_labels = []
        drawing = False
        ix, iy = -1, -1
        
        print(f"Labeling: {os.path.basename(file_path)}")
        
        while True:
            display_img = temp_img if drawing else current_img
            cv2.imshow('Labeler', display_img)
            k = cv2.waitKey(1) & 0xFF
            
            if k == ord('s'):
                if current_labels:
                    with open(label_path, 'w') as f:
                        f.write('\n'.join(current_labels))
                    print("Saved!")
                break
            elif k == ord('d'):
                print("Skipped.")
                break
            elif k == ord('c'):
                current_img = cv2.imread(file_path) # Reload to clear
                current_labels = []
                print("Cleared.")
            elif k == ord('q'):
                return

    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Hardcoded path to parked images for convenience
    DEFAULT_DIR = r"C:\Users\akshaya\Desktop\mini\dataset"
    label_images(DEFAULT_DIR)
