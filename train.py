from ultralytics import YOLO
import argparse
import os

def train_model(data_path: str, epochs: int = 50, img_size: int = 640):
    """
    Train a YOLOv8 Nano model.
    """
    # Load a model
    model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

    # Train the model
    print(f"Starting training with data={data_path}, epochs={epochs}...")
    results = model.train(data=data_path, epochs=epochs, imgsz=img_size, plots=True, workers=0)
    
    # Validate the model
    # metrics = model.val() # It evaluates automatically after training usually
    
    # Export the model
    # success = model.export(format='onnx')
    print("Training finished.")
    print(f"Best model saved at {results.save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8n for Vehicle Detection")
    parser.add_argument("--data", type=str, required=True, help="Path to data.yaml file")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data):
        print(f"Error: Data file {args.data} not found.")
    else:
        train_model(args.data, args.epochs, args.imgsz)
