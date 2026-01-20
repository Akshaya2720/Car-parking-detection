from ultralytics import YOLO
import numpy as np
from typing import List, Dict, Any

class ObjectDetector:
    """
    Wrapper for YOLOv8 model to detect vehicles.
    """
    def __init__(self, model_path: str = "yolov8n.pt", conf_threshold: float = 0.25):
        # Initialize YOLO model. This will download the model if not present.
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        # COCO classes for vehicles (car, motorcycle, bus, truck)
        # COCO classes: 2=car, 3=motorcycle, 5=bus, 7=truck
        # Custom trained classes: 0=car, 1=empty
        self.vehicle_classes = [0, 1, 2, 3, 5, 7] 

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Run inference on a frame.
        Returns a list of detections: {'box': [x1, y1, x2, y2], 'class': int, 'conf': float}
        """
        results = self.model(frame, conf=self.conf_threshold, verbose=False)
        detections = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                if cls_id in self.vehicle_classes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    detections.append({
                        'box': [x1, y1, x2, y2],
                        'class': cls_id,
                        'conf': conf,
                        'name': self.model.names[cls_id]
                    })
        return detections
        return detections

class RoboflowWrapper:
    """
    Wrapper for Roboflow Inference API.
    """
    def __init__(self, api_key: str, workspace: str, project: str, version: int):
        from roboflow import Roboflow
        self.rf = Roboflow(api_key=api_key)
        self.project = self.rf.workspace(workspace).project(project)
        self.model = self.project.version(version).model

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Run inference on a frame using Roboflow API.
        """
        # Save frame to temp file because Roboflow SDK prefers file paths or hosted URLs
        # (Though it can handle numpy arrays in some versions, temp file is safest for SDK)
        import tempfile
        import cv2
        import os
        
        # Create temp file
        fd, temp_path = tempfile.mkstemp(suffix=".jpg")
        os.close(fd)
        cv2.imwrite(temp_path, frame)
        
        try:
            # Predict
            prediction = self.model.predict(temp_path, confidence=40, overlap=30).json()
            
            detections = []
            for pred in prediction['predictions']:
                # Roboflow returns x, y (center), width, height
                x = pred['x']
                y = pred['y']
                w = pred['width']
                h = pred['height']
                
                # Convert to x1, y1, x2, y2
                x1 = x - w / 2
                y1 = y - h / 2
                x2 = x + w / 2
                y2 = y + h / 2
                
                cls_name = pred['class']
                conf = pred['confidence']
                
                # Map class name to ID if needed, or just use hash/arbitrary
                # For this app we mainly check if it's "car" or "empty"
                cls_id = 0 if cls_name == 'car' else 1
                
                detections.append({
                    'box': [x1, y1, x2, y2],
                    'class': cls_id,
                    'conf': conf,
                    'name': cls_name
                })
        finally:
            # Cleanup
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
        return detections
