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


