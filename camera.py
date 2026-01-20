import cv2
import time
from typing import Dict, Any, Generator, Optional, Union
import numpy as np

class CameraHandler:
    """
    Handles video input from various sources: ID (webcam), file path (video), or image path.
    Designed to yield frames efficiently.
    """
    def __init__(self, source: Union[int, str]):
        self.source = source
        self.is_image = False
        self.cap = None
        self.image = None
        
        if isinstance(source, str) and (source.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))):
            self.is_image = True
            self.image = cv2.imread(source)
            if self.image is None:
                raise ValueError(f"Could not load image from {source}")
        else:
            self.cap = cv2.VideoCapture(source)
            if not self.cap.isOpened():
                raise ValueError(f"Could not open video source {source}")

    def get_frame(self) -> Generator[np.ndarray, None, None]:
        """
        Yields frames from the source.
        If source is an image, yields the same image repeatedly (simulating a stream) 
        but handles control flow to not busy-loop too fast in a real app if needed, 
        or just yields it once depending on usage. 
        For this streaming app, we might want to just return it once per call or handle in UI.
        Here we simply read from the cap.
        """
        if self.is_image:
            # If it's an image, we just return it. 
            # In a loop context, the caller needs to decide when to stop.
            # But for compatibility with the loop structure, we can yield it.
            while True:
                yield self.image
                time.sleep(0.1) # Prevent CPU hogging
        else:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    break
                yield frame
                
    def read_once(self) -> Optional[np.ndarray]:
        """Read a single frame (useful for calibration or snapshot)"""
        if self.is_image:
            return self.image
        else:
            ret, frame = self.cap.read()
            return frame if ret else None

    def release(self):
        if self.cap:
            self.cap.release()
