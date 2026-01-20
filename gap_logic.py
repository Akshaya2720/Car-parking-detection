from typing import List, Dict, Any, Tuple

class ParkingGapAnalyzer:
    """
    Analyzes detections to find gaps suitable for parking.
    Assumes a somewhat linear parking arrangement for the simplistic geometric gap logic.
    """
    def __init__(self, min_gap_width: int):
        self.min_gap_width = min_gap_width

    def analyze_availability(self, detections: List[Dict[str, Any]], frame_width: int) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Determines if there is a parking slot available.
        
        Args:
            detections: List of detection dicts {'box': [x1, y1, x2, y2], ...}
            frame_width: Width of the frame (to check edges if needed)
            
        Returns:
            (is_available: bool, gaps: List[Dict])
            gaps is a list of dicts describing the open spaces found: {'start': x, 'end': x, 'width': w}
        """
        if not detections:
            # No cars detected -> Whole space is available
            return True, [{'start': 0, 'end': frame_width, 'width': frame_width}]

        # Sort detections by their left x-coordinate (x1)
        # We focus on horizontal linear parking for this MVP
        sorted_dets = sorted(detections, key=lambda d: d['box'][0])
        
        gaps = []
        
        # Check gap before the first car
        first_car_x1 = sorted_dets[0]['box'][0]
        if first_car_x1 > self.min_gap_width:
            gaps.append({'start': 0, 'end': first_car_x1, 'width': first_car_x1})

        # Check gaps between cars
        for i in range(len(sorted_dets) - 1):
            car1_x2 = sorted_dets[i]['box'][2]
            car2_x1 = sorted_dets[i+1]['box'][0]
            
            gap_width = car2_x1 - car1_x2
            
            if gap_width >= self.min_gap_width:
                gaps.append({
                    'start': car1_x2,
                    'end': car2_x1,
                    'width': gap_width
                })
                
        # Check gap after the last car
        last_car_x2 = sorted_dets[-1]['box'][2]
        remaining_space = frame_width - last_car_x2
        if remaining_space >= self.min_gap_width:
            gaps.append({'start': last_car_x2, 'end': frame_width, 'width': remaining_space})
            
        is_available = len(gaps) > 0
        return is_available, gaps
