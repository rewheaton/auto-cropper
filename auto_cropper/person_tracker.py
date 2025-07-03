"""Person tracking module for selecting and tracking specific people from detection data."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple


class PersonTracker:
    """Tracks a specific person across frames using detection data."""
    
    def __init__(self, verbose: bool = False):
        """
        Initialize the person tracker.
        
        Args:
            verbose: Enable verbose logging
        """
        self.verbose = verbose
        if verbose:
            logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def select_person_to_track(self, detection_file: str) -> str:
        """
        Select which person to track and create tracking data using the most consistent method.
        
        Args:
            detection_file: Path to detection JSON file
            
        Returns:
            Path to tracking data JSON file
        """
        with open(detection_file, 'r') as f:
            detection_data = json.load(f)
        
        self.logger.info("Selecting most consistent person to track")
        
        tracking_data = self._select_most_consistent_person(detection_data)
        
        # Save tracking data
        detection_path = Path(detection_file)
        tracking_file = detection_path.parent / f"{detection_path.stem.replace('_detections', '')}_tracking.json"
        
        with open(tracking_file, 'w') as f:
            json.dump(tracking_data, f, indent=2)
        
        self.logger.info(f"Tracking data saved to: {tracking_file}")
        return str(tracking_file)
    
    def _select_most_consistent_person(self, detection_data: Dict) -> Dict:
        """Select the person who appears in the most frames."""
        # This is a simplified version - a more sophisticated tracker would use
        # IoU overlap and temporal consistency
        person_appearances = {}
        
        for frame in detection_data["frames"]:
            for i, person in enumerate(frame["people"]):
                center_key = f"{person['center']['x']//50}_{person['center']['y']//50}"
                
                if center_key not in person_appearances:
                    person_appearances[center_key] = []
                person_appearances[center_key].append((frame["frame_number"], i))
        
        if not person_appearances:
            return self._create_empty_tracking_data(detection_data)
        
        most_consistent_key = max(person_appearances.keys(), key=lambda k: len(person_appearances[k]))
        
        return self._create_tracking_data(detection_data, person_appearances[most_consistent_key])
    
    def _create_tracking_data(self, detection_data: Dict, person_frames: List[Tuple]) -> Dict:
        """Create tracking data for a selected person."""
        tracking_data = {
            "video_info": detection_data["video_info"],
            "tracking_info": {
                "total_tracked_frames": len(person_frames),
                "tracking_coverage": round(len(person_frames) / len(detection_data["frames"]) * 100, 2)
            },
            "tracked_person": []
        }
        
        # Get person data for each frame
        for frame_info in person_frames:
            if len(frame_info) == 3:  # (value, frame_number, person_index)
                _, frame_number, person_index = frame_info
            else:  # (frame_number, person_index)
                frame_number, person_index = frame_info
            
            frame_data = detection_data["frames"][frame_number]
            if person_index < len(frame_data["people"]):
                person_data = frame_data["people"][person_index]
                
                tracking_entry = {
                    "frame_number": frame_number,
                    "timestamp": frame_data["timestamp"],
                    "bbox": person_data["bbox"],
                    "center": person_data["center"],
                    "confidence": person_data["confidence"]
                }
                
                tracking_data["tracked_person"].append(tracking_entry)
        
        # Sort by frame number
        tracking_data["tracked_person"].sort(key=lambda x: x["frame_number"])
        
        return tracking_data
    
    def _create_empty_tracking_data(self, detection_data: Dict) -> Dict:
        """Create empty tracking data when no person is found."""
        return {
            "video_info": detection_data["video_info"],
            "tracking_info": {
                "total_tracked_frames": 0,
                "tracking_coverage": 0.0
            },
            "tracked_person": []
        }
