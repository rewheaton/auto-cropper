"""Person detection module using YOLOv8."""

import cv2
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from ultralytics import YOLO
from tqdm import tqdm
import logging


class PersonDetector:
    """Detects and tracks people in video frames using YOLOv8."""
    
    def __init__(self, model_name: str = 'yolov8l.pt', confidence: float = 0.5, verbose: bool = False):
        """
        Initialize the person detector.
        
        Args:
            model_name: YOLOv8 model to use ('yolov8n.pt', 'yolov8s.pt', etc.)
            confidence: Minimum confidence threshold for detections
            verbose: Enable verbose logging
        """
        self.model_name = model_name
        self.confidence = confidence
        self.verbose = verbose
        
        # Setup logging
        if verbose:
            logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Load YOLO model
        self.logger.info(f"Loading YOLO model: {model_name}")
        self.model = YOLO(model_name)
        
        # COCO class ID for person is 0
        self.person_class_id = 0
    
    def detect_people_in_video(
        self, 
        video_path: str, 
        output_dir: str = "./output"
    ) -> str:
        """
        Detect people in all frames of a video and save detection data.
        
        Args:
            video_path: Path to input video file
            output_dir: Directory to save detection data
            
        Returns:
            Path to the detection data JSON file
        """
        video_path_obj = Path(video_path)
        output_dir_obj = Path(output_dir)
        output_dir_obj.mkdir(exist_ok=True)
        
        # Create output filename
        detection_file = output_dir_obj / f"{video_path_obj.stem}_detections.json"
        
        self.logger.info(f"Processing video: {video_path_obj}")
        self.logger.info(f"Detection data will be saved to: {detection_file}")
        
        # Open video
        cap = cv2.VideoCapture(str(video_path_obj))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path_obj}")
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        video_info = {
            "video_path": str(video_path_obj),
            "total_frames": total_frames,
            "fps": fps,
            "width": width,
            "height": height,
            "detection_settings": {
                "model": self.model_name,
                "confidence": self.confidence
            }
        }
        
        # Store all detections
        detections_data = {
            "video_info": video_info,
            "frames": []
        }
        
        frame_number = 0
        progress_bar = tqdm(total=total_frames, desc="Detecting people", disable=not self.verbose)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect people in current frame
            people_detections = self._detect_people_in_frame(frame, frame_number)
            
            frame_data = {
                "frame_number": frame_number,
                "timestamp": frame_number / fps,
                "people": people_detections
            }
            
            detections_data["frames"].append(frame_data)
            
            frame_number += 1
            progress_bar.update(1)
        
        cap.release()
        progress_bar.close()
        
        # Save detection data to JSON file
        with open(detection_file, 'w') as f:
            json.dump(detections_data, f, indent=2)
        
        self.logger.info(f"Detection complete. Found people in {len([f for f in detections_data['frames'] if f['people']])} frames")
        self.logger.info(f"Detection data saved to: {detection_file}")
        
        return str(detection_file)
    
    def _detect_people_in_frame(self, frame: np.ndarray, frame_number: int) -> List[Dict]:
        """
        Detect people in a single frame.
        
        Args:
            frame: Video frame as numpy array
            frame_number: Frame number for reference
            
        Returns:
            List of person detections with bounding boxes and confidence
        """
        # Run YOLO detection
        results = self.model(frame, verbose=False)
        
        people_detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Check if detection is a person and meets confidence threshold
                    class_id = int(box.cls.item())
                    confidence = float(box.conf.item())
                    
                    if class_id == self.person_class_id and confidence >= self.confidence:
                        # Get bounding box coordinates (xyxy format)
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        
                        detection = {
                            "bbox": {
                                "x1": int(x1),
                                "y1": int(y1),
                                "x2": int(x2),
                                "y2": int(y2),
                                "width": int(x2 - x1),
                                "height": int(y2 - y1)
                            },
                            "confidence": confidence,
                            "center": {
                                "x": int((x1 + x2) / 2),
                                "y": int((y1 + y2) / 2)
                            }
                        }
                        
                        people_detections.append(detection)
        
        return people_detections
    
    def get_detection_summary(self, detection_file: str) -> Dict:
        """
        Get a summary of detections from a detection file.
        
        Args:
            detection_file: Path to detection JSON file
            
        Returns:
            Summary statistics about the detections
        """
        with open(detection_file, 'r') as f:
            data = json.load(f)
        
        total_frames = len(data["frames"])
        frames_with_people = len([f for f in data["frames"] if f["people"]])
        total_detections = sum(len(f["people"]) for f in data["frames"])
        
        # Calculate average people per frame
        avg_people_per_frame = total_detections / total_frames if total_frames > 0 else 0
        
        # Find frame with most people
        max_people_frame = max(data["frames"], key=lambda f: len(f["people"]), default=None)
        max_people_count = len(max_people_frame["people"]) if max_people_frame else 0
        
        summary = {
            "video_info": data["video_info"],
            "total_frames": total_frames,
            "frames_with_people": frames_with_people,
            "frames_without_people": total_frames - frames_with_people,
            "total_detections": total_detections,
            "average_people_per_frame": round(avg_people_per_frame, 2),
            "max_people_in_frame": max_people_count,
            "detection_coverage": round((frames_with_people / total_frames * 100), 2) if total_frames > 0 else 0
        }
        
        return summary
