"""Person cropping functionality with real-time tracking."""

import cv2
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
import logging
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm


class PersonCropper:
    """Person tracking and video cropping functionality."""
    
    def __init__(
        self,
        confidence_threshold: float = 0.5,
        tracking_threshold: float = 0.7,
        margin_ratio: float = 0.1,
        smoothing_factor: float = 0.9,
        verbose: bool = False
    ):
        """
        Initialize the PersonCropper.
        
        Args:
            confidence_threshold: Minimum confidence for person detection
            tracking_threshold: Minimum IoU for tracking continuity
            margin_ratio: Additional margin around person as ratio of crop size
            smoothing_factor: Smoothing factor for crop box transitions (0-1)
            verbose: Enable verbose logging
        """
        self.confidence_threshold = confidence_threshold
        self.tracking_threshold = tracking_threshold
        self.margin_ratio = margin_ratio
        self.smoothing_factor = smoothing_factor
        self.verbose = verbose
        
        # Set up logging
        if verbose:
            logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize YOLO model
        try:
            self.model = YOLO('yolov8l.pt')  # Will download if not present
            if verbose:
                self.logger.info("YOLO model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load YOLO model: {e}")
            raise
        
        # Tracking state
        self.current_person_id = None
        self.last_crop_box = None
        self.person_history = []
    
    def crop_video(
        self,
        input_path: str,
        output_path: str,
        target_width: int = 1920,
        target_height: int = 1080
    ) -> bool:
        """
        Crop video to follow a person with 16:9 aspect ratio.
        
        Args:
            input_path: Path to input video file
            output_path: Path to save cropped video
            target_width: Width of output video
            target_height: Height of output video
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Open input video
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                self.logger.error(f"Could not open video file: {input_path}")
                return False
            
            # Get video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            if self.verbose:
                self.logger.info(f"Input video: {original_width}x{original_height}, {fps} FPS, {total_frames} frames")
            
            # Calculate aspect ratio
            aspect_ratio = target_width / target_height
            
            # Set up output video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (target_width, target_height))
            
            # Process frames
            frame_count = 0
            progress_bar = tqdm(total=total_frames, desc="Processing frames") if self.verbose else None
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Detect and track person
                person_box = self._detect_person(frame)
                
                if person_box is not None:
                    # Calculate crop region
                    crop_box = self._calculate_crop_box(
                        person_box, frame.shape, aspect_ratio
                    )
                    
                    # Apply smoothing
                    if self.last_crop_box is not None:
                        crop_box = self._smooth_crop_box(crop_box, self.last_crop_box)
                    
                    self.last_crop_box = crop_box
                else:
                    # Use last known position if person not detected
                    if self.last_crop_box is not None:
                        crop_box = self.last_crop_box
                    else:
                        # Default to center crop
                        crop_box = self._get_center_crop(frame.shape, aspect_ratio)
                
                # Crop and resize frame
                cropped_frame = self._crop_frame(frame, crop_box, target_width, target_height)
                
                # Write frame
                out.write(cropped_frame)
                
                frame_count += 1
                if progress_bar:
                    progress_bar.update(1)
            
            # Cleanup
            cap.release()
            out.release()
            if progress_bar:
                progress_bar.close()
            
            if self.verbose:
                self.logger.info(f"Successfully processed {frame_count} frames")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing video: {e}")
            return False
    
    def _detect_person(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect the best person in the frame.
        
        Returns:
            (x1, y1, x2, y2) bounding box or None if no person detected
        """
        try:
            # Run YOLO detection
            results = self.model(frame, verbose=False)
            
            if len(results) == 0:
                return None
            
            # Filter for person class (class 0 in COCO dataset)
            person_detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for i, cls in enumerate(boxes.cls):
                        if int(cls) == 0:  # Person class
                            confidence = float(boxes.conf[i])
                            if confidence >= self.confidence_threshold:
                                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                                person_detections.append({
                                    'box': (int(x1), int(y1), int(x2), int(y2)),
                                    'confidence': confidence,
                                    'area': (x2 - x1) * (y2 - y1)
                                })
            
            if not person_detections:
                return None
            
            # Select the best person to track
            if self.current_person_id is None:
                # Choose largest person with highest confidence
                best_person = max(
                    person_detections,
                    key=lambda p: p['confidence'] * p['area']
                )
                self.current_person_id = 0  # Simple ID assignment
                return best_person['box']
            else:
                # Try to maintain tracking continuity
                if self.last_crop_box is not None:
                    best_match = self._find_best_match(person_detections, self.last_crop_box)
                    if best_match:
                        return best_match['box']
                
                # Fall back to largest person
                if person_detections:
                    best_person = max(person_detections, key=lambda p: p['area'])
                    return best_person['box']
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Person detection failed: {e}")
            return None
    
    def _find_best_match(
        self,
        detections: List[Dict],
        last_box: Tuple[int, int, int, int]
    ) -> Optional[Dict]:
        """Find the detection that best matches the previous tracking box."""
        best_match = None
        best_iou = 0
        
        for detection in detections:
            iou = self._calculate_iou(detection['box'], last_box)
            if iou > best_iou and iou >= self.tracking_threshold:
                best_iou = iou
                best_match = detection
        
        return best_match
    
    def _calculate_iou(
        self,
        box1: Tuple[int, int, int, int],
        box2: Tuple[int, int, int, int]
    ) -> float:
        """Calculate Intersection over Union of two boxes."""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_crop_box(
        self,
        person_box: Tuple[int, int, int, int],
        frame_shape: Tuple[int, int, int],
        aspect_ratio: float
    ) -> Tuple[int, int, int, int]:
        """Calculate the crop box centered on the person with target aspect ratio."""
        height, width = frame_shape[:2]
        x1, y1, x2, y2 = person_box
        
        # Person center and dimensions
        person_center_x = (x1 + x2) // 2
        person_center_y = (y1 + y2) // 2
        person_width = x2 - x1
        person_height = y2 - y1
        
        # Calculate crop dimensions with margin
        margin_x = int(person_width * self.margin_ratio)
        margin_y = int(person_height * self.margin_ratio)
        
        content_width = person_width + 2 * margin_x
        content_height = person_height + 2 * margin_y
        
        # Adjust for target aspect ratio
        if content_width / content_height > aspect_ratio:
            # Content is too wide, increase height
            target_height = int(content_width / aspect_ratio)
            target_width = content_width
        else:
            # Content is too tall, increase width
            target_width = int(content_height * aspect_ratio)
            target_height = content_height
        
        # Ensure minimum size
        min_width = width // 3
        min_height = height // 3
        target_width = max(target_width, min_width)
        target_height = max(target_height, min_height)
        
        # Calculate crop coordinates
        crop_x1 = max(0, person_center_x - target_width // 2)
        crop_y1 = max(0, person_center_y - target_height // 2)
        crop_x2 = min(width, crop_x1 + target_width)
        crop_y2 = min(height, crop_y1 + target_height)
        
        # Adjust if crop exceeds boundaries
        if crop_x2 - crop_x1 < target_width:
            crop_x1 = max(0, crop_x2 - target_width)
        if crop_y2 - crop_y1 < target_height:
            crop_y1 = max(0, crop_y2 - target_height)
        
        return (crop_x1, crop_y1, crop_x2, crop_y2)
    
    def _get_center_crop(
        self,
        frame_shape: Tuple[int, int, int],
        aspect_ratio: float
    ) -> Tuple[int, int, int, int]:
        """Get a center crop with the target aspect ratio."""
        height, width = frame_shape[:2]
        
        if width / height > aspect_ratio:
            # Frame is too wide
            crop_width = int(height * aspect_ratio)
            crop_height = height
            crop_x1 = (width - crop_width) // 2
            crop_y1 = 0
        else:
            # Frame is too tall
            crop_width = width
            crop_height = int(width / aspect_ratio)
            crop_x1 = 0
            crop_y1 = (height - crop_height) // 2
        
        return (crop_x1, crop_y1, crop_x1 + crop_width, crop_y1 + crop_height)
    
    def _smooth_crop_box(
        self,
        current_box: Tuple[int, int, int, int],
        last_box: Tuple[int, int, int, int]
    ) -> Tuple[int, int, int, int]:
        """Apply smoothing to reduce jitter in crop box movement."""
        factor = self.smoothing_factor
        
        x1 = int(last_box[0] * factor + current_box[0] * (1 - factor))
        y1 = int(last_box[1] * factor + current_box[1] * (1 - factor))
        x2 = int(last_box[2] * factor + current_box[2] * (1 - factor))
        y2 = int(last_box[3] * factor + current_box[3] * (1 - factor))
        
        return (x1, y1, x2, y2)
    
    def _crop_frame(
        self,
        frame: np.ndarray,
        crop_box: Tuple[int, int, int, int],
        target_width: int,
        target_height: int
    ) -> np.ndarray:
        """Crop and resize frame to target dimensions."""
        x1, y1, x2, y2 = crop_box
        
        # Crop the frame
        cropped = frame[y1:y2, x1:x2]
        
        # Resize to target dimensions
        resized = cv2.resize(cropped, (target_width, target_height))
        
        return resized
