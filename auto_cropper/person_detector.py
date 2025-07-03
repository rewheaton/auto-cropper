"""Person detection module using YOLOv8."""

import cv2
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from auto_cropper.memory_monitor import MemoryMonitor
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
    
    def detect_people_in_video_chunked(
        self,
        video_path: str,
        output_dir: str = "./output",
        chunk_size_frames: Optional[int] = None,
        overlap_frames: int = 0,
        memory_monitor: Optional['MemoryMonitor'] = None,
        checkpoint_interval: int = 1000
    ) -> str:
        """
        Detect people in video using chunked processing for memory efficiency.
        
        Args:
            video_path: Path to input video file
            output_dir: Directory to save detection data
            chunk_size_frames: Number of frames per chunk (auto-calculated if None)
            overlap_frames: Number of overlapping frames between chunks
            memory_monitor: Optional memory monitor for tracking usage
            checkpoint_interval: Save progress every N frames
            
        Returns:
            Path to the detection data JSON file
            
        Raises:
            ValueError: If chunk parameters are invalid
            MemoryLimitException: If memory limit is exceeded
        """
        from auto_cropper.memory_monitor import MemoryMonitor
        from auto_cropper.streaming_json_writer import StreamingJSONWriter
        from auto_cropper.exceptions import MemoryLimitException
        
        # Validate parameters
        if chunk_size_frames is not None and chunk_size_frames <= 0:
            raise ValueError("Chunk size must be positive")
        if overlap_frames < 0:
            raise ValueError("Overlap frames must be non-negative")
        if chunk_size_frames is not None and overlap_frames >= chunk_size_frames:
            raise ValueError("Overlap cannot be larger than chunk size")
        
        video_path_obj = Path(video_path)
        output_dir_obj = Path(output_dir)
        output_dir_obj.mkdir(exist_ok=True)
        
        detection_file = output_dir_obj / f"{video_path_obj.stem}_detections.json"
        
        self.logger.info(f"Processing video with chunked detection: {video_path_obj}")
        
        # Open video and get properties
        cap = cv2.VideoCapture(str(video_path_obj))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path_obj}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create or use provided memory monitor
        if memory_monitor is None:
            memory_monitor = MemoryMonitor()
        
        # Calculate optimal chunk size if not provided
        if chunk_size_frames is None:
            chunk_size_frames = self._get_optimal_chunk_size(memory_monitor, width, height)
        
        self.logger.info(f"Using chunk size: {chunk_size_frames} frames")
        
        # Prepare video info
        video_info = {
            "video_path": str(video_path_obj),
            "total_frames": total_frames,
            "fps": fps,
            "width": width,
            "height": height,
            "detection_settings": {
                "model": self.model_name,
                "confidence": self.confidence,
                "chunked_processing": True,
                "chunk_size": chunk_size_frames,
                "overlap_frames": overlap_frames
            }
        }
        
        # Process video in chunks using streaming writer
        try:
            with StreamingJSONWriter(str(detection_file), video_info) as writer:
                current_frame = 0
                progress_bar = tqdm(total=total_frames, desc="Detecting people (chunked)", disable=not self.verbose)
                
                while current_frame < total_frames:
                    # Check memory usage
                    if not memory_monitor.check_memory_usage():
                        current_usage = memory_monitor.get_current_memory_usage()
                        raise MemoryLimitException(
                            current_usage_mb=current_usage,
                            max_memory_mb=memory_monitor.max_memory_mb,
                            message=f"Memory limit exceeded at frame {current_frame}. "
                                   f"Current usage: {current_usage:.1f}MB, "
                                   f"Limit: {memory_monitor.max_memory_mb}MB"
                        )
                    
                    # Calculate chunk boundaries
                    chunk_start = current_frame
                    chunk_end = min(current_frame + chunk_size_frames, total_frames)
                    
                    # Process chunk
                    chunk_frames = self._process_chunk(cap, chunk_start, chunk_end, current_frame, fps)
                    
                    # Write all frames from this chunk
                    for frame_data in chunk_frames:
                        writer.write_frame(frame_data)
                        progress_bar.update(1)
                    
                    # Update current frame position (accounting for overlap)
                    current_frame = chunk_end - overlap_frames if overlap_frames > 0 and chunk_end < total_frames else chunk_end
                    
                    # Cleanup GPU memory if available
                    try:
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except:
                        pass
                    
                    # Checkpoint if needed
                    if current_frame % checkpoint_interval == 0 and current_frame < total_frames:
                        self.logger.info(f"Checkpoint: processed {current_frame}/{total_frames} frames")
                
                progress_bar.close()
                
        finally:
            cap.release()
        
        frames_with_people = 0
        total_detections = 0
        
        # Read back the file to count detections (could be optimized)
        try:
            with open(detection_file, 'r') as f:
                data = json.load(f)
            frames_with_people = len([f for f in data["frames"] if f["people"]])
            total_detections = sum(len(f["people"]) for f in data["frames"])
        except:
            # If we can't read back for summary, that's ok
            pass
        
        self.logger.info(f"Chunked detection complete. Processed {total_frames} frames")
        self.logger.info(f"Frames with people: {frames_with_people}, Total detections: {total_detections}")
        self.logger.info(f"Detection data saved to: {detection_file}")
        
        return str(detection_file)
    
    def _get_optimal_chunk_size(self, memory_monitor: 'MemoryMonitor', width: int, height: int) -> int:
        """
        Calculate optimal chunk size based on available memory and frame size.
        
        Args:
            memory_monitor: Memory monitor instance
            width: Video frame width
            height: Video frame height
            
        Returns:
            Optimal chunk size in frames
        """
        frame_size_mb = self._estimate_frame_size_mb(width, height)
        base_chunk_size = memory_monitor.get_recommended_batch_size(frame_size_mb)
        
        # Apply safety factor and reasonable bounds
        safety_factor = 0.8  # Use 80% of recommended size for safety
        chunk_size = max(1, int(base_chunk_size * safety_factor))
        
        # Set reasonable bounds
        min_chunk_size = 10
        max_chunk_size = 1000
        
        return max(min_chunk_size, min(chunk_size, max_chunk_size))
    
    def _estimate_frame_size_mb(self, width: int, height: int) -> float:
        """
        Estimate memory usage per frame in MB.
        
        Args:
            width: Frame width in pixels
            height: Frame height in pixels
            
        Returns:
            Estimated frame size in MB
        """
        # Estimate based on RGB frame (3 bytes per pixel) plus processing overhead
        pixels = width * height
        rgb_size = pixels * 3  # 3 bytes per pixel for RGB
        
        # Add overhead for YOLO processing (roughly 2-3x the frame size)
        processing_overhead = 2.5
        total_bytes = rgb_size * processing_overhead
        
        # Convert to MB
        return total_bytes / (1024 * 1024)
    
    def _process_chunk(self, cap, chunk_start: int, chunk_end: int, 
                      absolute_start: int, fps: float) -> List[Dict]:
        """
        Process a chunk of video frames.
        
        Args:
            cap: OpenCV VideoCapture object
            chunk_start: Starting frame number for chunk
            chunk_end: Ending frame number for chunk  
            absolute_start: Absolute starting frame number for output
            fps: Video frame rate
            
        Returns:
            List of frame detection data
        """
        chunk_frames = []
        
        # Seek to chunk start
        cap.set(cv2.CAP_PROP_POS_FRAMES, chunk_start)
        
        current_frame = chunk_start
        while current_frame < chunk_end:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process all frames in the chunk
            people_detections = self._detect_people_in_frame(frame, current_frame)
            
            frame_data = {
                "frame_number": current_frame,
                "timestamp": current_frame / fps,
                "people": people_detections
            }
            
            chunk_frames.append(frame_data)
            current_frame += 1
        
        return chunk_frames
