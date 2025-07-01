"""Video cropping module that uses tracking data to crop videos around a tracked person."""

import cv2
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import logging


class VideoCropper:
    """Crops videos based on person tracking data with 16:9 aspect ratio."""
    
    def __init__(self, margin: int = 50, smoothing_window: int = 10, verbose: bool = False):
        """
        Initialize the video cropper.
        
        Args:
            margin: Margin around the person in pixels
            smoothing_window: Number of frames to use for smoothing crop positions
            verbose: Enable verbose logging
        """
        self.margin = margin
        self.smoothing_window = smoothing_window
        self.verbose = verbose
        
        # Target aspect ratio (16:9)
        self.target_aspect_ratio = 16 / 9
        
        # Setup logging
        if verbose:
            logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def crop_video(
        self, 
        video_path: str, 
        tracking_file: str, 
        output_path: Optional[str] = None,
        output_dir: str = "./output",
        duration_limit: Optional[int] = None
    ) -> str:
        """
        Crop a video based on tracking data.
        
        Args:
            video_path: Path to input video file
            tracking_file: Path to tracking JSON file
            output_path: Specific output path (optional)
            output_dir: Output directory if output_path not specified
            duration_limit: Limit cropping to first N seconds (optional)
            
        Returns:
            Path to the cropped video file
        """
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Create output filename if not specified
        if output_path is None:
            output_path = output_dir / f"{video_path.stem}_cropped.mp4"
        else:
            output_path = Path(output_path)
        
        self.logger.info(f"Cropping video: {video_path}")
        self.logger.info(f"Using tracking data: {tracking_file}")
        self.logger.info(f"Output will be saved to: {output_path}")
        
        # Load tracking data
        with open(tracking_file, 'r') as f:
            tracking_data = json.load(f)
        
        # Check if we have tracking data
        if not tracking_data["tracked_person"]:
            raise ValueError("No tracking data found. Cannot crop video without person tracking.")
        
        # Open input video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.logger.info(f"Original video: {original_width}x{original_height}, {total_frames} frames at {fps} FPS")
        
        # Calculate maximum frames to process if duration limit is specified
        max_frames_to_process = total_frames
        if duration_limit is not None:
            max_frames_to_process = min(total_frames, int(duration_limit * fps))
            self.logger.info(f"Duration limit: {duration_limit}s, will process {max_frames_to_process} frames")
        
        # Calculate crop dimensions and positions
        crop_positions = self._calculate_crop_positions(tracking_data, original_width, original_height)
        
        # Determine output dimensions (16:9 aspect ratio)
        crop_width, crop_height = self._calculate_output_dimensions(original_width, original_height)
        
        self.logger.info(f"Output video will be: {crop_width}x{crop_height}")
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (crop_width, crop_height))
        
        frame_number = 0
        progress_bar = tqdm(total=max_frames_to_process, desc="Cropping video", disable=not self.verbose)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Check if we've reached the duration limit
            if frame_number >= max_frames_to_process:
                break
            
            # Get crop position for this frame
            crop_x, crop_y = self._get_crop_position_for_frame(frame_number, crop_positions)
            
            # Crop the frame
            cropped_frame = self._crop_frame(frame, crop_x, crop_y, crop_width, crop_height)
            
            # Write cropped frame
            out.write(cropped_frame)
            
            frame_number += 1
            progress_bar.update(1)
        
        cap.release()
        out.release()
        progress_bar.close()
        
        self.logger.info(f"Video cropping complete. Output saved to: {output_path}")
        return str(output_path)
    
    def _calculate_crop_positions(
        self, 
        tracking_data: Dict, 
        video_width: int, 
        video_height: int
    ) -> List[Tuple[int, int]]:
        """
        Calculate crop positions for each frame based on tracking data.
        
        Args:
            tracking_data: Person tracking data
            video_width: Original video width
            video_height: Original video height
            
        Returns:
            List of (x, y) crop positions for each frame
        """
        # Calculate output dimensions
        crop_width, crop_height = self._calculate_output_dimensions(video_width, video_height)
        
        # Create a list of crop positions for all frames
        total_frames = tracking_data["video_info"]["total_frames"]
        crop_positions = []
        
        # Create a mapping of frame numbers to tracking data
        tracking_map = {entry["frame_number"]: entry for entry in tracking_data["tracked_person"]}
        
        for frame_num in range(total_frames):
            if frame_num in tracking_map:
                # Use actual tracking data
                person_center = tracking_map[frame_num]["center"]
                bbox = tracking_map[frame_num]["bbox"]
                
                # Calculate crop position to center on person
                crop_x, crop_y = self._calculate_crop_position(
                    person_center, bbox, crop_width, crop_height, video_width, video_height
                )
            else:
                # Interpolate or use last known position
                crop_x, crop_y = self._interpolate_crop_position(
                    frame_num, tracking_map, crop_width, crop_height, video_width, video_height
                )
            
            crop_positions.append((crop_x, crop_y))
        
        # Apply smoothing to reduce jitter
        if self.smoothing_window > 1:
            crop_positions = self._smooth_crop_positions(crop_positions)
        
        return crop_positions
    
    def _calculate_output_dimensions(self, video_width: int, video_height: int) -> Tuple[int, int]:
        """
        Calculate output dimensions maintaining 16:9 aspect ratio.
        
        Args:
            video_width: Original video width
            video_height: Original video height
            
        Returns:
            (crop_width, crop_height) for 16:9 aspect ratio
        """
        # Start with a reasonable crop size (e.g., 70% of original)
        max_crop_width = int(video_width * 0.8)
        max_crop_height = int(video_height * 0.8)
        
        # Adjust to maintain 16:9 aspect ratio
        if max_crop_width / max_crop_height > self.target_aspect_ratio:
            # Width is too large, adjust based on height
            crop_height = max_crop_height
            crop_width = int(crop_height * self.target_aspect_ratio)
        else:
            # Height is too large, adjust based on width
            crop_width = max_crop_width
            crop_height = int(crop_width / self.target_aspect_ratio)
        
        # Ensure dimensions are even (required for some codecs)
        crop_width = crop_width - (crop_width % 2)
        crop_height = crop_height - (crop_height % 2)
        
        return crop_width, crop_height
    
    def _calculate_crop_position(
        self,
        person_center: Dict,
        bbox: Dict,
        crop_width: int,
        crop_height: int,
        video_width: int,
        video_height: int
    ) -> Tuple[int, int]:
        """
        Calculate crop position for a frame with person tracking data.
        
        Args:
            person_center: Person center coordinates
            bbox: Person bounding box
            crop_width: Target crop width
            crop_height: Target crop height
            video_width: Original video width
            video_height: Original video height
            
        Returns:
            (crop_x, crop_y) position
        """
        # Start with person center
        center_x = person_center["x"]
        center_y = person_center["y"]
        
        # Adjust for person size and margin
        person_width = bbox["width"]
        person_height = bbox["height"]
        
        # Add margin around the person
        required_width = person_width + (2 * self.margin)
        required_height = person_height + (2 * self.margin)
        
        # Ensure crop area can contain the person with margin
        if required_width > crop_width or required_height > crop_height:
            # If person is too large, just center on them
            crop_x = max(0, min(center_x - crop_width // 2, video_width - crop_width))
            crop_y = max(0, min(center_y - crop_height // 2, video_height - crop_height))
        else:
            # Center crop on person
            crop_x = max(0, min(center_x - crop_width // 2, video_width - crop_width))
            crop_y = max(0, min(center_y - crop_height // 2, video_height - crop_height))
        
        return crop_x, crop_y
    
    def _interpolate_crop_position(
        self,
        frame_num: int,
        tracking_map: Dict,
        crop_width: int,
        crop_height: int,
        video_width: int,
        video_height: int
    ) -> Tuple[int, int]:
        """
        Interpolate crop position for frames without tracking data.
        
        Args:
            frame_num: Current frame number
            tracking_map: Map of frame numbers to tracking data
            crop_width: Target crop width
            crop_height: Target crop height
            video_width: Original video width
            video_height: Original video height
            
        Returns:
            (crop_x, crop_y) position
        """
        if not tracking_map:
            # No tracking data at all, center the crop
            return (video_width - crop_width) // 2, (video_height - crop_height) // 2
        
        # Find nearest frames with tracking data
        frame_numbers = sorted(tracking_map.keys())
        
        if frame_num <= frame_numbers[0]:
            # Before first tracked frame, use first position
            first_entry = tracking_map[frame_numbers[0]]
            return self._calculate_crop_position(
                first_entry["center"], first_entry["bbox"],
                crop_width, crop_height, video_width, video_height
            )
        elif frame_num >= frame_numbers[-1]:
            # After last tracked frame, use last position
            last_entry = tracking_map[frame_numbers[-1]]
            return self._calculate_crop_position(
                last_entry["center"], last_entry["bbox"],
                crop_width, crop_height, video_width, video_height
            )
        else:
            # Interpolate between two frames
            prev_frame = max(f for f in frame_numbers if f < frame_num)
            next_frame = min(f for f in frame_numbers if f > frame_num)
            
            prev_entry = tracking_map[prev_frame]
            next_entry = tracking_map[next_frame]
            
            # Linear interpolation
            alpha = (frame_num - prev_frame) / (next_frame - prev_frame)
            
            interp_center = {
                "x": int(prev_entry["center"]["x"] * (1 - alpha) + next_entry["center"]["x"] * alpha),
                "y": int(prev_entry["center"]["y"] * (1 - alpha) + next_entry["center"]["y"] * alpha)
            }
            
            # Use average bbox size
            avg_bbox = {
                "width": (prev_entry["bbox"]["width"] + next_entry["bbox"]["width"]) // 2,
                "height": (prev_entry["bbox"]["height"] + next_entry["bbox"]["height"]) // 2
            }
            
            return self._calculate_crop_position(
                interp_center, avg_bbox,
                crop_width, crop_height, video_width, video_height
            )
    
    def _smooth_crop_positions(self, crop_positions: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Apply smoothing to crop positions to reduce jitter.
        
        Args:
            crop_positions: List of (x, y) crop positions
            
        Returns:
            Smoothed crop positions
        """
        if len(crop_positions) < self.smoothing_window:
            return crop_positions
        
        smoothed_positions = []
        half_window = self.smoothing_window // 2
        
        for i in range(len(crop_positions)):
            # Define window bounds
            start_idx = max(0, i - half_window)
            end_idx = min(len(crop_positions), i + half_window + 1)
            
            # Calculate average position in window
            window_positions = crop_positions[start_idx:end_idx]
            avg_x = sum(pos[0] for pos in window_positions) // len(window_positions)
            avg_y = sum(pos[1] for pos in window_positions) // len(window_positions)
            
            smoothed_positions.append((avg_x, avg_y))
        
        return smoothed_positions
    
    def _get_crop_position_for_frame(
        self, 
        frame_number: int, 
        crop_positions: List[Tuple[int, int]]
    ) -> Tuple[int, int]:
        """
        Get crop position for a specific frame.
        
        Args:
            frame_number: Frame number
            crop_positions: List of crop positions
            
        Returns:
            (crop_x, crop_y) for the frame
        """
        if frame_number < len(crop_positions):
            return crop_positions[frame_number]
        else:
            # Use last available position
            return crop_positions[-1] if crop_positions else (0, 0)
    
    def _crop_frame(
        self, 
        frame: np.ndarray, 
        crop_x: int, 
        crop_y: int, 
        crop_width: int, 
        crop_height: int
    ) -> np.ndarray:
        """
        Crop a single frame.
        
        Args:
            frame: Input frame
            crop_x: Crop X position
            crop_y: Crop Y position
            crop_width: Crop width
            crop_height: Crop height
            
        Returns:
            Cropped frame
        """
        # Ensure crop coordinates are within frame bounds
        frame_height, frame_width = frame.shape[:2]
        
        crop_x = max(0, min(crop_x, frame_width - crop_width))
        crop_y = max(0, min(crop_y, frame_height - crop_height))
        
        # Crop the frame
        cropped = frame[crop_y:crop_y + crop_height, crop_x:crop_x + crop_width]
        
        # If cropped frame is smaller than expected (edge case), pad it
        if cropped.shape[0] != crop_height or cropped.shape[1] != crop_width:
            padded = np.zeros((crop_height, crop_width, 3), dtype=np.uint8)
            actual_height, actual_width = cropped.shape[:2]
            padded[:actual_height, :actual_width] = cropped
            return padded
        
        return cropped
