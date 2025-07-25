"""Tests for auto-cropper video processing functionality."""

import pytest
import tempfile
import json
import os
from pathlib import Path
import numpy as np
import cv2

from auto_cropper.video_cropper import VideoCropper


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield Path(tmpdirname)


@pytest.fixture
def sample_video(temp_dir):
    """Create a sample video for testing."""
    video_path = temp_dir / "test_video.mp4"
    
    # Create a simple test video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 30.0
    frame_size = (640, 480)
    
    out = cv2.VideoWriter(str(video_path), fourcc, fps, frame_size)
    
    # Create 30 frames (1 second of video)
    for i in range(30):
        # Create a frame with a moving rectangle (simulating a person)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:] = (50, 50, 50)  # Dark gray background
        
        # Moving rectangle position
        x = 100 + i * 5  # Move right over time
        y = 200
        
        # Draw a rectangle (simulating a person)
        cv2.rectangle(frame, (x, y), (x + 100, y + 200), (255, 255, 255), -1)
        
        out.write(frame)
    
    out.release()
    return video_path


@pytest.fixture
def sample_detection_data(temp_dir):
    """Create sample detection data."""
    detection_data = {
        "video_info": {
            "video_path": str(temp_dir / "test_video.mp4"),
            "total_frames": 30,
            "fps": 30.0,
            "width": 640,
            "height": 480,
            "detection_settings": {
                "model": "yolov8n.pt",
                "confidence": 0.5
            }
        },
        "frames": []
    }
    
    # Add detection data for each frame
    for i in range(30):
        x = 100 + i * 5  # Moving person
        y = 200
        
        frame_data = {
            "frame_number": i,
            "timestamp": i / 30.0,
            "people": [
                {
                    "bbox": {
                        "x1": x,
                        "y1": y,
                        "x2": x + 100,
                        "y2": y + 200,
                        "width": 100,
                        "height": 200
                    },
                    "confidence": 0.85,
                    "center": {
                        "x": x + 50,
                        "y": y + 100
                    }
                }
            ]
        }
        detection_data["frames"].append(frame_data)
    
    detection_file = temp_dir / "test_detections.json"
    with open(detection_file, 'w') as f:
        json.dump(detection_data, f)
    
    return detection_file



class TestVideoCropper:
    """Test cases for VideoCropper class."""
    
    def test_init_default(self):
        """Test default initialization."""
        cropper = VideoCropper()
        assert cropper.margin == 50
        assert cropper.smoothing_window == 10
        assert cropper.target_aspect_ratio == 16/9
    
    def test_init_custom(self):
        """Test custom initialization."""
        cropper = VideoCropper(margin=100, smoothing_window=10, verbose=True)
        assert cropper.margin == 100
        assert cropper.smoothing_window == 10
        assert cropper.verbose is True
    
    def test_calculate_output_dimensions(self):
        """Test output dimension calculation."""
        cropper = VideoCropper()
        width, height = cropper._calculate_output_dimensions(1920, 1080)
        
        # Should maintain 16:9 aspect ratio
        assert abs((width / height) - (16/9)) < 0.01
        
        # Dimensions should be even
        assert width % 2 == 0
        assert height % 2 == 0
        
        # Should be smaller than original
        assert width <= 1920
        assert height <= 1080
    
    def test_calculate_crop_position(self):
        """Test crop position calculation."""
        cropper = VideoCropper(margin=10)
        
        person_center = {"x": 320, "y": 240}
        bbox = {"width": 100, "height": 200}
        crop_width = 400
        crop_height = 300
        video_width = 640
        video_height = 480
        
        crop_x, crop_y = cropper._calculate_crop_position(
            person_center, bbox, crop_width, crop_height, video_width, video_height
        )
        
        # Crop position should be within video bounds
        assert 0 <= crop_x <= video_width - crop_width
        assert 0 <= crop_y <= video_height - crop_height
    
    def test_smooth_crop_positions(self):
        """Test crop position smoothing."""
        cropper = VideoCropper(smoothing_window=3)
        
        # Create positions with some jitter
        positions = [(100, 100), (105, 98), (98, 102), (102, 99), (101, 101)]
        
        smoothed = cropper._smooth_crop_positions(positions)
        
        assert len(smoothed) == len(positions)
        
        # Smoothed positions should be less extreme
        original_variance_x = np.var([p[0] for p in positions])
        smoothed_variance_x = np.var([p[0] for p in smoothed])
        
        assert smoothed_variance_x <= original_variance_x


if __name__ == '__main__':
    pytest.main([__file__])
